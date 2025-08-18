#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from picamera2 import Picamera2
import threading
import time
import numpy as np
import cv2
from config import *  # falls du dort später etwas ablegen willst

# =========================================================
# Kalibrierung & Fokus-Scoring (gewebe-aware)
# =========================================================
# Trage hier deinen gemessenen Wert ein (aus /measure_tool):
UM_PER_PX = 1.0            # µm pro Pixel (z.B. 1.0)

# Typischer Kerndurchmesser in H&E (µm)
NUCLEUS_DIAM_UM = 8.0

# Aus Kalibrierung abgeleitete Pixelwerte (für den Bandpass)
NUCLEUS_DIAM_PX = max(1.0, NUCLEUS_DIAM_UM / UM_PER_PX)
DOG_SIGMA1_PX   = max(0.6, NUCLEUS_DIAM_PX / 6.0)
DOG_SIGMA2_PX   = max(DOG_SIGMA1_PX + 0.4, NUCLEUS_DIAM_PX / 2.0)

# Gewebe-/Farbmasken-Parameter
TISSUE_S_MIN   = 0.20   # HSV-Sättigungsschwelle (0.15–0.25 testen)
TISSUE_V_MAX   = 0.97   # sehr helle Highlights raus
NUCLEI_WEIGHT  = 0.35   # Boost für bläuliche Kerne (0.2–0.6)
COVERAGE_ALPHA = 0.7    # Score-Skalierung nach Tissue-Deckung (0.5–1.0)

# Logging
VERBOSE_AF = True
def _log(msg: str):
    if VERBOSE_AF:
        print(msg)

# Sicherheitsgrenzen Z (an deine Bühne anpassen)
Z_MIN_UM = -10000
Z_MAX_UM = +10000

def clamp_z(z_um: float) -> float:
    """Begrenzt Z-Wert auf sicheren Bereich."""
    return max(Z_MIN_UM, min(Z_MAX_UM, z_um))


# ---------------- intern: Scoring-Helfer ----------------
def _tissue_mask_bgr(img_bgr):
    """Maske für Gewebe: genügend Sättigung und nicht zu hell."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    m = (S > TISSUE_S_MIN) & (V < TISSUE_V_MAX)
    mask = (m.astype(np.uint8)) * 255
    cov = float(np.mean(m))
    return mask, cov

def _nuclei_proxy_bgr(img_bgr):
    """Einfacher Proxy für Hematoxylin (bläulich/violett) + Sättigung."""
    EPS = 1e-6
    b, g, r = cv2.split(img_bgr.astype(np.float32) + EPS)
    bluish = np.clip(b / (0.5 * (r + g)), 0, 4.0) ** 0.8
    s = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    return bluish * (0.5 + 0.5 * s)

def _dog_bandpass(gray32, s1=DOG_SIGMA1_PX, s2=DOG_SIGMA2_PX):
    """Difference-of-Gaussians (Bandpass) im Zellmaßstab."""
    g1 = cv2.GaussianBlur(gray32, (0, 0), s1)
    g2 = cv2.GaussianBlur(gray32, (0, 0), s2)
    return cv2.absdiff(g1, g2)


# =========================================================
# Kamera & Autofokus
# =========================================================
class CameraController:
    def __init__(self):
        try:
            self.picam2 = Picamera2()
            # Variante: RGB888 aus Kamera, danach zu BGR (OpenCV) konvertieren
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"AwbEnable": 1, "AwbMode": 1}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.lock = threading.Lock()
            time.sleep(0.5)  # kurz stabilisieren lassen
            self._lock_exposure_awb()
            print("✅ Kamera erfolgreich gestartet & Belichtung fixiert!")
        except Exception as e:
            print("❌ Kamera konnte nicht gestartet werden:", e)
            self.picam2 = None

    def _lock_exposure_awb(self):
        """Belichtung/Weißabgleich fixieren → stabile Fokuswerte."""
        if not self.picam2:
            return
        meta = self.picam2.capture_metadata() or {}
        self.picam2.set_controls({
            "AeEnable": 0,
            "AwbEnable": 0,
            "ExposureTime": meta.get("ExposureTime", 10000),
            "AnalogueGain": meta.get("AnalogueGain", 1.0),
            "ColourGains": meta.get("ColourGains", (1.0, 1.0)),
            "NoiseReductionMode": 0,   # keine Glättung
            "Sharpness": 0.0,          # keine künstliche Schärfung
        })

    # ---------------- Scoring (gewebe-aware) ----------------
    def score_combined(self, image_bgr):
        """Fokus-Score, der Gewebe/kerne gewichtet und Glas/Halos abwertet."""
        if image_bgr is None:
            return 0.0
        mask_u8, coverage = _tissue_mask_bgr(image_bgr)
        if coverage < 0.02:
            return 1e-3  # quasi kein Gewebe sichtbar

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bp = _dog_bandpass(gray, DOG_SIGMA1_PX, DOG_SIGMA2_PX)

        # Tenengrad auf Bandpass
        sx = cv2.Sobel(bp, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(bp, cv2.CV_32F, 0, 1, ksize=3)
        ten = sx * sx + sy * sy

        m = (mask_u8 > 0)
        if not np.any(m):
            return 1e-3

        base_focus = float(np.median(ten[m]))
        nuc_boost  = float(np.median(_nuclei_proxy_bgr(image_bgr)[m]))  # 0..~2
        score_raw  = base_focus * (1.0 + NUCLEI_WEIGHT * nuc_boost)
        score      = score_raw * (coverage ** COVERAGE_ALPHA)
        return float(score)

    # ---------------- Mathe-Tools ----------------
    def _parabolic_refine(self, zs, scores):
        """Parabel-Fit um lokales Maximum (3-Punkt)."""
        i = int(np.argmax(scores))
        if i == 0 or i == len(scores) - 1:
            return float(zs[i]), float(scores[i])
        z1, z2, z3 = zs[i - 1], zs[i], zs[i + 1]
        f1, f2, f3 = scores[i - 1], scores[i], scores[i + 1]
        denom = (z1 - 2 * z2 + z3)
        if denom == 0:
            return float(z2), float(f2)
        z_peak = z2 + 0.5 * ((f1 - f3) / denom) * (z1 - z3)
        return float(z_peak), float(max(f1, f2, f3))

    # ---------------- Messen @ Z ----------------
    def _measure_at(self, target_z_um, current_z_um, move_z_func,
                    settle_s=0.30, discard=2, samples=3, overshoot_um=60):
        """Fahre (mit Backlash-Workaround) an Ziel-Z, warte und messe Median-Score."""
        target_clamped = clamp_z(target_z_um)
        approach = "direct"

        # Backlash-Workaround: möglichst von unten annähern
        if target_clamped < current_z_um:
            approach = f"undershoot+up({overshoot_um}µm)"
            delta1 = (target_clamped - overshoot_um) - current_z_um
            delta2 = overshoot_um
            _log(f"→ Move plan: curr={current_z_um:.1f} → {target_clamped:.1f} (clamped) via {approach}: Δ1={delta1:+.1f}, Δ2={delta2:+.1f}")
            move_z_func(delta1); time.sleep(settle_s)
            move_z_func(delta2)
        else:
            delta = target_clamped - current_z_um
            _log(f"→ Move plan: curr={current_z_um:.1f} → {target_clamped:.1f} (clamped) via {approach}: Δ={delta:+.1f}")
            move_z_func(delta)

        time.sleep(settle_s)

        # Mehrfach messen → Median
        vals = []
        for i in range(samples):
            frame = self.get_frame(discard=1 if i == 0 else 0)
            s = self.score_combined(frame) if frame is not None else 0.0
            vals.append(float(s))
        score = float(np.median(vals))

        _log(f"   Measured @ Z={target_clamped:+.1f} µm → samples={vals} → score(med)={score:.2f}")
        return score, target_clamped

    def _micro_bracket(self, z_center_um, current_z_um, move_z_func,
                        micro_step_um=10, span=2, settle_s=0.30):
        """Sehr feiner Check um den Parabel-Peak (±span Schritte à micro_step)."""
        zs = [z_center_um + i * micro_step_um for i in range(-span, span + 1)]
        scores = []
        cz = current_z_um
        _log(f"🔬 Micro bracket around {z_center_um:.1f} µm, step={micro_step_um} µm, span=±{span}")
        for z in zs:
            s, _ = self._measure_at(z, cz, move_z_func, settle_s=settle_s, discard=2, samples=3)
            scores.append(s); cz = z
        i = int(np.argmax(scores))
        return zs[i], scores[i], cz

    def _expand_window(self, move_z_func, start_z, init_halfspan_um, step_um,
                       grow=1.8, max_halfspan_um=4000, settle_s=0.25):
        """
        Sucht ein Fenster [z_lo,z_hi], in dem der Peak sicher liegt (steigt→Peak→fällt).
        Vergrößert halbwegs geometrisch, bis erkennbar oder Grenzen erreicht.
        """
        current_z = start_z
        # kleine Richtungsprobe
        s0, _     = self._measure_at(start_z, current_z, move_z_func, settle_s); current_z = start_z
        s_plus, _ = self._measure_at(clamp_z(start_z + step_um), current_z, move_z_func, settle_s); current_z = clamp_z(start_z + step_um)
        s_minus, _= self._measure_at(clamp_z(start_z - step_um), current_z, move_z_func, settle_s); current_z = clamp_z(start_z - step_um)
        prefer = "up" if s_plus >= s_minus else "down"

        half = float(init_halfspan_um)
        clipped = False

        while half <= max_halfspan_um:
            z_lo = clamp_z(start_z - half)
            z_hi = clamp_z(start_z + half)
            if z_lo == Z_MIN_UM or z_hi == Z_MAX_UM:
                clipped = True

            probes = [z_lo, clamp_z(start_z - half/2), start_z, clamp_z(start_z + half/2), z_hi]
            scores = []
            cz = current_z
            for z in (probes if prefer == "up" else list(reversed(probes))):
                s, _ = self._measure_at(z, cz, move_z_func, settle_s)
                scores.append(s); cz = z
            current_z = cz

            _log(f"   Window probe {z_lo:.0f}..{z_hi:.0f} µm → scores={['%.0f'%x for x in scores]}")
            peak_idx = int(np.argmax(scores))
            if 0 < peak_idx < len(scores) - 1 and scores[peak_idx] >= scores[0] and scores[peak_idx] >= scores[-1]:
                return dict(z_lo=z_lo, z_hi=z_hi, current_z=current_z, clipped=clipped)

            _log("   ↗ Peak am Rand/Trend unklar → Fenster vergrößern…")
            half = min(max_halfspan_um, half * grow)

        # Fallback: maximales Fenster
        return dict(z_lo=clamp_z(start_z - half), z_hi=clamp_z(start_z + half),
                    current_z=current_z, clipped=True)

    # ---------------- Haupt-Autofokus ----------------
    def autofocus(self, move_z_func,
                  start_pos_um=0,
                  coarse_step_um=200, fine_step_um=50,
                  init_halfspan_um=800, max_halfspan_um=4000,
                  grow=1.8, settle_s=0.25, fine_span=2):
        if move_z_func is None:
            raise ValueError("move_z_func fehlt")

        # 1) Fenster adaptiv finden
        win = self._expand_window(move_z_func, start_pos_um, init_halfspan_um,
                                  step_um=coarse_step_um, grow=grow,
                                  max_halfspan_um=max_halfspan_um, settle_s=settle_s)
        z_lo, z_hi, current_z = win["z_lo"], win["z_hi"], win["current_z"]
        print(f"🪟 Fenster: [{z_lo:.0f}, {z_hi:.0f}] µm (clipped={win['clipped']})")

        # 2) Coarse-Scan im Fenster
        zs, sc = [], []
        z = z_lo
        while z <= z_hi + 1e-6:
            s, _ = self._measure_at(z, current_z, move_z_func, settle_s)
            current_z = z
            zs.append(z); sc.append(s)
            print(f"📸 Coarse Z={z:+.0f} µm → Score={s:.2f}")
            z += coarse_step_um
        z_best_coarse = zs[int(np.argmax(sc))]

        # 3) Fine-Scan um Coarse-Peak
        zs2, sc2 = [], []
        z = clamp_z(z_best_coarse - fine_span * fine_step_um)
        z_end = clamp_z(z_best_coarse + fine_span * fine_step_um)
        while z <= z_end + 1e-6:
            s, _ = self._measure_at(z, current_z, move_z_func, settle_s)
            current_z = z
            zs2.append(z); sc2.append(s)
            print(f"🔎 Fine   Z={z:+.0f} µm → Score={s:.2f}")
            z += fine_step_um

        # 4) Parabel + Micro (best of both)
        z_refined, s_refined = self._parabolic_refine(np.array(zs2), np.array(sc2))
        z_micro, s_micro, current_z = self._micro_bracket(z_refined, current_z, move_z_func,
                                                          micro_step_um=10, span=2, settle_s=0.30)
        _log(f"🏁 Final pick after micro-bracket: Z={z_micro:.1f} µm (score={s_micro:.2f})")

        if s_micro >= s_refined:
            final_z, final_score, picked = z_micro, s_micro, "micro"
        else:
            final_z, final_score, picked = z_refined, s_refined, "parabola"

        move_z_func(final_z - current_z)
        time.sleep(0.25)
        print(f"🟢 Ergebnis: Z={final_z:.1f} µm ({picked}, Score≈{final_score:.2f})")

        return {
            "best_z_um": float(final_z),
            "score": float(final_score),
            "picked": picked,
            "window": [float(z_lo), float(z_hi)],
            "clipped": bool(win["clipped"]),
            "measurements": int(len(zs) + len(zs2) + 5)  # +5 für Micro-Punkte
        }

    # ---------------- Alte Version behalten (geflickt) ----------------
    def autofocus_old(self, move_z_func, start_pos_um=0, step_um=200,
                      initial_sweep_steps=4, sweep_extension_steps=4, max_total_steps=20):
        """Deine alte Logik – jetzt nutzt sie den kombinierten Score."""
        travel_speed_um_per_s = 300.0
        extension_direction = None
        current_step = 0
        total_measurements = 0
        best_score = -1.0
        best_z = start_pos_um
        visited_steps = set()

        sweep_min = -initial_sweep_steps
        sweep_max = initial_sweep_steps
        current_z = start_pos_um

        while abs(current_step) <= max_total_steps:
            scores, step_positions = [], []
            sweep_range = [s for s in range(sweep_min, sweep_max + 1) if s not in visited_steps]
            if not sweep_range:
                print("⛔ Keine neuen Schritte mehr zu testen.")
                break

            print(f"🔍 Starte Sweep: Schritte {sweep_range[0]} bis {sweep_range[-1]} ({len(sweep_range)} Schritte)")
            for target_step in sweep_range:
                delta_steps = target_step - current_step
                move_z_func(delta_steps * step_um)
                current_step = target_step
                visited_steps.add(current_step)
                current_z += delta_steps * step_um
                time.sleep(max(1.0, abs(delta_steps * step_um) / travel_speed_um_per_s))

                frame = self.get_frame(discard=2)
                if frame is None:
                    continue
                score = self.score_combined(frame)  # EIN Score
                scores.append(score)
                step_positions.append(current_z)
                total_measurements += 1
                print(f"📸 Z={current_z:+} µm → Score={score:.2f}")

            if not step_positions:
                break

            local_best_idx = int(np.argmax(scores))
            local_best_score = scores[local_best_idx]
            local_best_step = list(sweep_range)[local_best_idx]
            local_best_z = step_positions[local_best_idx]
            print(f"🟢 Bester Punkt im Sweep: Schritt {local_best_step}, Score={local_best_score:.3f}")

            if local_best_score > best_score:
                best_score = local_best_score
                best_z = local_best_z

            # Erweiterung prüfen
            if local_best_step == min(sweep_range) and sweep_min > -max_total_steps:
                if extension_direction in (None, "down"):
                    print("⚠️ Fokus am unteren Rand → erweitere nach unten")
                    sweep_min -= sweep_extension_steps
                    extension_direction = "down"
                else:
                    print("⛔ Erweiterung in andere Richtung verboten → Abbruch")
                    break

            elif local_best_step == max(sweep_range) and sweep_max < max_total_steps:
                if extension_direction in (None, "up"):
                    print("⚠️ Fokus am oberen Rand → erweitere nach oben")
                    sweep_max += sweep_extension_steps
                    extension_direction = "up"
                else:
                    print("⛔ Erweiterung in andere Richtung verboten → Abbruch")
                    break
            else:
                print("✅ Fokus innerhalb Bereich gefunden.")
                break

        # Rückfahrt
        delta_back = best_z - current_z
        print(f"↩️ Rückfahrt zur besten Z-Position: Z={best_z} µm (ΔZ={delta_back:+} µm)")
        move_z_func(delta_back)
        time.sleep(0.5)
        print(f"📊 Insgesamt {total_measurements} Messungen durchgeführt.")

        return {
            "best_z_um": float(best_z),
            "score": float(best_score) if best_score >= 0 else None,
            "measurements": int(total_measurements)
        }

    # ---------------- Frames holen ----------------
    def get_frame(self, discard: int = 0):
        """Holt einen Frame; wir arbeiten intern in BGR (OpenCV-Standard)."""
        if not self.picam2:
            print("⚠️ Fehler: Kamera nicht verfügbar!")
            return None
        with self.lock:
            for _ in range(max(0, int(discard))):
                _ = self.picam2.capture_array()
            frame = self.picam2.capture_array()
        # Kamera liefert RGB888 → zu BGR für OpenCV
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame is not None else None

    def _get_stable_frame(self, discard=1):
        return self.get_frame(discard=discard)

    # ---------------- optional: simpler Laplacian-Score ----------------
    def get_autofocus_score(self, frame):
        if frame is None:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    # ---------------- Zoom ----------------
    def zoom(self, direction):
        """Zoom-Steuerung mit Fehlerprüfung."""
        if not self.picam2:
            print("⚠️ Fehler: Kamera nicht verfügbar!")
            return

        with self.lock:
            metadata = self.picam2.capture_metadata()
            current_crop = metadata.get("ScalerCrop")
            if current_crop is None:
                print("⚠️ ScalerCrop nicht verfügbar")
                return

            x, y, w, h = current_crop
            zoom_factor = 1.2 if direction == 'in' else 0.8

            new_w = max(640, min(3280, int(w / zoom_factor)))
            new_h = max(480, min(2464, int(h / zoom_factor)))
            new_x = x + (w - new_w) // 2
            new_y = y + (h - new_h) // 2

            try:
                self.picam2.set_controls({"ScalerCrop": (new_x, new_y, new_w, new_h)})
                print(f"🔍 Zoom geändert auf: {new_w}x{new_h}")
            except Exception as e:
                print(f"❌ Fehler beim Zoom setzen: {e}")
