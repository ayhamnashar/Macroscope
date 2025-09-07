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
# Zentrales ROI gegen Rand-Halos/Glasreflexe
CENTER_ROI_RATIO = 0.7  # Anteil des Bildes (0..1), der zentral gewichtet wird
TEN_PERCENTILE = 75     # robuster Fokuswert statt Median (z.B. 70–85)
# Faser-/Linienunterdrückung (Anisotropie) und minimale Kerndichte
ANISO_SIGMA = 1.2       # Glättung des Struktur-Tensors
ANISO_WEIGHT = 0.25     # wie stark lineare Strukturen abgewertet werden (0.15–0.4)
NUCLEI_MIN_PCTL = 60    # Perzentil für Minimalanforderung an Kerndichte
NUCLEI_MIN_VALUE = 0.35 # Schwelle für bluish*sättigung (0.3–0.6)
# Minimale Gewebe-Deckung und Flachheits-Erkennung (gegen Glas/unscharfe Ebenen)
MIN_COVERAGE = 0.08     # unterhalb wird Kandidat stark abgewertet/verw.
FLAT_TEN_RATIO_MIN = 1.22  # p90/p50 von Tenengrad unterhalb → "flach"

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

def _anisotropy_map(gray32, sigma=ANISO_SIGMA):
    """Struktur-Tensor-Kohärenz (0≈isotrop, 1≈starke Linie)."""
    Ix = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)
    J11 = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma)
    J22 = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma)
    J12 = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma)
    eps = 1e-6
    trace = J11 + J22 + eps
    diff = J11 - J22
    lamb_diff = cv2.sqrt(diff * diff + 4.0 * J12 * J12)
    coherence = lamb_diff / trace
    return np.clip(coherence, 0.0, 1.0)

def _vesselness_frangi(gray32, sigmas=(0.8, 1.2, 1.8), beta=0.5, c=12.0):
    """Einfaches 2D-Frangi-Vesselness (Ridge-Detektor) über mehrere Skalen.
    Liefert Werte ~0..1; betont faser-/linienartige Strukturen.
    """
    gray = gray32.astype(np.float32)
    best = np.zeros_like(gray, dtype=np.float32)
    eps = 1e-6
    for s in sigmas:
        g = cv2.GaussianBlur(gray, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        # Hessian Eigenwerte (2x2)
        tmp = cv2.sqrt((dxx - dyy) * (dxx - dyy) + 4.0 * dxy * dxy)
        lam1 = 0.5 * (dxx + dyy + tmp)
        lam2 = 0.5 * (dxx + dyy - tmp)
        # sortiere nach |lam1| >= |lam2|
        swap = (np.abs(lam1) < np.abs(lam2))
        l1 = lam1.copy(); l2 = lam2.copy()
        l1[swap], l2[swap] = lam2[swap], lam1[swap]
        # Frangi-Kennwerte
        rb = (np.abs(l2) / (np.abs(l1) + eps))
        s2 = l1 * l1 + l2 * l2
        v = np.exp(-(rb * rb) / (2 * beta * beta)) * (1.0 - np.exp(-(s2) / (2 * c * c)))
        best = np.maximum(best, v)
    # Normalisieren
    m, M = float(np.min(best)), float(np.max(best) + eps)
    out = (best - m) / (M - m)
    return out


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

        # Zentrales ROI (Ellipse) multipliziert mit Tissue-Maske
        h, w = image_bgr.shape[:2]
        cy, cx = h // 2, w // 2
        ry = max(1, int(h * CENTER_ROI_RATIO * 0.5))
        rx = max(1, int(w * CENTER_ROI_RATIO * 0.5))
        yy, xx = np.ogrid[:h, :w]
        roi = ((yy - cy) ** 2) / (ry * ry) + ((xx - cx) ** 2) / (rx * rx) <= 1.0
        m = (mask_u8 > 0) & roi
        if not np.any(m):
            # Fallback: nur Tissue-Maske
            m = (mask_u8 > 0)
            if not np.any(m):
                return 1e-3

        # Robuster Fokuswert: oberes Perzentil statt Median
        base_focus = float(np.percentile(ten[m], TEN_PERCENTILE))
        nuc_map = _nuclei_proxy_bgr(image_bgr)
        nuc_boost  = float(np.median(nuc_map[m]))  # 0..~2

        # Anisotropie-Penalty (Fasern/Linien)
        aniso = _anisotropy_map(gray)
        aniso_med = float(np.median(aniso[m]))
        aniso_pen = max(0.6, 1.0 - ANISO_WEIGHT * aniso_med)

        score_raw  = base_focus * (1.0 + NUCLEI_WEIGHT * nuc_boost) * aniso_pen
        score      = score_raw * (coverage ** COVERAGE_ALPHA)

        # Mindest-Kerndichte (optional) – vermeidet Fokus auf farblose Fasern
        nuc_req = float(np.percentile(nuc_map[m], NUCLEI_MIN_PCTL))
        if nuc_req < NUCLEI_MIN_VALUE:
            score *= 0.75
        return float(score)

    # ---------------- Klassischer (einfacher) Fokus-Score ----------------
    def score_classic(self, image_bgr):
        """Einfacher, früher verwendeter Fokus-Score (Laplacian-Varianz über das ganze Bild).
        Keine Gewebe-/Kern-Heuristiken, schnell und robust für die klassische AF-Strategie."""
        if image_bgr is None:
            return 0.0
        gray32 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray32, cv2.CV_32F)
        return float(np.var(lap))

    # ---------------- Faser-/Ridge-orientierter Fokus-Score ----------------
    def score_fiber(self, image_bgr):
        """Fokus-Score, der lineare/faserige Strukturen betont.
        - nutzt Struktur-Tensor-Kohärenz (Anisotropie) als Gewicht
        - kombiniert mit Tenengrad (Kantenenergie)
        - zentrales ROI + Tissue-Maske, aber ohne Nuclei-Boost
        """
        if image_bgr is None:
            return 0.0
        mask_u8, coverage = _tissue_mask_bgr(image_bgr)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Bandpass zur Z-Sensitivität, dann Tenengrad
        bp = _dog_bandpass(gray, DOG_SIGMA1_PX, DOG_SIGMA2_PX)
        sx = cv2.Sobel(bp, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(bp, cv2.CV_32F, 0, 1, ksize=3)
        ten = sx * sx + sy * sy
        # Kohärenz (Anisotropie) → je höher, desto linearer die Struktur
        coh = _anisotropy_map(gray, sigma=ANISO_SIGMA)
        # Frangi-Vesselness über mehrere Skalen
        ves = _vesselness_frangi(gray, sigmas=(0.9, 1.4, 2.0), beta=0.5, c=12.0)

        h, w = image_bgr.shape[:2]
        cy, cx = h // 2, w // 2
        ry = max(1, int(h * CENTER_ROI_RATIO * 0.5))
        rx = max(1, int(w * CENTER_ROI_RATIO * 0.5))
        yy, xx = np.ogrid[:h, :w]
        roi = ((yy - cy) ** 2) / (ry * ry) + ((xx - cx) ** 2) / (rx * rx) <= 1.0
        # Maske: Tissue ODER starke Kante, damit blasse Fasern nicht rausfallen
        tmask = (mask_u8 > 0)
        edge_thresh = float(np.percentile(ten, 75))
        emask = (ten > edge_thresh)
        m = (tmask | emask) & roi
        if not np.any(m):
            m = roi

        # Gewicht aus Kohärenz + Vesselness
        wgt = np.clip(0.7 * coh + 0.6 * ves, 0.0, 1.5)
        fiber_energy = ten * wgt
        base = float(np.percentile(fiber_energy[m], TEN_PERCENTILE))
        # leichte Coverage-Skalierung, abgeschwächt für Fasern
        score = base * (coverage ** max(0.0, (COVERAGE_ALPHA * 0.4)))
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
                    settle_s=0.30, discard=2, samples=3, overshoot_um=60,
                    score_func=None):
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
            if frame is not None:
                s = (score_func or self.score_combined)(frame)
            else:
                s = 0.0
            vals.append(float(s))
        score = float(np.median(vals))

        _log(f"   Measured @ Z={target_clamped:+.1f} µm → samples={vals} → score(med)={score:.2f}")
        return score, target_clamped

    def _micro_bracket(self, z_center_um, current_z_um, move_z_func,
                        micro_step_um=10, span=2, settle_s=0.30, score_func=None):
        """Sehr feiner Check um den Parabel-Peak (±span Schritte à micro_step)."""
        zs = [z_center_um + i * micro_step_um for i in range(-span, span + 1)]
        scores = []
        cz = current_z_um
        _log(f"🔬 Micro bracket around {z_center_um:.1f} µm, step={micro_step_um} µm, span=±{span}")
        for z in zs:
            s, _ = self._measure_at(
                z, cz, move_z_func,
                settle_s=settle_s, discard=2, samples=3,
                score_func=score_func,
            )
            scores.append(s); cz = z
        i = int(np.argmax(scores))
        return zs[i], scores[i], cz

    def _expand_window(self, move_z_func, start_z, init_halfspan_um, step_um,
                       grow=1.8, max_halfspan_um=4000, settle_s=0.25,
                       max_iters=6, flat_rel_eps=0.08, score_func=None):
        """
        Sucht ein Fenster [z_lo,z_hi], in dem der Peak sicher liegt (steigt→Peak→fällt).
        Vergrößert halbwegs geometrisch, bis erkennbar oder Grenzen erreicht.
        """
        current_z = start_z
        # kleine Richtungsprobe
        s0, _ = self._measure_at(start_z, current_z, move_z_func, settle_s, score_func=score_func)
        current_z = start_z
        s_plus, _ = self._measure_at(clamp_z(start_z + step_um), current_z, move_z_func, settle_s, score_func=score_func)
        current_z = clamp_z(start_z + step_um)
        s_minus, _ = self._measure_at(clamp_z(start_z - step_um), current_z, move_z_func, settle_s, score_func=score_func)
        current_z = clamp_z(start_z - step_um)
        prefer = "up" if s_plus >= s_minus else "down"

        half = float(init_halfspan_um)
        clipped = False

        iters = 0
        while half <= max_halfspan_um and iters < max_iters:
            z_lo = clamp_z(start_z - half)
            z_hi = clamp_z(start_z + half)
            if z_lo == Z_MIN_UM or z_hi == Z_MAX_UM:
                clipped = True

            probes = [z_lo, clamp_z(start_z - half/2), start_z, clamp_z(start_z + half/2), z_hi]
            scores = []
            cz = current_z
            for z in (probes if prefer == "up" else list(reversed(probes))):
                s, _ = self._measure_at(z, cz, move_z_func, settle_s, score_func=score_func)
                scores.append(s)
                cz = z
            current_z = cz

            _log(f"   Window probe {z_lo:.0f}..{z_hi:.0f} µm → scores={['%.0f'%x for x in scores]}")
            # Flat-profile early exit: accept current window when variation is tiny
            smax = max(scores) if scores else 0.0
            smin = min(scores) if scores else 0.0
            rel_span = (smax - smin) / (smax + 1e-6)
            if rel_span < flat_rel_eps:
                _log("   ⚪ Flat profile detected → accept current window and stop expanding")
                return dict(z_lo=z_lo, z_hi=z_hi, current_z=current_z, clipped=clipped)
            peak_idx = int(np.argmax(scores))
            if 0 < peak_idx < len(scores) - 1 and scores[peak_idx] >= scores[0] and scores[peak_idx] >= scores[-1]:
                return dict(z_lo=z_lo, z_hi=z_hi, current_z=current_z, clipped=clipped)

            _log("   ↗ Peak am Rand/Trend unklar → Fenster vergrößern…")
            half = min(max_halfspan_um, half * grow)
            iters += 1

        # Fallback: maximales Fenster
        return dict(
            z_lo=clamp_z(start_z - half),
            z_hi=clamp_z(start_z + half),
            current_z=current_z,
            clipped=True,
        )

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
                # Klassischer, einfacher Fokus-Score (wie in frühen Versionen)
                score = self.score_classic(frame)
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

    # ---------------- Smarter, robust-heuristischer Autofokus ----------------
    def _recheck_score_with_coverage(self, move_z_func, target_z_um, current_z_um, settle_s=0.30):
        """Fahre zu Z, hole frischen Frame und berechne robuste Ensemble-Scores + Coverage.
        Liefert (smart_score, coverage, tenengrad_med, lap_var, nuclei_med, ten_p50, ten_p90, z, current_z_after).
        """
        # sicher zum Ziel (mit bestehender Backlash-Logik von _measure_at)
        s_combined, z_at = self._measure_at(
            target_z_um, current_z_um, move_z_func,
            settle_s=settle_s, discard=2, samples=3
        )
        frame = self.get_frame(discard=1)
        if frame is None:
            return s_combined, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, z_at, z_at

        # Tissue-Maske und Coverage
        mask_u8, coverage = _tissue_mask_bgr(frame)
        m = (mask_u8 > 0)
        gray32 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Bandpass + Tenengrad
        bp = _dog_bandpass(gray32, DOG_SIGMA1_PX, DOG_SIGMA2_PX)
        sx = cv2.Sobel(bp, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(bp, cv2.CV_32F, 0, 1, ksize=3)
        ten = sx * sx + sy * sy

        # robuste Tenengrad-Statistik
        ten_vals = ten[m] if np.any(m) else ten
        ten_med = float(np.median(ten_vals))
        ten_p50 = float(np.percentile(ten_vals, 50))
        ten_p90 = float(np.percentile(ten_vals, 90))

        # Laplacian-Varianz (klassisch)
        lap = cv2.Laplacian(gray32, cv2.CV_32F)
        lap_var = float(np.var(lap[m])) if np.any(m) else float(np.var(lap))

        # Nuclei-Proxy (blau/sättigung) als Relevanz-Boost
        nuc = _nuclei_proxy_bgr(frame)
        nuc_med = float(np.median(nuc[m])) if np.any(m) else float(np.median(nuc))

        # Ensemble-Score – gewichtet und coverage-abhängig
        W_TEN, W_LAP, W_REL = 0.45, 0.35, 0.20
        raw = (W_TEN * ten_med) + (W_LAP * lap_var) + (W_REL * nuc_med)
        smart_score = raw * (coverage ** 0.6)  # moderate Abwertung bei wenig Gewebe

        # zusätzlich mit vorherigem kombinierten Score harmonisieren
        smart_score = 0.7 * smart_score + 0.3 * s_combined
        return (
            float(smart_score), float(coverage), float(ten_med), float(lap_var),
            float(nuc_med), float(ten_p50), float(ten_p90), float(z_at), float(z_at)
        )

    def _scout_global(self, move_z_func, seed_z_um, halfspan_um=6000, step_um=600,
                      settle_s=0.25, min_sep_um=800, top_k=2):
        """Breite Vorerkundung: grobe Stichprobe über großen Z-Bereich, um gute Start-Seed(s) zu finden.
        Liefert (candidates_z_list, current_z_after, debug_dict).
        """
        z_lo = clamp_z(float(seed_z_um) - float(halfspan_um))
        z_hi = clamp_z(float(seed_z_um) + float(halfspan_um))
        if step_um <= 0:
            step_um = 600
        zs = []
        z = z_lo
        current_z = seed_z_um
        scores = []
        covs = []
        flats = []
        while z <= z_hi + 1e-6:
            s, cov, ten_med, lap_var, nuc_med, ten_p50, ten_p90, z_at, current_z = self._recheck_score_with_coverage(
                move_z_func, z, current_z, settle_s=max(0.30, settle_s)
            )
            zs.append(z)
            scores.append(s)
            covs.append(cov)
            is_flat = (ten_p90 / (ten_p50 + 1e-6) < FLAT_TEN_RATIO_MIN)
            flats.append(bool(is_flat))
            _log(f"🌐 Scout Z={z:+.0f} µm → smart={s:.2f}, cov={cov:.2f}, flat={is_flat}")
            z += step_um
        if not zs:
            return [], current_z, {"zs": [], "scores": [], "covs": []}

        zs = np.array(zs, dtype=float)
        scores = np.array(scores, dtype=float)

        # wähle Top-K mit Nicht-Maxima-Unterdrückung (Abstand min_sep_um)
        order = np.argsort(-scores)
        picked = []
        for idx in order:
            zc = float(zs[idx])
            if any(abs(zc - p) < min_sep_um for p in picked):
                continue
            picked.append(zc)
            if len(picked) >= max(1, int(top_k)):
                break
        return picked, current_z, {"zs": zs.tolist(), "scores": scores.tolist(), "covs": covs, "flats": flats}

    def autofocus_smart(self, move_z_func,
                        start_pos_um=0,
                        coarse_step_um=250, fine_step_um=80,
                        init_halfspan_um=700, max_halfspan_um=3500,
                        grow=1.7, settle_s=0.25, fine_span=2,
                        topk_validate=3,
                        use_global_scout=True,
                        scout_halfspan_um=6000, scout_step_um=600,
                        scout_top_k=2, scout_min_sep_um=800):
        """Intelligenter, robust-heuristischer Autofokus.
        Ideen:
         - adaptives Fenster (steigt→Peak→fällt) via _expand_window
         - coarse→fine Scan mit kombinierten, gewebe-bewussten Scores
         - Top-K Kandidaten werden mit zusätzlicher Coverage-/Nuclei-Heuristik validiert
         - Micro-Bracketing um den finalen Kandidaten zur Sub-Schritt-Optimierung

        Erwartet move_z_func(delta_um) (relative Bewegung in µm).
        Rückgabe: Dict mit best_z_um, score und Diagnosen.
        """
        if move_z_func is None:
            raise ValueError("move_z_func fehlt")

        # 0) Optional: globaler Scout zur Bestimmung eines guten Start-Seeds
        seed_z = float(start_pos_um)
        used_scout = False
        if use_global_scout:
            seeds, current_z_tmp, dbg = self._scout_global(
                move_z_func, seed_z_um=seed_z, halfspan_um=scout_halfspan_um,
                step_um=scout_step_um, settle_s=settle_s,
                min_sep_um=scout_min_sep_um, top_k=scout_top_k
            )
            if seeds:
                seed_z = float(seeds[0])
                _log(f"🌐 Scout pick seed Z={seed_z:.1f} µm (from {len(seeds)} candidates)")
                used_scout = True
        # 1) Fenster adaptiv bestimmen um den Seed
        win = self._expand_window(move_z_func, seed_z, init_halfspan_um,
                                  step_um=coarse_step_um, grow=grow,
                                  max_halfspan_um=max_halfspan_um, settle_s=settle_s)
        z_lo, z_hi, current_z = win["z_lo"], win["z_hi"], win["current_z"]
        print(f"🪟 SMART Fenster: [{z_lo:.0f}, {z_hi:.0f}] µm (clipped={win['clipped']})")

        # 2) Coarse-Scan im Fenster
        zs, sc = [], []
        z = z_lo
        while z <= z_hi + 1e-6:
            s, _ = self._measure_at(z, current_z, move_z_func, settle_s)
            current_z = z
            zs.append(z); sc.append(s)
            print(f"📸 SMART Coarse Z={z:+.0f} µm → Score={s:.2f}")
            z += coarse_step_um
        zs = np.array(zs, dtype=float)
        sc = np.array(sc, dtype=float)

        # 3) Fine-Scan um die besten 1–2 coarse Peaks
        # finde lokale Maxima grob
        peak_idx = int(np.argmax(sc))
        z_center = float(zs[peak_idx])
        # If coarse profile is nearly flat, skip fine scan and do a wider micro bracket
        coarse_flat = False
        if sc.size > 0:
            smax = float(np.max(sc))
            smin = float(np.min(sc))
            rel_span = (smax - smin) / (smax + 1e-6)
        else:
            rel_span = 0.0
        if rel_span < 0.05:
            _log("⚪ SMART: Coarse profile flat → try macro sweep over larger range before micro bracket")
            coarse_flat = True
            # Macro sweep: widen search range and increase step to find any improving region
            macro_half = min(float(max_halfspan_um) * 2.0, 8000.0)
            z_lo2 = clamp_z(z_center - macro_half)
            z_hi2 = clamp_z(z_center + macro_half)
            macro_step = max(coarse_step_um * 2, 400)
            macro_zs, macro_sc = [], []
            z = z_lo2
            while z <= z_hi2 + 1e-6:
                s, _ = self._measure_at(z, current_z, move_z_func, settle_s=max(0.30, settle_s))
                current_z = z
                macro_zs.append(z); macro_sc.append(s)
                _log(f"📈 SMART Macro Z={z:+.0f} µm → Score={s:.2f}")
                z += macro_step
            if macro_sc:
                macro_idx = int(np.argmax(macro_sc))
                z_pick = float(macro_zs[macro_idx])
                # If still flat, just micro bracket wider; else do a narrow fine around z_pick
                m_smax = float(np.max(macro_sc))
                m_smin = float(np.min(macro_sc))
                m_rel = (m_smax - m_smin) / (m_smax + 1e-6)
                if m_rel < 0.05:
                    _log("⚪ SMART: Macro still flat → use wide micro bracket")
                    z_micro, s_micro, current_z = self._micro_bracket(z_pick, current_z, move_z_func,
                                                                      micro_step_um=30, span=4, settle_s=max(0.35, settle_s))
                    # Backlash-safe approach
                    delta1 = (z_pick - 60) - current_z
                    move_z_func(delta1); time.sleep(0.20)
                    move_z_func(60); time.sleep(0.20)
                    move_z_func(z_micro - (z_pick)); time.sleep(0.20)
                    # Final coverage and simple confidence from macro profile
                    final_frame = self.get_frame(discard=1)
                    cov = 0.0
                    if final_frame is not None:
                        _, cov = _tissue_mask_bgr(final_frame)
                    # confidence: based on macro profile spread and margin to 2nd best
                    if len(macro_sc) >= 2:
                        arr = np.array(macro_sc, dtype=float)
                        order = np.argsort(-arr)
                        s1 = float(arr[order[0]])
                        s2 = float(arr[order[1]])
                        margin = max(0.0, (s1 - s2) / (s1 + 1e-6))
                        span_rel = max(0.0, (np.max(arr) - np.min(arr)) / (np.max(arr) + 1e-6))
                        conf = float(np.clip(0.5 * margin + 0.5 * span_rel, 0.0, 1.0))
                    else:
                        conf = 0.3
                    print(f"🟢 SMART Ergebnis (macro-flat): Z={z_micro:.1f} µm (score≈{s_micro:.2f})")
                    return {
                        "best_z_um": float(z_micro),
                        "score": float(s_micro),
                        "picked": "smart-macro-flat",
                        "window": [float(z_lo2), float(z_hi2)],
                        "clipped": bool(win["clipped"]),
                        "coarse_points": int(len(zs) + len(macro_zs)),
                        "fine_points": 0,
                        "confidence": conf,
                        "final_coverage": float(cov),
                        "flags": {"used_scout": bool(used_scout), "coarse_flat": True, "macro_used": True}
                    }
                else:
                    # Brief fine around macro best then micro bracket
                    fine_points, fine_scores = [], []
                    z = clamp_z(z_pick - fine_span * fine_step_um)
                    z_end = clamp_z(z_pick + fine_span * fine_step_um)
                    while z <= z_end + 1e-6:
                        s, _ = self._measure_at(z, current_z, move_z_func, settle_s)
                        current_z = z
                        fine_points.append(z); fine_scores.append(s)
                        _log(f"🔎 SMART Macro-Fine Z={z:+.0f} µm → Score={s:.2f}")
                        z += fine_step_um
                    fine_points = np.array(fine_points, dtype=float)
                    fine_scores = np.array(fine_scores, dtype=float)
                    z_refined2, s_refined2 = self._parabolic_refine(fine_points, fine_scores)
                    z_micro, s_micro, current_z = self._micro_bracket(z_refined2, current_z, move_z_func,
                                                                      micro_step_um=10, span=2, settle_s=max(0.30, settle_s))
                    delta1 = (z_refined2 - 40) - current_z
                    move_z_func(delta1); time.sleep(0.20)
                    move_z_func(40); time.sleep(0.20)
                    move_z_func(z_micro - (z_refined2)); time.sleep(0.20)
                    # coverage + confidence from fine macro
                    final_frame = self.get_frame(discard=1)
                    cov = 0.0
                    if final_frame is not None:
                        _, cov = _tissue_mask_bgr(final_frame)
                    # confidence: neighbor drop and curvature-like second diff
                    if len(fine_scores) >= 3:
                        i = int(np.argmax(fine_scores))
                        s1 = float(fine_scores[i])
                        s2 = float(np.partition(fine_scores, -2)[-2]) if len(fine_scores) >= 2 else 0.0
                        margin = max(0.0, (s1 - s2) / (s1 + 1e-6))
                        if 0 < i < len(fine_scores) - 1:
                            sec = float(fine_scores[i-1] - 2*fine_scores[i] + fine_scores[i+1])
                            curv = max(0.0, -sec / (abs(fine_scores[i]) + 1e-6))
                            curv = float(np.clip(curv * 5.0, 0.0, 1.0))
                        else:
                            curv = 0.3
                        conf = float(np.clip(0.6 * margin + 0.4 * curv, 0.0, 1.0))
                    else:
                        conf = 0.4
                    print(f"🟢 SMART Ergebnis (macro-refined): Z={z_micro:.1f} µm (score≈{s_micro:.2f})")
                    return {
                        "best_z_um": float(z_micro),
                        "score": float(s_micro),
                        "picked": "smart-macro",
                        "window": [float(z_lo2), float(z_hi2)],
                        "clipped": bool(win["clipped"]),
                        "coarse_points": int(len(zs) + len(macro_zs)),
                        "fine_points": int(len(fine_points)),
                        "confidence": conf,
                        "final_coverage": float(cov),
                        "flags": {"used_scout": bool(used_scout), "coarse_flat": True, "macro_used": True}
                    }
            # If no macro data (shouldn't happen), fall back to wide micro
            z_pick = z_center
            z_micro, s_micro, current_z = self._micro_bracket(z_pick, current_z, move_z_func,
                                                              micro_step_um=30, span=4, settle_s=max(0.35, settle_s))
            delta1 = (z_pick - 60) - current_z
            move_z_func(delta1); time.sleep(0.20)
            move_z_func(60); time.sleep(0.20)
            move_z_func(z_micro - (z_pick)); time.sleep(0.20)
            print(f"🟢 SMART Ergebnis (flat emergency): Z={z_micro:.1f} µm (score≈{s_micro:.2f})")
            # coverage + conservative confidence for emergency path
            final_frame = self.get_frame(discard=1)
            cov = 0.0
            if final_frame is not None:
                _, cov = _tissue_mask_bgr(final_frame)
            return {
                "best_z_um": float(z_micro),
                "score": float(s_micro),
                "picked": "smart-flat-emergency",
                "window": [float(z_lo), float(z_hi)],
                "clipped": bool(win["clipped"]),
                "coarse_points": int(len(zs)),
                "fine_points": 0,
                "confidence": 0.3,
                "final_coverage": float(cov),
                "flags": {"used_scout": bool(used_scout), "coarse_flat": True, "macro_used": True}
            }
        fine_points, fine_scores = [], []
        z = clamp_z(z_center - fine_span * fine_step_um)
        z_end = clamp_z(z_center + fine_span * fine_step_um)
        while z <= z_end + 1e-6:
            s, _ = self._measure_at(z, current_z, move_z_func, settle_s)
            current_z = z
            fine_points.append(z); fine_scores.append(s)
            print(f"🔎 SMART Fine   Z={z:+.0f} µm → Score={s:.2f}")
            z += fine_step_um

        fine_points = np.array(fine_points, dtype=float)
        fine_scores = np.array(fine_scores, dtype=float)
        z_refined, s_refined = self._parabolic_refine(fine_points, fine_scores)

        # 4) Kandidaten-Validierung mit Coverage/Nuclei auf Top-K Punkten
        # Wähle Top-K aus den (coarse+fine) Samplern
        all_z = np.concatenate([zs, fine_points])
        all_s = np.concatenate([sc, fine_scores])
        order = np.argsort(-all_s)
        chosen_idxs = order[:max(1, int(topk_validate))]

        best_pack = None
        cz = current_z
        best_nonflat = None
        best_nonflat_score = None
        for idx in chosen_idxs:
            z_cand = float(all_z[idx])
            smart_s, cov, ten_med, lap_var, nuc_med, ten_p50, ten_p90, z_at, cz = self._recheck_score_with_coverage(
                move_z_func, z_cand, cz, settle_s=max(settle_s, 0.30)
            )
            flat_ratio = ten_p90 / (ten_p50 + 1e-6)
            is_flat = flat_ratio < FLAT_TEN_RATIO_MIN
            print(f"✅ Validate Z={z_cand:.1f}µm → smart={smart_s:.2f}, cov={cov:.2f}, ten50={ten_p50:.1f}, ten90={ten_p90:.1f}, flatRatio={flat_ratio:.2f}, nuc={nuc_med:.2f}")
            # Mindest-Coverage verlangen, um Glas zu vermeiden
            if cov < MIN_COVERAGE:
                continue
            # Kerndichte schwach → leicht abwerten (vermeidet Faserfokus)
            if nuc_med < NUCLEI_MIN_VALUE:
                smart_s *= 0.85
            # Flache Ebenen stark abwerten
            if is_flat:
                smart_s *= 0.65
            if best_pack is None or smart_s > best_pack[0]:
                best_pack = (smart_s, z_cand, cov)
            # Merke besten NICHT-flachen Kandidaten separat
            if (not is_flat) and (best_nonflat is None or smart_s > best_nonflat_score):
                best_nonflat = z_cand
                best_nonflat_score = smart_s

        # Fallback auf z_refined, falls Validation nichts übrig ließ
        final_center = float(z_refined if best_pack is None else best_pack[1])

        # 5) Micro-Bracket für die Feineinstellung
        z_micro, s_micro, cz = self._micro_bracket(final_center, cz, move_z_func,
                                                   micro_step_um=8, span=2, settle_s=max(0.30, settle_s))
        print(f"🏁 SMART Final micro: Z={z_micro:.1f} µm (score={s_micro:.2f})")

        # 6) Anfahren der finalen Position mit kleiner Doppel-Anfahrt (Backlash-Entkopplung)
        delta1 = (final_center - 40) - cz
        move_z_func(delta1); time.sleep(0.20)
        move_z_func(40); time.sleep(0.20)
        move_z_func(z_micro - (final_center)); time.sleep(0.20)

        # Final diagnostics: coverage and flatness check at final focus
        final_frame = self.get_frame(discard=1)
        cov = 0.0
        flat_final = False
        if final_frame is not None:
            _, cov = _tissue_mask_bgr(final_frame)
            gray32 = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            bp = _dog_bandpass(gray32, DOG_SIGMA1_PX, DOG_SIGMA2_PX)
            sx = cv2.Sobel(bp, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(bp, cv2.CV_32F, 0, 1, ksize=3)
            ten = sx * sx + sy * sy
            ten_p50 = float(np.percentile(ten, 50))
            ten_p90 = float(np.percentile(ten, 90))
            flat_final = (ten_p90 / (ten_p50 + 1e-6) < FLAT_TEN_RATIO_MIN)

        # Falls Endpunkt flach ist und es einen guten nicht-flachen Kandidaten gibt → dorthin wechseln
        if flat_final and best_nonflat is not None:
            _log("⚠️ Final plane appears flat → switching to best non-flat candidate")
            z_micro2, s_micro2, cz2 = self._micro_bracket(best_nonflat, z_micro, move_z_func,
                                                          micro_step_um=10, span=2, settle_s=max(0.30, settle_s))
            delta1 = (best_nonflat - 40) - cz2
            move_z_func(delta1); time.sleep(0.20)
            move_z_func(40); time.sleep(0.20)
            move_z_func(z_micro2 - (best_nonflat)); time.sleep(0.20)
            z_micro, s_micro = z_micro2, s_micro2
            final_frame = self.get_frame(discard=1)
            if final_frame is not None:
                _, cov = _tissue_mask_bgr(final_frame)

        # Confidence based on coarse+fine prominence, coarse span, and fine curvature
        # coarse prominence
        if len(sc) >= 2:
            s1c = float(np.max(sc))
            s2c = float(np.partition(sc, -2)[-2])
            prom_c = max(0.0, (s1c - s2c) / (s1c + 1e-6))
        else:
            prom_c = 0.2
        # fine curvature and margin
        if len(fine_scores) >= 3:
            i = int(np.argmax(fine_scores))
            s1 = float(fine_scores[i])
            s2 = float(np.partition(fine_scores, -2)[-2]) if len(fine_scores) >= 2 else 0.0
            margin = max(0.0, (s1 - s2) / (s1 + 1e-6))
            if 0 < i < len(fine_scores) - 1:
                sec = float(fine_scores[i-1] - 2*fine_scores[i] + fine_scores[i+1])
                curv = max(0.0, -sec / (abs(fine_scores[i]) + 1e-6))
                curv = float(np.clip(curv * 5.0, 0.0, 1.0))
            else:
                curv = 0.3
        else:
            margin, curv = 0.3, 0.2
        # coarse relative span already computed as rel_span
        conf = float(np.clip(0.4 * margin + 0.3 * curv + 0.3 * max(0.0, rel_span), 0.0, 1.0))

        print(f"🟢 SMART Ergebnis: Z={z_micro:.1f} µm (score≈{s_micro:.2f}), cov={cov:.2f}, conf={conf:.2f}")
        return {
            "best_z_um": float(z_micro),
            "score": float(s_micro),
            "picked": "smart",
            "window": [float(z_lo), float(z_hi)],
            "clipped": bool(win["clipped"]),
            "coarse_points": len(zs),
            "fine_points": len(fine_points),
            "confidence": conf,
            "final_coverage": float(cov),
            "flags": {"used_scout": bool(used_scout), "coarse_flat": bool(coarse_flat), "macro_used": False}
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

    # ========================= Next-Gen AF: Helpers =========================
    def _make_ds_score(self, factor:int):
        """Return a score function that downsamples the frame before scoring."""
        factor = max(1, int(factor))
        def _fn(frame_bgr):
            if factor == 1:
                return self.score_combined(frame_bgr)
            h, w = frame_bgr.shape[:2]
            nh, nw = max(1, h // factor), max(1, w // factor)
            small = cv2.resize(frame_bgr, (nw, nh))
            return self.score_combined(small)
        return _fn
    def _measure_features_at(self, target_z_um, current_z_um, move_z_func,
                              settle_s=0.30, discard=1, downsample_factor=2):
        """Move and capture one frame, returning a rich feature dict for AI selection.
        Uses lap-var and Tenengrad, coverage, nuclei proxy, coherence, vesselness,
        plus both combined and fiber scores.
        """
        # move using backlash-safe _measure_at but capture frame ourselves once settled
        target_clamped = clamp_z(target_z_um)
        # plan movement similar to _measure_at
        if target_clamped < current_z_um:
            delta1 = (target_clamped - 60.0) - current_z_um
            delta2 = 60.0
            move_z_func(delta1); time.sleep(settle_s)
            move_z_func(delta2)
        else:
            delta = target_clamped - current_z_um
            move_z_func(delta)
        time.sleep(settle_s)

        frame = self.get_frame(discard=max(0, int(discard)))
        if frame is None:
            return {
                "z": float(target_clamped), "coverage": 0.0,
                "lap_var": 0.0, "ten_p50": 0.0, "ten_p90": 0.0,
                "nuclei": 0.0, "coherence": 0.0, "vessel": 0.0,
                "score_combined": 0.0, "score_fiber": 0.0,
            }, target_clamped

        fbgr = frame
        if downsample_factor and downsample_factor > 1:
            h, w = fbgr.shape[:2]
            fbgr = cv2.resize(fbgr, (max(1, w//downsample_factor), max(1, h//downsample_factor)))

        # coverage + tissue mask (uint8 → boolean for indexing)
        mask_u8, coverage = _tissue_mask_bgr(fbgr)
        mask_bool = mask_u8.astype(bool)
        # Tenengrad energy percentiles
        gray = cv2.cvtColor(fbgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        ten = gx*gx + gy*gy
        ten_vals = ten[mask_bool]
        ten_p50 = float(np.percentile(ten_vals, 50)) if ten_vals.size else 0.0
        ten_p90 = float(np.percentile(ten_vals, 90)) if ten_vals.size else 0.0
        # Laplacian variance
        lap_var = self.get_autofocus_score(fbgr)
        # nuclei proxy via color (blue/purple bias)
        b,g,r = cv2.split(fbgr)
        nuc = float(np.mean(np.clip(b.astype(np.float32) - r.astype(np.float32), 0, 255))) / 255.0
        # coherence (structure tensor)
        Jxx = cv2.GaussianBlur(gx*gx, (0,0), 1.0)
        Jyy = cv2.GaussianBlur(gy*gy, (0,0), 1.0)
        Jxy = cv2.GaussianBlur(gx*gy, (0,0), 1.0)
        lam1 = 0.5*(Jxx+Jyy + np.sqrt((Jxx-Jyy)**2 + 4*Jxy*Jxy))
        lam2 = 0.5*(Jxx+Jyy - np.sqrt((Jxx-Jyy)**2 + 4*Jxy*Jxy))
        coherence = float(np.mean(((lam1 - lam2) / (lam1 + lam2 + 1e-6))[mask_bool])) if ten_vals.size else 0.0
        # very light vesselness (Frangi-like proxy using DoG + coherence)
        g_small = cv2.GaussianBlur(gray, (0,0), 1.0)
        g_large = cv2.GaussianBlur(gray, (0,0), 2.0)
        dog = cv2.absdiff(g_small, g_large).astype(np.float32)
        vessel = float(np.mean((dog / (np.max(dog)+1e-6))[mask_bool])) if ten_vals.size else 0.0

        # scores
        sc_comb = float(self.score_combined(fbgr))
        try:
            sc_fib = float(self.score_fiber(fbgr))
        except Exception:
            sc_fib = sc_comb

        return {
            "z": float(target_clamped),
            "coverage": float(coverage),
            "lap_var": float(lap_var),
            "ten_p50": float(ten_p50),
            "ten_p90": float(ten_p90),
            "nuclei": float(nuc),
            "coherence": float(coherence),
            "vessel": float(vessel),
            "score_combined": sc_comb,
            "score_fiber": sc_fib,
        }, target_clamped

    def _make_ds_score_var(self, factor:int):
        """Downsample and compute simple Laplacian-variance (robust coarse metric)."""
        factor = max(1, int(factor))
        def _fn(frame_bgr):
            if factor == 1:
                return self.get_autofocus_score(frame_bgr)
            h, w = frame_bgr.shape[:2]
            nh, nw = max(1, h // factor), max(1, w // factor)
            small = cv2.resize(frame_bgr, (nw, nh))
            return self.get_autofocus_score(small)
        return _fn

    def _coarse_scan_nextgen(self, move_z_func, z_start, current_z,
                              step_um=15, n_steps=5, settle_s=0.25, downsample=True):
        zs, scores = [], []
        # Use robust Laplacian variance for coarse (coverage-independent)
        sc_fn = self._make_ds_score_var(3) if downsample else self.get_autofocus_score
        for i in range(-int(n_steps), int(n_steps) + 1):
            z = clamp_z(float(z_start) + i * float(step_um))
            s, _ = self._measure_at(z, current_z, move_z_func, settle_s=settle_s,
                                    discard=1, samples=2, score_func=sc_fn)
            current_z = z
            zs.append(z); scores.append(s)
            _log(f"📸 NextGen Coarse Z={z:+.0f} µm → Score={s:.2f}")
        i_best = int(np.argmax(scores)) if scores else 0
        return (float(zs[i_best]) if zs else float(z_start),
                float(scores[i_best]) if scores else 0.0,
                current_z, np.array(zs, dtype=float), np.array(scores, dtype=float))

    def _fine_scan_nextgen(self, move_z_func, z_start, current_z,
                           step_um=3, n_steps=5, settle_s=0.25, downsample=False,
                           score_func=None):
        zs, scores = [], []
        sc_fn = score_func or (self._make_ds_score(2) if downsample else None)
        for i in range(-int(n_steps), int(n_steps) + 1):
            z = clamp_z(float(z_start) + i * float(step_um))
            s, _ = self._measure_at(z, current_z, move_z_func, settle_s=settle_s,
                                    discard=2, samples=3, score_func=sc_fn)
            current_z = z
            zs.append(z); scores.append(s)
            _log(f"🔎 NextGen Fine   Z={z:+.0f} µm → Score={s:.2f}")
        i_best = int(np.argmax(scores)) if scores else 0
        return (float(zs[i_best]) if zs else float(z_start),
                float(scores[i_best]) if scores else 0.0,
                current_z, np.array(zs, dtype=float), np.array(scores, dtype=float))

    def _parabolic_fit_nextgen(self, zs_np, scores_np):
        if zs_np is None or scores_np is None or len(zs_np) < 3:
            i = int(np.argmax(scores_np)) if scores_np is not None and len(scores_np) else 0
            return (float(zs_np[i]) if zs_np is not None and len(zs_np) else 0.0,
                    float(scores_np[i]) if scores_np is not None and len(scores_np) else 0.0)
        return self._parabolic_refine(zs_np, scores_np)

    def _validate_candidate_nextgen(self, move_z_func, z_candidate, current_z, settle_s=0.30):
        s_valid, cov, *_ = self._recheck_score_with_coverage(
            move_z_func, z_candidate, current_z, settle_s=max(0.30, settle_s)
        )
        return float(z_candidate), float(s_valid), float(cov)

    # ========================= Next-Gen AF: Main =========================
    def autofocus_nextgen(self, move_z_func,
                           field_id=None, neighbor_focus=None,
                           start_pos_um=0,
                           coarse_step_um=1000, coarse_n=6,
                           fine_step_um=200, fine_n=4,
                           settle_s=0.30,
                           max_duration_s=60.0):
        """Next-Gen Autofokus:
        - Multi-Resolution: downsampled coarse, full-res fine
        - Optional focus map: reuse last focus per field
        - Parabolic fit + micro bracket + coverage check
        """
        if move_z_func is None:
            raise ValueError("move_z_func fehlt")

        if not hasattr(self, "focus_map"):
            self.focus_map = {}

        # 1) Startwert wählen
        if neighbor_focus is not None:
            z0 = float(neighbor_focus)
            _log(f"[NextGenAF] Starte nahe Nachbar-Fokus: {z0:.1f}")
        elif field_id in self.focus_map:
            z0 = float(self.focus_map[field_id])
            _log(f"[NextGenAF] Starte mit gespeichertem Fokus von Feld {field_id}: {z0:.1f}")
        else:
            z0 = float(start_pos_um)
            _log(f"[NextGenAF] Starte bei Startposition: {z0:.1f}")

        current_z = float(start_pos_um)

        # 2) Coarse (downsample) with AI feature gathering
        t0 = time.time()
        status = "ok"
        _log("[NextGenAF] Coarse-Scan (AI features)…")
        coarse_feats = []
        zs_c_list = []
        for i in range(-int(coarse_n), int(coarse_n)+1):
            # time budget guard
            if time.time() - t0 > max_duration_s:
                status = "timeout"
                _log("[NextGenAF] Abbruch: Zeitbudget für Coarse überschritten")
                break
            z = clamp_z(float(z0) + i*float(coarse_step_um))
            feat, current_z = self._measure_features_at(
                z, current_z, move_z_func, settle_s=max(0.30, settle_s), discard=1, downsample_factor=3
            )
            coarse_feats.append(feat); zs_c_list.append(z)
            _log(f"🤖 Coarse AI Z={z:+.0f}µm → cov={feat['coverage']:.2f}, ten90={feat['ten_p90']:.1f}, lapVar={feat['lap_var']:.1f}")
        zs_c = np.array(zs_c_list, dtype=float)
        s_c = np.array([f["lap_var"] for f in coarse_feats], dtype=float)

        # K-Means clustering (k=2) to separate tissue vs glass
        X = np.array([[f["coverage"], f["ten_p90"], f["lap_var"], f["nuclei"], f["coherence"], f["vessel"]] for f in coarse_feats], dtype=np.float32)
        if len(X) >= 4:
            # normalize features per stack
            mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-6
            Xn = (X - mu) / sigma
            K = 2
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
            compact, labels, centers = cv2.kmeans(Xn, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            centers_denorm = centers * sigma + mu
            # score clusters as tissue by heuristic
            tissue_scores = centers_denorm[:,0] + 0.5*centers_denorm[:,1] + 0.4*centers_denorm[:,2] + 0.3*centers_denorm[:,3]
            tissue_idx = int(np.argmax(tissue_scores))
            tissue_mask = (labels.ravel() == tissue_idx)
        else:
            tissue_mask = np.ones((len(X),), dtype=bool)

        # decide structure type
        if np.any(tissue_mask):
            coh_mean = float(np.mean([coarse_feats[i]["coherence"] for i in range(len(coarse_feats)) if tissue_mask[i]]))
            ves_mean = float(np.mean([coarse_feats[i]["vessel"] for i in range(len(coarse_feats)) if tissue_mask[i]]))
        else:
            coh_mean, ves_mean = 0.0, 0.0
        use_fiber_scoring = (coh_mean > 0.35 and ves_mean > 0.15)
        chosen_score_func = self.score_fiber if use_fiber_scoring else None

        # pick best seed z inside tissue cluster by appropriate score
        if np.any(tissue_mask):
            if use_fiber_scoring:
                idxs = [i for i in range(len(coarse_feats)) if tissue_mask[i]]
                i_best = int(max(idxs, key=lambda i: coarse_feats[i]["score_fiber"]))
            else:
                idxs = [i for i in range(len(coarse_feats)) if tissue_mask[i]]
                i_best = int(max(idxs, key=lambda i: coarse_feats[i]["score_combined"]))
        else:
            i_best = int(np.argmax(s_c)) if len(s_c) else 0
        zc = float(zs_c[i_best]) if len(zs_c) else float(z0)

        # if coarse uncertain (flat/edge), expand window
        coarse_rel_span = 0.0
        coarse_edge = (i_best == 0 or i_best == len(zs_c)-1)
        if len(s_c) >= 3:
            smax, smin = float(np.max(s_c)), float(np.min(s_c))
            coarse_rel_span = (smax - smin) / (smax + 1e-6)
        if coarse_rel_span < 0.05 or coarse_edge:
            _log("[NextGenAF] Coarse flat/edge → adaptive window expansion")
            try:
                win = self._expand_window(
                    move_z_func=move_z_func,
                    start_z=zc,
                    init_halfspan_um=max(800, float(coarse_step_um) * float(coarse_n)),
                    step_um=max(100.0, float(coarse_step_um) / 2.0),
                    grow=1.8, max_halfspan_um=6000, settle_s=settle_s,
                    score_func=self.get_autofocus_score,
                )
                z_lo, z_hi, current_z = win["z_lo"], win["z_hi"], win["current_z"]
                # rescan coarse densely inside window with lap-var
                zs_tmp, s_tmp = [], []
                z = z_lo
                step = max(100.0, float(coarse_step_um))
                while z <= z_hi + 1e-6:
                    if time.time() - t0 > max_duration_s:
                        status = "timeout"
                        _log("[NextGenAF] Abbruch: Zeitbudget im Window-Rescan überschritten")
                        break
                    f, current_z = self._measure_features_at(
                        z, current_z, move_z_func, settle_s=max(0.30, settle_s), discard=1, downsample_factor=2
                    )
                    zs_tmp.append(z); s_tmp.append(f["lap_var"])
                    _log(f"📸 NextGen Coarse(rescan) Z={z:+.0f} µm → lapVar={f['lap_var']:.2f}")
                    z += step
                if s_tmp:
                    zs_c = np.array(zs_tmp, dtype=float)
                    s_c = np.array(s_tmp, dtype=float)
                    zc = float(zs_c[int(np.argmax(s_c))])
            except Exception:
                pass

        # Early exit: no texture detected even after coarse
        if len(s_c) >= 3:
            smax, smin = float(np.max(s_c)), float(np.min(s_c))
            coarse_rel_span = (smax - smin) / (smax + 1e-6)
            if smax < 50.0 and coarse_rel_span < 0.03:  # thresholds for downsampled LapVar
                _log("[NextGenAF] Kein kontrastreiches Gewebe erkannt – Abbruch ohne Fein-Suche")
                return {
                    "best_z_um": float(zc),
                    "score": float(smax),
                    "picked": "nextgen",
                    "status": "no_structure",
                    "coarse_points": int(len(zs_c)),
                    "fine_points": 0,
                    "coverage": float(np.mean([f.get("coverage", 0.0) for f in coarse_feats]) if coarse_feats else 0.0),
                    "confidence": 0.0,
                    "ai": {"use_fiber": False, "coarse_tissue_points": int(len(zs_c))},
                    "coarse_profile": {"z": zs_c.tolist(), "s": s_c.tolist()},
                    "fine_profile": {"z": [], "s": []},
                }

        # 3) Fine (full-res)
        _log("[NextGenAF] Fine-Scan (full-res)...")
        if time.time() - t0 > max_duration_s:
            status = "timeout"
            _log("[NextGenAF] Abbruch: Zeitbudget vor Fine überschritten")
            return {
                "best_z_um": float(zc),
                "score": float(np.max(s_c) if len(s_c) else 0.0),
                "picked": "nextgen",
                "status": status,
                "coarse_points": int(len(zs_c)),
                "fine_points": 0,
                "coverage": float(np.mean([f.get("coverage", 0.0) for f in coarse_feats]) if coarse_feats else 0.0),
                "confidence": 0.0,
                "ai": {"use_fiber": False, "coarse_tissue_points": int(len(zs_c))},
                "coarse_profile": {"z": zs_c.tolist(), "s": s_c.tolist()},
                "fine_profile": {"z": [], "s": []},
            }
        # Run two fine scans (unless already decided fiber), pick clearer peak
        zf_c, sf_c, cz1, zs_fc, s_fc = self._fine_scan_nextgen(
            move_z_func, zc, current_z, step_um=fine_step_um, n_steps=fine_n,
            settle_s=settle_s, downsample=False, score_func=None
        )
        if use_fiber_scoring:
            zf_f, sf_f, cz2, zs_ff, s_ff = self._fine_scan_nextgen(
                move_z_func, zc, cz1, step_um=fine_step_um, n_steps=fine_n,
                settle_s=settle_s, downsample=False, score_func=self.score_fiber
            )
        else:
            zf_f, sf_f, cz2, zs_ff, s_ff = zf_c, sf_c, cz1, zs_fc, s_fc
        # decide by relative span; fallback to higher max if equal
        def _rel_span(arr):
            arr = np.asarray(arr)
            if len(arr) < 3:
                return 0.0
            return float((np.max(arr) - np.min(arr)) / (np.max(arr) + 1e-6))
        span_c = _rel_span(s_fc); span_f = _rel_span(s_ff)
        use_fiber = (span_f > span_c * 1.05) or ((span_f >= span_c) and (sf_f >= sf_c)) or use_fiber_scoring
        if use_fiber:
            zf, sf, current_z, zs_f, s_f = zf_f, sf_f, cz2, zs_ff, s_ff
            chosen_score_func = self.score_fiber
            _log("[NextGenAF] Fine uses fiber score")
        else:
            zf, sf, current_z, zs_f, s_f = zf_c, sf_c, cz1, zs_fc, s_fc
            chosen_score_func = None  # default combined
            _log("[NextGenAF] Fine uses combined score")

        # 4) Parabel-Fit
        z_fit, s_fit = self._parabolic_fit_nextgen(zs_f, s_f)
        _log(f"[NextGenAF] Parabel-Fit: z={z_fit:.1f}, Score={s_fit:.2f}")

        # 5) Micro-Bracket (kleine Schritte um Fit)
        # pick a much smaller micro step than fine-step
        micro_step = max(3, min(12, int(max(1, fine_step_um // 5))))
        z_micro, s_micro, current_z = self._micro_bracket(
            z_fit, current_z, move_z_func,
            micro_step_um=micro_step, span=2, settle_s=max(0.30, settle_s),
            score_func=chosen_score_func
        )
        _log(f"[NextGenAF] Micro-Bracket: z={z_micro:.1f}, Score={s_micro:.2f}")

        # 6) Validation (Coverage + robust score)
        z_valid, s_valid, cov = self._validate_candidate_nextgen(
            move_z_func, z_micro, current_z, settle_s=max(0.30, settle_s)
        )
        _log(f"[NextGenAF] Validierter Fokus: z={z_valid:.1f}, Score={s_valid:.2f}, cov={cov:.2f}")

        # 7) Fallbacks bei Unsicherheit: flache Kurve oder Rand-Peak
        final_z, final_score = z_valid, s_valid
        rel_span = 0.0
        edge_peak = False
        try:
            if len(s_f) >= 3:
                smax = float(np.max(s_f)); smin = float(np.min(s_f))
                rel_span = (smax - smin) / (smax + 1e-6)
                i_best = int(np.argmax(s_f))
                edge_peak = (i_best == 0 or i_best == len(s_f) - 1)
        except Exception:
            pass
        if s_valid < 0.1 or rel_span < 0.05 or edge_peak:
            _log("[NextGenAF] Unsicherer Fokus → Fallback auf Smart-Autofokus")
            try:
                res = self.autofocus_smart(
                    move_z_func=move_z_func,
                    start_pos_um=current_z,
                    coarse_step_um=250, fine_step_um=80,
                    init_halfspan_um=700, max_halfspan_um=3500,
                    grow=1.7, settle_s=max(0.30, settle_s), fine_span=2,
                    topk_validate=3, use_global_scout=True
                )
                final_z = float(res.get("best_z_um", z_valid))
                final_score = float(res.get("score", s_valid))
                current_z = final_z
            except Exception:
                pass

        # 8) Motor bewegen
        move_z_func(final_z - current_z)
        time.sleep(0.25)
        _log(f"[NextGenAF] Final: z={final_z:.1f}, score={final_score:.2f}")

        # 9) Fokus-Map aktualisieren
        if field_id is not None:
            self.focus_map[field_id] = float(final_z)

        # simple confidence estimate from fine profile
        confidence = 0.0
        try:
            if len(s_f) >= 3:
                i = int(np.argmax(s_f))
                s1 = float(s_f[i])
                s2 = float(np.partition(s_f, -2)[-2]) if len(s_f) >= 2 else 0.0
                diff = s1 - s2
                curvature = max(0.0, float(2 * s_f[i] - s_f[max(0, i-1)] - s_f[min(len(s_f)-1, i+1)]))
                confidence = float(max(0.0, diff) * (1.0 + 0.3 * curvature))
        except Exception:
            confidence = 0.0

        # ensure profiles for serialization
        fine_points_count = int(len(zs_f)) if 'zs_f' in locals() else 0
        fine_prof_z = zs_f.tolist() if 'zs_f' in locals() else []
        fine_prof_s = s_f.tolist() if 's_f' in locals() else []

        return {
            "best_z_um": float(final_z),
            "score": float(final_score),
            "picked": "nextgen",
            "status": status,
            "coarse_points": int(len(zs_c)),
            "fine_points": fine_points_count,
            "coverage": float(cov),
            "confidence": float(confidence),
            "ai": {
                "use_fiber": bool(use_fiber),
                "coarse_tissue_points": int(np.sum(tissue_mask)) if 'tissue_mask' in locals() else int(len(zs_c)),
            },
            "coarse_profile": {"z": zs_c.tolist(), "s": s_c.tolist()},
            "fine_profile": {"z": fine_prof_z, "s": fine_prof_s},
        }

    # ---------------- Smart (Legacy, erste Version) ----------------
    def autofocus_smart_legacy(self, move_z_func,
                               start_pos_um=0,
                               coarse_step_um=200, fine_step_um=50,
                               halfspan_um=1200, settle_s=0.25, fine_span=2):
        """Erste, einfachere Smart-Autofokus-Version:
        - fixes Fenster um Startposition
        - Coarse-Scan → Fine-Scan
        - Parabel-Refine und Micro-Bracketing
        - nutzt score_combined ohne zusätzliche Scouts/Heuristiken
        """
        if move_z_func is None:
            raise ValueError("move_z_func fehlt")

        seed = float(start_pos_um)
        z_lo = clamp_z(seed - float(halfspan_um))
        z_hi = clamp_z(seed + float(halfspan_um))
        current_z = seed
        print(f"🪟 SMART(legacy) Fenster: [{z_lo:.0f}, {z_hi:.0f}] µm")

        # Coarse
        zs, sc = [], []
        z = z_lo
        while z <= z_hi + 1e-6:
            s, _ = self._measure_at(z, current_z, move_z_func, settle_s)
            current_z = z
            zs.append(z); sc.append(s)
            print(f"📸 SMART(legacy) Coarse Z={z:+.0f} µm → Score={s:.2f}")
            z += float(coarse_step_um)
        if not zs:
            return {"best_z_um": float(seed), "score": 0.0, "picked": "none", "window": [float(z_lo), float(z_hi)], "coarse_points": 0, "fine_points": 0}

        zs = np.array(zs, dtype=float)
        sc = np.array(sc, dtype=float)
        z_center = float(zs[int(np.argmax(sc))])

        # Falls coarse-Profil nahezu flach → weite Makro-Suche
        if sc.size > 0:
            smax = float(np.max(sc)); smin = float(np.min(sc))
            rel_span = (smax - smin) / (smax + 1e-6)
        else:
            rel_span = 0.0
        if rel_span < 0.06:
            _log("⚪ SMART(fiber): Coarse profile flat → macro sweep")
            macro_half = min(float(halfspan_um) * 2.0, 8000.0)
            z_lo2 = clamp_z(z_center - macro_half)
            z_hi2 = clamp_z(z_center + macro_half)
            macro_step = max(int(coarse_step_um) * 2, 400)
            macro_zs, macro_sc = [], []
            z = z_lo2
            while z <= z_hi2 + 1e-6:
                s, _ = self._measure_at(
                    z, current_z, move_z_func, settle_s,
                    score_func=self.score_fiber,
                )
                current_z = z
                macro_zs.append(z); macro_sc.append(s)
                _log(f"📈 SMART(fiber) Macro Z={z:+.0f} µm → Score={s:.2f}")
                z += macro_step
            if macro_sc:
                idx = int(np.argmax(macro_sc))
                z_pick = float(macro_zs[idx])
                m_smax = float(np.max(macro_sc)); m_smin = float(np.min(macro_sc))
                m_rel = (m_smax - m_smin) / (m_smax + 1e-6)
                if m_rel < 0.05:
                    # direkte weite Micro-Klammer
                    z_micro, s_micro, current_z = self._micro_bracket(
                        z_pick, current_z, move_z_func,
                        micro_step_um=30, span=4, settle_s=max(0.35, settle_s),
                        score_func=self.score_fiber,
                    )
                    move_z_func(z_micro - current_z); current_z = z_micro
                    print(f"🟢 SMART(fiber) Ergebnis: Z={z_micro:.1f} µm (wide-micro, Score≈{s_micro:.2f})")
                    return {
                        "best_z_um": float(z_micro),
                        "score": float(s_micro),
                        "picked": "fiber-wide-micro",
                        "window": [float(z_lo2), float(z_hi2)],
                        "coarse_points": int(len(zs)),
                        "fine_points": 0,
                    }
                else:
                    z_center = z_pick
        # relative Spannweite im coarse-Profil (für einfache Konfidenz)
        smax = float(np.max(sc)) if len(sc) else 0.0
        smin = float(np.min(sc)) if len(sc) else 0.0
        rel_span = (smax - smin) / (smax + 1e-6)

        # Fine
        fz, fs = [], []
        step_f = max(1.0, float(fine_step_um))
        z = clamp_z(z_center - fine_span * step_f)
        z_end = clamp_z(z_center + fine_span * step_f)
        while z <= z_end + 1e-6:
            s, _ = self._measure_at(z, current_z, move_z_func, settle_s)
            current_z = z
            fz.append(z); fs.append(s)
            print(f"🔎 SMART(legacy) Fine   Z={z:+.0f} µm → Score={s:.2f}")
            z += step_f
        # Falls zu wenige Punkte (z.B. wegen sehr großer Schrittweite) → min. 3 Punkte um Peak
        if len(fz) < 3:
            # stelle ein symmetrisches Triplet um z_center sicher
            fz = [clamp_z(z_center - step_f), z_center, clamp_z(z_center + step_f)]
            fs = []
            cz = current_z
            for zz in fz:
                s, _ = self._measure_at(zz, cz, move_z_func, settle_s)
                fs.append(s); cz = zz
            current_z = cz
        fz = np.array(fz, dtype=float)
        fs = np.array(fs, dtype=float)

        # Parabel nur sinnvoll bei ≥3 Punkten, sonst best-of
        if len(fz) >= 3:
            z_ref, s_ref = self._parabolic_refine(fz, fs)
        else:
            i_best = int(np.argmax(fs)) if len(fs) else 0
            z_ref, s_ref = (float(fz[i_best]), float(fs[i_best])) if len(fs) else (z_center, float(smax))
        z_micro, s_micro, current_z = self._micro_bracket(
            z_ref, current_z, move_z_func,
            micro_step_um=10, span=2, settle_s=max(0.30, settle_s)
        )
        picked = "legacy-micro" if s_micro >= s_ref else "legacy-parabola"
        final_z = z_micro if picked == "legacy-micro" else z_ref
        final_s = s_micro if picked == "legacy-micro" else s_ref

        move_z_func(final_z - current_z); time.sleep(0.25)
        # einfache Konfidenz (ähnlich wie advanced, aber ohne zusätzliche Heuristiken)
        if len(fs) >= 3:
            i = int(np.argmax(fs))
            s1 = float(fs[i])
            s2 = float(np.partition(fs, -2)[-2]) if len(fs) >= 2 else 0.0
            margin = max(0.0, (s1 - s2) / (s1 + 1e-6))
            if 0 < i < len(fs) - 1:
                sec = float(fs[i-1] - 2*fs[i] + fs[i+1])
                curv = max(0.0, -sec / (abs(fs[i]) + 1e-6))
                curv = float(np.clip(curv * 5.0, 0.0, 1.0))
            else:
                curv = 0.3
        else:
            margin, curv = 0.3, 0.2
        conf = float(np.clip(0.4 * margin + 0.3 * curv + 0.3 * max(0.0, rel_span), 0.0, 1.0))

        # final Coverage (ohne zusätzliche Bewegung)
        try:
            _s, cov, *_rest = self._recheck_score_with_coverage(
                move_z_func, final_z, final_z, settle_s=max(0.30, settle_s)
            )
        except Exception:
            cov = None

        print(f"🟢 SMART(legacy) Ergebnis: Z={final_z:.1f} µm ({picked}, Score≈{final_s:.2f}), conf={conf:.2f}, cov={(cov if cov is not None else float('nan')):.2f}")
        return {
            "best_z_um": float(final_z),
            "score": float(final_s),
            "picked": picked,
            "window": [float(z_lo), float(z_hi)],
            "coarse_points": int(len(zs)),
            "fine_points": int(len(fz)),
            "confidence": float(conf),
            "final_coverage": (None if cov is None else float(cov)),
    }

    # ---------------- Smart (Fiber-focused) ----------------
    def autofocus_smart_fiber(self, move_z_func,
                               start_pos_um=0,
                               coarse_step_um=200, fine_step_um=50,
                               halfspan_um=1200, settle_s=0.25, fine_span=2,
                               adaptive=True):
        """Wie die Legacy-Smart-Variante, aber mit faser-orientiertem Scoring.
        Nützlich, wenn man gezielt auf kollagene Fasern/lineare Strukturen scharf stellen möchte.
        Keine Scouts/Flatness-Logik, nur Score-Tausch.
        """
        if move_z_func is None:
            raise ValueError("move_z_func fehlt")

        seed = float(start_pos_um)
        current_z = seed
        if adaptive:
            win = self._expand_window(
                move_z_func, seed, init_halfspan_um=float(halfspan_um),
                step_um=float(coarse_step_um), grow=1.7, max_halfspan_um=4000,
                settle_s=settle_s, score_func=self.score_fiber,
            )
            z_lo, z_hi, current_z = win["z_lo"], win["z_hi"], win["current_z"]
            print(f"🪟 SMART(fiber) Fenster(adaptive): [{z_lo:.0f}, {z_hi:.0f}] µm (clipped={win['clipped']})")
        else:
            z_lo = clamp_z(seed - float(halfspan_um))
            z_hi = clamp_z(seed + float(halfspan_um))
            print(f"🪟 SMART(fiber) Fenster: [{z_lo:.0f}, {z_hi:.0f}] µm")

        # Coarse
        zs, sc = [], []
        z = z_lo
        while z <= z_hi + 1e-6:
            s, _ = self._measure_at(
                z, current_z, move_z_func, settle_s,
                score_func=self.score_fiber,
            )
            current_z = z
            zs.append(z); sc.append(s)
            print(f"📸 SMART(fiber) Coarse Z={z:+.0f} µm → Score={s:.2f}")
            z += float(coarse_step_um)
        if not zs:
            return {"best_z_um": float(seed), "score": 0.0, "picked": "none", "window": [float(z_lo), float(z_hi)], "coarse_points": 0, "fine_points": 0}

        zs = np.array(zs, dtype=float)
        sc = np.array(sc, dtype=float)
        z_center = float(zs[int(np.argmax(sc))])

        # Fine
        fz, fs = [], []
        step_f = max(1.0, float(fine_step_um))
        z = clamp_z(z_center - fine_span * step_f)
        z_end = clamp_z(z_center + fine_span * step_f)
        while z <= z_end + 1e-6:
            s, _ = self._measure_at(
                z, current_z, move_z_func, settle_s,
                score_func=self.score_fiber,
            )
            current_z = z
            fz.append(z); fs.append(s)
            print(f"🔎 SMART(fiber) Fine   Z={z:+.0f} µm → Score={s:.2f}")
            z += step_f
        if len(fz) < 3:
            fz = [clamp_z(z_center - step_f), z_center, clamp_z(z_center + step_f)]
            fs = []
            cz = current_z
            for zz in fz:
                s, _ = self._measure_at(
                    zz, cz, move_z_func, settle_s,
                    score_func=self.score_fiber,
                )
                fs.append(s); cz = zz
            current_z = cz
        fz = np.array(fz, dtype=float)
        fs = np.array(fs, dtype=float)

        if len(fz) >= 3:
            z_ref, s_ref = self._parabolic_refine(fz, fs)
        else:
            i_best = int(np.argmax(fs)) if len(fs) else 0
            z_ref, s_ref = (float(fz[i_best]), float(fs[i_best])) if len(fs) else (z_center, float(np.max(sc)))

        z_micro, s_micro, current_z = self._micro_bracket(
            z_ref, current_z, move_z_func,
            micro_step_um=10, span=2, settle_s=max(0.30, settle_s),
            score_func=self.score_fiber,
        )
        picked = "fiber-micro" if s_micro >= s_ref else "fiber-parabola"
        final_z = z_micro if picked == "fiber-micro" else z_ref
        final_s = s_micro if picked == "fiber-micro" else s_ref

        move_z_func(final_z - current_z); time.sleep(0.25)
        print(f"🟢 SMART(fiber) Ergebnis: Z={final_z:.1f} µm ({picked}, Score≈{final_s:.2f})")
        return {
            "best_z_um": float(final_z),
            "score": float(final_s),
            "picked": picked,
            "window": [float(z_lo), float(z_hi)],
            "coarse_points": int(len(zs)),
            "fine_points": int(len(fz)),
        }

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
