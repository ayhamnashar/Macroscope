from picamera2 import Picamera2
import threading
import cv2
import time
import numpy as np


class CameraController:
    def __init__(self):
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "BGR888"},
                controls={
                    "AwbEnable": 1,  
                    "AwbMode": 1,   
                    "ColourGains": (1.0, 1.0)
                }
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.lock = threading.Lock()
            print("✅ Kamera erfolgreich gestartet!")
        except Exception as e:
            print("❌ Kamera konnte nicht gestartet werden:", e)
            self.picam2 = None

    def autofocus(self, move_z_func, delay=0.7, frames_to_average=3):
        # --- Grober, weiter Sweep ---
        grob_step = 500
        grob_steps = 20
        print("🔍 Starte Grobsweep (weiter Bereich)...")

        try:
            start_z = move_z_func(0, return_position=True)
        except Exception as e:
            print(f"⚠️ Fehler beim Holen der Z-Position: {e}")
            return

        positions = []
        sharpnesses = []

        for i in range(-grob_steps, grob_steps + 1):
            target_z = start_z + i * grob_step
            move_z_func(target_z)
            time.sleep(delay)
            # Mittelung mehrerer Frames
            scores = []
            for j in range(frames_to_average + 1):
                frame = self.get_frame()
                if j > 0:
                    scores.append(self.get_autofocus_score(frame))
                time.sleep(0.05)
            score = np.mean(scores)
            print(f"Grob Z={target_z:.2f} | Schärfe={score:.2f}")
            positions.append(target_z)
            sharpnesses.append(score)

        max_idx = int(np.argmax(sharpnesses))
        best_z = positions[max_idx]
        print(f"🎯 Grob-Maximum bei Z={best_z:.2f} (Score={sharpnesses[max_idx]:.2f})")

        # Backlash-Korrektur: immer von unten anfahren
        backlash = grob_step * 2
        pre_z = best_z - backlash
        print(f"⬇️ Fahre deutlich unter die beste Position: {pre_z:.2f}")
        move_z_func(pre_z)
        time.sleep(delay * 2)

        # Jetzt ZWEIMAL zur besten Position fahren, um Spiel auszugleichen
        print(f"⬆️ Fahre zur besten Position: {best_z:.2f} (1. Mal)")
        move_z_func(best_z)
        time.sleep(delay * 2)

        print(f"⬆️ Fahre zur besten Position: {best_z:.2f} (2. Mal)")
        move_z_func(best_z)
        time.sleep(delay * 2)

        # Noch einmal sicherheitshalber auf best_z fahren
        print(f"➡️ Fahre ein drittes Mal exakt auf best_z: {best_z:.2f}")
        move_z_func(best_z)
        time.sleep(delay * 2)

        # Dummy-Frame verwerfen
        _ = self.get_frame()
        time.sleep(0.1)

        # Jetzt das finale Bild holen
        frame = self.get_frame()
        score = self.get_autofocus_score(frame)
        print(f"✅ Endgültige Schärfe bei Z={best_z:.2f}: {score:.2f}")

        return best_z

    def get_frame(self):
        """Fängt ein Bild ein und prüft, ob die Kamera läuft."""
        if not self.picam2:
            print("⚠️ Fehler: Kamera nicht verfügbar!")
            return None

        with self.lock:
            frame = self.picam2.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame is not None else None

    def get_autofocus_score(self, frame):
        if frame is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # --- GANZES BILD als ROI, Laplacian für robustere Schärfemessung ---
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

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