from picamera2 import Picamera2
import threading
import cv2

class CameraController:
    def __init__(self):
        try:
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.preview_configuration)
            self.picam2.start()
            self.lock = threading.Lock()  # ✅ Lock initialisieren
        except RuntimeError as e:
            print("❌ Kamera konnte nicht gestartet werden:", e)
            self.picam2 = None

    def get_frame(self):
        with self.lock:  # ✅ Zugriff absichern
            frame = self.picam2.capture_array()
            #print("📸 Frame gesendet")
            return frame

    def get_autofocus_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def zoom(self, direction):
        with self.lock:  # ✅ Auch hier sperren
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
