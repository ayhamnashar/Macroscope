import os
import cv2
import numpy as np
import time

class StitchingSession:
    def __init__(self, base_dir="stitching_images"):
        self.base_dir = base_dir
        self.images = []
        self.positions = []
        self.active = False
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def start(self):
        self.images = []
        self.positions = []
        self.active = True
        print("🟢 Stitching-Session gestartet.")

    def add_image(self, img, pos):
        if not self.active:
            print("⚠️ Keine aktive Stitching-Session!")
            return
        timestamp = int(time.time() * 1000)
        filename = os.path.join(self.base_dir, f"img_{timestamp}.jpg")
        cv2.imwrite(filename, img)
        self.images.append(filename)
        self.positions.append(pos)
        print(f"📸 Bild gespeichert: {filename} @ {pos}")

    def finish(self):
        self.active = False
        print("🛑 Stitching-Session beendet.")

        # Lade alle gespeicherten Bilder
        imgs = [cv2.imread(img_path) for img_path in self.images if os.path.exists(img_path)]
        if len(imgs) < 2:
            print("⚠️ Zu wenige Bilder zum Stitchen!")
            return self.images, self.positions, None

        # Prüfe Bildgrößen
        shapes = [img.shape for img in imgs if img is not None]
        if len(set(shapes)) > 1:
            print("⚠️ Bilder haben unterschiedliche Größen!")
            return self.images, self.positions, None

        # Verkleinere alle Bilder auf 640x480 (oder deine Wunschgröße)
        resized_imgs = [cv2.resize(img, (640, 480)) for img in imgs if img is not None]

        # OpenCV Stitching
        try:
            stitcher = cv2.Stitcher_create()
            status, stitched = stitcher.stitch(resized_imgs)
            if status == cv2.Stitcher_OK:
                stitched_path = os.path.join(self.base_dir, "stitched_result.jpg")
                cv2.imwrite(stitched_path, stitched)
                print(f"✅ Stitching erfolgreich: {stitched_path}")
                return self.images, self.positions, stitched_path
            else:
                print(f"❌ Stitching fehlgeschlagen! Status: {status}")
                return self.images, self.positions, None
        except Exception as e:
            print(f"❌ Fehler beim Stitching: {e}")
            return self.images, self.positions, None

# Singleton-Session für das API
stitching_session = StitchingSession()