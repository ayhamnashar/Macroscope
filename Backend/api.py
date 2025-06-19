from flask import Blueprint, jsonify, request, Response
from cameraController import CameraController
from motorController import MotorController
from stitchingController import stitching_session
import cv2
import time
import gc
import base64
import traceback

api = Blueprint('api', __name__)

camera = CameraController()
motor = MotorController()

def encode_image(frame):
    """
    Kodiert ein Bild (OpenCV/Numpy) zu base64 JPEG
    """
    _, buffer = cv2.imencode('.jpg', frame)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return encoded

def gen_frames():
    try:
        while True:
            frame = camera.get_frame()
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)    # 20 FPS
            gc.collect()  # ✅ Speicher freigeben

    except GeneratorExit:
        print("🚫 Verbindung zum Client getrennt – Stream gestoppt.")
    except Exception as e:
        print(f"❌ Fehler im Stream: {e}")
        
@api.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@api.route('/zoom', methods=['POST'])
def zoom():
    direction = request.json.get('direction')
    camera.zoom(direction)
    return jsonify({'status': 'ok'})

#Extract Z position from serial response
def parse_z_from_response(response):
    try:
        parts = response.split()
        for part in parts:
            if part.startswith("Z:"):
                return float(part[2:])
    except Exception as e:
        print(f"Fehler beim Parsen der Z-Position: {e}")
    return 0.0

#Move Z motor or get position
def move_z_func(z, return_position=False):
    if return_position:
        response = motor.get_position()
        parsed_z = parse_z_from_response(response)
        print(f"📍 Aktuelle Z-Position laut Motor: {parsed_z}")
        return parsed_z
    motor.move(z=z)
    time.sleep(0.3)  # Mehr Zeit für Stabilität
    print(f"🔄 Z-Achse bewegt auf {z} µm")

@api.route('/autofocus', methods=['POST'])
def autofocus():
    try:
        best_z = camera.autofocus(move_z_func)  # Kein Parameter mehr nötig
        return jsonify({
            'status': 'success',
            'z_position': best_z,
            'message': f'Autofocus completed at Z position {best_z}'
        })
    except Exception as e:
        print(f"❌ Autofocus error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/image_dimensions')
def image_dimensions():
    frame = camera.get_frame()
    h, w = frame.shape[:2]
    pixel_size_um = 2
    return jsonify({
        'width_px': w,
        'height_px': h,
        'width_um': w * pixel_size_um,
        'height_um': h * pixel_size_um
    })

@api.route('/test_map', methods=['POST'])
def test_map():
    try:
        print("🧪 Testkarte gestartet...")

        rows, cols = 2, 2  # 3x3 Raster
        pixel_size_um = 2  # µm pro Pixel (kannst du später dynamisch machen)

        # Testbild holen, um Bildgröße zu kennen
        test_frame = camera.get_frame()
        height, width = test_frame.shape[:2]
        print(f"📐 Bildgröße: {width}x{height} px")

        # Schrittweite berechnen: volle Bildbreite in µm
        step = int(width * pixel_size_um)
        print(f"📏 Schrittweite automatisch berechnet: {step} µm")

        # Aktuelle Position
        
        motor.move_absolute(target_x=0, target_y=0, target_z=0)
        time.sleep(3.0)

        images = []

        for row in range(rows):
            for col in range(cols):
                if col != 0:
                    motor.move(x=step, y=0)  # Nur nach rechts bewegen
                    time.sleep(1.0)

                print("🔍 Autofokus...")
                camera.autofocus(move_z_func=move_z_func)

                print("📸 Aufnahme...")
                frame = camera.get_frame()
                encoded = encode_image(frame)
                images.append({'x': col * step, 'y': row * step, 'image': encoded})

            if row != rows - 1:
                # Nächste Zeile: erst zurück zur Spalte 0, dann 1 Schritt nach unten
                motor.move(x=-step * (cols - 1), y=step)
                time.sleep(1.0)

        print("🧩 Testkarte fertig.")
        return jsonify({'status': 'ok', 'images': images, 'step': step})

    except Exception as e:
        print("❌ Fehler bei /test_map:")
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@api.route('/move', methods=['POST'])
def move():
    try:
        data = request.json
        x = data.get('x', 0)
        y = data.get('y', 0)
        z = data.get('z', 0)
        response = motor.move(x, y, z)
        return jsonify({'status': 'moved', 'response': response})
    except Exception as e:
        print(f"❌ Fehler in /move: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/position', methods=['GET'])
def position():
    response = motor.get_position()
    if response.startswith("SerialException:"):
        return jsonify({'error': response}), 500
    return jsonify({'position': response})

@api.route('/release', methods=['POST'])
def release():
    response = motor.release_motors()
    return jsonify({'status': 'released', 'response': response})

@api.route('/stitching/start', methods=['POST'])
def stitching_start():
    stitching_session.start()
    return jsonify({'status': 'started'})

@api.route('/stitching/capture', methods=['POST'])
def stitching_capture():
    frame = camera.get_frame()
    pos = motor.get_position()
    stitching_session.add_image(frame, pos)
    return jsonify({'status': 'captured'})

@api.route('/stitching/finish', methods=['POST'])
def stitching_finish():
    images, positions, stitched_path = stitching_session.finish()
    return jsonify({
        'status': 'finished',
        'images': images,
        'positions': positions,
        'stitched': stitched_path
    })
