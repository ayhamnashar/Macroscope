from flask import Blueprint, jsonify, request, Response
from cameraController import CameraController
from motorController import MotorController
import cv2
import time
import gc

api = Blueprint('api', __name__)

camera = CameraController()
motor = MotorController()

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

@api.route('/autofocus', methods=['POST'])
def autofocus():
    frame = camera.get_frame()
    score = camera.get_autofocus_score(frame)
    return jsonify({'autofocus_score': score})

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
    return jsonify({'position': response})

@api.route('/release', methods=['POST'])
def release():
    response = motor.release_motors()
    return jsonify({'status': 'released', 'response': response})
