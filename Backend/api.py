from flask import Blueprint, jsonify, request, Response
from cameraController import CameraController
from motorController import MotorController
from stitchingController import stitching_session
import cv2
import time
import gc
import threading, time, cv2, base64, gc, traceback
from collections import deque

api = Blueprint('api', __name__)
camera = CameraController()
motor = MotorController()
_autofocus_lock = threading.Lock()
_frame_times = deque(maxlen=120)  # ~6 s bei 20 FPS


from collections import deque
import uuid

# Zustands-Container für aktuelle Bewegung (thread-safe)
_move_state = {
    "moving": False,
    "move_id": None,
    "axis": None,
    "dz_um": 0,
    "start_z": None,
    "target_z": None,
    "started_at": None,
    "eta_ms": None,
    "v_um_s": None,
}
_state_lock = threading.Lock()

# simple Geschwindigkeitsannahme (umgehen „Raten“ im FE):
# Wenn MotorController eine realistische max-Speed kennt, nutze die!
DEFAULT_V_UM_S = 3000  # 3 mm/s – anpassen, falls bekannt

def estimate_eta_ms(dist_um, v_um_s=DEFAULT_V_UM_S, ramp_factor=1.15):
    """ETA in ms für einfachen Trapez-/Dreiecks-Fahrplan (grob, aber stabil)."""
    dist_um = abs(float(dist_um))
    if v_um_s <= 0:
        v_um_s = DEFAULT_V_UM_S
    base = (dist_um / v_um_s) * 1000.0
    return int(base * ramp_factor)  # etwas Puffer für Rampen

_frame_times = deque(maxlen=120)  # ~6 s bei 20 FPS

def gen_frames():
    try:
        while True:
            frame = camera.get_frame()
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # Timestamp für FPS
            _frame_times.append(time.time())

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)    # 20 FPS soll
            gc.collect()
    except GeneratorExit:
        print("🚫 Verbindung zum Client getrennt – Stream gestoppt.")
    except Exception as e:
        print(f"❌ Fehler im Stream: {e}")

def move_z_func(delta_um, return_position=False):
    if return_position:
        return motor.get_position()
    motor.move(z=delta_um)
    print(f"🔄 Z relativ bewegt: {delta_um:+} µm")


@api.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return resp


@api.route('/video_stats', methods=['GET'])
def video_stats():
    fps = 0.0
    if len(_frame_times) >= 2:
        dt = _frame_times[-1] - _frame_times[0]
        if dt > 0:
            fps = (len(_frame_times) - 1) / dt
    return jsonify({"fps": fps})

@api.route('/autofocus', methods=['POST'])
def autofocus_route():
    if not _autofocus_lock.acquire(blocking=False):
        return jsonify({"status": "busy", "message": "Autofokus läuft bereits."}), 409
    try:
        payload = request.get_json(silent=True) or {}
        start_pos_um = float(payload.get("start_pos_um", 0))
        step_um = int(payload.get("step_um", 200))
        init_steps = int(payload.get("initial_sweep_steps", 4))
        ext_steps = int(payload.get("sweep_extension_steps", 4))
        max_steps = int(payload.get("max_total_steps", 20))

        """  result = camera.autofocus(
            move_z_func=move_z_func,
            start_pos_um=start_pos_um,
            step_um=step_um,
            initial_sweep_steps=init_steps,
            sweep_extension_steps=ext_steps,
            max_total_steps=max_steps,
        )
        """
        result = camera.autofocus(
            move_z_func=move_z_func
        )
        return jsonify({"status": "ok", **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        _autofocus_lock.release()

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

@api.route('/image_dimensions')
def image_dimensions():
    frame = camera.get_frame(discard=2)
    if frame is None:
        return jsonify({"status": "error", "message": "Kein Frame von der Kamera."}), 503
    h, w = frame.shape[:2]
    pixel_size_um = 2
    return jsonify({
        "status": "success",
        "width_px": w,
        "height_px": h,
        "width_um": w * pixel_size_um,
        "height_um": h * pixel_size_um,
    })

@api.route('/move', methods=['POST'])
def move():
    try:
        data = request.json or {}
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        z = int(data.get('z', 0))

        # Nur Z-ETA rechnen (du kannst analog X/Y ergänzen)
        start_pos_str = motor.get_position()              # z. B. "X:.. Y:.. Z:.."
        start_z = parse_z_from_response(start_pos_str)

        target_z = None
        if isinstance(z, (int, float)) and z != 0:
            # relative Bewegung → Ziel ist ungefähr:
            target_z = start_z + z
            eta_ms = estimate_eta_ms(z)
        else:
            eta_ms = 0

        move_id = uuid.uuid4().hex
        with _state_lock:
            _move_state.update({
                "moving": True,
                "move_id": move_id,
                "axis": {"x": x, "y": y, "z": z},
                "dz_um": z,
                "start_z": start_z,
                "target_z": target_z,
                "started_at": time.time(),
                "eta_ms": eta_ms,
                "v_um_s": DEFAULT_V_UM_S,
            })

        # Bewegung asynchron starten (damit /move sofort antwortet)
        def _worker():
            try:
                motor.move(x, y, z)  # deine bestehende Funktion (blocking)
            finally:
                with _state_lock:
                    _move_state["moving"] = False

        threading.Thread(target=_worker, daemon=True).start()

        return jsonify({
            'status': 'moving',
            'response': 'started',
            'move_id': move_id,
            'planned': {
                'dz_um': z,
                'eta_ms': eta_ms,
                'v_um_s': DEFAULT_V_UM_S,
                'start_z': start_z,
                'target_z': target_z,
            },
            'started_at': _move_state["started_at"],
        })
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

@api.route('/status', methods=['GET'])
def status():
    z_str = motor.get_position()
    z_now = parse_z_from_response(z_str)
    with _state_lock:
        st = dict(_move_state)  # shallow copy
    # restliche ETA grob abschätzen
    eta_left_ms = None
    if st["moving"] and st["target_z"] is not None and z_now is not None:
        remaining = abs(st["target_z"] - z_now)
        v = st["v_um_s"] or DEFAULT_V_UM_S
        eta_left_ms = int((remaining / max(v, 1)) * 1000)

    return jsonify({
        "moving": st["moving"],
        "move_id": st["move_id"],
        "axis": st["axis"],
        "z_now": z_now,
        "target_z": st["target_z"],
        "eta_planned_ms": st["eta_ms"],
        "eta_left_ms": eta_left_ms,
        "started_at": st["started_at"],
        "v_um_s": st["v_um_s"],
    })
