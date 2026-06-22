import cv2
import time
import shutil
import datetime
import psutil
from pathlib import Path
from flask import Flask, Response, send_from_directory

ROOT_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = ROOT_DIR / "webapp"
START_TIME = time.time()

app = Flask(__name__, static_folder=str(WEBAPP_DIR), static_url_path="/static")

CAMERA_MANAGER = None


@app.route("/")
def index():
    return send_from_directory(WEBAPP_DIR, "control_panel.html")
    # or use "index.html" if that is your real file name


@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    if CAMERA_MANAGER is None:
        return {"ok": False, "error": "Camera manager not initialized"}, 500

    try:
        acquiring = CAMERA_MANAGER.start_acquisition()
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

    return {"ok": True, "message": acquiring}


@app.route("/api/camera/stream")
def camera_stream():
    if CAMERA_MANAGER is None:
        return {"ok": False, "error": "Camera manager not initialized"}, 500

    def generate():
        while True:
            frame = CAMERA_MANAGER.get_frame_bgr()

            ok, jpg = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpg.tobytes()
                + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/stop", methods=["POST"])
@app.route("/api/system/stop_camera", methods=["POST"])
def stop_camera():
    if CAMERA_MANAGER is not None:
        CAMERA_MANAGER.close()

    return {"ok": True, "message": "Camera stopped"}

def run_server(camera_manager):
    global CAMERA_MANAGER
    CAMERA_MANAGER = camera_manager

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
    )

def get_time_sync_status():
    """
    Temporary stub.

    Later this will query the Time Service /
    GPS / NTP synchronization subsystem.
    """

    return {
        "time_sync": "UNKNOWN",
        "drift_ms": None
    }

@app.route("/system/status", methods=["GET"])
def system_status():
    disk = shutil.disk_usage("/")

    cpu_temp_c = 0.0
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            first_sensor = next(iter(temps.values()))
            if first_sensor:
                cpu_temp_c = first_sensor[0].current
    except Exception:
        cpu_temp_c = 0.0

    time_status = get_time_sync_status()

    return {
        "utc_now": datetime.datetime.utcnow().isoformat() + "Z",
        "cpu_temp_c": cpu_temp_c,
        "cpu_load_pct": psutil.cpu_percent(interval=0.1),
        "disk_used_gb": disk.used / 1024**3,
        "disk_total_gb": disk.total / 1024**3,
        "uptime_s": time.time() - START_TIME,

        # Time service
        "time_sync": time_status["time_sync"],
        "drift_ms": time_status["drift_ms"]
    }