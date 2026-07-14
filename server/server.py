import time
import shutil
import psutil
import datetime
import threading
import numpy as np

from pathlib import Path
from waitress import serve
from flask import Flask, Response, send_from_directory
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

# *********************************************** #
# JPEG encoder selection
# *********************************************** #
_turbo = TurboJPEG()

def encode_jpeg(frame, quality):
    if frame.ndim == 2:
        return _turbo.encode(np.ascontiguousarray(frame), quality=quality, pixel_format=TJPF_GRAY, jpeg_subsample=TJSAMP_GRAY,)
    return _turbo.encode(frame, quality=quality)

ENCODER = "turbojpeg"

# *********************************************** #
# Frame Hub
# *********************************************** #
class FrameHub:

    def __init__(self, name):
        self.name = name
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = None
        self._seq = 0  # Keeping track of each frame published

    def publish(self, frame):
        if frame is None:
            raise RuntimeError(f"FrameHub[{self.name}]: publish(None)")
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def wait_for_frame(self, last_seq, timeout=5.0):
        # Block until a frame with seq > last_seq exists.
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= last_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_seq
                self._cond.wait(remaining)
            return self._frame, self._seq


CAMERA_FRAMES = FrameHub("camera")
CELESTIAL_FRAMES = FrameHub("celestial")

# *********************************************** #
# Flask App Definition
# *********************************************** #
ROOT_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = ROOT_DIR / "webapp"
START_TIME = time.time()

app = Flask(__name__, static_folder=str(WEBAPP_DIR), static_url_path="/static")

CAMERA_MANAGER = None
CELESTIAL_WATCHER = None
RSO_TRACKER = None

@app.route("/")
def index():
    return send_from_directory(WEBAPP_DIR, "control_panel.html")

def mjpeg_response(hub, quality):
    def generate():
        seq = 0
        while True:
            frame, seq = hub.wait_for_frame(seq)
            if frame is None:
                # Timeout: no producer activity. Loop again; the
                # blocking wait keeps this costless. If the client
                # disconnected, the next yield raises and Flask
                # tears the generator down.
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encode_jpeg(frame, quality) # TODO: We can use H.264 here specially if we use a CM4, but for now, I am too lazy to add it.
                + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-cache"})


@app.route("/api/camera/stream")
def camera_stream():
    return mjpeg_response(CAMERA_FRAMES, quality=80)


@app.route("/api/celestial/processed_frame")
def celestial_processed_stream():
    return mjpeg_response(CELESTIAL_FRAMES, quality=60)


@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    if CAMERA_MANAGER is None:
        return {"ok": False, "error": "Camera manager not initialized"}, 500

    try:
        acquiring = CAMERA_MANAGER.start_acquisition()
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

    return {"ok": True, "message": acquiring}


@app.route("/api/camera/stop", methods=["POST"])
@app.route("/api/system/stop_camera", methods=["POST"])
def stop_camera():
    if CAMERA_MANAGER is not None:
        CAMERA_MANAGER.close()

    return {"ok": True, "message": "Camera stopped"}


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
        "utc_now": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cpu_temp_c": cpu_temp_c,
        "cpu_load_pct": psutil.cpu_percent(interval=0.1),
        "disk_used_gb": disk.used / 1024**3,
        "disk_total_gb": disk.total / 1024**3,
        "uptime_s": time.time() - START_TIME,

        # Time service
        "time_sync": time_status["time_sync"],
        "drift_ms": time_status["drift_ms"],
    }

@app.route("/api/rso/overlay")
def overlay():
    empty = {"capture_time": None, "frame_w": 4096, "frame_h": 3000,
             "rsos": [], "stars": []}
    if RSO_TRACKER is None:
        return empty
    result = RSO_TRACKER.get_latest_hits()
    if result is None:
        return empty
    capture_time, hits, _frame, star_matches = result
    if capture_time is None:
        return empty
    return {
        "capture_time": capture_time.isoformat(),
        "frame_w": 4096,
        "frame_h": 3000,
        "rsos": [{"name": n, "norad": nid, "x": x, "y": y, "vx": vx, "vy": vy}
                 for n, nid, x, y, vx, vy in hits],
        "stars": [{"fx": fx, "fy": fy, "ix": ix, "iy": iy}
                  for fx, fy, ix, iy in star_matches],
    }

@app.route("/api/rso/solved_frame")
def solved_frame():
    if RSO_TRACKER is None:
        return Response(status=204)
    result = RSO_TRACKER.get_latest_hits()
    if result is None:
        return Response(status=204)
    _, _, frame, _ = result
    if frame is None:
        return Response(status=204)
    return Response(encode_jpeg(np.ascontiguousarray(frame), quality=70),
                    mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache"})

def run_server(camera_manager, celestial_watcher, rso_tracker, host="0.0.0.0", port=5000):
    global CAMERA_MANAGER, CELESTIAL_WATCHER, RSO_TRACKER
    CAMERA_MANAGER = camera_manager
    CELESTIAL_WATCHER = celestial_watcher
    RSO_TRACKER = rso_tracker

    print(f"[stream_server] JPEG encoder: {ENCODER}")
    print(f"[stream_server] JPEG encoder: {ENCODER}")
    print(f"[stream_server] Website: http://127.0.0.1:{port}")
    print(f"[stream_server] Listening on: http://{host}:{port}")

    # Each live MJPEG stream pins one thread. 8 covers two streams
    # per client x a couple of clients, plus API calls.
    serve(app, host=host, port=port, threads=8)