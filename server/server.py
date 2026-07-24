import cv2
import time
import traceback
import shutil
import psutil
import datetime
import numpy as np

from pathlib import Path
from waitress.server import create_server
from flask import Flask, Response, request, send_from_directory
from werkzeug.exceptions import HTTPException
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

_turbo = TurboJPEG()

def encode_jpeg(frame, quality):
    if frame.ndim == 2:
        return _turbo.encode(np.ascontiguousarray(frame), quality=quality, pixel_format=TJPF_GRAY, jpeg_subsample=TJSAMP_GRAY,)
    return _turbo.encode(frame, quality=quality)

ENCODER = "turbojpeg"

MIN_STREAM_WIDTH = 64
MAX_STREAM_WIDTH = 8192

def parse_width():
    # explicit reject rather than silently serving full resolution, otherwise a
    # typo in the query string costs downlink bandwidth with no visible symptom
    raw = request.args.get("w")
    if raw is None:
        return None

    try:
        width = int(raw)
    except ValueError:
        raise ValueError(f"w must be an integer, got {raw!r}")

    if not MIN_STREAM_WIDTH <= width <= MAX_STREAM_WIDTH:
        raise ValueError(f"w must be in [{MIN_STREAM_WIDTH}, {MAX_STREAM_WIDTH}], got {width}")

    return width

def parse_stretch():
    raw = request.args.get("stretch")
    if raw is None:
        return False
    if raw not in ("0", "1"):
        raise ValueError(f"stretch must be 0 or 1, got {raw!r}")
    return raw == "1"

def apply_stretch(frame, low_pct=1.0, high_pct=99.5):
    # display only, never touches pipeline data. A raw sky frame sits at a few
    # ADU of background with stars a few hundred above it, so a linear 8-bit
    # render is indistinguishable from black even when the frame is fine.
    sample = frame[::8, ::8]  # percentiles on 12M pixels are not free
    lo, hi = np.percentile(sample, [low_pct, high_pct])

    if hi <= lo:
        return frame

    scaled = (frame.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)

def downscale(frame, width):
    if width is None or width >= frame.shape[1]:
        return frame

    factor = frame.shape[1] / width
    height = max(1, int(round(frame.shape[0] / factor)))

    # point sources vanish under area averaging: a single-pixel star binned 6x6
    # loses ~36x its amplitude. Dilate first so each star survives decimation
    # as the local maximum of its block.
    k = max(1, int(np.ceil(factor)))
    if k > 1:
        frame = cv2.dilate(frame, np.ones((k, k), np.uint8))

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

def prepare(frame, width, stretch):
    frame = downscale(frame, width)

    if stretch:
        return apply_stretch(frame)

    # 12-bit data arrives MSB-aligned in a uint16 container; JPEG needs 8-bit
    if frame.dtype == np.uint16:
        return (frame >> 8).astype(np.uint8)

    return frame

# *********************************************** #
# Flask App Definition
# *********************************************** #
ROOT_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = ROOT_DIR / "webapp"
START_TIME = time.time()

app = Flask(__name__, static_folder=str(WEBAPP_DIR), static_url_path="/static")

# set once in run_server. CAMERA_MANAGER is kept as a direct handle because the
# control routes issue commands to hardware; frame flow goes through CHANNELS.
CAMERA_MANAGER = None
CHANNELS = {}
_SERVER = None

@app.route("/")
def index():
    return send_from_directory(WEBAPP_DIR, "control_panel.html")

def mjpeg_response(channel, quality, width, stretch):
    def generate():
        seq = 0
        while True:
            item = channel.wait(seq)
            if item is None:
                # Timeout: no producer activity. Loop again; the blocking wait
                # keeps this costless. If the client disconnected, the next
                # yield raises and Flask tears the generator down.
                continue
            frame, _, seq = item
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encode_jpeg(prepare(frame, width, stretch), quality)
                + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-cache"})

@app.route("/api/camera/stream")
def camera_stream():
    try:
        width, stretch = parse_width(), parse_stretch()
    except ValueError as e:
        return {"ok": False, "error": str(e)}, 400

    return mjpeg_response(CHANNELS["raw"], quality=80, width=width, stretch=stretch)

@app.route("/api/celestial/processed_frame")
def celestial_processed_stream():
    try:
        width, stretch = parse_width(), parse_stretch()
    except ValueError as e:
        return {"ok": False, "error": str(e)}, 400

    return mjpeg_response(CHANNELS["processed"], quality=60, width=width, stretch=stretch)

@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    if CAMERA_MANAGER is None:
        return {"ok": False, "error": "Camera manager not initialized"}, 500

    try:
        CAMERA_MANAGER.start_acquisition()
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500

    return {
        "ok": True,
        "message": "Acquisition started",
        "time_source": getattr(CAMERA_MANAGER, "time_source", None),
    }

@app.route("/api/camera/stop", methods=["POST"])
@app.route("/api/system/stop_camera", methods=["POST"])
def stop_camera():
    # stop_acquisition, not close: keeps the device open so the latched clock
    # offset survives a viewfinder toggle. close() belongs in the shutdown path.
    # The camera simply stops publishing to the raw channel; downstream stages
    # block on the empty channel and idle until frames resume. No pump to stop.
    if CAMERA_MANAGER is not None:
        CAMERA_MANAGER.stop_acquisition()

    return {"ok": True, "message": "Acquisition stopped"}

@app.errorhandler(Exception)
def handle_unexpected(e):
    # without this every bug reaches the browser as Flask's generic 500 page and
    # the only copy of the traceback is in the server terminal
    if isinstance(e, HTTPException):
        return e

    traceback.print_exc()
    return {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500

@app.route("/camera/opt/exposure", methods=["POST"])
def set_exposure():
    if CAMERA_MANAGER is None:
        return {"ok": False, "error": "Camera manager not initialized"}, 500

    payload = request.get_json(silent=True) or {}
    us = payload.get("us")

    if not isinstance(us, (int, float)) or us <= 0:
        return {"ok": False, "error": f"us must be a positive number, got {us!r}"}, 400

    try:
        applied_us = CAMERA_MANAGER.set_exposure_us(float(us))
    except (RuntimeError, ValueError) as e:
        return {"ok": False, "error": str(e)}, 400

    return {"ok": True, "applied_us": applied_us, "applied_ms": applied_us / 1000.0}

@app.route("/api/camera/stats")
def camera_stats():
    # the fastest way to tell "no photons" from "photons, bad display"
    if CAMERA_MANAGER is None:
        return {"ok": False, "error": "Camera manager not initialized"}, 500

    peeked = CHANNELS["raw"].peek() if "raw" in CHANNELS else None
    if peeked is None:
        return {"ok": False, "error": "No frame yet. Start the viewfinder."}, 409

    frame, capture_time, _ = peeked
    ceiling = 65535 if frame.dtype == np.uint16 else 255
    sample = frame[::4, ::4]
    median, p99, p999 = np.percentile(sample, [50, 99, 99.9])

    return {
        "ok": True,
        "capture_time": capture_time.isoformat() if capture_time else None,
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "min": int(frame.min()),
        "max": int(frame.max()),
        "mean": round(float(frame.mean()), 2),
        "median": round(float(median), 2),
        "p99": round(float(p99), 2),
        "p99_9": round(float(p999), 2),
        "saturated_px": int((frame >= ceiling).sum()),
        "exposure_ms": round(CAMERA_MANAGER._exposure_s * 1000.0, 3),
        **CAMERA_MANAGER.read_analog_settings(),
    }

@app.route("/camera/opt/black_level", methods=["POST"])
def set_black_level():
    if CAMERA_MANAGER is None:
        return {"ok": False, "error": "Camera manager not initialized"}, 500

    payload = request.get_json(silent=True) or {}
    level = payload.get("level")

    if not isinstance(level, (int, float)):
        return {"ok": False, "error": f"level must be a number, got {level!r}"}, 400

    try:
        applied = CAMERA_MANAGER.set_black_level(float(level))
    except (RuntimeError, ValueError) as e:
        return {"ok": False, "error": str(e)}, 400

    return {"ok": True, "applied": applied}

@app.route("/camera/opt/gain", methods=["POST"])
def set_gain():
    if CAMERA_MANAGER is None:
        return {"ok": False, "error": "Camera manager not initialized"}, 500

    payload = request.get_json(silent=True) or {}
    db = payload.get("db")

    if not isinstance(db, (int, float)):
        return {"ok": False, "error": f"db must be a number, got {db!r}"}, 400

    try:
        applied = CAMERA_MANAGER.set_gain_db(float(db))
    except (RuntimeError, ValueError) as e:
        return {"ok": False, "error": str(e)}, 400

    return {"ok": True, "applied_db": applied}

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
    empty = {"capture_time": None, "frame_w": None, "frame_h": None,
             "rsos": [], "stars": []}
    peeked = CHANNELS["detections"].peek() if "detections" in CHANNELS else None
    if peeked is None:
        return empty
    (hits, frame, star_matches), capture_time, _ = peeked
    if capture_time is None or frame is None:
        return empty
    # dimensions come from the solved frame itself; a hardcoded 4096x3000 would
    # silently shift every overlay coordinate the day an ROI or binning changes
    return {
        "capture_time": capture_time.isoformat(),
        "frame_w": frame.shape[1],
        "frame_h": frame.shape[0],
        "rsos": [{"name": n, "norad": nid, "x": x, "y": y, "vx": vx, "vy": vy}
                 for n, nid, x, y, vx, vy in hits],
        "stars": [{"fx": fx, "fy": fy, "ix": ix, "iy": iy}
                  for fx, fy, ix, iy in star_matches],
    }

@app.route("/api/rso/solved_frame")
def solved_frame():
    try:
        width, stretch = parse_width(), parse_stretch()
    except ValueError as e:
        return {"ok": False, "error": str(e)}, 400

    peeked = CHANNELS["detections"].peek() if "detections" in CHANNELS else None
    if peeked is None:
        return Response(status=204)
    (_, frame, _), _, _ = peeked
    if frame is None:
        return Response(status=204)
    return Response(encode_jpeg(prepare(np.ascontiguousarray(frame), width, stretch), quality=70), mimetype="image/jpeg", headers={"Cache-Control": "no-cache"})

def run_server(camera, channels, host="0.0.0.0", port=5000):
    global CAMERA_MANAGER, CHANNELS, _SERVER
    CAMERA_MANAGER = camera
    CHANNELS = channels

    print(f"[stream_server] JPEG encoder: {ENCODER}")
    print(f"[stream_server] Website: http://127.0.0.1:{port}")
    print(f"[stream_server] Listening on: http://{host}:{port}")

    _SERVER = create_server(app, host=host, port=port, threads=16, connection_limit=64, channel_timeout=30)
    _SERVER.run()

def stop_server():
    if _SERVER is not None:
        _SERVER.close()