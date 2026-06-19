import cv2
from pathlib import Path
from flask import Flask, Response, send_from_directory

ROOT_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = ROOT_DIR / "webapp"

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

    return {"ok": True, "message": "Camera already started"}


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