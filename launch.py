#!/usr/bin/env python3

import os

from pathlib import Path
from camera.CameraManager import CameraManager
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

HOST = "0.0.0.0"
PORT = 8000
WEBAPP_DIR = Path(__file__).parent / "webapp"

def main():
    # Start the camera manager
    camera_manager = CameraManager()

    # Start the web server to serve the control panel
    os.chdir(WEBAPP_DIR)
    server = ThreadingHTTPServer((HOST, PORT), SimpleHTTPRequestHandler)
    print(f"Serving webapp on http://{HOST}:{PORT}/control_panel.html")
    server.serve_forever()


if __name__ == "__main__":
    main()