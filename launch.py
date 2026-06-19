#!/usr/bin/env python3

from server.server import run_server
from camera.CameraManager import CameraManager

def main():
    # Start the camera manager
    camera_manager = CameraManager()

    # Start the server
    run_server(camera_manager)


if __name__ == "__main__":
    main()