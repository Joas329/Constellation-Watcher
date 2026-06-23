#!/usr/bin/env python3
import signal
import threading

from server.server import run_server
from camera.CameraManager import CameraManager
from celestial_watcher.CelestialWatcherExecutive import CelestialWatcherExecutive

shutdown_event = threading.Event()

def handle_shutdown(signum, frame):
    print(f"\nShutdown signal received: {signum}")
    shutdown_event.set()
    raise KeyboardInterrupt

def main():
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    camera_manager = CameraManager()
    celestial_watcher = CelestialWatcherExecutive(camera_manager)

    try:
        # Start Celestial
        celestial_watcher.start()

        # Start the server
        run_server(camera_manager)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        print("Cleaning up...")
        celestial_watcher.stop()
        camera_manager.close()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()