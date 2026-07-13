#!/usr/bin/env python3
import signal
import threading

from server.server import run_server
from camera.CameraManager import CameraManager
from camera.FakeCameraManager import FakeCameraManager
from celestial_watcher.RSOTracker import RSOTracker
from celestial_watcher.CelestialWatcherExecutive import CelestialWatcherExecutive
from celestial_watcher.LocalPlateSolverExecutive import LocalPlateSolverExecutive

shutdown_event = threading.Event()

IMAGE_DIRECTORY = ("/media/joas329/My Passport/celestial_data_chunk")

def handle_shutdown(signum, frame):
    print(f"\nShutdown signal received: {signum}")
    shutdown_event.set()
    raise KeyboardInterrupt

def main():
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    camera_manager = FakeCameraManager(IMAGE_DIRECTORY, 20.0, True, False)

    # camera_manager = CameraManager()
    celestial_watcher = CelestialWatcherExecutive(camera_manager)
    plate_solver = LocalPlateSolverExecutive(camera_manager)
    rso_tracker = RSOTracker(plate_solver)

    try:
        # Start Celestial
        celestial_watcher.start()

        # Start Plate Solver
        plate_solver.start()

        # Start RSO Tracker
        rso_tracker.start()

        # Start the server
        run_server(camera_manager, celestial_watcher)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        print("Cleaning up...")
        plate_solver.stop()
        rso_tracker.stop()
        celestial_watcher.stop()
        camera_manager.close()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()