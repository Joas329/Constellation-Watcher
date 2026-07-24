#!/usr/bin/env python3
import signal

from transport.channel import Channel
from server.server import run_server
from camera.CameraManager import CameraManager
from camera.FakeCameraManager import FakeCameraManager
from celestial_watcher.RSOTrackerStage import RSOTrackerStage
from celestial_watcher.CelestialWatcherStage import CelestialWatcherStage
from celestial_watcher.PlateSolverStage import PlateSolverStage
from celestial_watcher.CelestialTools import shutdown_processing_pool

IMAGE_DIRECTORY = "/media/joas329/My Passport/celestial_data_chunk"
USE_FAKE_CAMERA = False

def main():
    # one channel per hop; the composition root is the only place the pipeline topology is written down. Each stage knows its source and sink, nothing else.
    raw_frames = Channel("raw_frames")
    processed_frames = Channel("processed_frames")
    solutions = Channel("solutions")
    detections = Channel("detections")

    if USE_FAKE_CAMERA:
        camera = FakeCameraManager(raw_frames, IMAGE_DIRECTORY, fps=20.0, loop=True, grayscale=True)
    else:
        camera = CameraManager(raw_frames, grayscale=True)

    watcher = CelestialWatcherStage(raw_frames, processed_frames)
    solver = PlateSolverStage(processed_frames, solutions)
    tracker = RSOTrackerStage(solutions, detections)
    stages = [watcher, solver, tracker]

    # SIGINT/SIGTERM unblock waitress by raising into the main thread
    def handle_shutdown(signum, frame):
        print(f"\nShutdown signal received: {signum}")
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        for stage in stages:
            stage.start()

        # camera acquisition is started by the HTTP viewfinder route, so the stages just block on empty channels until frames start flowing.
        run_server(camera, channels={"raw": raw_frames, "processed": processed_frames, "detections": detections})

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        print("Cleaning up...")

        for stage in reversed(stages):
            stage.stop()
        camera.close()
        shutdown_processing_pool()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()