#!/usr/bin/env python3
import signal

from transport.channel import Channel
from server.server import run_server, stop_server
from camera.CameraManager import CameraManager
from camera.FakeCameraManager import FakeCameraManager
from celestial_watcher.RSOTrackerStage import RSOTrackerStage
from celestial_watcher.GPSManager import GPSManager
from celestial_watcher.CelestialWatcherStage import CelestialWatcherStage
from celestial_watcher.PlateSolverStage import PlateSolverStage
from celestial_watcher.CelestialTools import shutdown_processing_pool

IMAGE_DIRECTORY = "/media/joas329/My Passport/celestial_data_chunk"
USE_FAKE_CAMERA = True

def main():
    # one channel per hop; the composition root is the only place the pipeline topology is written down. Each stage knows its source and sink, nothing else.
    raw_frames = Channel("raw_frames")
    processed_frames = Channel("processed_frames")
    solutions = Channel("solutions")
    detections = Channel("detections")
    channels = {"raw": raw_frames, "processed": processed_frames, "detections": detections}

    # Start Camera Manager
    if USE_FAKE_CAMERA:
        camera = FakeCameraManager(raw_frames, IMAGE_DIRECTORY, fps=20.0, loop=True, grayscale=True)
    else:
        camera = CameraManager(raw_frames, grayscale=True)

    # Start GPS Manager
    gps_manager = GPSManager()
    gps_manager.start()

    # Start Celestial Channels
    watcher = CelestialWatcherStage(raw_frames, processed_frames)
    solver = PlateSolverStage(processed_frames, solutions)
    tracker = RSOTrackerStage(solutions, detections, gps=gps_manager)
    stages = [watcher, solver, tracker]

    # SIGINT/SIGTERM unblock waitress by raising into the main thread
    def handle_shutdown(signum, frame):
        print(f"\nShutdown signal received: {signum}")
        camera.close() # source off: no new frames
        gps_manager.close()
        for stage in stages:
            stage.request_stop() # flag every stage (solver kills its solve)
        for channel in channels.values():
            channel.close() # wait() returns None -> no new frame pulled
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        for stage in stages:
            stage.start()

        # camera acquisition is started by the HTTP viewfinder route; stages block on empty channels until frames flow.
        run_server(camera, channels=channels, gps=gps_manager)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        print("Cleaning up...")

        # source, stages, and channels were already stopped/closed in the signal handler. This block is idempotent: it also covers a non-signal exit where the handler never ran.
        camera.close()
        gps_manager.close()
        for channel in channels.values():
            channel.close()

        # stop the server: stream threads already unblocked by the closed channels
        stop_server()

        # join stages: flags set + channels closed -> each returns in well under 1s
        for stage in reversed(stages):
            stage.stop()

        # executor last: no stage can schedule onto it anymore
        shutdown_processing_pool()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()