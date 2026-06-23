import time
import threading
from camera.CameraManager import CameraManager


class CelestialWatcherExecutive:
    def __init__(self, camera_manager: CameraManager):
        self.camera_manager = camera_manager
        self._celestial_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.frame_number = 0

    def start(self):
        with self._lock:
            if self._celestial_thread is not None and self._celestial_thread.is_alive():
                return "Celestial Watcher already running."

            self._stop_event.clear()

            self._celestial_thread = threading.Thread(
                target=self._run_celestial_logic,
                daemon=False
            )
            self._celestial_thread.start()

        print("Celestial Watcher started.")
        return "Celestial Watcher started."

    def stop(self):
        with self._lock:
            if self._celestial_thread is None:
                return "Celestial Watcher already stopped."

            self._stop_event.set()
            thread = self._celestial_thread

        thread.join(timeout=5.0)

        with self._lock:
            self._celestial_thread = None

        print("Celestial Watcher stopped.")
        return "Celestial Watcher stopped."

    def _run_celestial_logic(self):
        while not self._stop_event.is_set():
            frame = self.camera_manager.get_latest_frame()

            if frame is None:
                self._stop_event.wait(0.05)
                continue

            print(f"Processing frame {self.frame_number}: {frame.shape}")
            self.frame_number += 1

            self._stop_event.wait(0.1)

        print("Celestial Watcher thread stopped.")