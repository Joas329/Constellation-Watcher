import threading

from server import server
from camera.CameraManager import CameraManager
from camera.FakeCameraManager import FakeCameraManager
from celestial_watcher.CelestialTools import background_subtraction_parallel, build_background_model, gaussian_denoise_parallel, median_filter_parallel, threshold_parallel

PROCESSING_BATCH_SIZE = 3
MAX_WORKERS = 5
PUBLISH_FPS = 30

class CelestialWatcherExecutive:
    def __init__(self, camera_manager: CameraManager):
        self.camera_manager = camera_manager
        self._celestial_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.batch_number = 0
        self.frames_to_process = []

    def start(self):
        with self._lock:
            if self._celestial_thread is not None and self._celestial_thread.is_alive():
                return "Celestial Watcher already running."

            self._stop_event.clear()

            self._celestial_thread = threading.Thread(target=self._run_celestial_logic, daemon=False)
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

    def _publish_batch(self, frames):
        interval = 1.0 / PUBLISH_FPS
        for frame in frames:
            if self._stop_event.is_set():
                return
            server.CELESTIAL_FRAMES.publish(frame)
            self._stop_event.wait(interval)

    def _run_celestial_logic(self):
        while not self._stop_event.is_set():
            frame = self.camera_manager.get_latest_frame()

            if frame is None:
                self._stop_event.wait(0.05)
                continue

            self.frames_to_process.append(frame)
            if len(self.frames_to_process) < PROCESSING_BATCH_SIZE:
                continue

            self.batch_number += 1

            print(f"Batch {self.batch_number}: Processing {PROCESSING_BATCH_SIZE} frames...")

            batch = self.frames_to_process

            # MEDIAN FILTER
            filtered_batch = median_filter_parallel(batch, kernel_size=3, max_workers=MAX_WORKERS)

            # MEAN BACKGROUND SUBTRACTION
            background_model = build_background_model(filtered_batch, n_frames=PROCESSING_BATCH_SIZE//4)
            bg_sub_frames = background_subtraction_parallel(filtered_batch, background_model, max_workers=MAX_WORKERS)

            # GAUSSIAN DENOISING
            gaussian_batch = gaussian_denoise_parallel(bg_sub_frames, 5, 1.0, MAX_WORKERS)

            # INTENSITY THRESHOLDING
            threshold_batch = threshold_parallel(gaussian_batch, max_workers=MAX_WORKERS)

            # Stream the processed batch
            self._publish_batch(threshold_batch)

            self.frames_to_process = []

        print("Celestial Watcher thread stopped.")