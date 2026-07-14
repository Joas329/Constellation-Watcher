import collections
import threading

from server import server
from camera.CameraManager import CameraManager
from camera.FakeCameraManager import FakeCameraManager
from celestial_watcher.CelestialTools import background_subtraction_parallel, build_background_model, gaussian_denoise_parallel, median_filter_parallel, threshold_parallel

MAX_WORKERS = 5
BACKGROUND_FRAMES = 5 # frames used to (re)build the model
BACKGROUND_REFRESH_EVERY = 60 # rebuild after this many processed frames (~30 s at 2 fps)

class CelestialWatcherExecutive:
    def __init__(self, camera_manager: CameraManager):
        self.camera_manager = camera_manager
        self._celestial_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._recent_filtered = collections.deque(maxlen=BACKGROUND_FRAMES)
        self._background_model = None
        self._frames_since_refresh = 0

        self.latest_solve_frame = None # (thresholded frame, capture_time)
        self.latest_raw_frame = None

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

    def get_latest_frame_with_time(self):
        with self._lock:
            return self.latest_solve_frame

    def get_latest_raw_frame(self): # Since camera manger getter increases the frame count every get() call, lets just add a getter of the inital unadultered frame Celestial receives. They are in fact, the same.
        with self._lock:
            return self.latest_raw_frame

    def _run_celestial_logic(self):
        while not self._stop_event.is_set():
            result = self.camera_manager.get_latest_frame_with_time()

            if result is None:
                self._stop_event.wait(0.05)
                continue

            frame, capture_time = result

            with self._lock:
                self.latest_raw_frame = frame

            # MEDIAN FILTER (single frame)
            filtered = median_filter_parallel([frame], kernel_size=3,
                                              max_workers=MAX_WORKERS)[0]
            self._recent_filtered.append(filtered)

            # no background model yet
            if self._background_model is None:
                if len(self._recent_filtered) < BACKGROUND_FRAMES:
                    continue  # still collecting warm-up frames
                self._background_model = build_background_model(
                    list(self._recent_filtered), n_frames=BACKGROUND_FRAMES)
                print(f"Background model built from {BACKGROUND_FRAMES} frames.")

            # Periodic refresh from the trailing window
            self._frames_since_refresh += 1
            if self._frames_since_refresh >= BACKGROUND_REFRESH_EVERY:
                self._background_model = build_background_model(
                    list(self._recent_filtered), n_frames=BACKGROUND_FRAMES)
                self._frames_since_refresh = 0
                print("Background model refreshed.")

            # Per-frame pipeline against the standing model
            # MEAN BACKGROUND SUBTRACTION
            bg_sub = background_subtraction_parallel([filtered], self._background_model, max_workers=MAX_WORKERS)[0]

            # GAUSSIAN DENOISING
            gaussian = gaussian_denoise_parallel([bg_sub], 5, 1.0, MAX_WORKERS)[0]

            # INTENSITY THRESHOLDING
            # For the solver: all bright sources kept (big stars are the solver's best quad anchors)
            solve_frame = threshold_parallel([gaussian], max_workers=MAX_WORKERS, filter_components=False)[0]

            # For streak detection / the stream: component-filtered
            thresholded = threshold_parallel([gaussian], max_workers=MAX_WORKERS)[0]

            with self._lock:
                self.latest_solve_frame = (solve_frame, capture_time)

            # Stream the processed frame at camera cadence
            server.CELESTIAL_FRAMES.publish(thresholded)

        print("Celestial Watcher thread stopped.")