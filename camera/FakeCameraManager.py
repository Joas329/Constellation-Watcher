import re
import cv2
import threading
from datetime import datetime, timezone
from pathlib import Path

FILENAME_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

class FakeCameraManager:
    # Replays PNGs from disk into the raw channel at `fps`, one publish per frame, so downstream stages see the exact same interface as the real camera.
    def __init__(self, raw, image_directory: str, fps: float = 20.0,
                 loop: bool = True, grayscale: bool = False):
        self._raw = raw
        self.image_directory = Path(image_directory)
        self.fps = fps
        self.loop = loop
        self.grayscale = grayscale

        self._frame_delay = 1.0 / fps
        self._frame_paths: list[Path] = []

        # mirrors CameraManager's timing state so the stats route can read it
        self.time_source = "file"
        self._exposure_s = 0.0

        self._acquisition_thread = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()

        self._load_frame_paths()

    def _load_frame_paths(self) -> None:
        if not self.image_directory.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_directory}")

        if not self.image_directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.image_directory}")

        # Sort by filename. Correct when filenames have zero-padded numbers, such as frame_0001.png.
        self._frame_paths = sorted(self.image_directory.glob("*.png"), key=lambda path: path.name)

        if not self._frame_paths:
            raise RuntimeError(f"No PNG files found in: {self.image_directory}")

        print(f"FakeCameraManager found {len(self._frame_paths)} PNG frames.")

    @staticmethod
    def _parse_capture_time(frame_path: Path):
        match = FILENAME_TIMESTAMP_RE.search(frame_path.name)
        if match is None:
            return None
        date_str, time_str = match.groups()
        try:
            naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        except ValueError:
            return None
        return naive.replace(tzinfo=timezone.utc)

    def start_acquisition(self, index: int = 0) -> None:
        with self._state_lock:
            if self._acquisition_thread is not None and self._acquisition_thread.is_alive():
                print("Fake camera already acquiring.")
                return

            self._stop_event.clear()
            self._acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=False)
            self._acquisition_thread.start()

        print("Fake camera stream started.")

    def _acquisition_loop(self):
        read_mode = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        i = 0

        while not self._stop_event.is_set():
            if i >= len(self._frame_paths):
                if not self.loop:
                    print("Fake camera reached end of frames.")
                    break
                i = 0

            frame_path = self._frame_paths[i]
            frame = cv2.imread(str(frame_path), read_mode)

            if frame is None:
                print(f"Could not read frame: {frame_path}")
            else:
                # imread already returns an owned array, no view into a shared
                # buffer, so unlike the real camera no defensive copy is needed
                self._raw.publish(frame, self._parse_capture_time(frame_path))

            i += 1
            # interruptible pace: wait() returns early the moment stop is set
            self._stop_event.wait(self._frame_delay)

        print("Fake camera acquisition thread exited.")

    def stop_acquisition(self) -> None:
        with self._state_lock:
            if self._acquisition_thread is None:
                print("Fake camera already stopped.")
                return
            self._stop_event.set()
            thread = self._acquisition_thread

        thread.join(timeout=5.0)

        with self._state_lock:
            self._acquisition_thread = None

        print("[Camera Manager] Fake camera stopped.")

    def close(self):
        self.stop_acquisition()

    # settings shims so the camera-control routes don't 500 against the fake

    def read_analog_settings(self) -> dict:
        return {"gain_db": None, "black_level": None,
                "pixel_format": "Mono8" if self.grayscale else "BGR8"}