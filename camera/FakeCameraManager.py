import re
from datetime import datetime, timezone
from pathlib import Path

import threading
import time

import cv2

FILENAME_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")

class FakeCameraManager:
    def __init__(self, image_directory: str, fps: float = 20.0, loop: bool = True, grayscale: bool = False):
        self.image_directory = Path(image_directory)
        self.fps = fps
        self.loop = loop
        self.grayscale = grayscale

        self._frame_delay = 1.0 / fps
        self._frame_paths: list[Path] = []
        self._current_index = 0
        self._last_frame_time = 0.0

        self._lock = threading.Lock()
        self._running = False

        self._load_frame_paths()

    def _load_frame_paths(self) -> None:
        if not self.image_directory.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.image_directory}"
            )

        if not self.image_directory.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {self.image_directory}"
            )

        # Sort by filename. This works correctly when filenames have
        # zero-padded numbers, such as frame_0001.png.
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

    def start_acquisition(self) -> None:
        with self._lock:
            self._running = True
            self._current_index = 0
            self._last_frame_time = 0.0

        print("Fake camera stream started.")

    def close(self):
        with self._lock:
            self._running = False

    def start_acquisition(self) -> None:
        with self._lock:
            self._running = True
            self._start_time = time.monotonic()
        print("Fake camera stream started.")

    def get_latest_frame_with_time(self):
        with self._lock:
            if not self._running:
                return None
            elapsed = time.monotonic() - self._start_time
            index = int(elapsed * self.fps)
            if self.loop:
                index %= len(self._frame_paths)
            elif index >= len(self._frame_paths):
                self._running = False
                return None
            frame_path = self._frame_paths[index]
        # imread + return as before
        read_mode = (cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR)

        frame = cv2.imread(str(frame_path), read_mode)

        if frame is None:
            print(f"Could not read frame: {frame_path}")
            return None

        return frame, self._parse_capture_time(frame_path)