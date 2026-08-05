import cv2
import threading
from pathlib import Path
from datetime import datetime, timezone

from transport.stage import Stage
from celestial_watcher.CelestialTools import component_filter_parallel, median_filter_parallel, sigma_threshold_parallel, spatial_background_parallel

MAX_WORKERS = 5
DETECTION_NSIGMA = 2.5  # wide net for faint stars; the component filter removes the junk it lets in
MIN_SOURCE_AREA = 2     # drop single-pixel hot pixels / cosmic rays

class CelestialWatcherStage(Stage):
    def __init__(self, source, sink, save_dir):
        super().__init__("celestial_watcher", source, sink)
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)  # exist_ok: re-runs don't raise
        print(f"[Celestial Watcher Stage] CWS save_dir: {self._save_dir}")

        # one lock guards every cross-thread flag below; the Flask route sets them,
        # the pipeline thread reads and clears them in process()
        self._save_lock = threading.Lock()

        self._save_remaining = 0        # disk-save: frames still to write; 0 = idle
        self._download_pending = False  # download: arm capture of the next frame for HTTP
        self._download_buffer = None    # (filename, png_bytes) awaiting browser pickup

    # ---- disk save: "Save to Celestial device memory" ----

    # called from the Flask thread. Arms the writer for the next `count` frames.
    def request_save(self, count):
        with self._save_lock:
            self._save_remaining = count  # latest request wins (does not stack)
        return count

    def _claim_save_slot(self):
        with self._save_lock:
            if self._save_remaining <= 0:
                return False
            self._save_remaining -= 1
            return True

    # ---- HTTP handoff: "Save to local device" (download to viewing machine) ----

    # called from the Flask thread. Arms capture of the next single frame; the
    # pipeline buffers it in memory for pop_download to return. No disk write.
    def request_download(self):
        with self._save_lock:
            self._download_pending = True
        return True

    def _claim_download_slot(self):
        with self._save_lock:
            if not self._download_pending:
                return False
            self._download_pending = False  # one-shot; re-arm for the next frame
            return True

    # called from the Flask thread. Returns (filename, bytes) once buffered,
    # or None if the pipeline hasn't grabbed a frame yet.
    def pop_download(self):
        with self._save_lock:
            buf = self._download_buffer
            self._download_buffer = None
        return buf

    def process(self, frame, capture_time):
        # claim both flags up front so one frame can satisfy a save and a download
        # at once, and neither races the route on its counter
        save = self._claim_save_slot()
        grab = self._claim_download_slot()

        # MEDIAN FILTER (single frame)
        filtered = median_filter_parallel([frame], kernel_size=3, max_workers=MAX_WORKERS)[0]

        # SPATIAL BACKGROUND SUBTRACTION (sigma-clipped mesh, single frame)
        bg_sub, rms = spatial_background_parallel([filtered], max_workers=MAX_WORKERS, return_rms=True)[0]

        # SIGMA THRESHOLDING (noise-adaptive, thresholds the frame rms was measured on)
        thresholded = sigma_threshold_parallel([bg_sub], [rms], max_workers=MAX_WORKERS, nsigma=DETECTION_NSIGMA)[0]

        # COMPONENT FILTERING (hot-pixel / cosmic-ray rejection by shape)
        # sigma-threshold answers "above noise?"; this answers "star-shaped?".
        # on real frames ~96% of nsigma=2 detections are single-pixel junk.
        sources = component_filter_parallel([thresholded], max_workers=MAX_WORKERS, min_area=MIN_SOURCE_AREA)[0]

        if save:
            self._save_pair(frame, sources, capture_time)

        if grab:
            self._buffer_for_download(frame, sources, capture_time)

        return sources

    def _save_pair(self, raw, processed, capture_time):
        # raw + processed share one stamp so the pair is obvious on disk
        stamp = self._stamp(capture_time)
        self._write_frame(self._save_dir / f"frame_{stamp}_raw.png", raw)
        self._write_frame(self._save_dir / f"frame_{stamp}_processed.png", processed)

    def _buffer_for_download(self, raw, processed, capture_time):
        # encode both on the pipeline thread; the route zips and ships them.
        # one stamp ties the pair together, matching the disk-save naming.
        stamp = self._stamp(capture_time)
        files = []
        for suffix, arr in (("raw", raw), ("processed", processed)):
            ok, png = cv2.imencode(".png", arr)
            if not ok:
                raise RuntimeError(f"cv2.imencode failed for {suffix} download frame")
            files.append((f"frame_{stamp}_{suffix}.png", png.tobytes()))
        with self._save_lock:
            self._download_buffer = files   # list of (filename, bytes)

    def _write_frame(self, path, frame):
        # PNG is lossless; imwrite handles uint16 sky frames without truncating
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"cv2.imwrite failed for {path}")

    def _stamp(self, capture_time):
        # UTC so a burst sorts chronologically and never collides; fall back to
        # wall clock if the frame carries no timestamp
        return (capture_time or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S_%f")

    def stop(self, timeout=10.0):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise RuntimeError(f"{self.name} did not exit within {timeout}s")

        print("[Celestial Watcher Stage] CWS thread stop.")
        self._thread = None