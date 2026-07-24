from transport.stage import Stage
from celestial_watcher.CelestialTools import component_filter_parallel, median_filter_parallel, sigma_threshold_parallel, spatial_background_parallel

MAX_WORKERS = 5
DETECTION_NSIGMA = 2.0 # wide net for faint stars; the component filter removes the junk it lets in
MIN_SOURCE_AREA = 2 # drop single-pixel hot pixels / cosmic rays

class CelestialWatcherStage(Stage):
    def __init__(self, source, sink):
        super().__init__("celestial_watcher", source, sink)

    def process(self, frame, capture_time):

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

        return sources

    def stop(self, timeout=10.0):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise RuntimeError(f"{self.name} did not exit within {timeout}s")

        print("[Celestial Watcher Stage] CWS thread stop.")
        self._thread = None

