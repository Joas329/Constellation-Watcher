import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np

# ============================================================
# Threading configuration
# ============================================================
MAX_WORKERS = min(5, os.cpu_count() or 1)

# We parallelize frames ourselves. Prevent every OpenCV operation
# from creating another set of internal threads.
cv2.setNumThreads(1)

# Persistent executor: created once and reused for every batch.
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ============================================================
# Fast percentile
# ============================================================
def fast_percentile(image: np.ndarray, percentile: float, sample_step: int = 4) -> float:
    flat = image.reshape(-1)

    if sample_step > 1:
        flat = flat[::sample_step]

    if flat.size == 0:
        return 0.0

    index = int(np.clip(percentile / 100.0 * (flat.size - 1), 0, flat.size - 1))

    return float(np.partition(flat, index)[index])


# ============================================================
# Generic parallel processing
# ============================================================

def run_parallel(function, frames, *, max_workers: int = MAX_WORKERS):
    """
    Process frames in parallel while preserving their original order.

    max_workers remains in the API for compatibility, but the persistent
    executor uses MAX_WORKERS configured at module load time.
    """
    if not frames:
        return []

    # executor.map preserves ordering, so no indexes, futures,
    # as_completed(), or sorting are required.
    return list(_EXECUTOR.map(function, frames))


# ============================================================
# Median filtering
# ============================================================
def _median_filter_frame(frame: np.ndarray, kernel_size: int) -> np.ndarray | None:
    if frame is None:
        return None

    return cv2.medianBlur(frame, kernel_size)


def median_filter_parallel(frames, kernel_size: int = 3, max_workers: int = MAX_WORKERS):
    if not frames:
        raise RuntimeError("No frames provided")

    worker = partial(_median_filter_frame, kernel_size=kernel_size,)

    return run_parallel(worker, frames, max_workers=max_workers)


# ============================================================
# Background model
# ============================================================

def build_background_model(frames, n_frames: int = 20) -> np.ndarray:
    """
    Build a uint8 median background image.

    Returning uint8 allows cv2.subtract() to perform fast saturated
    subtraction without converting every frame to float32.
    """
    valid_frames = [
        frame
        for frame in frames[-n_frames:]
        if frame is not None
    ]

    if not valid_frames:
        raise RuntimeError("No valid frames available for background model")

    # np.asarray avoids an extra Python-side list transformation.
    stack = np.asarray(valid_frames, dtype=np.uint8)

    # Median produces float64 by default, so convert immediately.
    background = np.median(stack, axis=0, overwrite_input=False).astype(np.uint8)

    return np.ascontiguousarray(background)

# ============================================================
# Background subtraction
# ============================================================

def _background_subtraction_frame(frame: np.ndarray, background: np.ndarray, percentile_sample_step: int) -> np.ndarray | None:
    if frame is None:
        return None

    # Saturated subtraction:
    # negative results automatically become zero.
    difference = cv2.subtract(frame, background)

    p_low = fast_percentile(difference, 1.0, sample_step=percentile_sample_step)

    p_high = fast_percentile(difference, 99.7, sample_step=percentile_sample_step,)

    value_range = p_high - p_low

    if value_range <= 1e-6:
        return np.zeros_like(difference, dtype=np.uint8)

    # OpenCV performs the normalization in optimized native code.
    normalized = cv2.convertScaleAbs(difference, alpha=255.0 / value_range, beta=-p_low * 255.0 / value_range)

    return normalized


def background_subtraction_parallel(frames, background: np.ndarray, max_workers: int = MAX_WORKERS, percentile_sample_step: int = 4):
    if not frames:
        raise RuntimeError("No frames provided")

    if background.dtype != np.uint8:
        background = np.clip(background, 0, 255).astype(np.uint8)

    background = np.ascontiguousarray(background)

    worker = partial(_background_subtraction_frame, background=background, percentile_sample_step=percentile_sample_step)

    return run_parallel(worker, frames, max_workers=max_workers)

# ============================================================
# Gaussian denoising
# ============================================================
def _gaussian_denoise_frame(frame: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray | None:
    if frame is None:
        return None

    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)


def gaussian_denoise_parallel(frames, kernel_size: int = 5, sigma: float = 1.0, max_workers: int = MAX_WORKERS):
    if not frames:
        raise RuntimeError("No frames provided")

    worker = partial(_gaussian_denoise_frame, kernel_size=kernel_size, sigma=sigma)

    return run_parallel(worker, frames, max_workers=max_workers)

# ============================================================
# Intensity thresholding
# ============================================================
def _threshold_frame(frame: np.ndarray, percentile: float, percentile_sample_step: int, min_area: int, max_area: int, max_aspect: float, min_compactness: float) -> np.ndarray | None:
    if frame is None:
        return None

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY,)
    else:
        gray = frame

    # No float32 conversion is needed. Percentiles work directly on uint8.
    threshold_value = fast_percentile(gray, percentile, sample_step=percentile_sample_step)

    # OpenCV thresholding is faster than np.where().
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)

    if num_labels <= 1:
        return np.zeros_like(gray, dtype=np.uint8)

    component_stats = stats[1:]

    areas = component_stats[:, cv2.CC_STAT_AREA]
    widths = component_stats[:, cv2.CC_STAT_WIDTH]
    heights = component_stats[:, cv2.CC_STAT_HEIGHT]

    minimum_side = np.minimum(widths, heights)
    maximum_side = np.maximum(widths, heights)

    aspect_ratios = maximum_side / np.maximum(minimum_side, 1)

    bounding_box_areas = widths * heights

    compactness = areas / np.maximum(bounding_box_areas, 1)

    # Completely vectorized component filtering.
    valid_components = (
        (areas >= min_area)
        & (areas <= max_area)
        & (aspect_ratios <= max_aspect)
        & (compactness >= min_compactness)
    )

    # Lookup table:
    # label 0 is the background and remains invalid.
    label_lookup = np.zeros(num_labels, dtype=np.uint8)

    label_lookup[1:] = valid_components.astype(np.uint8) * 255

    # Much faster than np.isin(labels, valid_labels).
    cleaned = label_lookup[labels]

    return cv2.medianBlur(cleaned, 3)


def threshold_parallel(frames, max_workers: int = MAX_WORKERS, percentile: float = 99.95, percentile_sample_step: int = 4, min_area: int = 10, max_area: int = 150, max_aspect: float = 3.0, min_compactness: float = 0.3):
    if not frames:
        raise RuntimeError("No frames provided")

    worker = partial(_threshold_frame, percentile=percentile, percentile_sample_step=percentile_sample_step, min_area=min_area, max_area=max_area, max_aspect=max_aspect, min_compactness=min_compactness)

    return run_parallel(worker, frames, max_workers=max_workers,)

# ============================================================
# Optional shutdown
# ============================================================
def shutdown_processing_pool() -> None:
    _EXECUTOR.shutdown(wait=True, cancel_futures=True)