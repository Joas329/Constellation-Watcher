import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from scipy.ndimage import median_filter as ndi_median

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
def _threshold_frame(frame: np.ndarray, percentile: float, percentile_sample_step: int, min_area: int, max_area: int, max_aspect: float, min_compactness: float, filter_components: bool = True) -> np.ndarray | None:
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

    if not filter_components:
        return binary

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


def threshold_parallel(frames, max_workers: int = MAX_WORKERS, percentile: float = 99.95, percentile_sample_step: int = 4, min_area: int = 10, max_area: int = 150, max_aspect: float = 3.0, min_compactness: float = 0.3, filter_components: bool = True):
    if not frames:
        raise RuntimeError("No frames provided")

    worker = partial(_threshold_frame, percentile=percentile, percentile_sample_step=percentile_sample_step, min_area=min_area, max_area=max_area, max_aspect=max_aspect, min_compactness=min_compactness, filter_components=filter_components)

    return run_parallel(worker, frames, max_workers=max_workers,)

# ============================================================
# Spatial background estimation (single frame, sigma-clipped mesh)
# ============================================================
def _spatial_background_frame(frame, box_size, filter_size, sigma, maxiters, downsample, subtract, return_rms: bool = False):
    if frame is None:
        return None

    h, w = frame.shape[:2]
    data = frame.astype(np.float32)

    # decimate before boxing: sky stats survive it and it cuts the pixel count
    # the clip iterates over (downsample=4 -> ~16x fewer pixels, ~10x faster)
    data_s = data[::downsample, ::downsample] if downsample > 1 else data
    hs, ws = data_s.shape
    bs = max(1, box_size // downsample)
    ny, nx = hs // bs, ws // bs

    # crop to whole boxes, reshape so each box is the last axis of the array;
    # then sigma-clip every box at once with no Python per-box loop
    cropped = data_s[:ny * bs, :nx * bs]
    boxes = cropped.reshape(ny, bs, nx, bs).transpose(0, 2, 1, 3).reshape(ny, nx, bs * bs)

    mask = np.ones_like(boxes, dtype=bool)
    for _ in range(maxiters):
        med = np.median(np.where(mask, boxes, np.nan), axis=2, keepdims=True)
        mad = np.median(np.where(mask, np.abs(boxes - med), np.nan), axis=2, keepdims=True)
        s = 1.4826 * mad  # MAD scaled to Gaussian sigma; robust to star pixels
        new_mask = np.abs(boxes - med) <= sigma * np.where(s > 0, s, np.inf)
        # never empty a box completely: keep the old mask where a clip would
        # reject everything, so its median stays defined (no NaN in the surface)
        empties = new_mask.sum(axis=2, keepdims=True) == 0
        new_mask = np.where(empties, mask, new_mask)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    coarse = np.nanmedian(np.where(mask, boxes, np.nan), axis=2).astype(np.float32)

    # per-box noise level: MAD-scaled std of the CLIPPED (sky-only) pixels,
    # i.e. the same robust sigma the clip used, kept instead of discarded
    clipped_mad = np.nanmedian(np.where(mask, np.abs(boxes - coarse[..., None]), np.nan), axis=2)
    coarse_rms = (1.4826 * clipped_mad).astype(np.float32)

    RMS_FLOOR = 1.0
    # a truly flat/quantized box gives MAD=0 (Median Absolute Deviation)-> rms=0 -> the sigma threshold collapses to "anything above zero", flooding the frame with noise.
    # floor at a minimum physical noise level so nsigma*rms is always a real cut.
    coarse_rms = np.maximum(coarse_rms, RMS_FLOOR)

    # median-filter both low-res grids so one bad box can't spike the surface
    if filter_size > 1 and ny >= filter_size and nx >= filter_size:
        coarse = ndi_median(coarse, size=filter_size, mode="nearest")
        coarse_rms = ndi_median(coarse_rms, size=filter_size, mode="nearest")

    background = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    rms = cv2.resize(coarse_rms, (w, h), interpolation=cv2.INTER_LINEAR)

    if subtract:
        out = data - background
        np.clip(out, 0, None, out=out)
        if frame.dtype == np.uint8:
            result = np.clip(out, 0, 255).astype(np.uint8)
        elif frame.dtype == np.uint16:
            result = np.clip(out, 0, 65535).astype(np.uint16)
        else:
            result = out.astype(frame.dtype)
    else:
        result = background

    return (result, rms) if return_rms else result


def spatial_background_parallel(frames, max_workers: int = MAX_WORKERS, box_size: int = 64, filter_size: int = 3, sigma: float = 3.0, maxiters: int = 5, downsample: int = 4, subtract: bool = True, return_rms: bool = False):
    if not frames:
        raise RuntimeError("No frames provided")

    worker = partial(_spatial_background_frame, box_size=box_size, filter_size=filter_size, sigma=sigma, maxiters=maxiters, downsample=downsample, subtract=subtract, return_rms=return_rms)

    return run_parallel(worker, frames, max_workers=max_workers)

# ============================================================
# Sigma thresholding (absolute, noise-adaptive)
# ============================================================
def _sigma_threshold_frame(frame, rms, nsigma):
    if frame is None:
        return None

    # frame is already background-subtracted (sky ~ 0), so detection is simply
    # "n local noise sigmas above zero". per-pixel rms means a noisy region demands a proportionally taller peak, unlike a single global cut.
    threshold = nsigma * rms
    binary = (frame.astype(np.float32) > threshold).astype(np.uint8) * 255
    return binary

def sigma_threshold_parallel(frames, rms_maps, max_workers: int = MAX_WORKERS, nsigma: float = 5.0):
    if not frames:
        raise RuntimeError("No frames provided")

    # each frame pairs with its own rms map, so zip rather than a single partial
    worker = partial(_sigma_threshold_frame, nsigma=nsigma)
    return list(_EXECUTOR.map(worker, frames, rms_maps))

# ============================================================
# Component filtering (hot-pixel / cosmic-ray rejection)
# ============================================================
def _component_filter_frame(frame, min_area, max_area, max_aspect, min_compactness):
    if frame is None:
        return None

    # frame is a binary mask (0/255) from the sigma threshold. Reject blobs that aren't star-shaped: single-pixel hot pixels and cosmic rays fail min_area, elongated artifacts fail max_aspect, sparse ones fail min_compactness.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(frame, connectivity=8, ltype=cv2.CV_32S)

    if num_labels <= 1:
        return np.zeros_like(frame, dtype=np.uint8)

    component_stats = stats[1:] # row 0 is the background label
    areas = component_stats[:, cv2.CC_STAT_AREA]
    widths = component_stats[:, cv2.CC_STAT_WIDTH]
    heights = component_stats[:, cv2.CC_STAT_HEIGHT]

    minimum_side = np.minimum(widths, heights)
    maximum_side = np.maximum(widths, heights)
    aspect_ratios = maximum_side / np.maximum(minimum_side, 1)
    compactness = areas / np.maximum(widths * heights, 1)

    valid = ( (areas >= min_area) & (areas <= max_area) & (aspect_ratios <= max_aspect) & (compactness >= min_compactness))

    # label 0 (background) stays invalid; map surviving labels back to 255
    label_lookup = np.zeros(num_labels, dtype=np.uint8)
    label_lookup[1:] = valid.astype(np.uint8) * 255
    return label_lookup[labels]

def component_filter_parallel(frames, max_workers: int = MAX_WORKERS, min_area: int = 2, max_area: int = 150, max_aspect: float = 3.0, min_compactness: float = 0.3):
    if not frames:
        raise RuntimeError("No frames provided")

    worker = partial(_component_filter_frame, min_area=min_area, max_area=max_area, max_aspect=max_aspect, min_compactness=min_compactness)

    return run_parallel(worker, frames, max_workers=max_workers)

# ============================================================
# Optional shutdown
# ============================================================
def shutdown_processing_pool() -> None:
    print("Celestil Tools' executor thread closed")
    _EXECUTOR.shutdown(wait=True, cancel_futures=True)