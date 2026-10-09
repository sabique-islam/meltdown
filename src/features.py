"""Contour, moment, and HOG features used by the four pipeline stages."""

from __future__ import annotations

import cv2
import numpy as np

ORIENTATION_SIZE = (64, 128)  # width, height -> 3780-D HOG
WING_SIZE = (80, 96)  # width, height -> 3564-D HOG
BODY_THRESHOLD = 100


def body_mask(gray: np.ndarray) -> np.ndarray:
    """Keep dark fly bodies. Wings and the bright well stay out at this cutoff.

    On the 1530x1530 courtship frames, threshold 100 puts every labeled body
    point inside a contour and leaves background specks under about 350 px.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, BODY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return binary


def fly_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) >= 4]


def contour_features(contour: np.ndarray) -> tuple[float, float]:
    area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    if moments["m00"] <= 1e-6:
        return area, 1.0
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]
    spread = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)
    major = max(mu20 + mu02 + spread, 1e-6)
    minor = max(mu20 + mu02 - spread, 1e-6)
    return area, float(np.sqrt(major / minor))


def centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] <= 1e-6:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w / 2.0, y + h / 2.0
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def major_axis_angle(mask: np.ndarray) -> float:
    """Undirected major-axis angle in degrees, clockwise from +x. Range (-90, 90]."""
    moments = cv2.moments(mask)
    return float(0.5 * np.degrees(np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])))


def axis_to_heading(axis_angle: float) -> float:
    """One of the two headings along a major axis, clockwise degrees from image-up."""
    return float((axis_angle + 90.0) % 360.0)


def angular_difference(left: float, right: float) -> float:
    return float((left - right + 180.0) % 360.0 - 180.0)


def hog_descriptor(image: np.ndarray, kind: str) -> np.ndarray:
    """Dalal-Triggs HOG: 8x8 cells, 9 bins, 2x2 blocks with an 8-pixel stride.

    A 64x128 window is 3780 dimensions. An 80x96 window is 3564 dimensions.
    Those are the descriptor lengths used for orientation and wing angle.
    """
    window = ORIENTATION_SIZE if kind == "orientation" else WING_SIZE
    resized = cv2.resize(image, window, interpolation=cv2.INTER_AREA).astype(np.float32)
    gradient_x = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    angle = np.rad2deg(np.arctan2(gradient_y, gradient_x)) % 180.0

    cell = 8
    bins = 9
    cells_y = resized.shape[0] // cell
    cells_x = resized.shape[1] // cell
    magnitude = magnitude[: cells_y * cell, : cells_x * cell].reshape(cells_y, cell, cells_x, cell)
    angle = angle[: cells_y * cell, : cells_x * cell].reshape(cells_y, cell, cells_x, cell)
    magnitude = magnitude.transpose(0, 2, 1, 3)
    angle = angle.transpose(0, 2, 1, 3)

    bin_float = angle / (180.0 / bins)
    bin_low = np.floor(bin_float).astype(np.int32) % bins
    bin_high = (bin_low + 1) % bins
    high_weight = (bin_float - np.floor(bin_float)).astype(np.float32)
    histograms = np.zeros((cells_y, cells_x, bins), np.float32)
    for bin_index in range(bins):
        histograms[:, :, bin_index] += (magnitude * (1.0 - high_weight) * (bin_low == bin_index)).sum(axis=(2, 3))
        histograms[:, :, bin_index] += (magnitude * high_weight * (bin_high == bin_index)).sum(axis=(2, 3))

    blocks_y = cells_y - 1
    blocks_x = cells_x - 1
    blocks = np.concatenate(
        [
            histograms[0:blocks_y, 0:blocks_x],
            histograms[0:blocks_y, 1 : blocks_x + 1],
            histograms[1 : blocks_y + 1, 0:blocks_x],
            histograms[1 : blocks_y + 1, 1 : blocks_x + 1],
        ],
        axis=-1,
    ).reshape(blocks_y * blocks_x, -1)
    blocks /= np.linalg.norm(blocks, axis=1, keepdims=True) + 1e-6
    return blocks.ravel().astype(np.float64)


def _rotate(image: np.ndarray, angle_deg: float, border: int) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderValue=border,
    )


def orientation_patch(gray: np.ndarray, contour: np.ndarray) -> tuple[np.ndarray, float]:
    """Moment-align a fly so its body axis is vertical, and return a base heading.

    The heading still has a 180 degree ambiguity. The orientation classifier
    decides whether to add 180 degrees.
    """
    x, y, w, h = cv2.boundingRect(contour)
    pad = int(0.45 * max(w, h))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, gray.shape[1])
    y1 = min(y + h + pad, gray.shape[0])
    crop = gray[y0:y1, x0:x1]
    mask = np.zeros(crop.shape, np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x0
    shifted[:, 0, 1] -= y0
    cv2.drawContours(mask, [shifted], -1, 255, -1)
    axis = major_axis_angle(mask)
    aligned = _rotate(crop, axis - 90.0, border=208)
    aligned_mask = _rotate(mask, axis - 90.0, border=0)
    points = cv2.findNonZero(aligned_mask)
    if points is not None:
        bx, by, bw, bh = cv2.boundingRect(points)
        grow = int(0.2 * max(bw, bh))
        bx0 = max(bx - grow, 0)
        by0 = max(by - grow, 0)
        aligned = aligned[by0 : by + bh + grow, bx0 : bx + bw + grow]
    return aligned, axis_to_heading(axis)


def upright_fly(gray: np.ndarray, cx: float, cy: float, heading: float, box: int = 640) -> np.ndarray:
    """Rotate a square crop so the given heading points up."""
    half = box // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    canvas = np.full((box, box), 208, np.uint8)
    src_x0 = max(x0, 0)
    src_y0 = max(y0, 0)
    src_x1 = min(x0 + box, gray.shape[1])
    src_y1 = min(y0 + box, gray.shape[0])
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    canvas[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = gray[
        src_y0:src_y1, src_x0:src_x1
    ]
    # Positive OpenCV angles are counter-clockwise, matching a clockwise heading.
    return _rotate(canvas, heading, border=208)


def wing_patches(upright: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop around the fly center, keep the wings, and drop the thin legs.

    Returns the right half and a mirrored left half so one regressor can
    score both wings.
    """
    height, width = upright.shape
    center_y, center_x = height // 2, width // 2
    half_w, half_h = 160, 180
    crop = upright[
        max(center_y - half_h, 0) : min(center_y + half_h, height),
        max(center_x - half_w, 0) : min(center_x + half_w, width),
    ]
    blurred = cv2.medianBlur(crop, 5)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        6,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.erode(binary, kernel, iterations=1)
    mid = binary.shape[1] // 2
    right = binary[:, mid:]
    left = cv2.flip(binary[:, :mid], 1)
    return right, left
