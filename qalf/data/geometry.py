"""Motion-decoupled landmark representation."""

from __future__ import annotations

import numpy as np

# Stable Face Mesh points covering contour, brows, eyes, nose, mouth, and chin.
DEFAULT_LANDMARK_INDICES = (
    10, 21, 33, 54, 58, 67, 93, 103, 109, 127, 132, 133, 145, 152, 155, 159,
    172, 176, 181, 191, 197, 205, 234, 246, 249, 263, 276, 284, 288, 297, 323,
    326, 332, 338, 356, 361, 362, 374, 378, 382, 386, 397, 400, 405, 415, 425,
    454, 1, 2, 4, 5, 6, 13, 14, 17, 61, 78, 291, 308,
)
GEOMETRY_FEATURE_MODES = {"normalized", "aligned", "motion_only", "aligned_motion"}


def _fill_missing(points: np.ndarray, detected: np.ndarray) -> np.ndarray:
    output = points.copy()
    times = np.arange(points.shape[0])
    valid = np.flatnonzero(detected & np.isfinite(points).all(axis=(1, 2)))
    if valid.size == 0:
        raise ValueError("The clip contains no valid landmark frame")
    for landmark in range(points.shape[1]):
        for coordinate in range(points.shape[2]):
            values = points[valid, landmark, coordinate]
            output[:, landmark, coordinate] = np.interp(times, valid, values)
    return output


def _normalize_frame(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    return centered / max(scale, 1e-6)


def _align_to_reference(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    covariance = points.T @ reference
    left, _, right_t = np.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return points @ rotation


def build_geometry_features(
    landmarks: np.ndarray,
    detected: np.ndarray,
    timestamps_sec: np.ndarray | None = None,
    indices: tuple[int, ...] = DEFAULT_LANDMARK_INDICES,
    feature_mode: str = "aligned_motion",
) -> tuple[np.ndarray, np.ndarray]:
    if feature_mode not in GEOMETRY_FEATURE_MODES:
        raise ValueError(f"Unsupported geometry feature mode: {feature_mode}")
    if landmarks.ndim != 3 or landmarks.shape[-1] < 2:
        raise ValueError(f"Expected [T,K,C] landmarks, got {landmarks.shape}")
    if max(indices) >= landmarks.shape[1]:
        raise ValueError("Configured landmark index exceeds cache point count")
    points = landmarks[:, indices, :2].astype(np.float32)
    detected = np.asarray(detected, dtype=bool)
    points = _fill_missing(points, detected)
    normalized = np.stack([_normalize_frame(frame) for frame in points])
    reference = normalized[0]
    aligned = np.stack([_align_to_reference(frame, reference) for frame in normalized])

    if timestamps_sec is None or len(timestamps_sec) != len(aligned):
        timestamps_sec = np.arange(len(aligned), dtype=np.float32)
    else:
        timestamps_sec = np.asarray(timestamps_sec, dtype=np.float32)
    deltas = np.diff(timestamps_sec, prepend=timestamps_sec[0])
    positive = deltas[deltas > 1e-6]
    fallback_delta = float(np.median(positive)) if positive.size else 1.0
    deltas[0] = fallback_delta
    deltas = np.maximum(deltas / max(fallback_delta, 1e-6), 1e-3)

    velocity = np.zeros_like(aligned)
    velocity[1:] = np.diff(aligned, axis=0) / deltas[1:, None, None]
    acceleration = np.zeros_like(aligned)
    acceleration[1:] = np.diff(velocity, axis=0) / deltas[1:, None, None]
    frame_valid = detected.astype(np.float32)[:, None]
    if feature_mode == "normalized":
        components = [normalized.reshape(len(normalized), -1)]
    elif feature_mode == "aligned":
        components = [aligned.reshape(len(aligned), -1)]
    elif feature_mode == "motion_only":
        components = [
            velocity.reshape(len(aligned), -1),
            acceleration.reshape(len(aligned), -1),
        ]
    else:
        components = [
            aligned.reshape(len(aligned), -1),
            velocity.reshape(len(aligned), -1),
            acceleration.reshape(len(aligned), -1),
        ]
    features = np.concatenate([*components, frame_valid], axis=1).astype(np.float32)
    delta_cv = float(np.std(deltas) / max(np.mean(deltas), 1e-6))
    quality = np.asarray(
        [
            detected.mean(),
            1.0 - detected.mean(),
            min(float(np.log1p(np.mean(np.abs(velocity)))), 1.0),
            min(float(np.log1p(np.mean(np.abs(acceleration)))), 1.0),
            delta_cv,
        ],
        dtype=np.float32,
    )
    return features, quality


def geometry_input_dim(
    indices: tuple[int, ...] = DEFAULT_LANDMARK_INDICES,
    feature_mode: str = "aligned_motion",
) -> int:
    if feature_mode not in GEOMETRY_FEATURE_MODES:
        raise ValueError(f"Unsupported geometry feature mode: {feature_mode}")
    coordinate_multipliers = {
        "normalized": 2,
        "aligned": 2,
        "motion_only": 4,
        "aligned_motion": 6,
    }
    return len(indices) * coordinate_multipliers[feature_mode] + 1
