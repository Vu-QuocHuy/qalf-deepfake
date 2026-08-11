"""Pose-decoupled landmark representations for the geometry branch."""

from __future__ import annotations

import numpy as np

# Face Mesh points covering contour, brows, eyes, nose, mouth, and chin.
DEFAULT_LANDMARK_INDICES = (
    10,
    21,
    33,
    54,
    58,
    67,
    93,
    103,
    109,
    127,
    132,
    133,
    145,
    152,
    155,
    159,
    172,
    176,
    181,
    191,
    197,
    205,
    234,
    246,
    249,
    263,
    276,
    284,
    288,
    297,
    323,
    326,
    332,
    338,
    356,
    361,
    362,
    374,
    378,
    382,
    386,
    397,
    400,
    405,
    415,
    425,
    454,
    1,
    2,
    4,
    5,
    6,
    13,
    14,
    17,
    61,
    78,
    291,
    308,
)

# Relatively expression-stable points used only to estimate the rigid transform.
# The transform is then applied to DEFAULT_LANDMARK_INDICES, including expressive regions.
RIGID_ALIGNMENT_INDICES = (
    1,
    4,
    5,
    6,
    10,
    33,
    133,
    152,
    168,
    197,
    234,
    263,
    362,
    454,
)

# The original names are retained as explicit legacy 2D baselines. New experiments should
# use the *_3d modes so yaw and pitch are not interpreted as non-rigid motion.
LEGACY_2D_MODES = {"normalized", "aligned", "motion_only", "aligned_motion"}
POSE_3D_MODES = {
    "aligned_3d",
    "motion_3d",
    "aligned_motion_3d",
    "aligned_motion_rigid_3d",
}
GEOMETRY_FEATURE_MODES = LEGACY_2D_MODES | POSE_3D_MODES
DEFAULT_GEOMETRY_FEATURE_MODE = "aligned_motion_3d"
RIGID_FEATURE_DIM = 30


def _fill_missing(points: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Interpolate missing frames or individual coordinates along time."""

    output = points.copy()
    times = np.arange(points.shape[0])
    if not np.any(detected):
        raise ValueError("The clip contains no valid landmark frame")
    for landmark in range(points.shape[1]):
        for coordinate in range(points.shape[2]):
            valid = np.flatnonzero(detected & np.isfinite(points[:, landmark, coordinate]))
            if valid.size == 0:
                raise ValueError(
                    f"Landmark {landmark} coordinate {coordinate} has no valid observation"
                )
            output[:, landmark, coordinate] = np.interp(
                times,
                valid,
                points[valid, landmark, coordinate],
            )
    return output


def _normalize_frame(points: np.ndarray) -> tuple[np.ndarray, float]:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    return centered / max(scale, 1e-6), scale


def _kabsch_rotation(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return a proper row-vector rotation mapping points to reference."""

    covariance = points.T @ reference
    left, _, right_t = np.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return rotation.astype(np.float32)


def _alignment_medoid(frames: np.ndarray) -> np.ndarray:
    """Estimate a robust temporal reference and return its nearest observed frame.

    A medoid keeps the reference on the observed face manifold and avoids making the
    entire clip depend on a potentially noisy first frame. Two generalized-Procrustes
    median updates make the selection insensitive to rigid pose without the quadratic
    cost of comparing every frame pair.
    """

    if len(frames) == 0:
        raise ValueError("Cannot select an alignment reference from an empty sequence")
    if len(frames) == 1:
        return frames[0]
    reference = frames[len(frames) // 2]
    for _ in range(2):
        aligned = np.stack([frame @ _kabsch_rotation(frame, reference) for frame in frames])
        reference = np.median(aligned, axis=0).astype(np.float32)
        reference, _ = _normalize_frame(reference)

    residuals: list[float] = []
    aligned_frames: list[np.ndarray] = []
    for frame in frames:
        aligned = frame @ _kabsch_rotation(frame, reference)
        aligned_frames.append(aligned)
        residuals.append(float(np.sqrt(np.mean((aligned - reference) ** 2))))
    return aligned_frames[int(np.argmin(residuals))].astype(np.float32)


def _legacy_2d_alignment(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_frames: list[np.ndarray] = []
    scales: list[float] = []
    for frame in points:
        normalized, scale = _normalize_frame(frame)
        normalized_frames.append(normalized)
        scales.append(scale)
    normalized = np.stack(normalized_frames).astype(np.float32)
    reference = _alignment_medoid(normalized)
    aligned = np.stack([frame @ _kabsch_rotation(frame, reference) for frame in normalized]).astype(
        np.float32
    )
    residuals = np.sqrt(np.mean((aligned - reference[None, ...]) ** 2, axis=(1, 2)))
    return normalized, aligned, np.asarray([scales, residuals], dtype=np.float32)


def _rigid_3d_alignment(
    feature_points: np.ndarray,
    anchor_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize and remove 3D rigid pose estimated from expression-stable anchors."""

    normalized_features: list[np.ndarray] = []
    normalized_anchors: list[np.ndarray] = []
    scales: list[float] = []
    for points, anchors in zip(feature_points, anchor_points, strict=True):
        center = anchors.mean(axis=0, keepdims=True)
        centered_anchors = anchors - center
        scale = float(np.sqrt(np.mean(np.sum(centered_anchors**2, axis=1))))
        scale = max(scale, 1e-6)
        normalized_features.append((points - center) / scale)
        normalized_anchors.append(centered_anchors / scale)
        scales.append(scale)

    normalized = np.stack(normalized_features).astype(np.float32)
    anchors = np.stack(normalized_anchors).astype(np.float32)
    reference = _alignment_medoid(anchors)
    aligned_frames: list[np.ndarray] = []
    residuals: list[float] = []
    rigid_features: list[np.ndarray] = []
    for points, frame_anchors, center, scale in zip(
        normalized,
        anchors,
        [anchors.mean(axis=0, keepdims=True) for anchors in anchor_points],
        scales,
        strict=True,
    ):
        rotation = _kabsch_rotation(frame_anchors, reference)
        aligned_frames.append(points @ rotation)
        residuals.append(float(np.sqrt(np.mean((frame_anchors @ rotation - reference) ** 2))))
        # A continuous six-dimensional rotation representation plus translation and
        # log-scale preserves the rigid motion discarded from the aligned landmarks.
        rigid_features.append(
            np.concatenate(
                [
                    rotation[:, :2].reshape(-1),
                    center.reshape(-1),
                    np.asarray([np.log(max(float(scale), 1e-6))], dtype=np.float32),
                ]
            ).astype(np.float32)
        )
    rigid = np.stack(rigid_features).astype(np.float32)
    # Translation and scale in MediaPipe coordinates contain clip/crop-specific offsets.
    # Keep only their within-clip motion so the rigid stream cannot memorize a dataset's
    # face-cropping convention. The rotation already maps each frame to the clip medoid.
    rigid[:, 6:10] -= np.median(rigid[:, 6:10], axis=0, keepdims=True)
    return (
        normalized,
        np.stack(aligned_frames).astype(np.float32),
        np.asarray([scales, residuals], dtype=np.float32),
        rigid,
    )


def _temporal_derivatives(
    aligned: np.ndarray,
    timestamps_sec: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if timestamps_sec is None or len(timestamps_sec) != len(aligned):
        timestamps = np.arange(len(aligned), dtype=np.float32)
    else:
        timestamps = np.asarray(timestamps_sec, dtype=np.float32)
    deltas = np.diff(timestamps, prepend=timestamps[0])
    positive = deltas[deltas > 1e-6]
    fallback_delta = float(np.median(positive)) if positive.size else 1.0
    deltas[0] = fallback_delta
    deltas = np.maximum(deltas / max(fallback_delta, 1e-6), 1e-3)

    velocity = np.zeros_like(aligned)
    velocity[1:] = np.diff(aligned, axis=0) / deltas[1:, None, None]
    acceleration = np.zeros_like(aligned)
    acceleration[1:] = np.diff(velocity, axis=0) / deltas[1:, None, None]
    return velocity, acceleration, deltas


def build_geometry_features(
    landmarks: np.ndarray,
    detected: np.ndarray,
    timestamps_sec: np.ndarray | None = None,
    indices: tuple[int, ...] = DEFAULT_LANDMARK_INDICES,
    feature_mode: str = DEFAULT_GEOMETRY_FEATURE_MODE,
    alignment_indices: tuple[int, ...] = RIGID_ALIGNMENT_INDICES,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-frame geometry features and five measurement-quality descriptors.

    Quality dimensions are detection ratio (high is good), finite-point ratio (high is
    good), rigid alignment residual, crop/face scale coefficient of variation, and sampling
    interval coefficient of variation (the final three are lower-is-better).
    """

    if feature_mode not in GEOMETRY_FEATURE_MODES:
        raise ValueError(f"Unsupported geometry feature mode: {feature_mode}")
    if landmarks.ndim != 3 or landmarks.shape[-1] < 3:
        raise ValueError(f"Expected [T,K,3+] landmarks, got {landmarks.shape}")
    if not indices or max(indices) >= landmarks.shape[1]:
        raise ValueError("Configured landmark index exceeds cache point count")
    if not alignment_indices or max(alignment_indices) >= landmarks.shape[1]:
        raise ValueError("Configured alignment landmark index exceeds cache point count")

    detected = np.asarray(detected, dtype=bool)
    if detected.shape != (len(landmarks),):
        raise ValueError(f"Expected detected shape {(len(landmarks),)}, got {detected.shape}")

    coordinate_count = 2 if feature_mode in LEGACY_2D_MODES else 3
    raw_points = landmarks[:, indices, :coordinate_count].astype(np.float32)
    finite_point_ratio = float(np.mean(np.isfinite(raw_points).all(axis=2)))
    points = _fill_missing(raw_points, detected)

    if feature_mode in LEGACY_2D_MODES:
        normalized, aligned, diagnostics = _legacy_2d_alignment(points)
        rigid_base = None
    else:
        raw_anchors = landmarks[:, alignment_indices, :3].astype(np.float32)
        anchors = _fill_missing(raw_anchors, detected)
        normalized, aligned, diagnostics, rigid_base = _rigid_3d_alignment(points, anchors)

    velocity, acceleration, deltas = _temporal_derivatives(aligned, timestamps_sec)
    frame_valid = detected.astype(np.float32)[:, None]
    if feature_mode in {"normalized"}:
        components = [normalized.reshape(len(normalized), -1)]
    elif feature_mode in {"aligned", "aligned_3d"}:
        components = [aligned.reshape(len(aligned), -1)]
    elif feature_mode in {"motion_only", "motion_3d"}:
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
        if feature_mode == "aligned_motion_rigid_3d":
            assert rigid_base is not None
            rigid_velocity, rigid_acceleration, _ = _temporal_derivatives(
                rigid_base[:, None, :], timestamps_sec
            )
            components.append(
                np.concatenate(
                    [rigid_base, rigid_velocity[:, 0], rigid_acceleration[:, 0]],
                    axis=1,
                )
            )
    features = np.concatenate([*components, frame_valid], axis=1).astype(np.float32)

    scales, residuals = diagnostics
    scale_cv = float(np.std(scales) / max(np.mean(scales), 1e-6))
    delta_cv = float(np.std(deltas) / max(np.mean(deltas), 1e-6))
    quality = np.asarray(
        [
            detected.mean(),
            finite_point_ratio,
            min(float(np.mean(residuals)), 1.0),
            min(scale_cv, 1.0),
            min(delta_cv, 1.0),
        ],
        dtype=np.float32,
    )
    return features, quality


def geometry_input_dim(
    indices: tuple[int, ...] = DEFAULT_LANDMARK_INDICES,
    feature_mode: str = DEFAULT_GEOMETRY_FEATURE_MODE,
) -> int:
    if feature_mode not in GEOMETRY_FEATURE_MODES:
        raise ValueError(f"Unsupported geometry feature mode: {feature_mode}")
    coordinate_multipliers = {
        "normalized": 2,
        "aligned": 2,
        "motion_only": 4,
        "aligned_motion": 6,
        "aligned_3d": 3,
        "motion_3d": 6,
        "aligned_motion_3d": 9,
        "aligned_motion_rigid_3d": 9,
    }
    rigid_dim = RIGID_FEATURE_DIM if feature_mode == "aligned_motion_rigid_3d" else 0
    return len(indices) * coordinate_multipliers[feature_mode] + rigid_dim + 1


def geometry_feature_layout(
    indices: tuple[int, ...] = DEFAULT_LANDMARK_INDICES,
    feature_mode: str = DEFAULT_GEOMETRY_FEATURE_MODE,
) -> tuple[int, int, int]:
    """Return node count, per-node feature width, and rigid feature width.

    Graph encoders require a structured aligned-position/motion layout. Legacy
    feature modes remain valid for the original flat temporal encoder.
    """

    if feature_mode == "aligned_motion_3d":
        return len(indices), 9, 0
    if feature_mode == "aligned_motion_rigid_3d":
        return len(indices), 9, RIGID_FEATURE_DIM
    return len(indices), 0, 0
