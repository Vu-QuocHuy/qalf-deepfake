"""Training-only, temporally coherent Self-Blended Image generation.

The implementation follows the data-generation idea from Self-Blended Images
(Shiohara and Yamasaki, CVPR 2022) while remaining native to QALF's OpenCV and
NumPy input pipeline. It deliberately operates on an aligned clip, not on
independent frames, so the synthetic artifact cannot become a temporal-flicker
shortcut.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

SAMPLE_REAL = "real"
SAMPLE_ORIGINAL_FAKE = "original_fake"
SAMPLE_SBI = "sbi"
SBI_SAMPLE_TYPES = (SAMPLE_REAL, SAMPLE_ORIGINAL_FAKE, SAMPLE_SBI)

# Ordered MediaPipe Face Mesh points around the outer face oval.
FACE_OVAL_INDICES = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

DEFAULT_SBI_CONFIG: dict[str, object] = {
    "enabled": False,
    "mixture": {
        SAMPLE_REAL: 0.50,
        SAMPLE_ORIGINAL_FAKE: 0.25,
        SAMPLE_SBI: 0.25,
    },
    "blend_strengths": [0.25, 0.50, 0.75, 1.00],
    "translation_fraction": 0.03,
    "scale_min": 0.95,
    "scale_max": 1.05,
    "rotation_degrees": 3.0,
    "elastic_probability": 0.75,
    "elastic_grid_size": 8,
    "elastic_strength": 3.0,
    "mask_erode_fraction_max": 0.04,
    "mask_blur_fraction_min": 0.015,
    "mask_blur_fraction_max": 0.050,
    "rgb_shift_limit": 20.0,
    "brightness_limit": 0.10,
    "contrast_limit": 0.12,
    "gamma_min": 0.80,
    "gamma_max": 1.20,
    "downsample_probability": 0.50,
    "downsample_min_scale": 0.50,
    "blur_probability": 0.25,
    "blur_sigma_max": 1.20,
    "sharpen_probability": 0.25,
    "sharpen_amount_max": 0.75,
    "temporal_coherence": "clip",
}


def resolve_sbi_config(config: Mapping[str, object] | None = None) -> dict[str, object]:
    """Validate and return a complete, JSON-serializable SBI configuration."""

    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_SBI_CONFIG)
    if unknown:
        raise ValueError(f"Unknown SBI configuration keys: {sorted(unknown)}")
    resolved = {**DEFAULT_SBI_CONFIG, **supplied}
    resolved["enabled"] = bool(resolved["enabled"])

    mixture = dict(resolved["mixture"])
    if set(mixture) != set(SBI_SAMPLE_TYPES):
        raise ValueError(f"SBI mixture must contain exactly {list(SBI_SAMPLE_TYPES)}")
    mixture = {name: float(mixture[name]) for name in SBI_SAMPLE_TYPES}
    if any(value <= 0.0 for value in mixture.values()):
        raise ValueError("Every SBI mixture share must be positive")
    if not np.isclose(sum(mixture.values()), 1.0, atol=1e-8):
        raise ValueError("SBI mixture shares must sum to one")
    resolved["mixture"] = mixture

    strengths = [float(value) for value in resolved["blend_strengths"]]
    if not strengths or any(not 0.0 < value <= 1.0 for value in strengths):
        raise ValueError("SBI blend strengths must be non-empty and in (0, 1]")
    resolved["blend_strengths"] = strengths

    non_negative = (
        "translation_fraction",
        "rotation_degrees",
        "elastic_strength",
        "mask_erode_fraction_max",
        "mask_blur_fraction_min",
        "mask_blur_fraction_max",
        "rgb_shift_limit",
        "brightness_limit",
        "contrast_limit",
        "blur_sigma_max",
        "sharpen_amount_max",
    )
    for name in non_negative:
        resolved[name] = float(resolved[name])
        if resolved[name] < 0.0:
            raise ValueError(f"SBI {name} cannot be negative")

    probabilities = (
        "elastic_probability",
        "downsample_probability",
        "blur_probability",
        "sharpen_probability",
    )
    for name in probabilities:
        resolved[name] = float(resolved[name])
        if not 0.0 <= resolved[name] <= 1.0:
            raise ValueError(f"SBI {name} must be in [0, 1]")

    for name in ("scale_min", "scale_max", "gamma_min", "gamma_max"):
        resolved[name] = float(resolved[name])
        if resolved[name] <= 0.0:
            raise ValueError(f"SBI {name} must be positive")
    if resolved["scale_min"] > resolved["scale_max"]:
        raise ValueError("SBI scale range is invalid")
    if resolved["gamma_min"] > resolved["gamma_max"]:
        raise ValueError("SBI gamma range is invalid")

    resolved["downsample_min_scale"] = float(resolved["downsample_min_scale"])
    if not 0.0 < resolved["downsample_min_scale"] <= 1.0:
        raise ValueError("SBI downsample_min_scale must be in (0, 1]")
    resolved["elastic_grid_size"] = int(resolved["elastic_grid_size"])
    if resolved["elastic_grid_size"] < 2:
        raise ValueError("SBI elastic_grid_size must be at least two")
    if resolved["mask_blur_fraction_min"] > resolved["mask_blur_fraction_max"]:
        raise ValueError("SBI mask blur range is invalid")
    if resolved["temporal_coherence"] != "clip":
        raise ValueError("Only clip-coherent SBI generation is supported")
    return resolved


def stratum_sampling_weights(
    strata: Sequence[str],
    mixture: Mapping[str, float],
) -> list[float]:
    """Return item weights whose total mass matches each requested stratum share."""

    counts = Counter(strata)
    if set(counts) != set(SBI_SAMPLE_TYPES) or any(counts[name] == 0 for name in SBI_SAMPLE_TYPES):
        raise ValueError(f"SBI training requires all sample strata: counts={dict(counts)}")
    if set(mixture) != set(SBI_SAMPLE_TYPES):
        raise ValueError(f"SBI mixture must contain exactly {list(SBI_SAMPLE_TYPES)}")
    return [float(mixture[name]) / counts[name] for name in strata]


def face_mask_from_aligned_landmarks(
    landmarks: np.ndarray | None,
    image_size: int,
) -> np.ndarray:
    """Construct a bounded face mask in aligned-image coordinates."""

    if image_size < 2:
        raise ValueError("image_size must be at least two")
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    points: np.ndarray | None = None
    if landmarks is not None:
        candidate = np.asarray(landmarks, dtype=np.float32)
        if (
            candidate.ndim == 2
            and candidate.shape[1] >= 2
            and candidate.shape[0] > max(FACE_OVAL_INDICES)
        ):
            candidate = candidate[np.asarray(FACE_OVAL_INDICES), :2]
            if np.isfinite(candidate).all():
                points = candidate
    if points is None:
        # Alignment places the eyes near y=.38 and mouth near y=.72. This
        # conservative fallback is used only when a sampled texture frame lacks
        # valid landmarks; it never reaches outside the canonical face crop.
        center = (int(round(0.50 * image_size)), int(round(0.52 * image_size)))
        axes = (int(round(0.39 * image_size)), int(round(0.47 * image_size)))
        cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 1.0, thickness=-1)
    else:
        hull = cv2.convexHull(np.rint(points).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
    mask[[0, -1], :] = 0.0
    mask[:, [0, -1]] = 0.0
    return mask


@dataclass(frozen=True)
class SBIParameters:
    alter_source: bool
    rgb_shift: tuple[float, float, float]
    brightness: float
    contrast: float
    gamma: float
    downsample_scale: float
    blur_sigma: float
    sharpen_amount: float
    angle_degrees: float
    scale: float
    translate_x_fraction: float
    translate_y_fraction: float
    elastic_enabled: bool
    elastic_seed: int
    erode_fraction: float
    blur_fraction: float
    blend_strength: float


def _sample_parameters(config: Mapping[str, object], rng: np.random.Generator) -> SBIParameters:
    downsample_scale = 1.0
    if rng.random() < float(config["downsample_probability"]):
        downsample_scale = float(rng.uniform(config["downsample_min_scale"], 0.95))
    blur_sigma = 0.0
    sharpen_amount = 0.0
    choice = rng.random()
    if choice < float(config["blur_probability"]):
        blur_sigma = float(rng.uniform(0.10, max(0.10, float(config["blur_sigma_max"]))))
    elif choice < float(config["blur_probability"]) + float(config["sharpen_probability"]):
        sharpen_amount = float(rng.uniform(0.10, config["sharpen_amount_max"]))
    translation = float(config["translation_fraction"])
    return SBIParameters(
        alter_source=bool(rng.integers(0, 2)),
        rgb_shift=tuple(
            float(value)
            for value in rng.uniform(
                -float(config["rgb_shift_limit"]),
                float(config["rgb_shift_limit"]),
                size=3,
            )
        ),
        brightness=float(
            rng.uniform(-float(config["brightness_limit"]), config["brightness_limit"])
        ),
        contrast=float(
            rng.uniform(
                1.0 - float(config["contrast_limit"]), 1.0 + float(config["contrast_limit"])
            )
        ),
        gamma=float(rng.uniform(config["gamma_min"], config["gamma_max"])),
        downsample_scale=downsample_scale,
        blur_sigma=blur_sigma,
        sharpen_amount=sharpen_amount,
        angle_degrees=float(
            rng.uniform(-float(config["rotation_degrees"]), config["rotation_degrees"])
        ),
        scale=float(rng.uniform(config["scale_min"], config["scale_max"])),
        translate_x_fraction=float(rng.uniform(-translation, translation)),
        translate_y_fraction=float(rng.uniform(-translation, translation)),
        elastic_enabled=bool(rng.random() < float(config["elastic_probability"])),
        elastic_seed=int(rng.integers(0, 2**31 - 1)),
        erode_fraction=float(rng.uniform(0.0, config["mask_erode_fraction_max"])),
        blur_fraction=float(
            rng.uniform(config["mask_blur_fraction_min"], config["mask_blur_fraction_max"])
        ),
        blend_strength=float(rng.choice(config["blend_strengths"])),
    )


def _apply_appearance(image: np.ndarray, parameters: SBIParameters) -> np.ndarray:
    output = image.astype(np.float32)
    output = output * parameters.contrast + parameters.brightness * 255.0
    output += np.asarray(parameters.rgb_shift, dtype=np.float32)
    output = np.clip(output, 0.0, 255.0)
    output = 255.0 * np.power(output / 255.0, parameters.gamma)
    height, width = output.shape[:2]
    if parameters.downsample_scale < 1.0:
        small = (
            max(16, int(round(width * parameters.downsample_scale))),
            max(16, int(round(height * parameters.downsample_scale))),
        )
        output = cv2.resize(output, small, interpolation=cv2.INTER_AREA)
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LINEAR)
    if parameters.blur_sigma > 0.0:
        output = cv2.GaussianBlur(output, (0, 0), sigmaX=parameters.blur_sigma)
    elif parameters.sharpen_amount > 0.0:
        smooth = cv2.GaussianBlur(output, (0, 0), sigmaX=1.0)
        output = output + parameters.sharpen_amount * (output - smooth)
    return np.clip(output, 0.0, 255.0).astype(np.float32)


def _elastic_maps(
    height: int,
    width: int,
    config: Mapping[str, object],
    parameters: SBIParameters,
) -> tuple[np.ndarray, np.ndarray]:
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    if not parameters.elastic_enabled or float(config["elastic_strength"]) == 0.0:
        return grid_x, grid_y
    rng = np.random.default_rng(parameters.elastic_seed)
    grid_size = int(config["elastic_grid_size"])
    displacement_x = rng.normal(size=(grid_size, grid_size)).astype(np.float32)
    displacement_y = rng.normal(size=(grid_size, grid_size)).astype(np.float32)
    displacement_x = cv2.resize(displacement_x, (width, height), interpolation=cv2.INTER_CUBIC)
    displacement_y = cv2.resize(displacement_y, (width, height), interpolation=cv2.INTER_CUBIC)
    for displacement in (displacement_x, displacement_y):
        deviation = float(displacement.std())
        if deviation > 1e-6:
            displacement /= deviation
    strength = float(config["elastic_strength"])
    return grid_x + strength * displacement_x, grid_y + strength * displacement_y


def generate_self_blended_clip(
    frames: np.ndarray,
    face_mask: np.ndarray,
    config: Mapping[str, object] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, SBIParameters]:
    """Generate an SBI clip and its coherent alpha masks.

    Args:
        frames: uint8 RGB array with shape ``[T, H, W, 3]``.
        face_mask: face-region mask with shape ``[H, W]``.
        config: complete or partial SBI configuration.
        rng: optional generator used by deterministic tests.
    """

    resolved = resolve_sbi_config(config)
    clip = np.asarray(frames)
    if clip.ndim != 4 or clip.shape[-1] != 3 or clip.shape[0] < 1:
        raise ValueError("SBI frames must have shape [T, H, W, 3]")
    if clip.dtype != np.uint8:
        raise ValueError("SBI frames must use uint8 RGB values")
    height, width = clip.shape[1:3]
    mask = np.asarray(face_mask, dtype=np.float32)
    if mask.shape != (height, width) or not np.isfinite(mask).all():
        raise ValueError("SBI face mask shape or values are invalid")
    mask = np.clip(mask, 0.0, 1.0)
    if float(mask.sum()) <= 0.0 or bool(np.all(mask > 0.0)):
        raise ValueError("SBI face mask must be non-empty and bounded")

    if rng is None:
        # The process-wide NumPy RNG is seeded per DataLoader worker. Drawing one
        # seed here provides deterministic clip-level parameters without sharing
        # mutable generator state across frames.
        rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
    parameters = _sample_parameters(resolved, rng)

    center = (0.5 * (width - 1), 0.5 * (height - 1))
    affine = cv2.getRotationMatrix2D(center, parameters.angle_degrees, parameters.scale)
    affine[0, 2] += parameters.translate_x_fraction * width
    affine[1, 2] += parameters.translate_y_fraction * height
    affine_mask = cv2.warpAffine(
        mask,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    map_x, map_y = _elastic_maps(height, width, resolved, parameters)
    transformed_mask = cv2.remap(
        affine_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    erosion = int(round(parameters.erode_fraction * min(height, width)))
    if erosion > 0:
        kernel_size = max(1, 2 * erosion + 1)
        transformed_mask = cv2.erode(
            transformed_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )
    sigma = max(0.1, parameters.blur_fraction * min(height, width))
    transformed_mask = cv2.GaussianBlur(transformed_mask, (0, 0), sigmaX=sigma)
    maximum = float(transformed_mask.max())
    if maximum <= 1e-6:
        raise ValueError("SBI transform produced an empty blend mask")
    alpha = np.clip(transformed_mask / maximum, 0.0, 1.0)
    alpha = (alpha * parameters.blend_strength).astype(np.float32)

    outputs: list[np.ndarray] = []
    alpha_3d = alpha[:, :, None]
    for frame in clip:
        target = frame.astype(np.float32)
        source = target.copy()
        if parameters.alter_source:
            source = _apply_appearance(source, parameters)
        else:
            target = _apply_appearance(target, parameters)
        source = cv2.warpAffine(
            source,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        source = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        blended = alpha_3d * source + (1.0 - alpha_3d) * target
        outputs.append(np.clip(blended, 0.0, 255.0).astype(np.uint8))
    masks = np.repeat(alpha[None, :, :], len(outputs), axis=0)
    return np.stack(outputs), masks, parameters
