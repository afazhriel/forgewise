"""Computer-vision helpers for ForgeWise — honest, classical, offline.

Two independent tools are provided, both fully deterministic and dependency-light
(Pillow + NumPy only). Neither one diagnoses a machine or claims to detect
defects:

1. **Image-quality / usability analysis** (:func:`analyze_quality`) — measures
   resolution, brightness, contrast, sharpness (variance of the Laplacian) and
   colour balance, then returns a *usability* verdict (``USABLE`` / ``RECAPTURE``).
   This is an evidence-quality gate, the visual analogue of the sensor
   completeness checks — not a condition assessment.

2. **Reference comparison / visual difference** (:func:`compare_images`) —
   compares a "known-good" reference image against a current image of the same
   view and reports the changed fraction, mean absolute difference, a global SSIM
   approximation, and the bounding box of the changes. This is classical change
   detection: it reports *visual differences*, **not** defects, and always
   requires human review.

An interface for a future trained defect detector (:func:`load_detector`) is
included, but it deliberately returns "no detector configured" until a model
artifact and its runtime are actually present — the UI must never fabricate
detections.

Keeping these functions free of Streamlit makes them unit-testable with synthetic
images (see ``tests/test_vision.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np
from PIL import Image

ImageLike = Union[str, Path, bytes, Image.Image]

# Documented, adjustable quality thresholds (see README).
MIN_WIDTH = 640
MIN_HEIGHT = 480
MIN_BRIGHTNESS = 0.15
MAX_BRIGHTNESS = 0.92
MIN_CONTRAST = 0.05
MIN_SHARPNESS = 40.0  # variance of the Laplacian on an 8-bit (0..255) luma image

# Documented comparison thresholds.
CHANGE_PIXEL_THRESHOLD = 25  # per-pixel |difference| in 8-bit luma
NO_CHANGE_FRACTION = 0.005
SIGNIFICANT_CHANGE_FRACTION = 0.08


def load_image(source: ImageLike) -> Image.Image:
    """Load an image from a path, raw bytes, or an existing PIL image as RGB."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(source))).convert("RGB")
    return Image.open(source).convert("RGB")


def to_luma(image: Image.Image) -> np.ndarray:
    """Return the luma channel as a float array in 0..255."""
    return np.asarray(image.convert("L"), dtype=np.float64)


def _laplacian(luma: np.ndarray) -> np.ndarray:
    """3x3 Laplacian response (vectorised, interior only)."""
    out = np.zeros_like(luma)
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return out
    out[1:-1, 1:-1] = (
        luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
        - 4.0 * luma[1:-1, 1:-1]
    )
    return out


@dataclass
class QualityMetrics:
    width: int
    height: int
    megapixels: float
    brightness: float
    contrast: float
    sharpness: float
    color_balance: float
    verdict: str  # "USABLE" | "RECAPTURE"
    issues: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.verdict == "USABLE"


def analyze_quality(img: ImageLike) -> QualityMetrics:
    """Measure objective image-quality metrics and return a usability verdict."""
    image = load_image(img)
    width, height = image.size
    luma = to_luma(image)

    brightness = float(luma.mean() / 255.0) if luma.size else 0.0
    contrast = float(luma.std() / 255.0) if luma.size else 0.0
    sharpness = float(_laplacian(luma).var())

    arr = np.asarray(image, dtype=np.float64)
    channel_means = arr.reshape(-1, 3).mean(axis=0)
    cast = float((channel_means.max() - channel_means.min()) / 255.0)
    color_balance = float(np.clip(1.0 - cast, 0.0, 1.0))

    issues: list[str] = []
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        issues.append(
            f"Resolution too low ({width}x{height}); need at least "
            f"{MIN_WIDTH}x{MIN_HEIGHT}."
        )
    if brightness < MIN_BRIGHTNESS:
        issues.append(f"Image is too dark (brightness {brightness:.2f}).")
    elif brightness > MAX_BRIGHTNESS:
        issues.append(f"Image is over-exposed (brightness {brightness:.2f}).")
    if contrast < MIN_CONTRAST:
        issues.append(f"Contrast is too low ({contrast:.2f}).")
    if sharpness < MIN_SHARPNESS:
        issues.append(f"Image appears blurry (sharpness {sharpness:.1f}).")

    verdict = "USABLE" if not issues else "RECAPTURE"
    return QualityMetrics(
        width=width,
        height=height,
        megapixels=round(width * height / 1_000_000.0, 3),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        sharpness=round(sharpness, 2),
        color_balance=round(color_balance, 4),
        verdict=verdict,
        issues=issues,
    )


@dataclass
class ComparisonResult:
    width: int
    height: int
    changed_fraction: float
    mean_abs_diff: float
    structural_similarity: float
    bounding_box: Optional[tuple[int, int, int, int]]  # (x0, y0, x1, y1)
    verdict: str  # "NO_MATERIAL_CHANGE" | "REVIEW" | "SIGNIFICANT_CHANGE"
    note: str


def _global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Single-window (global) SSIM. Documented as an approximation, not local SSIM."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denominator == 0:
        return 1.0
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def compare_images(reference: ImageLike, current: ImageLike) -> ComparisonResult:
    """Compare a current image against a reference of the same view.

    The current image is resized to the reference size. Returns change statistics
    for human review — this is visual *difference* detection, not defect
    classification.
    """
    ref_img = load_image(reference)
    cur_img = load_image(current).resize(ref_img.size, Image.Resampling.BILINEAR)

    ref = to_luma(ref_img)
    cur = to_luma(cur_img)
    diff = np.abs(ref - cur)

    changed = diff > CHANGE_PIXEL_THRESHOLD
    changed_fraction = float(changed.mean()) if changed.size else 0.0
    mean_abs_diff = float(diff.mean())

    bbox: Optional[tuple[int, int, int, int]] = None
    if changed.any():
        ys, xs = np.where(changed)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    ssim = _global_ssim(ref, cur)

    if changed_fraction < NO_CHANGE_FRACTION:
        verdict = "NO_MATERIAL_CHANGE"
    elif changed_fraction < SIGNIFICANT_CHANGE_FRACTION:
        verdict = "REVIEW"
    else:
        verdict = "SIGNIFICANT_CHANGE"

    note = (
        "Change detection compares two images of the same view. It reports visual "
        "differences only — it does not identify or name defects, and a human must "
        "review any difference in context."
    )
    return ComparisonResult(
        width=ref_img.size[0],
        height=ref_img.size[1],
        changed_fraction=round(changed_fraction, 5),
        mean_abs_diff=round(mean_abs_diff, 3),
        structural_similarity=round(ssim, 4),
        bounding_box=bbox,
        verdict=verdict,
        note=note,
    )


# ---------------------------------------------------------------------------
# Pluggable defect-detector interface (honest placeholder — no trained model)
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1


class Detector(Protocol):
    name: str

    def detect(self, image: Image.Image) -> list[Detection]:  # pragma: no cover
        ...


DETECTOR_DIR = Path(__file__).resolve().parent.parent / "models" / "vision"


def detector_status() -> dict:
    """Report what defect-detection capability is actually present.

    Always honest: unless a model artifact *and* a supported runtime are both
    available, this reports that no detector is configured.
    """
    runtime = None
    for module_name in ("onnxruntime", "tflite_runtime", "tensorflow", "torch"):
        try:  # pragma: no cover - depends on the environment
            __import__(module_name)
            runtime = module_name
            break
        except Exception:
            continue

    models = []
    if DETECTOR_DIR.exists():
        models = [p.name for p in DETECTOR_DIR.iterdir() if p.suffix in {".onnx", ".tflite", ".pt"}]

    if not models:
        message = (
            "No defect-detection model is installed. This is an interface "
            "placeholder: it will not invent detections. Use the image-quality and "
            "reference-comparison tools, which work today."
        )
    elif runtime is None:
        message = (
            f"Found model artifact(s) {models} but no supported runtime "
            "(onnxruntime / tflite / torch). Install the matching runtime to enable it."
        )
    else:
        message = f"Model artifact(s) {models} found with runtime '{runtime}'."

    return {"runtime": runtime, "models": models, "configured": bool(models) and runtime is not None, "message": message}


def load_detector() -> tuple[Optional[Detector], str]:
    """Return ``(detector, message)``. Returns ``(None, reason)`` when unavailable."""
    status = detector_status()
    if not status["configured"]:
        return None, status["message"]
    # A real integration would construct the detector here. We deliberately do not
    # fabricate one; without a committed model this branch is unreachable.
    return None, status["message"]  # pragma: no cover
