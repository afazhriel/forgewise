"""Tests for the classical computer-vision helpers (deterministic, offline)."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from src import vision


def _noise(width: int = 800, height: int = 600, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((height, width, 3)) * 255).astype("uint8"))


def _solid(value: int, width: int = 800, height: int = 600) -> Image.Image:
    return Image.fromarray((np.ones((height, width, 3)) * value).astype("uint8"))


# --- image quality ----------------------------------------------------------
def test_noise_image_is_usable():
    metrics = vision.analyze_quality(_noise())
    assert metrics.verdict == "USABLE"
    assert metrics.usable is True
    assert metrics.issues == []
    assert metrics.width == 800 and metrics.height == 600


def test_tiny_image_needs_recapture():
    metrics = vision.analyze_quality(_noise(width=50, height=40))
    assert metrics.verdict == "RECAPTURE"
    assert any("Resolution too low" in issue for issue in metrics.issues)


def test_dark_image_needs_recapture():
    metrics = vision.analyze_quality(_solid(10))
    assert metrics.verdict == "RECAPTURE"
    assert any("too dark" in issue for issue in metrics.issues)


def test_blurred_image_needs_recapture():
    blurred = _noise(seed=2).filter(ImageFilter.GaussianBlur(12))
    metrics = vision.analyze_quality(blurred)
    assert metrics.verdict == "RECAPTURE"
    assert any("blurry" in issue for issue in metrics.issues)


def test_load_image_accepts_bytes():
    buffer = BytesIO()
    _noise(width=100, height=80, seed=3).save(buffer, format="PNG")
    image = vision.load_image(buffer.getvalue())
    assert image.mode == "RGB"
    assert image.size == (100, 80)


# --- reference comparison ---------------------------------------------------
def test_identical_images_have_no_material_change():
    reference = _noise(seed=4)
    current = _noise(seed=4)
    result = vision.compare_images(reference, current)
    assert result.verdict == "NO_MATERIAL_CHANGE"
    assert result.changed_fraction == 0.0
    assert result.structural_similarity == 1.0
    assert result.bounding_box is None


def test_changed_region_is_detected():
    rng = np.random.default_rng(5)
    base = (rng.random((600, 800, 3)) * 255).astype("uint8")
    reference = Image.fromarray(base)
    modified = base.copy()
    modified[200:320, 300:430] = 255  # a solid block
    result = vision.compare_images(reference, Image.fromarray(modified))

    assert result.changed_fraction > 0.005
    assert result.verdict in {"REVIEW", "SIGNIFICANT_CHANGE"}
    assert result.structural_similarity < 1.0
    x0, y0, x1, y1 = result.bounding_box
    assert (x0, y0, x1, y1) == (300, 200, 429, 319)


def test_comparison_resizes_current_to_reference():
    reference = _noise(width=800, height=600, seed=6)
    current = _noise(width=400, height=300, seed=6)
    result = vision.compare_images(reference, current)
    assert (result.width, result.height) == (800, 600)


# --- detector interface (honest placeholder) --------------------------------
def test_detector_is_honest_when_no_model_present():
    detector, message = vision.load_detector()
    assert detector is None
    assert "No defect-detection model" in message

    status = vision.detector_status()
    assert status["configured"] is False
