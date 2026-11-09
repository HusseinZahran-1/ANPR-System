"""Layout compatibility and reading-order contracts, using synthetic images."""
import numpy as np
import pytest
from PIL import Image

from app.ocr.layout import prepare_layout, rectify_oblique_single_row


def test_legacy_mode_preserves_pixels_and_input():
    rgb = np.random.default_rng(42).integers(0, 256, (50, 40, 3), dtype=np.uint8)
    source = Image.fromarray(rgb)
    assert np.array_equal(np.array(prepare_layout(source, 'none')), rgb)
    assert np.array_equal(np.array(source), rgb)


def test_single_row_is_not_unfolded():
    rgb = np.random.default_rng(42).integers(0, 256, (30, 180, 3), dtype=np.uint8)
    assert np.array_equal(np.array(prepare_layout(Image.fromarray(rgb), 'unfold_v1')), rgb)


def test_upper_row_precedes_lower_row_without_mirroring():
    # Colored landmarks mark reading order independently of OCR predictions.
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:48, :50] = (255, 0, 0)
    rgb[:48, 50:] = (0, 255, 0)
    rgb[48:, :50] = (0, 0, 255)
    rgb[48:, 50:] = (255, 255, 0)
    result = np.array(prepare_layout(Image.fromarray(rgb), 'unfold_v1'))
    assert result.shape[1] > result.shape[0] * 3
    samples = [result[result.shape[0] // 2, round(result.shape[1] * f)]
               for f in (.125, .375, .625, .875)]
    assert np.array_equal(samples, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])


def test_unknown_checkpoint_transform_is_rejected():
    with pytest.raises(ValueError, match='unsupported OCR layout'):
        prepare_layout(Image.new('RGB', (100, 50)), 'unfold_v999')


def test_oblique_long_plate_rectification_preserves_reading_order():
    import cv2
    rgb = np.zeros((144, 144, 3), dtype=np.uint8)
    box = cv2.boxPoints(((72, 72), (140, 35), -35)).astype(np.int32)
    cv2.fillConvexPoly(rgb, box, (220, 220, 220))
    # Left/right dark markers remain left/right after straightening.
    cv2.circle(rgb, (40, 94), 4, (20, 0, 0), -1)
    cv2.circle(rgb, (104, 50), 4, (0, 0, 20), -1)
    im = Image.fromarray(rgb)
    result = rectify_oblique_single_row(im)
    assert result is not None and result.width / result.height > 3
    arr = np.asarray(result)
    left, right = arr[:, :arr.shape[1]//2], arr[:, arr.shape[1]//2:]
    assert (left[..., 0].astype(int) - left[..., 2]).sum() > 0
    assert (right[..., 2].astype(int) - right[..., 0]).sum() > 0
    assert np.array_equal(np.asarray(im), rgb)


def test_short_stacked_region_does_not_trigger_single_row_rectification():
    import cv2
    rgb = np.zeros((160, 160, 3), dtype=np.uint8)
    box = cv2.boxPoints(((80, 80), (125, 65), -20)).astype(np.int32)
    cv2.fillConvexPoly(rgb, box, (220, 220, 220))
    im = Image.fromarray(rgb)
    assert rectify_oblique_single_row(im) is None
    assert np.array_equal(np.asarray(prepare_layout(im, 'rectify_unfold_v2')),
                          np.asarray(prepare_layout(im, 'unfold_v1')))


def test_blank_crop_geometry_falls_back_without_failure():
    im = Image.new('RGB', (60, 60), 'black')
    assert rectify_oblique_single_row(im) is None
    assert np.array_equal(np.asarray(prepare_layout(im, 'rectify_unfold_v2')),
                          np.asarray(prepare_layout(im, 'unfold_v1')))
