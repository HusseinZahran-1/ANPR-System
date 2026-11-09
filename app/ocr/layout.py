"""Versioned, label-independent plate layout transforms for OCR checkpoints."""
from PIL import Image

SUPPORTED_LAYOUTS = ('none', 'unfold_v1', 'rectify_unfold_v2',
                     'rectify_unfold_wide_v3')


def rectify_oblique_single_row(image: Image.Image,
                               max_aspect_ratio: float = 1.6) -> Image.Image | None:
    """Conservatively find a long bright plate within a short oblique crop.

    Experimental geometry only: no transcript, OCR score or country rule is
    consulted. Return None when a sufficiently large elongated region cannot
    be found. The warp derives solely from its minimum-area rectangle.
    """
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    if min(h, w) < 16 or w / h >= max_aspect_ratio:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < 0.22 * w * h:
            break
        rect = cv2.minAreaRect(contour)
        a, b = rect[1]
        if min(a, b) < 8 or max(a, b) / min(a, b) < 2.8:
            continue
        if area / (a * b) < 0.65:
            continue
        points = cv2.boxPoints(rect)
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        indices = [sums.argmin(), differences.argmin(), sums.argmax(), differences.argmax()]
        if len(set(indices)) != 4:
            continue
        src = points[indices].astype(np.float32)
        width = round(max(np.linalg.norm(src[1]-src[0]), np.linalg.norm(src[2]-src[3])))
        height = round(max(np.linalg.norm(src[3]-src[0]), np.linalg.norm(src[2]-src[1])))
        if width / max(height, 1) < 2.8:
            continue
        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(rgb, matrix, (width, height),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped).convert(image.mode)
    return None


def prepare_layout(image: Image.Image, mode: str = 'none') -> Image.Image:
    """Unfold geometrically short plates from top/bottom into left/right.

    This is an experimental checkpoint-specific transform, not a plate type
    classifier. Threshold and cut are fixed; no recognized text is consulted.
    Keeping a version in checkpoint metadata ensures training and inference use
    the identical transform. The input image is never modified.
    """
    if mode not in SUPPORTED_LAYOUTS:
        raise ValueError(f'unsupported OCR layout preprocessing: {mode}')
    max_aspect_ratio = 1.8 if mode == 'rectify_unfold_wide_v3' else 1.6
    if mode in ('rectify_unfold_v2', 'rectify_unfold_wide_v3'):
        rectified = rectify_oblique_single_row(image, max_aspect_ratio)
        if rectified is not None:
            return rectified
    w, h = image.size
    if mode == 'none' or w / h >= max_aspect_ratio or h < 8:
        return image
    cut = round(h * 0.48)
    pieces = [image.crop((0, 0, w, cut)), image.crop((0, cut, w, h))]
    height = max(p.height for p in pieces)
    pieces = [p.resize((round(p.width * height / p.height), height),
                       Image.Resampling.BICUBIC) for p in pieces]
    canvas = Image.new(image.mode, (sum(p.width for p in pieces), height))
    canvas.paste(pieces[0], (0, 0))
    canvas.paste(pieces[1], (pieces[0].width, 0))
    return canvas
