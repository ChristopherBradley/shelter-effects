"""Isolate the dominant plant in a street-level image so it fills the frame before
sending to a plant-ID API.

We found Pl@ntNet returns noise on whole street scenes (road/sky/buildings) but works
on tight single-plant crops. A full object detector (SegFormer etc.) needs PyTorch,
which has no wheels for this Intel-mac env, so we use a lightweight, dependency-free
vegetation segmentation instead:

  Excess-Green index (ExG = 2G - R - B) -> threshold -> morphological cleanup ->
  largest connected vegetation blob -> padded bounding-box crop.

Crude but effective for roadside trees/shrubs (which are green and contiguous), and it
directly achieves "make the plant most of the frame". numpy + scipy + PIL only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class VegCrop:
    image: Image.Image
    bbox: tuple[int, int, int, int]   # left, top, right, bottom in source px
    veg_fraction: float               # fraction of the crop that is vegetation
    area_fraction: float              # blob area / full image area


def _exg(rgb: np.ndarray) -> np.ndarray:
    """Excess-Green vegetation index in [-1,1]-ish; high for green vegetation."""
    rgb = rgb.astype("float32") / 255.0
    s = rgb.sum(axis=2) + 1e-6
    r, g, b = rgb[..., 0] / s, rgb[..., 1] / s, rgb[..., 2] / s
    return 2 * g - r - b


def vegetation_mask(img: Image.Image, thresh: float = 0.12) -> np.ndarray:
    """Boolean vegetation mask via Excess-Green + light morphological cleanup."""
    from scipy.ndimage import binary_opening, binary_closing
    rgb = np.asarray(img.convert("RGB"))
    mask = _exg(rgb) > thresh
    mask = binary_opening(mask, iterations=2)
    mask = binary_closing(mask, iterations=3)
    return mask


def largest_plant_crop(
    img: Image.Image,
    thresh: float = 0.12,
    min_area_frac: float = 0.02,
    pad_frac: float = 0.08,
    square: bool = True,
) -> VegCrop | None:
    """Find the largest contiguous vegetation region and return a padded crop of it.
    Returns None if no vegetation blob exceeds min_area_frac of the image."""
    from scipy.ndimage import label
    img = img.convert("RGB")
    W, H = img.size
    mask = vegetation_mask(img, thresh=thresh)
    lab, n = label(mask)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background
    biggest = int(sizes.argmax())
    area_frac = sizes[biggest] / (W * H)
    if area_frac < min_area_frac:
        return None

    ys, xs = np.where(lab == biggest)
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    # pad
    ph, pw = int((bottom - top) * pad_frac), int((right - left) * pad_frac)
    top, bottom = max(0, top - ph), min(H, bottom + ph)
    left, right = max(0, left - pw), min(W, right + pw)
    if square:
        # expand the shorter side to make a square-ish crop (Pl@ntNet prefers this)
        bw, bh = right - left, bottom - top
        if bw > bh:
            grow = (bw - bh) // 2
            top, bottom = max(0, top - grow), min(H, bottom + grow)
        else:
            grow = (bh - bw) // 2
            left, right = max(0, left - grow), min(W, right + grow)

    crop = img.crop((left, top, right, bottom))
    veg_in_crop = float(mask[top:bottom, left:right].mean())
    return VegCrop(crop, (left, top, right, bottom), veg_in_crop, float(area_frac))


def overlay_mask(img: Image.Image, bbox=None, thresh: float = 0.12) -> Image.Image:
    """Visual: tint vegetation green and draw the chosen crop box (for inspection)."""
    from PIL import ImageDraw
    img = img.convert("RGB")
    mask = vegetation_mask(img, thresh=thresh)
    arr = np.asarray(img).copy()
    arr[mask] = (0.5 * arr[mask] + 0.5 * np.array([0, 255, 0])).astype("uint8")
    out = Image.fromarray(arr)
    if bbox:
        ImageDraw.Draw(out).rectangle(bbox, outline=(255, 0, 0), width=6)
    return out
