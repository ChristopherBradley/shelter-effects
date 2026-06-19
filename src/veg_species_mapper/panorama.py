"""Crop rectilinear (perspective) views out of an equirectangular 360 panorama.

We render a normal-looking photo looking in a given direction so a plant-ID API
(which expects ordinary photos, not 360s) has a fair chance. Critically, the
*yaw* of each crop relative to the panorama, plus the image's compass angle,
gives an absolute bearing to whatever is in the crop -- the input to triangulation.

Numpy + Pillow only (works on Python 3.14). Sampling is nearest-neighbour, which
is plenty for feeding an ID model.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def crop_perspective(
    pano: Image.Image,
    yaw_deg: float,
    pitch_deg: float = -5.0,
    fov_deg: float = 90.0,
    out_size: tuple[int, int] = (768, 768),
) -> Image.Image:
    """Render a perspective view from an equirectangular panorama.

    yaw_deg:   horizontal look direction relative to the panorama centre column,
               clockwise. (Add the image compass_angle to get an absolute bearing.)
    pitch_deg: vertical look direction; slightly negative looks down toward roadside
               vegetation. positive looks up.
    fov_deg:   horizontal field of view.
    """
    pano_rgb = pano.convert("RGB")
    src = np.asarray(pano_rgb)
    H, W = src.shape[:2]
    out_w, out_h = out_size

    # Camera intrinsics: focal length in pixels from horizontal FOV.
    f = (out_w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)

    # Pixel grid -> camera-space rays (z forward, x right, y down).
    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = u - out_w / 2.0
    y = v - out_h / 2.0
    z = np.full_like(x, f)
    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Rotate about x-axis (pitch), then y-axis (yaw).
    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    cy, sy = np.cos(yaw), np.sin(yaw)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    d = dirs @ rx.T @ ry.T

    # Direction -> equirectangular lon/lat. Centre column == forward (z+).
    lon = np.arctan2(d[..., 0], d[..., 2])   # -pi..pi, +x to the right
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))  # -pi/2..pi/2, +y downward

    sx = (lon / (2 * np.pi) + 0.5) * W
    sy = (lat / np.pi + 0.5) * H
    sx = np.clip(sx.astype(np.int32), 0, W - 1)
    sy = np.clip(sy.astype(np.int32), 0, H - 1)

    out = src[sy, sx]
    return Image.fromarray(out)


def sweep_yaws(n: int = 12, start: float = 0.0) -> list[float]:
    """Evenly spaced yaw angles around the full panorama, e.g. every 30 deg for n=12."""
    step = 360.0 / n
    return [(start + i * step) % 360.0 for i in range(n)]
