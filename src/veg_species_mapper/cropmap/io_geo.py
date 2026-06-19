"""Write categorical GeoTIFFs (with colour tables) and summary PNG maps/plots.

All functions take a `classes` legend dict {id: (name, (R,G,B))} so the same code
serves the US (CDL) and Australian (NLUM) legends.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .legend import CLASSES as CLASSES_US


def _names(classes):
    return {k: v[0] for k, v in classes.items()}


def _colormap(classes):
    cm = {k: (*rgb, 255) for k, (_, rgb) in classes.items()}
    cm[255] = (0, 0, 0, 0)
    return cm


def write_class_geotiff(arr2d, like, path, classes=CLASSES_US):
    import rioxarray  # noqa
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(arr2d.astype("uint8"), coords={"y": like.y, "x": like.x},
                      dims=("y", "x")).rio.write_crs(like.rio.crs)
    da.rio.write_nodata(255, inplace=True)
    da.rio.to_raster(path, dtype="uint8")
    import rasterio
    with rasterio.open(path, "r+") as ds:
        ds.write_colormap(1, _colormap(classes))
    return path


def write_float_geotiff(arr2d, like, path):
    import rioxarray  # noqa
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(arr2d.astype("float32"), coords={"y": like.y, "x": like.x},
                      dims=("y", "x")).rio.write_crs(like.rio.crs)
    da.rio.to_raster(path, dtype="float32")
    return path


def _legend_handles(class_ids, classes):
    from matplotlib.patches import Patch
    names = _names(classes)
    return [Patch(facecolor=np.array(classes[c][1]) / 255, label=f"{c} {names[c]}")
            for c in class_ids if c in classes]


def save_class_map_png(arr2d, class_ids, title, path, classes=CLASSES_US):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    maxc = max(classes) + 1
    lut = np.zeros((maxc, 3))
    for c, (_, rgb) in classes.items():
        lut[c] = np.array(rgb) / 255
    cmap = ListedColormap(lut)
    norm = BoundaryNorm(np.arange(-0.5, maxc + 0.5), maxc)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = np.where(arr2d == 255, 0, arr2d)
    ax.imshow(disp, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title); ax.axis("off")
    ax.legend(handles=_legend_handles(class_ids, classes), loc="center left",
              bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def save_confusion_png(cm, labels, path, classes=CLASSES_US,
                       title="Confusion matrix (spatial holdout)"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names_map = _names(classes)
    cm = np.array(cm, dtype=float)
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    names = [names_map.get(l, str(l)) for l in labels]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def save_training_sources_map(train_idx, y, shape, source_name, path,
                              classes=CLASSES_US, extra_sources=None):
    """Scatter of training-sample locations coloured by class, panelled by source."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names_map = _names(classes)
    ny, nx = shape
    sources = {source_name: (train_idx, y)}
    if extra_sources:
        sources.update(extra_sources)
    n = len(sources)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), squeeze=False)
    for ax, (sname, (idx, yy)) in zip(axes[0], sources.items()):
        rows, cols = np.unravel_index(idx, (ny, nx))
        for c in sorted(np.unique(yy[idx])):
            sel = yy[idx] == c
            ax.scatter(cols[sel], rows[sel], s=2, color=np.array(classes[c][1]) / 255,
                       label=names_map.get(c, str(c)))
        ax.set_title(f"Training samples — source: {sname}  (n={len(idx)})")
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7,
                  frameon=False, markerscale=3)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path
