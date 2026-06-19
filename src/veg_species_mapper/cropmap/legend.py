"""Map USDA CDL crop codes onto a reduced legend that mirrors the Australian
crop list of interest (wheat / canola / barley / legumes / pasture), plus the
other classes that dominate the North Dakota test area.

Keeping a small, balanced legend makes the classifier and the maps interpretable.
CDL codes: https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php
"""
from __future__ import annotations

# reduced class id -> (name, RGB colour)
CLASSES: dict[int, tuple[str, tuple[int, int, int]]] = {
    0:  ("Other/Non-crop", (180, 180, 180)),
    1:  ("Wheat",          (214, 176, 96)),
    2:  ("Canola",         (240, 220, 40)),
    3:  ("Barley",         (220, 150, 80)),
    4:  ("Legumes/Pulses", (130, 90, 200)),
    5:  ("Corn",           (255, 140, 0)),
    6:  ("Soybeans",       (40, 160, 60)),
    7:  ("Alfalfa/Hay",    (120, 200, 120)),
    8:  ("Pasture/Grass",  (90, 170, 90)),
    9:  ("Sugarbeets",     (200, 60, 120)),
    10: ("Fallow/Idle",    (200, 190, 160)),
}

NAME_BY_ID = {k: v[0] for k, v in CLASSES.items()}
COLOR_BY_ID = {k: v[1] for k, v in CLASSES.items()}

# CDL native code -> reduced class id
_CDL_TO_CLASS: dict[int, int] = {}
def _m(code, cls):
    _CDL_TO_CLASS[code] = cls

for c in (22, 23, 24):            _m(c, 1)   # durum / spring / winter wheat
_m(31, 2)                                    # canola
_m(21, 3)                                    # barley
for c in (42, 51, 52, 53):       _m(c, 4)   # dry beans, chickpeas, lentils, peas
_m(1, 5)                                     # corn
_m(5, 6)                                     # soybeans
for c in (36, 37):               _m(c, 7)   # alfalfa, other hay/non-alfalfa
for c in (176,):                 _m(c, 8)   # grassland/pasture
_m(41, 9)                                    # sugarbeets
for c in (61, 152):              _m(c, 10)  # fallow/idle, shrubland-ish


def cdl_to_class(arr):
    """Vectorised remap of a CDL code array to reduced class ids (default 0)."""
    import numpy as np
    out = np.zeros_like(arr, dtype="uint8")
    for code, cls in _CDL_TO_CLASS.items():
        out[arr == code] = cls
    return out


def colormap() -> dict[int, tuple[int, int, int, int]]:
    """rasterio-style colour table {value: (R,G,B,A)} for writing categorical GeoTIFFs."""
    cm = {k: (*rgb, 255) for k, (_, rgb) in CLASSES.items()}
    cm[255] = (0, 0, 0, 0)  # nodata transparent
    return cm
