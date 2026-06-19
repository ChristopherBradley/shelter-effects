"""Australian crop legend, derived from NLUM v7 ABARES commodity probability surfaces.

NOTE: NLUM lumps wheat/barley/oats into 'Winter Cereals' — wheat vs barley cannot be
separated from NLUM alone (would need an independent label source, e.g. the NVT trials).
"""
from __future__ import annotations

# reduced class id -> (name, RGB)
CLASSES: dict[int, tuple[str, tuple[int, int, int]]] = {
    0: ("Other/Non-ag",       (190, 190, 190)),
    1: ("Winter cereals",     (214, 176, 96)),   # wheat/barley/oats
    2: ("Canola/Oilseeds",    (240, 220, 40)),
    3: ("Legumes/Pulses",     (130, 90, 200)),
    4: ("Pasture (modified)", (120, 200, 120)),
    5: ("Grazing native veg", (90, 140, 70)),
    6: ("Hay",                (90, 200, 200)),
    7: ("Summer crops",       (255, 120, 0)),
    8: ("Horticulture",       (220, 70, 140)),
}

# NLUM filename substring -> reduced class id. Order doesn't matter (argmax handles it).
NLUM_KEY_TO_CLASS: list[tuple[str, int]] = [
    ("W_CER", 1),
    ("W_OILSEEDS", 2),
    ("W_LEGUMES", 3),
    ("GRAZ_NOTIMBNP", 4),   # ABARES 2.1.0 grazing modified pastures
    ("GRAZ_NOTIMBSP", 5),   # ABARES 3.2.0 grazing native vegetation
    ("HAY", 6),
    ("S_CER_EX_RICE", 7), ("S_OILSEEDS", 7), ("S_LEGUMES", 7),
    ("RICE", 7), ("COTTON", 7), ("SUGAR_CANE", 7),
    ("APPLES", 8), ("PEARS_OTH_PME", 8), ("ST_FRT_EX_TRP", 8), ("TROP_STONE_FR", 8),
    ("NUTS", 8), ("BERRY_FRT", 8), ("CITRUS", 8), ("GRAPES", 8),
    ("VEGETABLES", 8), ("PLANTATION_FR", 8), ("ONCC", 8),
]

NAME_BY_ID = {k: v[0] for k, v in CLASSES.items()}
