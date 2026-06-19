"""Pl@ntNet identification API wrapper -- the auto-labelling step.

Pl@ntNet offers a free-tier REST API that takes ordinary plant photos and returns
ranked species with confidence scores. Get a key at https://my.plantnet.org/
(account -> settings -> API key). Set PLANTNET_API_KEY in your environment.

Free tier is rate/quota limited, so the POC caches results per image.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests

API = "https://my-api.plantnet.org/v2/identify/all"


@dataclass
class IdResult:
    scientific_name: str
    common_names: list[str]
    score: float  # 0..1 confidence

    def __str__(self) -> str:
        common = f" ({', '.join(self.common_names)})" if self.common_names else ""
        return f"{self.scientific_name}{common}: {self.score:.2f}"


def _key() -> str:
    k = os.environ.get("PLANTNET_API_KEY")
    if not k:
        raise RuntimeError(
            "PLANTNET_API_KEY not set. Get one at https://my.plantnet.org/ and put it in .env"
        )
    return k


def identify(image_path: str | Path, organs: str = "auto") -> list[IdResult]:
    """Identify the plant in image_path. Returns ranked results (best first).

    organs: 'auto' lets Pl@ntNet decide; or 'leaf'/'flower'/'fruit'/'bark'.
            'bark' or 'leaf' tends to help for trees like Pinus radiata.
    """
    image_path = Path(image_path)
    with open(image_path, "rb") as fh:
        files = [("images", (image_path.name, fh, "image/jpeg"))]
        data = {"organs": [organs]}
        resp = requests.post(
            API, files=files, data=data, params={"api-key": _key()}, timeout=60
        )
    if resp.status_code == 404:
        # Pl@ntNet returns 404 when it can't identify anything in the image.
        return []
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", []):
        sp = r.get("species", {})
        out.append(
            IdResult(
                scientific_name=sp.get("scientificNameWithoutAuthor", "?"),
                common_names=sp.get("commonNames", []),
                score=float(r.get("score", 0.0)),
            )
        )
    return out


def best_match_for(results: list[IdResult], target_genus_or_name: str) -> IdResult | None:
    """Return the highest-scoring result whose name contains the target string
    (case-insensitive), e.g. 'Pinus radiata' or 'Eragrostis' for lovegrass."""
    t = target_genus_or_name.lower()
    hits = [r for r in results if t in r.scientific_name.lower()]
    return max(hits, key=lambda r: r.score) if hits else None
