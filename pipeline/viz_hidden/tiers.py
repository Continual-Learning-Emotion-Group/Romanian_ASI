"""Emotion tiers for the hidden-state circumplex experiment.

`emotion_category` in the benchmark is the seed word's own label (a WN-Affect
leaf like "happiness"/"negative-fear", or a bare Plutchik name the seed authors
used directly like "trust"/"fear"). We roll every label up to one of Plutchik's
8 basic emotions using the WN-Affect hierarchy (pipeline/seed/wn-affect-1.1/
a-hierarchy.xml), plus a few explicit overrides for labels absent from the tree.

Tiers (progressive experiment: circle -> fill -> manifold):
  Tier 1  Plutchik-8 basic emotions   (the circumplex anchors)
  Tier 2  core emotions (enriched)    (still "true emotions")
  Tier 3  full ASI                    (everything, incl. complex states)

This module is import-safe with no heavy deps (stdlib only).
"""
from __future__ import annotations

import re
from pathlib import Path

_HIER = Path(__file__).resolve().parents[1] / "seed" / "wn-affect-1.1" / "a-hierarchy.xml"

# WN-Affect subtree roots -> Plutchik emotion. Any label at/under one of these
# nodes rolls up to the given Plutchik-8 basic emotion.
PLUTCHIK_ANCHORS = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "negative-fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "astonishment": "surprise",       # surprise<->astonishment mini-cycle in WN-Affect
    "anticipation": "anticipation",
    "positive-expectation": "anticipation",
}

# Bare labels the seed used directly that are NOT WN-Affect nodes (or not anchors).
# Checked before the tree walk.
EXTRA_LABEL_MAP = {
    "fear": "fear",          # seed's bare label; tree only has negative-fear
    "trust": "trust",        # Plutchik trust; not present in WN-Affect at all
    "anticipation": "anticipation",
}

PLUTCHIK8 = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

# Valence sign per Plutchik emotion for labels resolved via EXTRA_LABEL_MAP
# (tree-resolved labels get their sign from the positive/negative branch).
_EXTRA_VALENCE = {"fear": -1, "trust": +1, "anticipation": +1}


def build_parent_map() -> dict[str, str]:
    txt = _HIER.read_text()
    parent: dict[str, str] = {}
    for m in re.finditer(r'<categ name="([^"]+)"(?:\s+isa="([^"]+)")?/>', txt):
        child, par = m.group(1), m.group(2)
        if par:
            parent[child] = par
    return parent


class PlutchikMapper:
    def __init__(self) -> None:
        self.parent = build_parent_map()

    def _ancestors(self, label: str):
        seen, cur = set(), label
        while cur is not None and cur not in seen:
            seen.add(cur)
            yield cur
            cur = self.parent.get(cur)

    def plutchik(self, label: str) -> str | None:
        """Roll a single emotion_category label up to a Plutchik-8 emotion."""
        if label in EXTRA_LABEL_MAP:
            return EXTRA_LABEL_MAP[label]
        for anc in self._ancestors(label):
            if anc in PLUTCHIK_ANCHORS:
                return PLUTCHIK_ANCHORS[anc]
        return None

    def valence(self, label: str) -> int:
        """+1 positive, -1 negative, 0 ambiguous/neutral (from WN-Affect branch)."""
        if label in _EXTRA_VALENCE:
            return _EXTRA_VALENCE[label]
        for anc in self._ancestors(label):
            if anc == "positive-emotion":
                return +1
            if anc == "negative-emotion":
                return -1
            if anc in ("ambiguous-emotion", "neutral-emotion"):
                return 0
        return 0

    def row_plutchik(self, emotion_category: list[str]) -> str | None:
        """Assign a row to a Plutchik emotion only if its labels map unambiguously
        to exactly one basic emotion (else return None -> excluded from Tier 1)."""
        mapped = {self.plutchik(e) for e in emotion_category}
        mapped.discard(None)
        return next(iter(mapped)) if len(mapped) == 1 else None

    def row_valence(self, emotion_category: list[str]) -> int:
        vals = [self.valence(e) for e in emotion_category]
        return round(sum(vals) / len(vals)) if vals else 0
