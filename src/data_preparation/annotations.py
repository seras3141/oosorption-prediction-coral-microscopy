"""Shared GeoJSON annotation-property parsing helpers.

``extract_stage`` is the canonical stage-number parser for QuPath-exported
annotation *properties* blocks (as opposed to full GeoJSON *features*). It was
promoted here from :mod:`scripts.geojson_to_yolo` because
:mod:`src.data_preparation.generate_split_manifest` needs the same logic.

Note on naming: two other, unrelated helpers named ``_extract_stage`` exist in
:mod:`src.data_preparation.generate_tiles` and
:mod:`src.data_preparation.sample_review_tiles`. Both are private, take a full
GeoJSON *feature* (not just ``properties``), and return a display string
(``"Stage 3"`` or ``"Unknown"``) from ``classification.name`` only - a
different contract from this module's ``extract_stage`` (properties-only
input, ``int | None`` output, three-field fallback). That is pre-existing
duplication elsewhere in the codebase and is out of scope here; the two are
not the same helper and should not be merged.
"""

from __future__ import annotations

import re

STAGE_PATTERN = re.compile(r"stage\s*(\d+)", re.IGNORECASE)
FEATURE_COLLECTION = "FeatureCollection"


def extract_stage(props: dict) -> int | None:
    """Pull a stage number (0-4) out of a QuPath-style properties block.

    Checks, in priority order, ``metadata.ANNOTATION_DESCRIPTION``, then
    ``name``, then ``classification.name`` - the first field containing a
    ``stage <N>`` pattern wins.

    Parameters
    ----------
    props : dict
        A GeoJSON feature's ``properties`` block.

    Returns
    -------
    int or None
        The parsed stage number, or ``None`` if no field matched.

    Examples
    --------
    >>> extract_stage({"name": "Oosorption Stage 0"})
    0
    >>> extract_stage({"metadata": {"ANNOTATION_DESCRIPTION": "Oosorption Stage 0"},
    ...                 "name": "Oosorption Stage 3"})
    0
    >>> extract_stage({"classification": {"name": "Stage 3"}})
    3
    >>> extract_stage({"name": "no stage here"}) is None
    True
    """
    candidates: list[str] = []
    meta = props.get("metadata")
    if isinstance(meta, dict):
        desc = meta.get("ANNOTATION_DESCRIPTION")
        if desc:
            candidates.append(str(desc))
    name = props.get("name")
    if name:
        candidates.append(str(name))
    cls = props.get("classification")
    if isinstance(cls, dict) and cls.get("name"):
        candidates.append(str(cls["name"]))

    for text in candidates:
        m = STAGE_PATTERN.search(text)
        if m:
            return int(m.group(1))
    return None
