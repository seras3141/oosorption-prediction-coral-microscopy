"""Deterministic slide-level train/val/test split manifest generation.

Implements the algorithm specified in ``docs_local/plan_addendum_m6_split.md``
so every downstream training step (M7 YOLOv8-seg, the tile-classification
baseline, M10 stage classifier) reads splits from one file instead of
re-deriving its own ad hoc split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.data_preparation.annotations import extract_stage
from src.data_preparation.remap_annotations import FEATURE_COLLECTION

LOG = logging.getLogger(__name__)

FORCE_TRAIN_IF_STAGE_PRESENT: tuple[int, ...] = (1,)
DEFAULT_SEED: int = 42
DEFAULT_QUOTAS: dict[tuple[str, bool], tuple[int, int]] = {
    ("CHN", False): (2, 1),
    ("CHN", True): (0, 0),
    ("LHP", False): (1, 1),
    ("LHP", True): (1, 1),
}
MANIFEST_VERSION: str = "v1"


def compute_stage_distribution(
    geojson_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Compute per-slide location and stage-annotation counts.

    Parameters
    ----------
    geojson_dir : str or Path
        Directory containing one ``{stem}.geojson`` file per slide
        (``data/dataset_28_04/``).

    Returns
    -------
    dict
        ``{stem: {"location": str, "stage_counts": dict[int, int]}}``.
        ``location`` is parsed as the first underscore-delimited token of
        the stem. Features with no extractable stage (``extract_stage``
        returns ``None``) are skipped and logged at DEBUG.

    Examples
    --------
    >>> import json, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     fc = {"type": "FeatureCollection",
    ...           "features": [{"properties": {"name": "Oosorption Stage 0"}}]}
    ...     _ = (Path(tmp) / "CHN_A_1_1-2.geojson").write_text(json.dumps(fc))
    ...     dist = compute_stage_distribution(tmp)
    >>> dist["CHN_A_1_1-2"]["location"], dist["CHN_A_1_1-2"]["stage_counts"]
    ('CHN', {0: 1})
    """
    geojson_dir = Path(geojson_dir)
    if not geojson_dir.is_dir():
        raise FileNotFoundError(f"geojson_dir does not exist: {geojson_dir}")

    distribution: dict[str, dict[str, Any]] = {}
    for geojson_path in sorted(geojson_dir.glob("*.geojson")):
        stem = geojson_path.stem
        location = stem.split("_")[0]
        with geojson_path.open("r", encoding="utf-8") as fp:
            geojson = json.load(fp)
        if geojson.get("type") != FEATURE_COLLECTION:
            raise ValueError(f"{geojson_path} is not a GeoJSON FeatureCollection")

        stage_counts: dict[int, int] = {}
        for feature in geojson.get("features", []):
            props = feature.get("properties") or {}
            stage = extract_stage(props)
            if stage is None:
                LOG.debug(
                    "Skipping unlabelled feature %s in %s",
                    feature.get("id"),
                    geojson_path.name,
                )
                continue
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        distribution[stem] = {"location": location, "stage_counts": stage_counts}
    return distribution
