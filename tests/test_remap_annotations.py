"""Synthetic tests for annotation remapping."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_preparation.remap_annotations import (
    _compute_centroid,
    _translate_coords,
    remap_annotations,
)


def _manifest() -> dict:
    return {
        "source_ndpi": "TEST_slide.ndpi",
        "level0_dimensions": [120000, 40000],
        "mpp_x": 0.23,
        "mpp_y": 0.23,
        "cuts": [
            {
                "index": 0,
                "name": "TEST_slide_cut000",
                "level0_bbox": {"x0": 1000, "y0": 500, "x1": 50000, "y1": 39500},
                "level0_size": [49000, 39000],
            },
            {
                "index": 1,
                "name": "TEST_slide_cut001",
                "level0_bbox": {"x0": 60000, "y0": 500, "x1": 119000, "y1": 39500},
                "level0_size": [59000, 39000],
            },
        ],
    }


def _feature(feature_id: str, geometry: dict, stage: int = 0) -> dict:
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": geometry,
        "properties": {
            "objectType": "annotation",
            "name": f"Oosorption Stage {stage}",
            "metadata": {"ANNOTATION_DESCRIPTION": f"Oosorption Stage {stage}"},
        },
    }


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


class RemapAnnotationsTest(unittest.TestCase):
    def test_translate_coords(self) -> None:
        coords = [[38541.76, 9591.36], [38652.63, 9778.99], [38700.0, 9600.0]]
        expected = [[34341.76, 7791.36], [34452.63, 7978.99], [34500.0, 7800.0]]
        self.assertEqual(_translate_coords(coords, 4200, 1800), expected)

    def test_compute_centroid(self) -> None:
        self.assertEqual(
            _compute_centroid([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
            (5.0, 5.0),
        )

    def test_happy_path_unassigned_vertex_polygon_multipolygon_and_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "TEST_slide_cuts.json"
            geojson_path = root / "TEST_slide.geojson"
            output_dir = root / "out"
            _write_json(manifest_path, _manifest())

            features = [
                _feature(
                    "feat_a",
                    {
                        "type": "LineString",
                        "coordinates": [
                            [20000.0, 10000.0],
                            [20200.0, 10000.0],
                            [20200.0, 10200.0],
                            [20000.0, 10200.0],
                        ],
                    },
                    stage=0,
                ),
                _feature(
                    "feat_b",
                    {
                        "type": "LineString",
                        "coordinates": [
                            [90000.0, 20000.0],
                            [90200.0, 20000.0],
                            [90200.0, 20200.0],
                            [90000.0, 20200.0],
                        ],
                    },
                    stage=3,
                ),
                _feature(
                    "feat_c",
                    {
                        "type": "LineString",
                        "coordinates": [[200000.0, 50000.0], [200100.0, 50000.0]],
                    },
                    stage=1,
                ),
                _feature(
                    "feat_d",
                    {
                        "type": "LineString",
                        "coordinates": [
                            [800.0, 9900.0],
                            [1100.0, 9900.0],
                            [1100.0, 10100.0],
                            [800.0, 10100.0],
                        ],
                    },
                    stage=2,
                ),
                _feature(
                    "feat_e",
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [25000.0, 15000.0],
                                [25300.0, 15000.0],
                                [25300.0, 15300.0],
                                [25000.0, 15300.0],
                                [25000.0, 15000.0],
                            ]
                        ],
                    },
                    stage=4,
                ),
                _feature(
                    "feat_f",
                    {
                        "type": "MultiPolygon",
                        "coordinates": [
                            [
                                [
                                    [10000.0, 5000.0],
                                    [10400.0, 5000.0],
                                    [10400.0, 5400.0],
                                    [10000.0, 5400.0],
                                    [10000.0, 5000.0],
                                ]
                            ],
                            [
                                [
                                    [10500.0, 5000.0],
                                    [10600.0, 5000.0],
                                    [10600.0, 5100.0],
                                    [10500.0, 5100.0],
                                    [10500.0, 5000.0],
                                ]
                            ],
                        ],
                    },
                    stage=0,
                ),
            ]
            _write_json(
                geojson_path,
                {"type": "FeatureCollection", "features": features},
            )

            with self.assertLogs(
                "src.data_preparation.remap_annotations",
                level="WARNING",
            ) as logs:
                report = remap_annotations(geojson_path, manifest_path, output_dir)
            self.assertIn("feat_d", "\n".join(logs.output))

            self.assertEqual(report["n_assigned"], 5)
            self.assertEqual(report["n_unassigned"], 1)
            self.assertEqual(report["unassigned_ids"], ["feat_c"])

            cut0 = _read_json(output_dir / "TEST_slide_cut000_annotations.geojson")
            cut1 = _read_json(output_dir / "TEST_slide_cut001_annotations.geojson")
            self.assertEqual(len(cut0["features"]), 4)
            self.assertEqual(len(cut1["features"]), 1)

            by_id = {feature["id"]: feature for feature in cut0["features"]}
            self.assertEqual(by_id["feat_a"]["geometry"]["coordinates"][0], [19000.0, 9500.0])
            self.assertEqual(
                by_id["feat_a"]["properties"]["assignment_method"],
                "centroid_in_bbox",
            )
            self.assertEqual(
                by_id["feat_d"]["properties"]["assignment_method"],
                "vertex_in_bbox",
            )
            self.assertEqual(by_id["feat_d"]["geometry"]["coordinates"][0], [-200.0, 9400.0])
            self.assertEqual(by_id["feat_e"]["geometry"]["type"], "Polygon")
            self.assertEqual(by_id["feat_e"]["geometry"]["coordinates"][0][0], [24000.0, 14500.0])
            self.assertEqual(by_id["feat_f"]["geometry"]["type"], "Polygon")
            self.assertEqual(by_id["feat_f"]["geometry"]["coordinates"][0][0], [9000.0, 4500.0])

            cut1_feature = cut1["features"][0]
            self.assertEqual(cut1_feature["id"], "feat_b")
            self.assertEqual(cut1_feature["geometry"]["coordinates"][0], [30000.0, 19500.0])

            report_path = output_dir / "TEST_slide_remap_report.json"
            before_mtime = report_path.stat().st_mtime_ns
            skipped_report = remap_annotations(
                geojson_path,
                manifest_path,
                output_dir,
                skip_if_exists=True,
            )
            self.assertEqual(skipped_report, report)
            self.assertEqual(report_path.stat().st_mtime_ns, before_mtime)

    def test_empty_geojson(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "TEST_slide_cuts.json"
            geojson_path = root / "TEST_slide.geojson"
            output_dir = root / "out"
            _write_json(manifest_path, _manifest())
            _write_json(geojson_path, {"type": "FeatureCollection", "features": []})

            report = remap_annotations(geojson_path, manifest_path, output_dir)
            self.assertEqual(report["n_annotations_total"], 0)
            self.assertEqual(report["n_assigned"], 0)
            self.assertEqual(report["n_unassigned"], 0)
            self.assertEqual(
                _read_json(output_dir / "TEST_slide_cut000_annotations.geojson")["features"],
                [],
            )


if __name__ == "__main__":
    unittest.main()
