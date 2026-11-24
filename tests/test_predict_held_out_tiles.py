"""Command-line entry point for predicting every held-out tile of CV runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import predict_held_out_tiles
from src.modeling.held_out_predictions import DEFAULT_OUTPUT_ROOT


def _cv_results(root: Path, name: str, cv_run_id: str | None = None) -> str:
    path = root / "cv" / name / "cv_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"cv_run_id": cv_run_id or name}), encoding="utf-8")
    return str(path)


@pytest.fixture
def run_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    def fake_run(cv_results, **kwargs):
        calls.append((cv_results, kwargs))
        return Path(kwargs["output_root"]) / Path(cv_results).parent.name

    monkeypatch.setattr(predict_held_out_tiles, "run_held_out_predictions", fake_run)
    return calls


def test_defaults_score_on_cuda_into_the_default_root_without_overwriting() -> None:
    args = predict_held_out_tiles.parse_args(["--cv-results", "a/cv_results.json"])

    assert (args.device, args.output_root, args.num_workers, args.overwrite) == (
        "cuda", DEFAULT_OUTPUT_ROOT, 0, False)


@pytest.mark.parametrize("arguments", [
    [], ["--cv-results"], ["--cv-results", "a.json", "--num-workers", "-1"],
])
def test_invalid_arguments_are_refused(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        predict_held_out_tiles.parse_args(arguments)


def test_each_cv_run_is_passed_through_with_the_given_options(
    tmp_path: Path, run_calls: list
) -> None:
    a, b = _cv_results(tmp_path, "a"), _cv_results(tmp_path, "b")
    out = str(tmp_path / "out")

    status = predict_held_out_tiles.main(["--cv-results", a, b, "--output-root", out,
                                          "--device", "cpu", "--num-workers", "4", "--overwrite"])

    assert status == 0
    expected = {"output_root": out, "device": "cpu", "num_workers": 4, "overwrite": True}
    assert run_calls == [(a, expected), (b, expected)]


def test_a_refused_run_stops_the_remaining_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_run(cv_results, **kwargs):
        calls.append(cv_results)
        raise ValueError("drifted")

    monkeypatch.setattr(predict_held_out_tiles, "run_held_out_predictions", fake_run)
    a, b = _cv_results(tmp_path, "a"), _cv_results(tmp_path, "b")

    with pytest.raises(ValueError, match="drifted"):
        predict_held_out_tiles.main(["--cv-results", a, b, "--output-root", str(tmp_path / "out")])
    assert calls == [a]


def test_an_existing_output_anywhere_in_the_batch_is_refused_before_any_run(
    tmp_path: Path, run_calls: list
) -> None:
    a, b = _cv_results(tmp_path, "a"), _cv_results(tmp_path, "b")
    (tmp_path / "out" / "b").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="--overwrite to replace them"):
        predict_held_out_tiles.main(["--cv-results", a, b, "--output-root", str(tmp_path / "out")])
    assert run_calls == []


@pytest.mark.parametrize("second", ["same_path", "same_cv_run_id"])
def test_two_runs_writing_to_one_output_are_refused_before_any_run(
    tmp_path: Path, run_calls: list, second: str
) -> None:
    a = _cv_results(tmp_path, "a")
    b = a if second == "same_path" else _cv_results(tmp_path, "b", cv_run_id="a")

    with pytest.raises(ValueError, match="more than one --cv-results writes to"):
        predict_held_out_tiles.main(["--cv-results", a, b, "--output-root", str(tmp_path / "out")])
    assert run_calls == []


def test_overwrite_lets_a_batch_with_existing_outputs_run(
    tmp_path: Path, run_calls: list
) -> None:
    a, b = _cv_results(tmp_path, "a"), _cv_results(tmp_path, "b")
    (tmp_path / "out" / "b").mkdir(parents=True)

    predict_held_out_tiles.main(["--cv-results", a, b, "--output-root", str(tmp_path / "out"),
                                 "--overwrite"])

    assert [path for path, _ in run_calls] == [a, b]
