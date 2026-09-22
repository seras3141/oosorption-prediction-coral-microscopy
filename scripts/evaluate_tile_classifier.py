"""CLI for evaluating a trained tile classifier.

Scores a finished run's checkpoint on the validation and test splits, writes
``results.json`` and a per-tile prediction log beside it, and prints a summary with each
accuracy next to the all-negative floor it has to beat.

Usage
-----
    uv run coral-evaluate-tile-classifier data/tile_classifier/<run_id>
    uv run coral-evaluate-tile-classifier <run_dir> --decision-threshold 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.evaluate_tile_classifier import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    EVALUATION_SPLITS,
    evaluate,
)


def _unit_interval(value: str) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1), got {number}")
    return number


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="A training run's output directory.")
    parser.add_argument(
        "--decision-threshold", type=_unit_interval, default=None,
        help="Score an externally chosen operating point. By default the threshold is "
             "chosen to maximise F1 on validation and applied unchanged to test, so "
             "test does not select its own cut.",
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Specimen-level resamples for the confidence intervals. Default: %(default)s.",
    )
    parser.add_argument(
        "--eval-min-oocyte-area-fraction", type=_unit_interval, default=None,
        help="Score against this coverage threshold instead of the one the model was "
             "trained under, so models trained at different thresholds can be compared "
             "on one yardstick. Always writes results_eval_area<bp>.json rather than "
             "overwriting the run's own results.json -- including when the value equals "
             "the trained one, so a sweep scored at a single threshold produces one "
             "file per run. Refused for the centroid rule, which ignores coverage.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = evaluate(
        args.run_dir,
        bootstrap_samples=args.bootstrap_samples,
        decision_threshold=args.decision_threshold,
        num_workers=args.num_workers,
        eval_min_oocyte_area_fraction=args.eval_min_oocyte_area_fraction,
    )

    print(f"\nrun {results['run_id']}")
    if results["is_rescored"]:
        print(f"  RESCORED             trained at "
              f"{results['min_oocyte_area_fraction']}, scored at "
              f"{results['eval_min_oocyte_area_fraction']}")
    print(f"  decision threshold   {results['decision_threshold']:.4f} "
          f"({results['threshold_selection']})")
    print(f"  positive fraction    {results['positive_fraction_per_batch']} requested, "
          f"{results['positive_fraction_observed']} observed")
    print(f"  split manifest       {results['split_manifest_version']} "
          f"{str(results['split_manifest_sha256'])[:12]}")
    for split in EVALUATION_SPLITS:
        metrics = results["metrics"][split]
        interval = metrics["confidence_interval_95"]["auprc"]
        baseline = metrics["trivial_baseline"]
        print(f"\n  {split}: {metrics['n_tiles']:,} tiles, {metrics['n_positive']:,} "
              f"positive, {metrics['n_slides']} slides")
        print(f"    AUPRC     {_fmt(metrics['auprc'])}  95% CI "
              f"[{_fmt(interval[0])}, {_fmt(interval[1])}]  "
              f"(chance {_fmt(baseline['positive_rate'])})")
        print(f"    ROC AUC   {_fmt(metrics['roc_auc'])}")
        print(f"    F1        {_fmt(metrics['f1'])}   precision {_fmt(metrics['precision'])}"
              f"   recall {_fmt(metrics['recall'])}")
        print(f"    bal. acc  {_fmt(metrics['balanced_accuracy'])}   MCC {_fmt(metrics['mcc'])}")
        print(f"    accuracy  {_fmt(metrics['accuracy'])}  vs "
              f"{_fmt(baseline['all_negative_accuracy'])} for answering negative to "
              f"everything")
    print(f"\n  predictions          {results['predictions_path']}")
    return 0


def _fmt(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
