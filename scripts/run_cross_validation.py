#!/usr/bin/env python3
"""Run specimen-grouped cross-validation of the tile classifier."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_tile_classifier import build_parser, config_from_args
from src.modeling.cross_validation import (
    DEFAULT_CV_OUTPUT_DIR,
    DEFAULT_FOLD_MANIFEST,
    aggregate,
    fold_config,
    load_fold_manifest,
    run_fold,
    write_cv_results,
)

DEFAULT_FOLD_RUNS_DIR = f"{DEFAULT_CV_OUTPUT_DIR}/runs"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    build_parser(parser)
    # Keep fold runs apart from M7a runs.
    parser.set_defaults(output_dir=DEFAULT_FOLD_RUNS_DIR)
    group = parser.add_argument_group("cross-validation")
    group.add_argument("--fold-manifest", default=DEFAULT_FOLD_MANIFEST)
    group.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="Folds to train and evaluate. Default: every fold in the manifest.",
    )
    group.add_argument(
        "--skip-training", action="store_true",
        help="Aggregate from finished fold runs instead of training them.",
    )
    group.add_argument(
        "--no-aggregate", action="store_true",
        help="Train and evaluate the folds only; aggregate later with --skip-training.",
    )
    group.add_argument(
        "--allow-partial", action="store_true",
        help="Aggregate fewer folds than the manifest defines. Smoke runs only.",
    )
    group.add_argument("--bootstrap-samples", type=int, default=2000)
    group.add_argument("--cv-output-dir", default=DEFAULT_CV_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.skip_training and args.no_aggregate:
        parser.error("--skip-training with --no-aggregate would do nothing")
    return args


def check_fold_selection(
    folds: list[int], manifest_folds: list[int], *, aggregate: bool, allow_partial: bool
) -> None:
    """Refuse, before any training, a fold subset the aggregate would reject."""
    if len(folds) != len(set(folds)):
        raise SystemExit("--folds must not contain duplicates")
    unknown = sorted(set(folds) - set(manifest_folds))
    if unknown:
        raise SystemExit(f"fold(s) {unknown} are not in the fold manifest")
    if aggregate and not allow_partial and set(folds) != set(manifest_folds):
        raise SystemExit(
            f"--folds {' '.join(map(str, folds))} is a subset of the manifest's folds; "
            "pass --no-aggregate for a per-fold job, or --allow-partial for a smoke run"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base = config_from_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    fold_manifest = load_fold_manifest(args.fold_manifest)
    manifest_folds = [f["fold"] for f in fold_manifest["folds"]]
    folds = args.folds or manifest_folds
    check_fold_selection(folds, manifest_folds, aggregate=not args.no_aggregate,
                         allow_partial=args.allow_partial)

    fold_results = {}
    for fold in folds:
        if args.skip_training:
            config = fold_config(base, fold_manifest, fold)
            path = REPO_ROOT / config.output_dir / config.run_id() / "results.json"
            if not path.exists():
                raise SystemExit(f"fold {fold}: no finished evaluation at {path}")
            fold_results[fold] = json.loads(path.read_text(encoding="utf-8"))
        else:
            fold_results[fold] = run_fold(
                base, fold_manifest, fold, bootstrap_samples=args.bootstrap_samples
            )
        test = fold_results[fold]["metrics"]["test"]
        print(f"fold {fold}: {fold_results[fold]['run_id']}  test AUPRC {test['auprc']}  "
              f"positive rate {test['trivial_baseline']['positive_rate']}")

    if args.no_aggregate:
        return 0

    results = aggregate(base, fold_manifest, fold_results, allow_partial=args.allow_partial)
    path = write_cv_results(results, args.cv_output_dir)
    lift = results["summary"]["auprc_lift"]
    print(f"\ncv run {results['cv_run_id']}"
          f"{'  (PARTIAL)' if results['is_partial'] else ''}"
          f"{'  (SUBSET RUN)' if results['is_subset_run'] else ''}")
    for fold in results["per_fold"]:
        print(f"  fold {fold['fold']}: lift {fold['auprc_lift']}  AUPRC {fold['auprc']}  "
              f"rate {fold['positive_rate']}")
    print(f"  lift mean {lift['mean']}  range [{lift['min']}, {lift['max']}]  "
          f"({results['summary']['note']})")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
