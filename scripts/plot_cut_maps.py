#!/usr/bin/env python3
"""Draw oocyte overlays, tile-label maps and per-model error maps for every cut."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.cut_maps import (
    MODELS,
    find_default_cv_results,
    held_out_predictions,
    list_cuts,
    load_cut_view,
    plot_oocyte_overlay,
    plot_prediction_errors,
    plot_tile_labels,
    save,
)

FIGURES = ("overlay", "labels", "errors")
DEFAULT_OUTPUT = Path("data/figures/cut_maps")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cuts", nargs="+", default=None,
                        help="Cut names, e.g. LHP_W_10_28-30_cut000. Default: every cut.")
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[512, 1024],
                        choices=(128, 256, 512, 1024))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--figures", nargs="+", default=list(FIGURES), choices=FIGURES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    out = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    cuts = list_cuts()
    if args.cuts:
        wanted = set(args.cuts)
        cuts = [c for c in cuts if c[1] in wanted]
        missing = wanted - {c[1] for c in cuts}
        if missing:
            raise SystemExit(f"unknown cut(s): {sorted(missing)}")

    predictions = {}
    if "errors" in args.figures:
        for model in args.models:
            for size in args.tile_sizes:
                path = find_default_cv_results(model, size, args.seed)
                logging.info("%s %d px: %s", model, size, path.parent.name)
                predictions[(model, size)] = held_out_predictions(path)

    for stem, cut_name in cuts:
        view = load_cut_view(stem, cut_name, tile_sizes=tuple(args.tile_sizes))
        if "overlay" in args.figures:
            save(plot_oocyte_overlay(view), out / "overlays" / f"{cut_name}.jpg")
        for size in args.tile_sizes:
            if "labels" in args.figures:
                save(plot_tile_labels(view, size), out / f"labels_{size:04d}" / f"{cut_name}.jpg")
            if "errors" in args.figures:
                for model in args.models:
                    save(plot_prediction_errors(view, size, predictions[(model, size)],
                                                MODELS[model]["label"]),
                         out / f"errors_{model}_{size:04d}" / f"{cut_name}.jpg")
        logging.info("Drew %s", cut_name)
    print(f"{len(cuts)} cut(s) written under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
