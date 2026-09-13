"""CLI for training the tile-level oocyte classifier.

Usage
-----
    uv run coral-train-tile-classifier --tile-size 512
    uv run coral-train-tile-classifier --tile-size 1024 --hard-negative-mining
    uv run coral-train-tile-classifier --tile-size 512 --max-train-slides 2 \
        --epochs 2 --max-batches-per-epoch 5      # smoke test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.train_tile_classifier import ARCHITECTURES, TrainingConfig, train
from src.modeling.encoders import DEFAULT_ENCODER, SUPPORTED_ENCODERS
from src.modeling.tile_index import LABEL_RULES
from src.modeling.tile_labels import DEFAULT_MIN_OOCYTE_AREA_FRACTION


def _positive_int(value: str) -> int:
    """An argparse type for counts that must be at least 1.

    Without it, a zero or negative cap fails deep inside the run -- after the whole tile
    index has been parsed -- with a message that blames something else.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {number}")
    return number


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument(
        "--encoder-name", default=None, choices=SUPPORTED_ENCODERS,
        help="Frozen-encoder arm only. Default: %s." % DEFAULT_ENCODER,
    )
    parser.add_argument(
        "--tile-size", type=int, choices=(512, 1024), default=512,
        help="One size per run: the dataset refuses a mixed index.",
    )
    parser.add_argument("--label-rule", choices=LABEL_RULES, default="area")
    parser.add_argument(
        "--min-oocyte-area-fraction", type=float,
        default=DEFAULT_MIN_OOCYTE_AREA_FRACTION,
        help="Coverage at or above which a tile is positive. Default: %(default)s.",
    )
    parser.add_argument("--epochs", type=_positive_int, default=30)
    parser.add_argument("--patience", type=_positive_int, default=5, help="On validation AP.")
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Default depends on the arm: 1e-4 fine-tuning ResNet-18, 1e-3 for the "
             "frozen encoder's probe head.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-fraction", type=float, default=0.25)
    parser.add_argument(
        "--hard-negative-mining", action="store_true",
        help="Refused under the centroid label rule, which it would train on the noise of.",
    )
    parser.add_argument("--warmup-epochs", type=_positive_int, default=3)
    parser.add_argument("--hard-negative-pool-fraction", type=float, default=0.25)
    parser.add_argument("--hard-negative-share", type=float, default=0.5)
    parser.add_argument(
        "--head-hidden-dim", type=_positive_int, default=None,
        help="Frozen-encoder head width. Recorded with the run, because a head-only "
             "checkpoint stores exactly these layers.",
    )
    parser.add_argument("--head-dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-train-slides", type=_positive_int, default=None,
        help="Smoke tests only: cap the number of train slides indexed.",
    )
    parser.add_argument(
        "--max-batches-per-epoch", type=_positive_int, default=None,
        help="Smoke tests only: shorten the epoch. Default is one pass over the negatives.",
    )
    parser.add_argument(
        "--max-val-tiles", type=_positive_int, default=None,
        help="Smoke tests only: cap validation tiles, class-stratified. Default is the "
             "whole val split, which is what a real run must evaluate on.",
    )
    parser.add_argument("--output-dir", default="data/tile_classifier")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.encoder_name and args.architecture != "frozen_encoder":
        # It would otherwise change the run id and the recorded config while the model
        # ignored it, so two "different" runs would be the same experiment.
        raise SystemExit(
            f"--encoder-name is only meaningful with --architecture frozen_encoder; "
            f"got --architecture {args.architecture}"
        )
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = TrainingConfig(
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        tile_size=args.tile_size,
        label_rule=args.label_rule,
        min_oocyte_area_fraction=args.min_oocyte_area_fraction,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        positive_fraction=args.positive_fraction,
        hard_negative_mining=args.hard_negative_mining,
        warmup_epochs=args.warmup_epochs,
        hard_negative_pool_fraction=args.hard_negative_pool_fraction,
        hard_negative_share=args.hard_negative_share,
        seed=args.seed,
        num_workers=args.num_workers,
        **(
            {"head_hidden_dim": args.head_hidden_dim}
            if args.head_hidden_dim is not None
            else {}
        ),
        **({"head_dropout": args.head_dropout} if args.head_dropout is not None else {}),
        max_train_slides=args.max_train_slides,
        max_batches_per_epoch=args.max_batches_per_epoch,
        max_val_tiles=args.max_val_tiles,
        output_dir=args.output_dir,
    )
    result = train(config)

    print(f"\nrun {result.run_id}")
    for name, sizes in result.split_sizes.items():
        print(
            f"  {name:>5}: {sizes['n_tiles']:,} tiles, {sizes['n_positive']:,} positive, "
            f"{sizes['n_slides']} slide(s)"
        )
    print(f"  epochs run           {result.epochs_run}")
    print(f"  best epoch           {result.best_epoch}")
    print(f"  best val AP          {result.best_val_average_precision:.4f}")
    print(f"  early stopped        {result.early_stopped}")
    print(f"  checkpoint           {result.checkpoint_path}")
    print(f"  batch composition    {result.batch_composition_log_path}")
    if result.encoder_licence:
        print(f"  encoder licence      {result.encoder_licence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
