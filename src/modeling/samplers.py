"""Batch sampling for the tile classifier: forced positive ratio and hard negatives.

Oocyte-positive tiles are the minority at both trained scales -- roughly 1 in 6.5 at
512 px and 1 in 5 at 1024 px under the default coverage label -- so batches drawn
uniformly would carry few positives and the loss would be dominated by background. This
module forces a fixed positive share in every training batch, and optionally biases the
negative draw toward negatives the model currently scores highest.

A batch sampler is just an iterable of index lists, which ``DataLoader`` accepts directly
as ``batch_sampler``, so nothing here imports ``torch`` -- see
:mod:`src.modeling.tile_index` for the same layering rule.

Only training batches are resampled. Validation and test run on the natural distribution,
so reported metrics describe deployment conditions rather than the training-time balance.

Every batch is assembled through a single shared ``seen`` set, and checked before being
yielded, so no dataset position can appear twice in one batch. Three separate draw
sites -- positives, hard negatives, uniform negatives -- each previously found a way to
duplicate a tile, which loads it twice and double-weights it in one gradient step, so the
guarantee is enforced structurally rather than at each site.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

DEFAULT_POSITIVE_FRACTION: float = 0.25
DEFAULT_HARD_NEGATIVE_POOL_FRACTION: float = 0.25
DEFAULT_HARD_NEGATIVE_SHARE: float = 0.5
DEFAULT_SEED: int = 42

LABEL_RULE_AREA = "area"
LABEL_RULE_CENTROID = "centroid"
LABEL_RULES = (LABEL_RULE_AREA, LABEL_RULE_CENTROID)


def check_mining_allowed(label_rule: str, hard_negative_mining: bool) -> None:
    """Refuse hard-negative mining under the centroid label rule.

    Mining ranks negatives by predicted positive probability and up-weights the highest
    scorers. Under the centroid rule the highest-scoring negatives are largely the tiles
    that rule mislabels -- interiors of oocytes too large to fit one tile, of which the
    corpus has 2,491 at 512 px that are more than half covered. Mining would therefore
    train hardest on the label noise, so the combination is rejected rather than left to
    produce a quietly bad model.

    :meth:`ForcedRatioBatchSampler.set_hard_negatives` calls this itself, so the refusal
    holds even if a caller forgets to.

    Raises
    ------
    ValueError
        If ``label_rule`` is not a known rule, or if ``hard_negative_mining`` is set
        while it is ``"centroid"``. An unrecognised value is rejected rather than
        treated as "not centroid": a typo such as ``"Centroid"`` would otherwise pass
        this check and quietly enable the combination it exists to forbid.

    Examples
    --------
    >>> check_mining_allowed("area", True) is None
    True
    """
    if label_rule not in LABEL_RULES:
        raise ValueError(
            f"label_rule must be one of {LABEL_RULES}, got {label_rule!r}; refusing to "
            "assume an unknown rule is safe for hard-negative mining"
        )
    if hard_negative_mining and label_rule == LABEL_RULE_CENTROID:
        raise ValueError(
            "hard-negative mining is not allowed with the centroid label rule: it would "
            "preferentially sample the oocyte-interior tiles that rule mislabels as "
            "negative and train hardest on that error. Use the area label rule, or "
            "disable mining."
        )


class ForcedRatioBatchSampler:
    """Yield batches of dataset positions holding a fixed share of positives.

    Negatives come from a shuffled pass over the negative pool, so an epoch is roughly
    one pass over the majority class; when mining is active the epoch lengthens so the
    shortened uniform stream still covers comparable ground. Positives are cycled through
    a reshuffled permutation, which oversamples them evenly rather than leaving coverage
    to chance the way sampling with replacement would. No dataset position appears twice
    in one batch.

    Parameters
    ----------
    positive_positions, negative_positions : sequence of int
        Dataset positions of each class, from
        :class:`src.modeling.tile_classification_dataset.TileClassificationDataset`.
    batch_size : int
        Total positions per batch. Every batch is full-size; the trailing partial batch
        an epoch would otherwise end on is dropped, so the forced ratio holds exactly for
        every batch the model sees.
    positive_fraction : float, optional
        Share of each batch that is positive. The default 0.25 is a 1:3
        positive:negative ratio, inside the 1:1-1:5 band the approach survey recommends.
    batches_per_epoch : int, optional
        Defaults to roughly one pass over the negatives, adjusted when mining is on.
    seed : int, optional
        Seeds this sampler's own RNG; no global random state is touched.
    composition_log_path : str or Path, optional
        JSONL file appended with one record per batch. Verifying that the ratio actually
        used matches the ratio configured needs the realised composition, not the flag.
    label_rule : str, optional
        The label rule the index was built under, one of :data:`LABEL_RULES`. Held so
        that :meth:`set_hard_negatives` can refuse mining under the centroid rule.

    Raises
    ------
    ValueError
        If either pool is empty, holds too few *distinct* positions to fill its
        per-batch quota, or overlaps the other; if ``batch_size`` is not positive; if
        ``positive_fraction`` would round to zero positives or to a whole batch of them;
        or if ``label_rule`` is unknown. The distinctness and overlap checks matter
        because every draw site fills a batch with distinct positions, so a pool padded
        with repeats -- or pools sharing a position -- makes the batch unsatisfiable and
        would otherwise spin the draw loops forever.

    Examples
    --------
    >>> sampler = ForcedRatioBatchSampler([0, 1], [2, 3, 4, 5, 6, 7], batch_size=4)
    >>> len(next(iter(sampler)))
    4
    """

    def __init__(
        self,
        positive_positions: Sequence[int],
        negative_positions: Sequence[int],
        batch_size: int,
        positive_fraction: float = DEFAULT_POSITIVE_FRACTION,
        batches_per_epoch: int | None = None,
        seed: int = DEFAULT_SEED,
        composition_log_path: str | Path | None = None,
        label_rule: str = LABEL_RULE_AREA,
    ) -> None:
        if not positive_positions:
            raise ValueError("positive_positions is empty; nothing to oversample")
        if not negative_positions:
            raise ValueError("negative_positions is empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must be in the open interval (0, 1)")
        if label_rule not in LABEL_RULES:
            raise ValueError(f"label_rule must be one of {LABEL_RULES}, got {label_rule!r}")

        overlap = set(positive_positions) & set(negative_positions)
        if overlap:
            raise ValueError(
                f"{len(overlap)} position(s) appear in both pools (e.g. "
                f"{sorted(overlap)[:3]}); a tile cannot be both classes, and the "
                "batches would be unsatisfiable"
            )

        self._positives = list(positive_positions)
        self._negatives = list(negative_positions)
        self._batch_size = int(batch_size)
        self._positive_fraction = float(positive_fraction)
        self._label_rule = label_rule

        self._n_positive = round(batch_size * positive_fraction)
        if self._n_positive == 0 or self._n_positive >= batch_size:
            raise ValueError(
                f"positive_fraction {positive_fraction} with batch_size {batch_size} "
                f"gives {self._n_positive} positives per batch; choose a fraction that "
                "leaves at least one tile of each class"
            )
        self._n_negative = batch_size - self._n_positive

        # Without these, a pool smaller than its per-batch quota yields an under-full
        # batch (or one repeating the same tile) while the reported fraction still
        # claims the configured balance -- which bites hardest on the small subset and
        # smoke runs where the balance is taken on trust.
        # Count distinct positions, not list entries: every draw site fills a batch with
        # distinct positions, so a pool padded with repeats is unsatisfiable and the
        # draw loops would spin forever -- a wedged job with no output, rather than an
        # error, which is the worst way for this to fail on a batch allocation.
        n_distinct_positive = len(set(self._positives))
        n_distinct_negative = len(set(self._negatives))
        if n_distinct_positive < self._n_positive:
            raise ValueError(
                f"{n_distinct_positive} distinct positive(s) cannot fill "
                f"{self._n_positive} positive slots per batch without repeating a tile "
                "within the batch; lower batch_size or positive_fraction"
            )
        if n_distinct_negative < self._n_negative:
            raise ValueError(
                f"{n_distinct_negative} distinct negative(s) cannot fill "
                f"{self._n_negative} negative slots per batch; lower batch_size or "
                "raise positive_fraction"
            )

        if batches_per_epoch is not None and int(batches_per_epoch) <= 0:
            raise ValueError("batches_per_epoch must be positive")
        self._explicit_batches_per_epoch = (
            int(batches_per_epoch) if batches_per_epoch is not None else None
        )
        self._batches_per_epoch = 0

        self._rng = random.Random(seed)
        self._positive_cycle: list[int] = []
        self._epoch = 0
        self._hard_negatives: list[int] = []
        self._hard_negative_share = 0.0
        self._composition_log_path = (
            Path(composition_log_path) if composition_log_path else None
        )
        self._records: list[dict[str, Any]] = []
        self._total_positive = 0
        self._total_sampled = 0
        self._recompute_epoch_length()

    @property
    def positives_per_batch(self) -> int:
        """Positives in every batch, after rounding the requested fraction."""
        return self._n_positive

    @property
    def realised_positive_fraction(self) -> float:
        """The fraction actually delivered, which rounding may shift off the request.

        Recorded alongside the requested value so a sweep's records are unambiguous.
        """
        return self._n_positive / self._batch_size

    @property
    def mining_active(self) -> bool:
        """Whether a hard-negative pool is currently biasing the negative draw."""
        return bool(self._hard_negatives) and self._hard_negative_share > 0.0

    def _uniform_per_batch(self) -> int:
        """Negatives per batch coming from the uniform stream rather than the hard pool."""
        return self._n_negative - round(self._n_negative * self._hard_negative_share)

    def _recompute_epoch_length(self) -> None:
        """Size an epoch at roughly one uniform-stream pass over the negatives.

        Mining takes part of each batch's negatives from the hard pool, which shortens
        the uniform stream. Holding the batch count fixed would then quietly halve the
        easy background seen per epoch, so the count grows to compensate. Coverage is
        approximate, not exact: the uniform stream also skips candidates already in the
        batch, which consumes the pass slightly faster than the returned count suggests.
        Mining is switched on between epochs, and ``DataLoader`` re-reads ``__len__``
        each epoch, so changing here is safe.
        """
        if self._explicit_batches_per_epoch is not None:
            self._batches_per_epoch = self._explicit_batches_per_epoch
            return
        per_batch = self._uniform_per_batch()
        if per_batch <= 0:
            # The share rounds to every negative slot, so no negative reaches a batch
            # except through the hard pool. One pass then means one pass over that pool,
            # not over all negatives: sizing from the full pool would keep drawing the
            # same few hard tiles for a whole epoch's worth of batches and stretch the
            # schedule for nothing.
            self._batches_per_epoch = max(1, len(self._hard_negatives) // self._n_negative)
            return
        self._batches_per_epoch = max(1, len(self._negatives) // per_batch)

    def __len__(self) -> int:
        return self._batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        self._epoch += 1
        epoch = self._epoch
        # Per-iteration state stays local: two live iterators over the same sampler
        # would otherwise interleave one cursor and one record list, producing short or
        # duplicated batches.
        records: list[dict[str, Any]] = []
        self._records = records
        flushed = False
        order = self._negatives[:]
        self._rng.shuffle(order)
        cursor = 0

        try:
            for batch_index in range(self._batches_per_epoch):
                seen: set[int] = set()
                positives = self._draw_positives(seen)
                hard = self._draw_hard_negatives(
                    round(self._n_negative * self._hard_negative_share), seen
                )
                uniform, cursor = self._draw_uniform_negatives(
                    order, cursor, self._n_negative - len(hard), seen
                )

                batch = positives + hard + uniform
                if len(set(batch)) != len(batch):
                    raise RuntimeError(
                        f"batch {batch_index} of epoch {epoch} repeats a position; "
                        "this would double-weight a tile in one gradient step"
                    )
                self._rng.shuffle(batch)
                self._record(records, epoch, batch_index, len(positives), len(hard),
                             len(uniform))
                yield batch
        finally:
            # An epoch abandoned by break, early stopping or a killed DataLoader must
            # still leave its composition on disk; that log is how the realised ratio is
            # verified afterwards. Each generator flushes its own list exactly once, so
            # a stale generator closed later cannot re-append a newer epoch's records.
            if not flushed:
                self._flush_records(records)
                flushed = True

    def set_hard_negatives(
        self,
        negative_scores: Sequence[float],
        pool_fraction: float = DEFAULT_HARD_NEGATIVE_POOL_FRACTION,
        share: float = DEFAULT_HARD_NEGATIVE_SHARE,
    ) -> list[int]:
        """Bias later epochs toward the negatives the model scores highest.

        Parameters
        ----------
        negative_scores : sequence of float
            Predicted positive probability for each entry of ``negative_positions``, in
            that same order.
        pool_fraction : float, optional
            Share of the negative pool treated as hard, taken from the top of the
            ranking.
        share : float, optional
            Share of each batch's negatives drawn deliberately from the hard pool. The
            remainder is drawn from the uniform stream over *all* negatives, so the hard
            pool is represented slightly above ``share``, by the amount that stream
            happens to land in it. Draws are not restricted to the hard pool entirely,
            which would drop the easy background the model must also keep classifying
            correctly. Calling this lengthens the epoch to compensate for the shortened
            uniform stream.

        Returns
        -------
        list of int
            The hard pool, hardest first.

        Raises
        ------
        ValueError
            If the label rule forbids mining, if ``negative_scores`` does not align with
            ``negative_positions``, if either proportion is outside its valid range, or
            if ``pool_fraction`` yields a pool too small to supply ``share`` of every
            batch's negatives.
        """
        check_mining_allowed(self._label_rule, True)
        if len(negative_scores) != len(self._negatives):
            raise ValueError(
                f"negative_scores has {len(negative_scores)} entries but there are "
                f"{len(self._negatives)} negatives; they must align positionally"
            )
        if not 0.0 < pool_fraction <= 1.0:
            raise ValueError("pool_fraction must be in (0, 1]")
        if not 0.0 <= share <= 1.0:
            raise ValueError("share must be in [0, 1]")

        ranked = sorted(
            range(len(self._negatives)),
            key=lambda position: negative_scores[position],
            reverse=True,
        )
        pool_size = max(1, math.ceil(len(ranked) * pool_fraction))
        wanted_per_batch = round(self._n_negative * share)
        if pool_size < wanted_per_batch:
            raise ValueError(
                f"a hard pool of {pool_size} cannot supply {wanted_per_batch} hard "
                f"negative(s) per batch; raise pool_fraction above "
                f"{wanted_per_batch / len(ranked):.4f} or lower share. Delivering fewer "
                "would leave mining_active true while the composition silently differs "
                "from the one configured"
            )
        self._hard_negatives = [self._negatives[index] for index in ranked[:pool_size]]
        self._hard_negative_share = share
        self._recompute_epoch_length()
        LOG.info(
            "Hard-negative pool: %d of %d negatives (top %.0f%%), %.0f%% of each "
            "batch's negatives",
            pool_size,
            len(self._negatives),
            100 * pool_fraction,
            100 * share,
        )
        return list(self._hard_negatives)

    def clear_hard_negatives(self) -> None:
        """Return to uniform negative sampling."""
        self._hard_negatives = []
        self._hard_negative_share = 0.0
        self._recompute_epoch_length()

    @property
    def composition_records(self) -> list[dict[str, Any]]:
        """This epoch's per-batch composition, for callers not writing a log file.

        Scoped to the current epoch so it cannot grow without bound over a long run;
        :meth:`observed_positive_fraction` keeps running totals for the whole run.
        """
        return list(self._records)

    def observed_positive_fraction(self, current_epoch_only: bool = False) -> float:
        """Positive share actually delivered, across the run or just this epoch.

        Raises
        ------
        ValueError
            If no batch has been drawn yet.
        """
        if current_epoch_only:
            if not self._records:
                raise ValueError("no batches drawn in the current epoch")
            positives = sum(record["n_positive"] for record in self._records)
            total = sum(record["batch_size"] for record in self._records)
            return positives / total
        if not self._total_sampled:
            raise ValueError("no batches drawn yet; iterate the sampler first")
        return self._total_positive / self._total_sampled

    def _draw_positives(self, seen: set[int]) -> list[int]:
        """Take the next positives from a reshuffling cycle over the positive pool.

        A candidate already in this batch is set aside and pushed back onto the cycle
        afterwards, rather than dropped: without that, a batch straddling the cycle
        boundary could re-draw a position it had already taken, and dropping it outright
        would bias the even coverage the cycle exists to provide.
        """
        drawn: list[int] = []
        deferred: list[int] = []
        attempts = 0
        limit = 2 * len(self._positives) + self._n_positive
        while len(drawn) < self._n_positive:
            attempts += 1
            if attempts > limit:
                raise RuntimeError(
                    f"could not draw {self._n_positive} distinct positives in {limit} "
                    "attempts; the positive pool cannot fill a batch"
                )
            if not self._positive_cycle:
                self._positive_cycle = self._positives[:]
                self._rng.shuffle(self._positive_cycle)
            candidate = self._positive_cycle.pop()
            if candidate in seen:
                deferred.append(candidate)
                continue
            drawn.append(candidate)
            seen.add(candidate)
        self._positive_cycle.extend(deferred)
        return drawn

    def _draw_hard_negatives(self, wanted: int, seen: set[int]) -> list[int]:
        if wanted <= 0 or not self._hard_negatives:
            return []
        available = [position for position in self._hard_negatives if position not in seen]
        if wanted >= len(available):
            drawn = available
        else:
            drawn = self._rng.sample(available, wanted)
        seen.update(drawn)
        return drawn

    def _draw_uniform_negatives(
        self,
        order: list[int],
        cursor: int,
        wanted: int,
        seen: set[int],
    ) -> tuple[list[int], int]:
        """Take the next negatives from the shuffled pass, skipping ones already drawn.

        ``seen`` covers both the hard negatives taken for this batch and the uniform
        ones taken so far, so neither the hard/uniform boundary nor this pass wrapping
        mid-batch can place the same tile in a batch twice.
        """
        drawn: list[int] = []
        attempts = 0
        limit = 2 * len(order) + wanted
        while len(drawn) < wanted:
            attempts += 1
            if attempts > limit:
                raise RuntimeError(
                    f"could not draw {wanted} distinct negatives in {limit} attempts; "
                    "the negative pool cannot fill a batch alongside the hard negatives"
                )
            if cursor >= len(order):
                self._rng.shuffle(order)
                cursor = 0
            candidate = order[cursor]
            cursor += 1
            if candidate in seen:
                continue
            drawn.append(candidate)
            seen.add(candidate)
        return drawn, cursor

    def _record(
        self,
        records: list[dict[str, Any]],
        epoch: int,
        batch_index: int,
        n_positive: int,
        n_hard_negative: int,
        n_uniform_negative: int,
    ) -> None:
        records.append(
            {
                "epoch": epoch,
                "batch_index": batch_index,
                "batch_size": n_positive + n_hard_negative + n_uniform_negative,
                "n_positive": n_positive,
                "n_negative": n_hard_negative + n_uniform_negative,
                "n_hard_negative": n_hard_negative,
                "requested_positive_fraction": self._positive_fraction,
            }
        )
        self._total_positive += n_positive
        self._total_sampled += n_positive + n_hard_negative + n_uniform_negative

    def _flush_records(self, records: list[dict[str, Any]]) -> None:
        if self._composition_log_path is None or not records:
            return
        self._composition_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._composition_log_path.open("a", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record) + "\n")
