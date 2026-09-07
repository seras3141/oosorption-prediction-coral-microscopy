"""Synthetic tests for the forced-ratio batch sampler and hard-negative mining."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.samplers import ForcedRatioBatchSampler, check_mining_allowed

POSITIVES = list(range(0, 20))
NEGATIVES = list(range(20, 220))


def _sampler(**overrides) -> ForcedRatioBatchSampler:
    kwargs = {
        "positive_positions": POSITIVES,
        "negative_positions": NEGATIVES,
        "batch_size": 20,
        "positive_fraction": 0.25,
        "seed": 7,
    }
    kwargs.update(overrides)
    return ForcedRatioBatchSampler(**kwargs)


def test_every_batch_holds_the_forced_ratio_exactly() -> None:
    """The ratio must hold per batch, not merely on average across an epoch."""
    sampler = _sampler()
    positive_set = set(POSITIVES)

    batches = list(sampler)

    assert len(batches) == len(sampler)
    for batch in batches:
        assert len(batch) == 20
        assert sum(1 for position in batch if position in positive_set) == 5


def test_forced_ratio_holds_over_many_epochs() -> None:
    sampler = _sampler()
    positive_set = set(POSITIVES)
    seen = 0

    for _ in range(20):
        for batch in sampler:
            assert sum(1 for p in batch if p in positive_set) == 5
            seen += 1

    assert sampler.observed_positive_fraction() == pytest.approx(0.25)
    assert seen == 20 * len(sampler)


def test_realised_fraction_reports_rounding() -> None:
    """batch_size 10 at 0.24 rounds to 2 positives, i.e. 0.2 actually delivered."""
    sampler = _sampler(batch_size=10, positive_fraction=0.24)

    assert sampler.positives_per_batch == 2
    assert sampler.realised_positive_fraction == pytest.approx(0.2)
    next(iter(sampler))
    assert sampler.observed_positive_fraction() == pytest.approx(0.2)


def test_epoch_covers_the_negative_pool_once_by_default() -> None:
    sampler = _sampler()
    # 200 negatives, 15 per batch -> 13 full batches.
    assert len(sampler) == 13

    negative_counts = collections.Counter(
        position for batch in sampler for position in batch if position in set(NEGATIVES)
    )
    assert max(negative_counts.values()) == 1, "negatives repeated inside one epoch"


def test_positives_are_cycled_rather_than_drawn_with_replacement() -> None:
    """Even oversampling: no positive is used twice before all have been used once."""
    sampler = _sampler()
    positive_set = set(POSITIVES)

    drawn = [p for batch in sampler for p in batch if p in positive_set]
    first_pass = drawn[: len(POSITIVES)]

    assert sorted(first_pass) == sorted(POSITIVES)


def test_batches_are_shuffled_not_class_ordered() -> None:
    """A batch laid out positives-then-negatives would correlate with batch position."""
    sampler = _sampler()
    positive_set = set(POSITIVES)

    offsets = {
        index
        for batch in sampler
        for index, position in enumerate(batch)
        if position in positive_set
    }

    assert len(offsets) > 5


def test_seed_makes_batches_reproducible() -> None:
    assert list(_sampler(seed=3)) == list(_sampler(seed=3))
    assert list(_sampler(seed=3)) != list(_sampler(seed=4))


def test_hard_negative_ranking_selects_the_highest_scoring_negatives() -> None:
    sampler = _sampler()
    # Ascending scores, so the hardest negatives are the last entries.
    scores = [index / len(NEGATIVES) for index in range(len(NEGATIVES))]

    pool = sampler.set_hard_negatives(scores, pool_fraction=0.1)

    assert pool == list(reversed(NEGATIVES[-20:]))
    assert pool[0] == NEGATIVES[-1]
    assert sampler.mining_active


def test_mining_biases_but_does_not_monopolise_the_negative_draw() -> None:
    """Drawing only from the hard pool would discard most of the background."""
    sampler = _sampler()
    scores = [index / len(NEGATIVES) for index in range(len(NEGATIVES))]
    pool = set(sampler.set_hard_negatives(scores, pool_fraction=0.1, share=0.5))
    negative_set = set(NEGATIVES)

    from_pool = outside_pool = 0
    for batch in sampler:
        for position in batch:
            if position not in negative_set:
                continue
            if position in pool:
                from_pool += 1
            else:
                outside_pool += 1

    assert from_pool > 0 and outside_pool > 0
    # 15 negatives per batch; round(15 * 0.5) = 8 drawn deliberately from the pool, and
    # the remaining 7 come uniformly from all 200 negatives, of which the 20-strong pool
    # is 10%. So the pool's realised share sits just above the requested one.
    expected = 8 / 15 + (7 / 15) * 0.1
    assert from_pool / (from_pool + outside_pool) == pytest.approx(expected, abs=0.02)
    assert expected > 0.5


def test_mining_share_of_one_uses_only_hard_negatives() -> None:
    sampler = _sampler()
    scores = [index / len(NEGATIVES) for index in range(len(NEGATIVES))]
    pool = set(sampler.set_hard_negatives(scores, pool_fraction=0.5, share=1.0))
    negative_set = set(NEGATIVES)

    for batch in sampler:
        assert all(p in pool for p in batch if p in negative_set)


def test_mining_records_hard_negative_counts() -> None:
    sampler = _sampler()
    sampler.set_hard_negatives([0.5] * len(NEGATIVES), pool_fraction=0.5, share=0.5)

    next(iter(sampler))

    record = sampler.composition_records[0]
    assert record["n_hard_negative"] > 0
    assert record["n_positive"] + record["n_negative"] == record["batch_size"]


def test_clearing_hard_negatives_restores_uniform_sampling() -> None:
    sampler = _sampler()
    sampler.set_hard_negatives([0.5] * len(NEGATIVES))
    sampler.clear_hard_negatives()

    next(iter(sampler))

    assert not sampler.mining_active
    assert sampler.composition_records[0]["n_hard_negative"] == 0


def test_misaligned_scores_raise() -> None:
    sampler = _sampler()
    with pytest.raises(ValueError, match="must align positionally"):
        sampler.set_hard_negatives([0.5, 0.5])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pool_fraction": 0.0}, r"pool_fraction must be in \(0, 1\]"),
        ({"pool_fraction": 1.5}, r"pool_fraction must be in \(0, 1\]"),
        ({"share": -0.1}, r"share must be in \[0, 1\]"),
        ({"share": 1.1}, r"share must be in \[0, 1\]"),
    ],
)
def test_invalid_mining_proportions_raise(kwargs: dict, match: str) -> None:
    sampler = _sampler()
    with pytest.raises(ValueError, match=match):
        sampler.set_hard_negatives([0.5] * len(NEGATIVES), **kwargs)


def test_composition_log_is_written_as_jsonl(tmp_path: Path) -> None:
    """Verification re-derives the ratio actually used from this, not from the flag."""
    log_path = tmp_path / "nested" / "composition.jsonl"
    sampler = _sampler(composition_log_path=log_path)

    for batch in sampler:
        pass

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == len(sampler)
    assert {record["epoch"] for record in records} == {1}
    assert all(record["n_positive"] == 5 for record in records)
    assert all(record["requested_positive_fraction"] == 0.25 for record in records)

    for batch in sampler:
        pass
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert {record["epoch"] for record in records} == {1, 2}


def test_observed_fraction_before_iterating_raises() -> None:
    with pytest.raises(ValueError, match="no batches drawn yet"):
        _sampler().observed_positive_fraction()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"positive_positions": []}, "positive_positions is empty"),
        ({"negative_positions": []}, "negative_positions is empty"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"positive_fraction": 0.0}, r"open interval \(0, 1\)"),
        ({"positive_fraction": 1.0}, r"open interval \(0, 1\)"),
        ({"batch_size": 4, "positive_fraction": 0.1}, "leaves at least one tile"),
        ({"batch_size": 4, "positive_fraction": 0.9}, "leaves at least one tile"),
    ],
)
def test_invalid_construction_raises(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _sampler(**kwargs)


def test_explicit_batches_per_epoch_reshuffles_when_negatives_run_out() -> None:
    """Batches must stay full and correctly balanced past one negative pass."""
    sampler = _sampler(batches_per_epoch=40)
    positive_set = set(POSITIVES)

    batches = list(sampler)

    assert len(batches) == 40
    for batch in batches:
        assert len(batch) == 20
        assert sum(1 for p in batch if p in positive_set) == 5


def test_mining_is_refused_under_the_centroid_rule() -> None:
    """It would preferentially sample the tiles that rule mislabels."""
    with pytest.raises(ValueError, match="not allowed with the centroid label rule"):
        check_mining_allowed("centroid", True)


def test_mining_allowed_under_the_area_rule() -> None:
    assert check_mining_allowed("area", True) is None
    assert check_mining_allowed("centroid", False) is None


def test_pool_too_small_to_fill_a_batch_is_refused() -> None:
    """Otherwise the batch is short and unbalanced while the reported ratio still
    claims the configured value -- worst in the subset and smoke runs that trust it."""
    with pytest.raises(ValueError, match="cannot fill 15 negative slots"):
        ForcedRatioBatchSampler(list(range(6)), [6, 7, 8], batch_size=20)

    with pytest.raises(ValueError, match="without repeating a tile within the batch"):
        ForcedRatioBatchSampler([0, 1, 2], list(range(10, 200)), batch_size=20)


def test_no_position_appears_twice_in_a_batch_under_mining() -> None:
    """The hard draw and the uniform draw must not select the same tile: it would be
    loaded twice and double-weighted in one gradient step."""
    sampler = _sampler()
    sampler.set_hard_negatives(
        [index / len(NEGATIVES) for index in range(len(NEGATIVES))],
        pool_fraction=0.1,
        share=0.5,
    )

    for batch in sampler:
        assert len(set(batch)) == len(batch)


def test_mining_lengthens_the_epoch_to_keep_negative_coverage() -> None:
    """Holding the batch count fixed would halve the easy background seen per epoch."""
    negative_set = set(NEGATIVES)

    plain = _sampler()
    plain_batches = len(plain)
    plain_seen = {p for batch in plain for p in batch if p in negative_set}

    mining = _sampler()
    mining.set_hard_negatives(
        [index / len(NEGATIVES) for index in range(len(NEGATIVES))],
        pool_fraction=0.1,
        share=0.5,
    )
    mining_seen = {p for batch in mining for p in batch if p in negative_set}

    assert len(mining) > plain_batches
    assert len(mining_seen) >= len(plain_seen)
    assert len(mining_seen) == len(NEGATIVES)


def test_clearing_mining_restores_the_original_epoch_length() -> None:
    sampler = _sampler()
    original = len(sampler)
    sampler.set_hard_negatives([0.5] * len(NEGATIVES), share=0.5)
    assert len(sampler) != original

    sampler.clear_hard_negatives()
    assert len(sampler) == original


def test_explicit_batches_per_epoch_is_not_overridden_by_mining() -> None:
    sampler = _sampler(batches_per_epoch=40)
    sampler.set_hard_negatives([0.5] * len(NEGATIVES), share=0.5)
    assert len(sampler) == 40


def test_records_are_flushed_when_an_epoch_is_abandoned(tmp_path: Path) -> None:
    """Early stopping or a killed DataLoader must still leave the composition on disk;
    that log is how the realised ratio gets verified afterwards."""
    log_path = tmp_path / "composition.jsonl"
    sampler = _sampler(composition_log_path=log_path)

    for position, _batch in enumerate(sampler):
        if position == 2:
            break

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 3


def test_records_are_scoped_to_the_current_epoch() -> None:
    """_records must not grow for the whole run; lifetime stats use running totals."""
    sampler = _sampler()

    for _batch in sampler:
        pass
    first_epoch = len(sampler.composition_records)
    for _batch in sampler:
        pass

    assert len(sampler.composition_records) == first_epoch
    assert sampler.composition_records[0]["epoch"] == 2


def test_observed_fraction_scopes_over_run_or_epoch() -> None:
    sampler = _sampler()

    for _batch in sampler:
        pass
    for _batch in sampler:
        pass

    assert sampler.observed_positive_fraction() == pytest.approx(0.25)
    assert sampler.observed_positive_fraction(current_epoch_only=True) == pytest.approx(0.25)


def test_current_epoch_fraction_before_iterating_raises() -> None:
    with pytest.raises(ValueError, match="no batches drawn in the current epoch"):
        _sampler().observed_positive_fraction(current_epoch_only=True)


@pytest.mark.parametrize("seed", range(6))
def test_no_duplicate_when_a_batch_straddles_the_positive_cycle_boundary(seed: int) -> None:
    """20 positives at 3 per batch is not an exact division, so a batch spans the
    cycle refill; the refilled permutation must not return a position already taken."""
    sampler = ForcedRatioBatchSampler(
        POSITIVES, NEGATIVES, batch_size=12, positive_fraction=0.25, seed=seed
    )
    assert len(POSITIVES) % sampler.positives_per_batch != 0

    for _epoch in range(5):
        for batch in sampler:
            assert len(set(batch)) == len(batch)


def test_no_duplicate_when_the_uniform_pass_wraps_mid_batch() -> None:
    """Past one negative pass the stream reshuffles; it must still exclude what this
    batch already holds, not only the hard negatives."""
    sampler = _sampler(batches_per_epoch=40, seed=1)

    for batch in sampler:
        assert len(set(batch)) == len(batch)
        assert len(batch) == 20


@pytest.mark.parametrize(
    ("batch_size", "positive_fraction"),
    [(8, 0.25), (12, 0.2), (20, 0.4), (64, 0.125)],
)
def test_batches_are_unique_and_full_across_configurations(
    batch_size: int, positive_fraction: float
) -> None:
    """Three draw sites each found a way to duplicate a tile, so sweep the shapes."""
    sampler = ForcedRatioBatchSampler(
        list(range(37)),
        list(range(100, 431)),
        batch_size=batch_size,
        positive_fraction=positive_fraction,
        seed=3,
    )
    sampler.set_hard_negatives([index / 331 for index in range(331)], pool_fraction=0.15)

    for _epoch in range(3):
        for batch in sampler:
            assert len(batch) == batch_size
            assert len(set(batch)) == batch_size


def test_epoch_without_mining_is_a_clean_single_pass() -> None:
    """The exact claim: no negative twice, and only the dropped partial batch unseen."""
    sampler = _sampler()
    negative_set = set(NEGATIVES)

    counts = collections.Counter(
        position for batch in sampler for position in batch if position in negative_set
    )

    assert max(counts.values()) == 1
    assert len(counts) == len(sampler) * (20 - sampler.positives_per_batch)


def test_two_live_iterators_do_not_share_cursor_or_records() -> None:
    """Nested loops or a retained DataLoader iterator would otherwise interleave state
    and yield short or duplicated batches."""
    sampler = _sampler(seed=5)
    first, second = iter(sampler), iter(sampler)

    for _ in range(8):
        for batch in (next(first), next(second)):
            assert len(batch) == 20
            assert len(set(batch)) == 20


def test_stale_generator_does_not_rewrite_a_newer_epochs_records(tmp_path: Path) -> None:
    """Its finally block must flush its own records, not whatever the sampler holds now."""
    import gc

    log_path = tmp_path / "composition.jsonl"
    sampler = _sampler(composition_log_path=log_path)

    abandoned = iter(sampler)
    next(abandoned)
    for _batch in sampler:
        pass
    del abandoned
    gc.collect()

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    keys = collections.Counter((r["epoch"], r["batch_index"]) for r in records)
    assert [key for key, count in keys.items() if count > 1] == []


def test_sampler_itself_refuses_mining_under_the_centroid_rule() -> None:
    """The guard must not depend on a training script remembering to call it."""
    sampler = _sampler(label_rule="centroid")

    with pytest.raises(ValueError, match="not allowed with the centroid label rule"):
        sampler.set_hard_negatives([0.5] * len(NEGATIVES))


def test_area_rule_sampler_accepts_mining() -> None:
    sampler = _sampler(label_rule="area")
    assert sampler.set_hard_negatives([0.5] * len(NEGATIVES))


def test_pool_padded_with_duplicates_is_refused() -> None:
    """Batches hold distinct positions, so a repeat-padded pool is unsatisfiable and
    the draw loop would spin forever -- a wedged job with no output, not an error."""
    with pytest.raises(ValueError, match="1 distinct positive"):
        ForcedRatioBatchSampler([0, 0, 0, 0], list(range(10, 100)), batch_size=8)

    with pytest.raises(ValueError, match="distinct negative"):
        ForcedRatioBatchSampler(list(range(20)), [20, 20, 20], batch_size=20)


def test_overlapping_pools_are_refused() -> None:
    """A tile cannot be both classes; the batch could never be filled."""
    with pytest.raises(ValueError, match="appear in both pools"):
        ForcedRatioBatchSampler(list(range(10)), list(range(15)), batch_size=20)


@pytest.mark.parametrize("label_rule", ["Centroid", "centroid_rule", "", "area "])
def test_unknown_label_rule_is_refused(label_rule: str) -> None:
    """Treating an unrecognised rule as "not centroid" would silently re-enable the
    combination the guard exists to forbid."""
    with pytest.raises(ValueError, match="label_rule must be one of"):
        ForcedRatioBatchSampler(POSITIVES, NEGATIVES, batch_size=20, label_rule=label_rule)

    with pytest.raises(ValueError, match="label_rule must be one of"):
        check_mining_allowed(label_rule, True)


def test_hard_pool_too_small_for_the_requested_share_is_refused() -> None:
    """Delivering fewer would leave mining_active true while the realised composition
    differs from the configured one, and __len__ would overstate coverage."""
    sampler = _sampler()

    with pytest.raises(ValueError, match="cannot supply 15 hard negative"):
        sampler.set_hard_negatives([0.5] * len(NEGATIVES), pool_fraction=0.005, share=1.0)


def test_hard_pool_exactly_large_enough_is_accepted() -> None:
    sampler = _sampler()
    # 15 negatives per batch at share 1.0 needs a pool of at least 15 of 200 = 0.075.
    pool = sampler.set_hard_negatives(
        [0.5] * len(NEGATIVES), pool_fraction=0.075, share=1.0
    )

    assert len(pool) >= 15
    for batch in sampler:
        assert len(set(batch)) == len(batch)
