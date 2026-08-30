from __future__ import annotations

import pandas as pd

from src.pipelines.alpha_factory.pipeline import (
    _split_into_batches,
    _upsert_by_candidate_id,
)


def _candidate_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"candidate_id": [f"c{i}" for i in range(n)], "value": range(n)}
    )


def test_split_into_batches_preserves_order_and_row_count():
    frame = _candidate_frame(23)
    batches = _split_into_batches(frame, batch_size=5)

    assert [len(batch) for batch in batches] == [5, 5, 5, 5, 3]
    assert (
        pd.concat(batches).reset_index(drop=True).equals(frame)
    )


def test_split_into_batches_batch_size_zero_returns_single_batch():
    frame = _candidate_frame(10)
    batches = _split_into_batches(frame, batch_size=0)

    assert len(batches) == 1
    assert batches[0].equals(frame)


def test_split_into_batches_batch_size_larger_than_population_returns_single_batch():
    frame = _candidate_frame(10)
    batches = _split_into_batches(frame, batch_size=1000)

    assert len(batches) == 1
    assert batches[0].equals(frame)


def test_split_into_batches_empty_frame_returns_no_batches():
    frame = _candidate_frame(0)

    assert _split_into_batches(frame, batch_size=5) == []


def test_upsert_twice_with_disjoint_subsets_matches_one_call_with_union(tmp_path):
    """
    Pins down the load-bearing assumption the whole batching design rests
    on: calling _upsert_by_candidate_id once per batch with disjoint
    candidate subsets must produce a result identical to calling it once
    with the full union.
    """
    full = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(10)],
            "metric": [float(i) for i in range(10)],
        }
    )
    batch_a, batch_b = full.iloc[:4], full.iloc[4:]

    path_batched = tmp_path / "batched.csv"
    _upsert_by_candidate_id(path_batched, batch_a)
    result_batched = _upsert_by_candidate_id(path_batched, batch_b)

    path_single = tmp_path / "single.csv"
    result_single = _upsert_by_candidate_id(path_single, full)

    pd.testing.assert_frame_equal(
        result_batched.reset_index(drop=True),
        result_single.reset_index(drop=True),
        check_dtype=False,
    )


def test_upsert_disjoint_subsets_result_independent_of_batch_order(tmp_path):
    full = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(6)],
            "metric": [float(i) for i in range(6)],
        }
    )
    first, second, third = full.iloc[:2], full.iloc[2:4], full.iloc[4:]

    path_forward = tmp_path / "forward.csv"
    result_forward = None
    for chunk in (first, second, third):
        result_forward = _upsert_by_candidate_id(path_forward, chunk)

    path_reverse = tmp_path / "reverse.csv"
    result_reverse = None
    for chunk in (third, second, first):
        result_reverse = _upsert_by_candidate_id(path_reverse, chunk)

    pd.testing.assert_frame_equal(
        result_forward.reset_index(drop=True),
        result_reverse.reset_index(drop=True),
        check_dtype=False,
    )
