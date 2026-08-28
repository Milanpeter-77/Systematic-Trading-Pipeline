from __future__ import annotations

import pytest

from src.factory.generator import expand_parameter_grid, generate_candidates
from src.factory.registry import STRATEGY_REGISTRY


def test_expand_parameter_grid_produces_cartesian_product():
    combinations = expand_parameter_grid(
        {"a": [1, 2], "b": [10, 20, 30]}
    )
    assert len(combinations) == 6
    assert {"a": 1, "b": 10} in combinations
    assert {"a": 2, "b": 30} in combinations


def test_expand_parameter_grid_rejects_empty_grid():
    with pytest.raises(ValueError):
        expand_parameter_grid({})


def test_generate_candidates_produces_no_duplicate_ids_across_full_registry():
    fx_symbols = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD"]
    all_symbols = fx_symbols + ["XAUUSD", "XAGUSD", "VOD", "SAP"]
    pair_symbols = ["AUDUSD_VOD", "XAGUSD_XAUUSD"]

    overrides: dict[str, list[str]] = {}
    for family in STRATEGY_REGISTRY:
        if family == "carry" or family.startswith("carry_"):
            overrides[family] = fx_symbols
        if family == "stat_arb" or family.startswith("stat_arb_"):
            overrides[family] = pair_symbols

    candidates = generate_candidates(
        symbols=all_symbols,
        family_symbol_overrides=overrides,
    )

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    assert len(candidate_ids) == len(set(candidate_ids))
    assert len(candidates) > 0


def test_carry_and_stat_arb_families_are_routed_to_their_own_symbol_universe():
    fx_symbols = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD"]
    all_symbols = fx_symbols + ["XAUUSD", "XAGUSD", "VOD", "SAP"]
    pair_symbols = ["AUDUSD_VOD", "XAGUSD_XAUUSD"]

    overrides: dict[str, list[str]] = {}
    for family in STRATEGY_REGISTRY:
        if family == "carry" or family.startswith("carry_"):
            overrides[family] = fx_symbols
        if family == "stat_arb" or family.startswith("stat_arb_"):
            overrides[family] = pair_symbols

    candidates = generate_candidates(
        symbols=all_symbols,
        family_symbol_overrides=overrides,
    )

    carry_symbols = {
        candidate.symbol
        for candidate in candidates
        if candidate.family == "carry" or candidate.family.startswith("carry_")
    }
    stat_arb_symbols = {
        candidate.symbol
        for candidate in candidates
        if candidate.family == "stat_arb"
        or candidate.family.startswith("stat_arb_")
    }

    assert carry_symbols == set(fx_symbols)
    assert stat_arb_symbols == set(pair_symbols)
