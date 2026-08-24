from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, TypeVar

import pandas as pd


T = TypeVar("T")
R = TypeVar("R")

_worker_market_data: dict[str, pd.DataFrame] | None = None


def _init_worker(market_data_by_symbol: dict[str, pd.DataFrame]) -> None:
    """
    Pool initializer: load read-only market data into this worker process
    once, at pool startup, instead of re-pickling/re-sending the full
    per-symbol DataFrames on every task submitted to the pool.
    """
    global _worker_market_data
    _worker_market_data = market_data_by_symbol


def get_worker_market_data() -> dict[str, pd.DataFrame]:
    """
    Return the market data loaded into this worker process by the pool
    initializer. Only valid inside a worker process spawned by
    create_executor() -- callers that don't go through a pool (the
    sequential fallback) should use their own already-in-scope market-data
    argument instead of calling this.
    """
    if _worker_market_data is None:
        raise RuntimeError(
            "Worker market data is not initialized. "
            "get_worker_market_data() is only valid inside a worker "
            "process started by create_executor()."
        )

    return _worker_market_data


def create_executor(
    max_workers: int,
    market_data_by_symbol: dict[str, pd.DataFrame],
) -> ProcessPoolExecutor | None:
    """
    Create a process pool for parallel candidate testing, reused across
    every stage of one pipeline run, or return None to signal that callers
    should fall back to their plain sequential loop.
    """
    if max_workers <= 1:
        return None

    return ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(market_data_by_symbol,),
    )


def run_parallel_map(
    executor: ProcessPoolExecutor | None,
    fn: Callable[[T], R],
    items: Iterable[T],
) -> list[R]:
    """
    Apply fn to every item in parallel via executor, or sequentially if
    executor is None.
    """
    items = list(items)

    if executor is None:
        return [fn(item) for item in items]

    return list(executor.map(fn, items))
