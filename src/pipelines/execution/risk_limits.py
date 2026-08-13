from __future__ import annotations

from typing import Any


def check_pre_trade_risk(*args: Any, **kwargs: Any) -> bool:
    """
    Run pre-trade risk checks before placing/adjusting a live order.

    Intended checks once implemented: max position size per instrument, max
    daily loss, max total gross exposure across all deployed strategies.

    Raises:
        NotImplementedError: always -- this is an unimplemented stub.
    """
    raise NotImplementedError(
        "Pre-trade risk checking is not yet implemented."
    )
