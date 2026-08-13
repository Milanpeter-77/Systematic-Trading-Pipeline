from __future__ import annotations

from typing import Any


class LiveTradingClient:
    """
    Stub for the live IBKR trading session (order placement + account state).

    Intended shape once implemented: an EWrapper/EClient subclass similar in
    spirit to src.data.ibkr.historical.HistoricalDataClient, but covering
    order placement/status callbacks and position/account queries --
    placeOrder, orderStatus, execDetails, reqPositions, reqAccountUpdates.
    Should reuse src.data.ibkr.contracts.build_contract() for contract
    construction rather than duplicating contract-building logic.
    """

    def connect_and_start(self, host: str, port: int, client_id: int) -> None:
        """Connect to IBKR TWS/Gateway for order placement and account queries."""
        raise NotImplementedError

    def place_order(self, *args: Any, **kwargs: Any) -> None:
        """Submit a new order to IBKR."""
        raise NotImplementedError

    def cancel_order(self, *args: Any, **kwargs: Any) -> None:
        """Cancel a previously submitted order."""
        raise NotImplementedError

    def get_current_position(self, symbol: str) -> int:
        """Query the current live position size for one instrument."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        raise NotImplementedError
