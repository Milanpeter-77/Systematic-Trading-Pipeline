from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class OrderState:
    """
    Tracked state of one live order/position for one deployed candidate.

    Structural placeholder only -- no logic anywhere reads or writes this
    yet. Fields sketch what the eventual order-tracking loop will need.
    """

    candidate_id: str
    symbol: str
    target_position: int
    current_position: int
    status: str  # e.g. "pending", "filled", "cancelled", "rejected"
    created_at: datetime
    updated_at: datetime
