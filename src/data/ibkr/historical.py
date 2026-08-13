from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalRequest:
    request_id: int
    duration: str
    bar_size: str
    what_to_show: str = "TRADES"
    use_regular_trading_hours: bool = True


class HistoricalDataClient(EWrapper, EClient):
    """Minimal synchronous wrapper around IBKR's asynchronous API."""

    def __init__(self) -> None:
        EClient.__init__(self, self)

        self._connected_event = threading.Event()
        self._historical_complete_event = threading.Event()

        self._bars: list[dict[str, object]] = []
        self._errors: list[tuple[int, int, str]] = []

    def nextValidId(self, order_id: int) -> None:
        self._connected_event.set()

    def historicalData(self, req_id: int, bar: BarData) -> None:
        self._bars.append(
            {
                "request_id": req_id,
                "timestamp": bar.date,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": self._to_float(bar.volume),
                "wap": self._to_float(bar.wap),
                "trade_count": int(bar.barCount),
            }
        )

    def historicalDataEnd(
        self,
        req_id: int,
        start: str,
        end: str,
    ) -> None:
        logger.debug(f"Historical request {req_id} completed: {start} to {end}")
        self._historical_complete_event.set()

    def error(
        self,
        req_id: int,
        error_time: int,
        error_code: int,
        error_message: str,
        advanced_order_reject_json: str = "",
    ) -> None:
        """Handle IBKR errors and informational connection messages."""

        informational_codes = {
            2104,  # Market data farm connection is OK
            2106,  # Historical data farm connection is OK
            2107,  # Historical data farm is inactive but available
            2108,  # Market data farm is inactive but available
            2158,  # Sec-def data farm connection is OK
        }

        if error_code in informational_codes:
            logger.debug(f"IBKR status {error_code}: {error_message}")
            return

        self._errors.append(
            (
                req_id,
                error_code,
                error_message,
            )
        )

        logger.error(
            f"IBKR error: request={req_id}, "
            f"time={error_time}, "
            f"code={error_code}, "
            f"message={error_message}"
        )

        if advanced_order_reject_json:
            logger.error(
                "Advanced order rejection details: "
                f"{advanced_order_reject_json}"
            )

    def connect_and_start(
        self,
        host: str,
        port: int,
        client_id: int,
        timeout: float = 10.0,
    ) -> None:
        self.connect(host, port, client_id)

        thread = threading.Thread(
            target=self.run,
            name="ibkr-api-thread",
            daemon=True,
        )
        thread.start()

        if not self._connected_event.wait(timeout=timeout):
            self.disconnect()
            raise TimeoutError("IBKR connection was not confirmed.")

    def request_historical_bars(
        self,
        contract: Contract,
        request: HistoricalRequest,
        timeout: float = 60.0,
    ) -> pd.DataFrame:
        self._bars.clear()
        self._errors.clear()
        self._historical_complete_event.clear()

        self.reqHistoricalData(
            reqId=request.request_id,
            contract=contract,
            endDateTime="",
            durationStr=request.duration,
            barSizeSetting=request.bar_size,
            whatToShow=request.what_to_show,
            useRTH=int(request.use_regular_trading_hours),
            formatDate=2,
            keepUpToDate=False,
            chartOptions=[],
        )

        if not self._historical_complete_event.wait(timeout=timeout):
            self.cancelHistoricalData(request.request_id)
            raise TimeoutError("Historical-data request timed out.")

        if self._errors:
            raise RuntimeError(f"IBKR request failed: {self._errors}")

        frame = pd.DataFrame(self._bars)

        if frame.empty:
            raise RuntimeError("IBKR returned no historical bars.")

        frame["timestamp"] = pd.to_datetime(
            pd.to_numeric(frame["timestamp"], errors="raise"),
            unit="s",
            utc=True,
        )

        frame = (
            frame.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )

        return frame

    @staticmethod
    def _to_float(value: Decimal | float | int) -> float:
        return float(value)