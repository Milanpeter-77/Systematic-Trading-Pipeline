"""IBKR market-data subscription probe.

Connects to the local paper-trading TWS session and attempts a short
historical-data pull for a list of candidate instruments spanning the
sec_types the pipeline already knows how to build contracts for (STK,
CASH, CRYPTO, CMDTY, IND) across a spread of exchanges/regions. Prints a
pass/fail table so it's clear which instruments the current IBKR account
subscription actually supports, before adding them to
config/instruments.yml.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ibapi.client import EClient  # noqa: E402
from ibapi.wrapper import EWrapper  # noqa: E402

from src.data.ibkr.contracts import (  # noqa: E402
    cash_contract,
    commodity_contract,
    crypto_contract,
    index_contract,
    stock_contract,
)
from src.data.ibkr.historical import (  # noqa: E402
    HistoricalDataClient,
    HistoricalRequest,
)
from src.environment import IBKR_HOST, IBKR_PORT  # noqa: E402


HOST = IBKR_HOST
PORT = IBKR_PORT
CLIENT_ID = 2

MARKET_DATA_TYPE_NAMES = {
    1: "live",
    2: "frozen",
    3: "delayed",
    4: "delayed-frozen",
}


@dataclass(frozen=True)
class Probe:
    name: str
    sec_type: str
    contract_factory: object
    contract_kwargs: dict
    what_to_show: str
    use_regular_trading_hours: bool


PROBES: list[Probe] = [
    # Non-US equities: does the account's data subscription cover
    # non-US STK, even though US equities/index (SPXUSD/SPY) are
    # confirmed blocked by error 2188?
    Probe(
        "VOD", "STK", stock_contract,
        {"symbol": "VOD", "currency": "GBP", "exchange": "SMART"},
        "TRADES", True,
    ),
    Probe(
        "SAP", "STK", stock_contract,
        {"symbol": "SAP", "currency": "EUR", "exchange": "SMART"},
        "TRADES", True,
    ),
    # US equity control probe: expected to fail the same way as the
    # already-confirmed SPY fallback, included for a fresh, direct
    # confirmation rather than relying on the earlier SPXUSD note.
    Probe(
        "AAPL", "STK", stock_contract,
        {"symbol": "AAPL", "currency": "USD", "exchange": "SMART"},
        "TRADES", True,
    ),
    # Non-US index.
    Probe(
        "DAX", "IND", index_contract,
        {"symbol": "DAX", "currency": "EUR", "exchange": "EUREX"},
        "TRADES", True,
    ),
    # Additional FX majors (CASH already confirmed broadly working via
    # USDJPY, but confirm a few more before adding them to config).
    Probe(
        "EURUSD", "CASH", cash_contract,
        {"symbol": "EUR", "currency": "USD", "exchange": "IDEALPRO"},
        "MIDPOINT", False,
    ),
    Probe(
        "GBPUSD", "CASH", cash_contract,
        {"symbol": "GBP", "currency": "USD", "exchange": "IDEALPRO"},
        "MIDPOINT", False,
    ),
    Probe(
        "AUDUSD", "CASH", cash_contract,
        {"symbol": "AUD", "currency": "USD", "exchange": "IDEALPRO"},
        "MIDPOINT", False,
    ),
    # Additional crypto pair.
    Probe(
        "BTCUSD", "CRYPTO", crypto_contract,
        {"symbol": "BTC", "currency": "USD", "exchange": "PAXOS"},
        "AGGTRADES", False,
    ),
    # Additional spot commodity.
    Probe(
        "XAGUSD", "CMDTY", commodity_contract,
        {"symbol": "XAGUSD", "currency": "USD", "exchange": "SMART"},
        "MIDPOINT", False,
    ),
]


@dataclass
class ProbeResult:
    probe: Probe
    passed: bool
    detail: str


# A rejected request's cancel-acknowledgment can arrive asynchronously
# well after request_historical_bars() has already returned, so a fixed
# gap between probes is needed -- otherwise a stale error meant for probe
# N can land in probe N+1's freshly-cleared error list and produce a false
# failure for an instrument that never actually errored (observed first-hand
# during this investigation: XAGUSD was reported failing with an error that
# was still tagged with the *previous* probe's request_id).
INTER_PROBE_DELAY_SECONDS = 3.0


def run_probe(client: HistoricalDataClient, request_id: int, probe: Probe) -> ProbeResult:
    contract = probe.contract_factory(**probe.contract_kwargs)

    request = HistoricalRequest(
        request_id=request_id,
        duration="1 D",
        bar_size="1 hour",
        what_to_show=probe.what_to_show,
        use_regular_trading_hours=probe.use_regular_trading_hours,
    )

    try:
        frame = client.request_historical_bars(contract, request, timeout=20.0)
    except (RuntimeError, TimeoutError) as error:
        return ProbeResult(probe, passed=False, detail=str(error))
    finally:
        time.sleep(INTER_PROBE_DELAY_SECONDS)

    return ProbeResult(
        probe, passed=True, detail=f"{len(frame)} bars received"
    )


class MarketDataProbeClient(EWrapper, EClient):
    """
    Minimal client to probe streaming market data (reqMktData) and the
    market-data type IBKR actually grants (live/frozen/delayed), decoupled
    from reqHistoricalData -- error 2188 specifically talks about
    "up-to-the-second historical data", which is a different permission
    from live streaming ticks or from historical data anchored to a fixed
    past timestamp.
    """

    def __init__(self) -> None:
        EClient.__init__(self, self)

        self._connected_event = threading.Event()
        self.granted_type: int | None = None
        self.received_tick = False
        self._errors: list[tuple[int, int, str]] = []

    def nextValidId(self, order_id: int) -> None:
        self._connected_event.set()

    def marketDataType(self, reqId: int, marketDataType: int) -> None:
        self.granted_type = marketDataType

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: object) -> None:
        if price > 0:
            self.received_tick = True

    def tickSize(self, reqId: int, tickType: int, size: object) -> None:
        if size and size > 0:
            self.received_tick = True

    def error(
        self,
        req_id: int,
        error_time: int,
        error_code: int,
        error_message: str,
        advanced_order_reject_json: str = "",
    ) -> None:
        informational_codes = {2104, 2106, 2107, 2108, 2158}

        if error_code in informational_codes:
            return

        self._errors.append((req_id, error_code, error_message))

    def connect_and_start(self, host: str, port: int, client_id: int, timeout: float = 10.0) -> None:
        self.connect(host, port, client_id)

        thread = threading.Thread(target=self.run, name="ibkr-mktdata-thread", daemon=True)
        thread.start()

        if not self._connected_event.wait(timeout=timeout):
            self.disconnect()
            raise TimeoutError("IBKR connection was not confirmed.")

    def probe_streaming_data(
        self,
        contract: object,
        request_id: int,
        requested_type: int,
        listen_seconds: float = 5.0,
    ) -> dict:
        self.granted_type = None
        self.received_tick = False
        self._errors.clear()

        self.reqMarketDataType(requested_type)
        self.reqMktData(request_id, contract, "", False, False, [])

        time.sleep(listen_seconds)

        self.cancelMktData(request_id)

        return {
            "requested_type": MARKET_DATA_TYPE_NAMES.get(requested_type, requested_type),
            "granted_type": MARKET_DATA_TYPE_NAMES.get(self.granted_type, self.granted_type),
            "received_tick": self.received_tick,
            "errors": list(self._errors),
        }


@dataclass
class DeepDiveCase:
    label: str
    market_data_type: int
    end_date_time: str


def run_historical_deep_dive(client: HistoricalDataClient, contract: object) -> None:
    two_days_ago_utc = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).strftime("%Y%m%d-%H:%M:%S")

    cases = [
        DeepDiveCase("delayed type + blank endDateTime (now)", 3, ""),
        DeepDiveCase("live type + fixed past endDateTime", 1, two_days_ago_utc),
        DeepDiveCase("delayed type + fixed past endDateTime", 3, two_days_ago_utc),
    ]

    for offset, case in enumerate(cases, start=100):
        client.reqMarketDataType(case.market_data_type)

        request = HistoricalRequest(
            request_id=offset,
            duration="1 D",
            bar_size="1 hour",
            what_to_show="TRADES",
            use_regular_trading_hours=True,
            end_date_time=case.end_date_time,
        )

        try:
            frame = client.request_historical_bars(contract, request, timeout=20.0)
        except (RuntimeError, TimeoutError) as error:
            print(f"  [FAIL] {case.label} -> {error}")
            time.sleep(INTER_PROBE_DELAY_SECONDS)
            continue

        print(f"  [PASS] {case.label} -> {len(frame)} bars received")
        time.sleep(INTER_PROBE_DELAY_SECONDS)


def run_streaming_deep_dive(contract: object) -> None:
    client = MarketDataProbeClient()
    client.connect_and_start(HOST, PORT, CLIENT_ID + 1)

    for label, requested_type in [("live (type 1)", 1), ("delayed (type 3)", 3)]:
        outcome = client.probe_streaming_data(
            contract, request_id=200 + requested_type, requested_type=requested_type
        )
        status = "PASS" if outcome["received_tick"] else "FAIL"
        print(
            f"  [{status}] streaming, requested={label} -> "
            f"granted={outcome['granted_type']}, tick received={outcome['received_tick']}, "
            f"errors={outcome['errors']}"
        )

    client.disconnect()


def main() -> None:
    client = HistoricalDataClient()

    print(f"Connecting to TWS at {HOST}:{PORT}...")
    client.connect_and_start(HOST, PORT, CLIENT_ID)
    print("Connected. Probing instruments...\n")

    results: list[ProbeResult] = []

    for index, probe in enumerate(PROBES, start=1):
        result = run_probe(client, request_id=index, probe=probe)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {probe.name:<8} {probe.sec_type:<6} -> {result.detail}")

    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} probes succeeded.")

    print(
        "\nDeep dive on AAPL (STK): isolating whether error 2188 is about "
        "'up-to-the-second' historical requests specifically, vs a broader "
        "STK data-permission gap.\n"
    )

    aapl_contract = stock_contract(symbol="AAPL", currency="USD", exchange="SMART")

    print("Historical-data variants:")
    run_historical_deep_dive(client, aapl_contract)

    client.disconnect()

    print("\nStreaming market-data variants:")
    run_streaming_deep_dive(aapl_contract)


if __name__ == "__main__":
    main()
