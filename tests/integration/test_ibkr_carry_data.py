"""IBKR carry-data feasibility probe.

Connects to the local paper-trading TWS session and checks two independent,
unconfirmed data sources needed for a real carry strategy:

  * FX forward points: does IBKR resolve a CASH contract with a forward
    lastTradeDateOrContractMonth (as opposed to spot), and if so, does
    historical/market data actually come back for it?
  * Equity fundamentals: does reqFundamentalData return real dividend data
    for the account's active equities (VOD, SAP)?

Neither is confirmed anywhere in this repo or in IBKR's own official
contract samples, so this prints raw results rather than assuming a shape
for either response -- the carry strategy design should be built from
whatever comes back here, not from a guess.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ibapi.client import EClient  # noqa: E402
from ibapi.contract import Contract, ContractDetails  # noqa: E402
from ibapi.wrapper import EWrapper  # noqa: E402

from src.data.ibkr.historical import (  # noqa: E402
    HistoricalDataClient,
    HistoricalRequest,
)


HOST = "127.0.0.1"
PORT = 7497

INFORMATIONAL_ERROR_CODES = {2104, 2106, 2107, 2108, 2158}


class ContractDetailsProbeClient(EWrapper, EClient):
    """Minimal client to check whether IBKR resolves a given contract at all."""

    def __init__(self) -> None:
        EClient.__init__(self, self)

        self._connected_event = threading.Event()
        self._details_end_event = threading.Event()
        self.details: list[ContractDetails] = []
        self.errors: list[tuple[int, int, str]] = []

    def nextValidId(self, order_id: int) -> None:
        self._connected_event.set()

    def contractDetails(self, reqId: int, contractDetails: ContractDetails) -> None:
        self.details.append(contractDetails)

    def contractDetailsEnd(self, reqId: int) -> None:
        self._details_end_event.set()

    def error(
        self,
        req_id: int,
        error_time: int,
        error_code: int,
        error_message: str,
        advanced_order_reject_json: str = "",
    ) -> None:
        if error_code in INFORMATIONAL_ERROR_CODES:
            return

        self.errors.append((req_id, error_code, error_message))
        self._details_end_event.set()

    def connect_and_start(self, host: str, port: int, client_id: int, timeout: float = 10.0) -> None:
        self.connect(host, port, client_id)

        thread = threading.Thread(target=self.run, name="ibkr-contractdetails-thread", daemon=True)
        thread.start()

        if not self._connected_event.wait(timeout=timeout):
            self.disconnect()
            raise TimeoutError("IBKR connection was not confirmed.")

    def probe_contract(self, contract: Contract, request_id: int, timeout: float = 10.0) -> dict:
        self.details.clear()
        self.errors.clear()
        self._details_end_event.clear()

        self.reqContractDetails(request_id, contract)

        if not self._details_end_event.wait(timeout=timeout):
            return {"resolved": False, "detail_count": 0, "errors": [("timeout", "no response")]}

        return {
            "resolved": len(self.details) > 0,
            "detail_count": len(self.details),
            "errors": list(self.errors),
        }


class FundamentalDataProbeClient(EWrapper, EClient):
    """Minimal client to check whether reqFundamentalData returns real data."""

    def __init__(self) -> None:
        EClient.__init__(self, self)

        self._connected_event = threading.Event()
        self._data_event = threading.Event()
        self.raw_xml: str | None = None
        self.errors: list[tuple[int, int, str]] = []

    def nextValidId(self, order_id: int) -> None:
        self._connected_event.set()

    def fundamentalData(self, reqId: int, data: str) -> None:
        self.raw_xml = data
        self._data_event.set()

    def error(
        self,
        req_id: int,
        error_time: int,
        error_code: int,
        error_message: str,
        advanced_order_reject_json: str = "",
    ) -> None:
        if error_code in INFORMATIONAL_ERROR_CODES:
            return

        self.errors.append((req_id, error_code, error_message))
        self._data_event.set()

    def connect_and_start(self, host: str, port: int, client_id: int, timeout: float = 10.0) -> None:
        self.connect(host, port, client_id)

        thread = threading.Thread(target=self.run, name="ibkr-fundamental-thread", daemon=True)
        thread.start()

        if not self._connected_event.wait(timeout=timeout):
            self.disconnect()
            raise TimeoutError("IBKR connection was not confirmed.")

    def probe_fundamentals(
        self,
        contract: Contract,
        request_id: int,
        report_type: str = "ReportSnapshot",
        timeout: float = 15.0,
    ) -> dict:
        self.raw_xml = None
        self.errors.clear()
        self._data_event.clear()

        self.reqFundamentalData(request_id, contract, report_type, [])

        if not self._data_event.wait(timeout=timeout):
            return {"received": False, "xml": None, "errors": [("timeout", "no response")]}

        return {
            "received": self.raw_xml is not None,
            "xml": self.raw_xml,
            "errors": list(self.errors),
        }


def probe_fx_forward() -> None:
    print("=== FX forward feasibility (EURUSD) ===\n")

    # A short (30-day) forward's interest-rate-differential effect can be
    # smaller than IBKR's quote rounding, making it look identical to spot
    # even if forward pricing genuinely is being computed. Use a full year
    # so any real forward-points adjustment is unmistakable either way.
    forward_date = (
        datetime.now(timezone.utc) + timedelta(days=365)
    ).strftime("%Y%m%d")

    spot_contract = Contract()
    spot_contract.symbol = "EUR"
    spot_contract.secType = "CASH"
    spot_contract.currency = "USD"
    spot_contract.exchange = "IDEALPRO"

    forward_contract = Contract()
    forward_contract.symbol = "EUR"
    forward_contract.secType = "CASH"
    forward_contract.currency = "USD"
    forward_contract.exchange = "IDEALPRO"
    forward_contract.lastTradeDateOrContractMonth = forward_date

    details_client = ContractDetailsProbeClient()
    details_client.connect_and_start(HOST, PORT, 10)

    spot_result = details_client.probe_contract(spot_contract, request_id=1)
    print(f"Spot CASH contract resolves: {spot_result}")

    time.sleep(2.0)

    forward_result = details_client.probe_contract(forward_contract, request_id=2)
    print(f"Forward-dated CASH contract ({forward_date}) resolves: {forward_result}")

    details_client.disconnect()

    if not forward_result["resolved"]:
        print(
            "\n-> Forward-dated CASH contract did NOT resolve. FX forward "
            "points are likely not accessible this way via the API.\n"
        )
        return

    print(
        "\n-> Forward-dated CASH contract resolved. Comparing its price "
        "against spot -- resolving is not proof of real forward pricing; "
        "IBKR may silently return spot-identical data for a forward-dated "
        "contract it doesn't actually price differently.\n"
    )

    time.sleep(2.0)

    historical_client = HistoricalDataClient()
    historical_client.connect_and_start(HOST, PORT, 11)

    def fetch_last_close(contract: Contract, request_id: int) -> float | None:
        request = HistoricalRequest(
            request_id=request_id,
            duration="1 D",
            bar_size="1 hour",
            what_to_show="MIDPOINT",
            use_regular_trading_hours=False,
        )

        try:
            frame = historical_client.request_historical_bars(contract, request, timeout=20.0)
            return float(frame["close"].iloc[-1])
        except (RuntimeError, TimeoutError) as error:
            print(f"  historical data FAILED: {error}")
            return None

    spot_close = fetch_last_close(spot_contract, request_id=3)
    print(f"Spot last close: {spot_close}")

    time.sleep(2.0)

    forward_close = fetch_last_close(forward_contract, request_id=4)
    print(f"1-year forward last close: {forward_close}")

    historical_client.disconnect()

    if spot_close is None or forward_close is None:
        print("\n-> Could not compare -- at least one leg failed to fetch.\n")
    elif spot_close == forward_close:
        print(
            "\n-> IDENTICAL to spot. This confirms the forward date is being "
            "accepted but not actually priced -- not usable for a real "
            "carry signal via this path.\n"
        )
    else:
        print(
            f"\n-> DIFFERS from spot by {forward_close - spot_close:+.5f}. "
            "This looks like real forward-points pricing.\n"
        )


def probe_equity_fundamentals() -> None:
    print("\n=== Equity fundamentals feasibility (VOD, SAP) ===\n")

    client = FundamentalDataProbeClient()
    client.connect_and_start(HOST, PORT, 12)

    for offset, (symbol, currency) in enumerate([("VOD", "GBP"), ("SAP", "EUR")]):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.currency = currency
        contract.exchange = "SMART"

        result = client.probe_fundamentals(contract, request_id=20 + offset)

        if result["received"]:
            xml = result["xml"] or ""
            print(f"[PASS] {symbol}: received {len(xml)} chars of XML")
            print(xml[:1500])
            print("...\n" if len(xml) > 1500 else "\n")
        else:
            print(f"[FAIL] {symbol}: {result['errors']}\n")

        time.sleep(3.0)

    client.disconnect()


def main() -> None:
    probe_fx_forward()
    probe_equity_fundamentals()


if __name__ == "__main__":
    main()
