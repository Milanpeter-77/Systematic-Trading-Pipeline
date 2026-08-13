"""Minimal IBKR TWS API connection test.

Connects to the local paper-trading TWS session, requests the IBKR
server time, prints the result, and disconnects.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from ibapi.client import EClient
from ibapi.wrapper import EWrapper


class IBKRConnectionTest(EWrapper, EClient):
    """Minimal TWS API client used only to test connectivity."""

    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.received_server_time = threading.Event()

    def nextValidId(self, order_id: int) -> None:
        """Called after a successful API connection."""
        print(f"Connected successfully. Next valid order ID: {order_id}")
        self.reqCurrentTime()

    def currentTime(self, epoch_seconds: int) -> None:
        """Receive and print IBKR server time."""
        server_time = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        print(f"IBKR server time: {server_time.isoformat()}")
        self.received_server_time.set()
        self.disconnect()

    def error(self, *args: object) -> None:
        """Handle API messages across recent IBKR API signatures."""
        print("IBKR message:", *args)


def main() -> None:
    host = "127.0.0.1"
    port = 7497
    client_id = 1

    app = IBKRConnectionTest()

    print(f"Connecting to TWS at {host}:{port}...")
    app.connect(host, port, client_id)

    api_thread = threading.Thread(
        target=app.run,
        name="ibkr-api-thread",
        daemon=True,
    )
    api_thread.start()

    if not app.received_server_time.wait(timeout=10):
        print(
            "No server-time response received within 10 seconds. "
            "Check that paper TWS is open and API socket access is enabled."
        )
        app.disconnect()
        raise SystemExit(1)

    api_thread.join(timeout=2)
    print("Connection test completed successfully.")


if __name__ == "__main__":
    main()