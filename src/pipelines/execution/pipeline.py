from __future__ import annotations

import logging
from pathlib import Path

from src.data.retrieval import DEFAULT_HOST, DEFAULT_PORT
from src.environment import IBKR_EXECUTION_CLIENT_ID
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# IBKR connection settings for the LIVE EXECUTION session.
#
# host/port reuse the same defaults as data ingestion (src.data.retrieval),
# but client_id is DELIBERATELY DIFFERENT (data_ingestion uses 2 -- see
# src.data.retrieval.DEFAULT_CLIENT_ID) because a data-ingestion session and
# a live-execution session may run concurrently, and IBKR requires each
# concurrent connection to use a unique client id.
#
# IBKR_PORT MUST stay defaulted to 7497 (paper trading). 7496 is the LIVE
# trading port. Switching to 7496 must be a conscious, explicit override at
# call time -- never change this default.
# ---------------------------------------------------------------------------
IBKR_HOST = DEFAULT_HOST
IBKR_PORT = DEFAULT_PORT  # 7497 = paper. NEVER default this to 7496 (live).
IBKR_CLIENT_ID = IBKR_EXECUTION_CLIENT_ID  # distinct from data_ingestion's client id (2)


def run_execution_loop() -> None:
    """
    Run the live execution loop (NOT YET IMPLEMENTED).

    Intended end-to-end responsibility once implemented:
      1. Load promoted strategies via
         src.pipelines.execution.strategy_loader.load_deployed_strategies().
      2. Connect a broker session via
         src.pipelines.execution.broker.live_client.LiveTradingClient.
      3. Loop:
         a. Fetch latest market data per instrument via
            src.pipelines.execution.market_data.get_latest_bars()
            (poll-vs-stream is an OPEN decision -- see that module's
            docstring).
         b. For each deployed CandidateSpec, compute the target position via
            src.pipelines.execution.signals.compute_target_position() (which
            MUST reuse src.factory.registry.create_strategy() and
            BaseStrategy.generate_positions() -- the exact code path used
            in backtesting -- never a reimplementation).
         c. Compare target position to the current live position (via the
            broker session).
         d. Run src.pipelines.execution.risk_limits.check_pre_trade_risk()
            and place or adjust orders through the broker session if it
            passes.
         e. Wait for the next interval and repeat.

    Raises:
        NotImplementedError: always -- this is an unimplemented stub.
    """
    raise NotImplementedError(
        "Execution loop is not yet implemented. Live market-data method "
        "(poll vs. stream) is an open decision -- see "
        "src.pipelines.execution.market_data."
    )


def main() -> None:
    """
    Public command-line entry point.

    Performs real, working logging setup (not trading logic), then attempts
    to start the execution loop, which currently always raises
    NotImplementedError.
    """
    log_file_path = configure_logging(PROJECT_ROOT)
    logger.info(f"Logging to {log_file_path}")
    logger.info(
        "Execution engine is not yet implemented -- this will raise "
        "NotImplementedError."
    )

    run_execution_loop()


if __name__ == "__main__":
    main()
