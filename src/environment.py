"""
Central home for sensitive/changeable environment variables (API keys,
IBKR host/port/client ids, etc.) -- see .env.example for the full list.

Loads repo-root .env (if present) once, then exposes each setting as a
module-level constant with the same default it used to have as a scattered
hardcoded value, so every caller reads from here instead of redeclaring
its own copy.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

FRED_API_KEY = os.environ.get("FRED_API_KEY")

IBKR_HOST = os.environ.get("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.environ.get("IBKR_PORT", "7497"))  # 7497 = paper TWS. 7496 = live TWS, 4001/4002 = IB Gateway.

# Concurrent IBKR sessions each need a distinct client id -- data ingestion
# and live execution run at the same time, so they get separate defaults.
IBKR_DATA_CLIENT_ID = int(os.environ.get("IBKR_DATA_CLIENT_ID", "2"))
IBKR_EXECUTION_CLIENT_ID = int(os.environ.get("IBKR_EXECUTION_CLIENT_ID", "3"))
