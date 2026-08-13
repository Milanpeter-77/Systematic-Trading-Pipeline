from __future__ import annotations

from typing import Any

import pandas as pd


def get_latest_bars(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Fetch the latest market data bar(s) needed to compute live signals.

    OPEN DECISION -- NOT YET RESOLVED: how live market data is obtained is
    explicitly undecided and must be resolved before this is implemented.
    Two options under consideration, described neutrally:

      * Poll: bounded historical-data requests in the style of
        src.data.retrieval.fetch_instrument_history(), issued on a fixed
        interval to pull the most recent bar(s).
      * Stream: true real-time subscriptions via IBKR's reqMktData or
        reqRealTimeBars, pushed to the client as they arrive.

    Do not implement either approach until this decision is made.

    Raises:
        NotImplementedError: always -- this is an unimplemented stub.
    """
    raise NotImplementedError(
        "Live market-data retrieval method (poll vs. stream) is undecided."
    )
