from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS test_events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id  TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    family        TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    tested_at     TEXT NOT NULL,
    layer1_pass   INTEGER,
    layer2_pass   INTEGER,
    layer3_pass   INTEGER,
    layer4_pass   INTEGER,
    final_pass    INTEGER,
    status        TEXT NOT NULL DEFAULT 'completed',
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_test_events_candidate_id ON test_events(candidate_id);
CREATE INDEX IF NOT EXISTS idx_test_events_tested_at ON test_events(tested_at);
"""


@dataclass(frozen=True)
class TestEvent:
    """
    One record of a single candidate having been tested in a specific run.
    """

    candidate_id: str
    run_id: str
    family: str
    symbol: str
    tested_at: str
    layer1_pass: bool | None = None
    layer2_pass: bool | None = None
    layer3_pass: bool | None = None
    layer4_pass: bool | None = None
    final_pass: bool | None = None
    status: str = "completed"
    error_message: str | None = None


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Open (creating if needed) the test-history database and ensure its schema exists.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()

    return conn


def _to_int_or_none(value: bool | None) -> int | None:
    return None if value is None else int(value)


def record_test_events(
    conn: sqlite3.Connection,
    events: list[TestEvent],
) -> None:
    """
    Append a batch of test events in one transaction.

    This is the single writer to the history database -- call it once per
    run, from the parent process only. Worker processes must never call
    this directly.
    """
    if not events:
        return

    rows = [
        (
            event.candidate_id,
            event.run_id,
            event.family,
            event.symbol,
            event.tested_at,
            _to_int_or_none(event.layer1_pass),
            _to_int_or_none(event.layer2_pass),
            _to_int_or_none(event.layer3_pass),
            _to_int_or_none(event.layer4_pass),
            _to_int_or_none(event.final_pass),
            event.status,
            event.error_message,
        )
        for event in events
    ]

    with conn:
        conn.executemany(
            """
            INSERT INTO test_events (
                candidate_id, run_id, family, symbol, tested_at,
                layer1_pass, layer2_pass, layer3_pass, layer4_pass, final_pass,
                status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def get_last_tested_at(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Return the most recent tested_at timestamp per candidate_id, as one
    bulk query -- cheap regardless of how large the history grows.
    """
    frame = pd.read_sql_query(
        """
        SELECT candidate_id, MAX(tested_at) AS last_tested_at
        FROM test_events
        GROUP BY candidate_id
        """,
        conn,
    )

    frame["last_tested_at"] = pd.to_datetime(
        frame["last_tested_at"],
        utc=True,
    )

    return frame


def candidates_due_for_testing(
    conn: sqlite3.Connection,
    candidate_frame: pd.DataFrame,
    cooldown_days: float,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a candidate population into those due for testing and those still
    within their cooldown window.

    A candidate with no prior test_events row is always due. A candidate
    last tested less than `cooldown_days` ago is skipped.
    """
    if candidate_frame.empty:
        return candidate_frame.copy(), candidate_frame.copy()

    now = now if now is not None else pd.Timestamp.now(tz="UTC")

    last_tested_at = get_last_tested_at(conn)

    # A left merge (not Series.map) so this works regardless of the
    # candidate_id column's string backend (numpy object vs. pandas 3.x's
    # default pyarrow-backed string dtype).
    merged = candidate_frame.merge(
        last_tested_at,
        on="candidate_id",
        how="left",
    )

    cooldown_expires_at = merged["last_tested_at"] + pd.Timedelta(
        days=cooldown_days
    )

    is_due = (
        merged["last_tested_at"].isna() | (now >= cooldown_expires_at)
    ).to_numpy()

    due_frame = candidate_frame.loc[is_due].reset_index(drop=True)
    skipped_frame = candidate_frame.loc[~is_due].reset_index(drop=True)

    return due_frame, skipped_frame


def get_test_history(
    conn: sqlite3.Connection,
    candidate_id: str,
) -> pd.DataFrame:
    """
    Return the full test-event history for one candidate, most recent first.
    """
    return pd.read_sql_query(
        """
        SELECT *
        FROM test_events
        WHERE candidate_id = ?
        ORDER BY tested_at DESC
        """,
        conn,
        params=(candidate_id,),
    )
