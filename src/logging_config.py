from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


CONSOLE_FORMAT = "%(levelname)s: %(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILENAME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def configure_logging(
    project_root: Path,
    pipeline_name: str,
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
) -> Path:
    """
    Configure the root logger with a console handler and a per-run file handler.

    Safe to call multiple times in one process: only the first call attaches
    handlers, later calls just return the already-configured log file path.
    """
    root_logger = logging.getLogger()

    existing_log_file = getattr(root_logger, "_configured_log_file", None)

    if existing_log_file is not None:
        return existing_log_file

    log_directory = project_root / "logs" / pipeline_name
    log_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(
        LOG_FILENAME_TIMESTAMP_FORMAT
    )

    log_file_path = (
        log_directory / f"{timestamp}.log"
    )

    root_logger.setLevel(min(console_level, file_level))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT)
    )

    file_handler = logging.FileHandler(
        log_file_path,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(
            FILE_FORMAT,
            datefmt=FILE_DATE_FORMAT,
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # ibapi logs every request/response at INFO, including the full
    # historical-bar payload of each answer -- left alone, a single
    # data-ingestion run turns into tens of thousands of lines of wire-
    # protocol noise. WARNING+ still surfaces any real ibapi problem.
    logging.getLogger("ibapi").setLevel(logging.WARNING)

    root_logger._configured_log_file = log_file_path
    root_logger._configured_run_id = timestamp

    return log_file_path


def new_run_id() -> str:
    """
    Generate a fresh run id for one pipeline pass, independent of the log
    file (which is configured once via configure_logging() and persists
    for the whole process, even across many passes of a long-running loop).
    """
    run_id = datetime.now().strftime(LOG_FILENAME_TIMESTAMP_FORMAT)
    logging.getLogger()._configured_run_id = run_id

    return run_id


def get_current_run_id() -> str:
    """
    Return this process's run id -- the same YYYYMMDD_HHMMSS timestamp
    embedded in the current log filename (see configure_logging()), so a
    structured test-history row (src.factory.test_history) can be
    cross-referenced back to its free-text log file for free.

    Falls back to a fresh timestamp if configure_logging() hasn't been
    called yet in this process (e.g. a standalone/test invocation).
    """
    root_logger = logging.getLogger()

    run_id = getattr(root_logger, "_configured_run_id", None)

    if run_id is not None:
        return run_id

    return datetime.now().strftime(LOG_FILENAME_TIMESTAMP_FORMAT)
