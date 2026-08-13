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
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
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

    log_directory = project_root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(
        LOG_FILENAME_TIMESTAMP_FORMAT
    )

    log_file_path = (
        log_directory / f"pipeline_{timestamp}.log"
    )

    root_logger.setLevel(logging.DEBUG)

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

    root_logger._configured_log_file = log_file_path

    return log_file_path
