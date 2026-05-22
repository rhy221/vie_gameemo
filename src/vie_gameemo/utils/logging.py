"""Centralized logging configuration.

Wraps Python's `logging` module with project-standard formatting,
optional file output, and optional WandB integration.

Example:
    >>> from vie_gameemo.utils.logging import setup_logging, get_logger
    >>> setup_logging(level="INFO", log_file="outputs/logs/run.log")
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started")
"""

import logging
from pathlib import Path

_NOISY_LOGGERS = [
    "transformers",
    "transformers.tokenization_utils",
    "transformers.modeling_utils",
    "urllib3",
    "urllib3.connectionpool",
    "filelock",
    "huggingface_hub",
    "datasets",
]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
    format_str: str | None = None,
) -> None:
    """Configure root logger for the project.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file. Parent dirs created if missing.
        console: Whether to also log to stdout.
        format_str: Custom log format string. If None, uses project default.

    Raises:
        ValueError: If `level` is not a valid logging level.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    fmt = format_str or _DEFAULT_FORMAT
    formatter = logging.Formatter(fmt, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Remove any existing handlers to avoid duplicate output on repeated calls
    root.handlers.clear()

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name, inheriting project config.

    Args:
        name: Logger name, typically `__name__` of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
