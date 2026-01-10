"""Logging utilities for the rewarduq package."""

from __future__ import annotations

import logging
import sys
import warnings

try:
    from accelerate import PartialState
    from accelerate import logging as logging_accelerate

    get_logger = logging_accelerate.get_logger
    PartialState()  # Ensure PartialState is initialized before using get_logger
except ImportError:
    get_logger = logging.getLogger


### SETUP UTILS ###


def setup_logging(
    level: int = logging.INFO,
    format: str | None = None,
    dateformat: str | None = None,
    filterwarnings: list | None = None,
):
    """Configure the logging system for the application."""
    if format is None:
        format = "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)-5d %(message)s"
    if dateformat is None:
        dateformat = "%Y-%m-%d %H:%M:%S"
    if filterwarnings is None:
        filterwarnings = []

    # Configure logging system
    logging.basicConfig(
        stream=sys.stdout,  # Log to stdout instead of stderr to sync with print()
        level=level,
        format=format,
        datefmt=dateformat,
    )
    logging.captureWarnings(True)

    # Suppress certain warnings
    for filter in filterwarnings:
        warnings.filterwarnings(**filter)
