"""
Readable logging setup.

Produces clean, easy-to-interpret logs in both the terminal and a file.
The format prioritizes human reading: time, level and message.
"""

import logging
import sys

import config


def setup_logger(name: str = "benchmark") -> logging.Logger:
    """Create (or reuse) a logger that writes to the terminal and a file."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured, avoid duplicate handlers
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Terminal output.
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File output (full run history).
    file_handler = logging.FileHandler(config.LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# Single logger used across the whole project.
log = setup_logger()
