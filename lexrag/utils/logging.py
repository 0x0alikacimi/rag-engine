from __future__ import annotations

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config import settings


_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_initialized: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _initialized:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_FORMATTER)
    logger.addHandler(ch)

    # File handler
    log_file = settings.log_dir / "lexrag.log"
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FORMATTER)
    logger.addHandler(fh)

    logger.propagate = False
    _initialized.add(name)
    return logger
