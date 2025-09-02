"""Logging utilities."""

import logging
from typing import Optional


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger.

    Parameters
    ----------
    verbose:
        When ``True`` sets level to ``DEBUG`` otherwise ``INFO``.
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
