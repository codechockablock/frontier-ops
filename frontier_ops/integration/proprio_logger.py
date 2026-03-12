"""
Structured logging for the proprioceptive wrapper.

All modules share a single 'proprio' logger that writes to stderr.
Non-invasive: logging failures never propagate. The agent is unaffected
even if this entire subsystem fails.

Usage:
    from frontier_ops.integration.proprio_logger import logger
    logger.warning("something happened: %s", detail)
"""

import logging
import sys

logger = logging.getLogger("proprio")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("[proprio] %(levelname)s %(name)s.%(funcName)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)
