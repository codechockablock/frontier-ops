"""
Cold-Start Detection and Suppression
======================================

Detects when the sidecar is in a cold-start / warmup phase after a restart
and suppresses market alerts that would otherwise produce a BLOCK storm.

Observed pattern (2026-03-20, session openclaw-20260320):
  - HMM state stuck in "initializing" for extended stretches
  - e_values growing exponentially (quintillions)
  - All verdicts BLOCK regardless of action type
  - ~95+ consecutive BLOCKs before the HMM stabilizes

Detection signals:
  1. HMM in "initializing" state (dominant in recent window)
  2. BLOCK rate > threshold in sliding window
  3. Monotonically increasing e_values (optional confirmation)

Recovery:
  - BLOCK rate drops below recovery threshold
  - HMM leaves "initializing" (< 50% of recent window)

Emits a single notification on cold-start entry and on recovery.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["ColdStartDetector", "ColdStartEvent"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColdStartEvent:
    """Emitted once on cold-start detection and once on recovery."""
    event_type: str  # "cold_start_detected" | "cold_start_recovered"
    timestamp: float
    step: int
    block_rate: float
    initializing_rate: float
    message: str


@dataclass
class _StepRecord:
    verdict: str
    hmm_state: str
    e_value: float
    timestamp: float


class ColdStartDetector:
    """
    Identifies when the sidecar is in cold-start mode and signals
    the MarketGate to suppress alerts.

    Parameters
    ----------
    window_size : int
        Number of recent steps to consider for rate calculations.
    block_rate_threshold : float
        BLOCK rate above this in the window triggers cold-start detection.
    recovery_block_rate : float
        BLOCK rate must drop below this to recover from cold-start.
    initializing_threshold : float
        Fraction of window steps in "initializing" HMM state to confirm
        cold-start.
    min_steps_before_detection : int
        Minimum steps observed before cold-start can be declared
        (avoids false positive on the very first step).
    """

    def __init__(
        self,
        window_size: int = 20,
        block_rate_threshold: float = 0.80,
        recovery_block_rate: float = 0.50,
        initializing_threshold: float = 0.50,
        min_steps_before_detection: int = 5,
    ):
        self._window_size = window_size
        self._block_rate_threshold = block_rate_threshold
        self._recovery_block_rate = recovery_block_rate
        self._initializing_threshold = initializing_threshold
        self._min_steps = min_steps_before_detection

        self._window: deque[_StepRecord] = deque(maxlen=window_size)
        self._suppressing: bool = False
        self._step_count: int = 0
        self._cold_start_step: Optional[int] = None
        self._notification_emitted: bool = False
        self._recovery_emitted: bool = False

    # -- Public API ----------------------------------------------------------

    @property
    def is_suppressing(self) -> bool:
        """True while cold-start suppression is active."""
        return self._suppressing

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def cold_start_duration(self) -> Optional[int]:
        """Number of steps since cold-start was detected, or None."""
        if self._cold_start_step is None:
            return None
        return self._step_count - self._cold_start_step

    def observe(
        self,
        verdict: str,
        hmm_state: str,
        e_value: float,
        timestamp: Optional[float] = None,
    ) -> Optional[ColdStartEvent]:
        """
        Feed one sidecar observation.

        Returns a ColdStartEvent on state transitions (detection / recovery),
        otherwise None.
        """
        self._step_count += 1
        ts = timestamp or time.time()

        self._window.append(_StepRecord(
            verdict=verdict,
            hmm_state=hmm_state,
            e_value=e_value,
            timestamp=ts,
        ))

        block_rate = self._block_rate()
        init_rate = self._initializing_rate()

        if not self._suppressing:
            return self._check_enter_cold_start(block_rate, init_rate, ts)
        else:
            return self._check_recovery(block_rate, init_rate, ts)

    def reset(self) -> None:
        """Reset detector state (e.g. on session boundary)."""
        self._window.clear()
        self._suppressing = False
        self._step_count = 0
        self._cold_start_step = None
        self._notification_emitted = False
        self._recovery_emitted = False

    # -- Internal ------------------------------------------------------------

    def _block_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(1 for r in self._window if r.verdict == "block") / len(self._window)

    def _initializing_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(1 for r in self._window if r.hmm_state == "initializing") / len(self._window)

    def _check_enter_cold_start(
        self, block_rate: float, init_rate: float, ts: float
    ) -> Optional[ColdStartEvent]:
        if self._step_count < self._min_steps:
            return None

        if (
            block_rate >= self._block_rate_threshold
            and init_rate >= self._initializing_threshold
        ):
            self._suppressing = True
            self._cold_start_step = self._step_count
            self._notification_emitted = True
            self._recovery_emitted = False

            msg = (
                f"Cold start detected at step {self._step_count}: "
                f"BLOCK rate {block_rate:.0%}, "
                f"initializing rate {init_rate:.0%}. "
                f"Suppressing market alerts."
            )
            logger.warning(msg)

            return ColdStartEvent(
                event_type="cold_start_detected",
                timestamp=ts,
                step=self._step_count,
                block_rate=block_rate,
                initializing_rate=init_rate,
                message=msg,
            )
        return None

    def _check_recovery(
        self, block_rate: float, init_rate: float, ts: float
    ) -> Optional[ColdStartEvent]:
        if (
            block_rate < self._recovery_block_rate
            and init_rate < self._initializing_threshold
        ):
            self._suppressing = False
            duration = self._step_count - (self._cold_start_step or 0)

            msg = (
                f"Cold start recovered at step {self._step_count} "
                f"(duration: {duration} steps). "
                f"BLOCK rate {block_rate:.0%}, "
                f"initializing rate {init_rate:.0%}. "
                f"Resuming market alerts."
            )
            logger.info(msg)

            self._cold_start_step = None
            self._recovery_emitted = True

            return ColdStartEvent(
                event_type="cold_start_recovered",
                timestamp=ts,
                step=self._step_count,
                block_rate=block_rate,
                initializing_rate=init_rate,
                message=msg,
            )
        return None
