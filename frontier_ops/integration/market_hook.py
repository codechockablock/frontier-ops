"""
Market Hook — Integration with OpenClaw Sidecar
=================================================

Wires the proprioceptive market architecture into the existing sidecar
pipeline. Intercepts per-step verdicts and timestamps, feeds them to the
MarketGate, and records evaluations to the governance audit chain.

Integration point:
    ProprioceptionManager.update() → verdict, timestamp
                                    ↓
    MarketHook.on_step() → SeveritySignal + IntervalAnomalySignal
                          → MarketGate.evaluate()
                          → MarketAuditChain.record()
                          → qualitative label (or None)
                          → writes market_state to disk

The hook is completely non-invasive. If it crashes, the sidecar continues
operating normally — market evaluation is advisory-only.

Usage:
    from frontier_ops.integration.market_hook import MarketHook

    hook = MarketHook.from_telemetry(
        telemetry_path="~/.openclaw/workspace/proprioception-log.jsonl"
    )
    # After each sidecar step:
    market_label = hook.on_step(verdict="FLAG", timestamp=time.time())
    # market_label is None (nominal) or a qualitative string

See CORRECTNESS_SPEC.md §5 for the qualitative-only constraint.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from frontier_ops.sensing.market_signals import SeveritySignal, IntervalAnomalySignal
from frontier_ops.sensing.market_gate import MarketGate, MarketSignalState
from frontier_ops.sensing.market_entropy import (
    MarketHealthMonitor,
)
from frontier_ops.governance.market_audit import MarketAuditChain

__all__ = ["MarketHook"]


# Default paths
_DEFAULT_STATE_PATH = os.path.expanduser(
    "~/.openclaw/workspace/market_state.json"
)
_DEFAULT_LOG_PATH = os.path.expanduser(
    "~/.openclaw/workspace/proprioception-log.jsonl"
)


class MarketHook:
    """
    Non-invasive hook that wires market signals into the sidecar pipeline.

    Lifecycle:
      1. Construct with from_telemetry() or from_intervals() for calibration
      2. Call on_step(verdict, timestamp) after each sidecar evaluation
      3. Read get_state() for current market snapshot
      4. Read get_label() for the last qualitative label (or None)

    All market evaluation results are signed into a tamper-evident audit chain.
    Market state is written to disk for agent consumption (qualitative only).
    """

    def __init__(
        self,
        d_signal: SeveritySignal,
        s_signal: IntervalAnomalySignal,
        audit_chain: Optional[MarketAuditChain] = None,
        health_monitor: Optional[MarketHealthMonitor] = None,
        state_path: str = _DEFAULT_STATE_PATH,
        verbose: bool = False,
    ):
        self._audit = audit_chain or MarketAuditChain()
        self._health = health_monitor or MarketHealthMonitor()
        self._gate = MarketGate(d_signal, s_signal, audit_chain=self._audit)
        self._state_path = state_path
        self._verbose = verbose
        self._last_label: Optional[str] = None
        self._n_steps: int = 0
        self._n_alerts: int = 0

    @classmethod
    def from_telemetry(
        cls,
        telemetry_path: str = _DEFAULT_LOG_PATH,
        max_calibration_entries: int = 50000,
        severity_window: int = 10,
        d_cusum_params: Optional[Dict] = None,
        s_cusum_params: Optional[Dict] = None,
        state_path: str = _DEFAULT_STATE_PATH,
        verbose: bool = False,
    ) -> "MarketHook":
        """
        Construct a MarketHook calibrated from existing telemetry.

        Reads the proprioception-log.jsonl, extracts PASS-verdict inter-action
        intervals as the benign baseline for the S signal.

        Falls back to default intervals if telemetry is unavailable.
        """
        telemetry_path = os.path.expanduser(telemetry_path)
        benign_intervals = cls._extract_benign_intervals(
            telemetry_path, max_entries=max_calibration_entries,
        )

        d_signal = SeveritySignal(
            severity_window=severity_window,
            cusum_params=d_cusum_params,
        )
        s_signal = IntervalAnomalySignal(cusum_params=s_cusum_params)

        if len(benign_intervals) >= 10:
            s_signal.calibrate(benign_intervals)
            if verbose:
                median = np.median(benign_intervals)
                print(
                    f"[market] Calibrated from {len(benign_intervals)} benign intervals "
                    f"(median={median:.2f}s)",
                    file=__import__('sys').stderr,
                )
        else:
            # Fallback: synthetic benign intervals (exponential, median ~6s)
            rng = np.random.default_rng(42)
            s_signal.calibrate(sorted(rng.exponential(scale=8.0, size=500)))
            if verbose:
                print(
                    "[market] Using synthetic calibration (no telemetry found)",
                    file=__import__('sys').stderr,
                )

        return cls(
            d_signal=d_signal,
            s_signal=s_signal,
            state_path=state_path,
            verbose=verbose,
        )

    @classmethod
    def from_intervals(
        cls,
        benign_intervals: List[float],
        severity_window: int = 10,
        d_cusum_params: Optional[Dict] = None,
        s_cusum_params: Optional[Dict] = None,
        state_path: str = _DEFAULT_STATE_PATH,
        verbose: bool = False,
    ) -> "MarketHook":
        """Construct with explicit benign interval calibration data."""
        d_signal = SeveritySignal(
            severity_window=severity_window,
            cusum_params=d_cusum_params,
        )
        s_signal = IntervalAnomalySignal(cusum_params=s_cusum_params)
        s_signal.calibrate(benign_intervals)

        return cls(
            d_signal=d_signal,
            s_signal=s_signal,
            state_path=state_path,
            verbose=verbose,
        )

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    def on_step(
        self,
        verdict: str,
        timestamp: Optional[float] = None,
        hmm_state: str = "unknown",
        e_value: float = 0.0,
    ) -> Optional[str]:
        """
        Process one sidecar step. Returns qualitative label or None.

        Args:
            verdict: Sidecar verdict — "pass", "monitor", "flag", or "block".
            timestamp: Action timestamp (defaults to now).
            hmm_state: HMM task state from sidecar (for cold-start detection).
            e_value: Cumulative e-value from sidecar (for cold-start detection).

        Returns:
            None if all market signals are nominal.
            A qualitative label string if any signal is elevated.
        """
        if timestamp is None:
            timestamp = time.time()

        self._n_steps += 1

        # Feed to market gate (atomic D.step + S.step + audit)
        label = self._gate.evaluate(verdict, timestamp, hmm_state=hmm_state, e_value=e_value)
        self._last_label = label

        if label is not None:
            self._n_alerts += 1

        # Update market health monitor
        state = self._gate.get_state()
        sv = state.signal_vector
        total = float(np.sum(np.abs(sv)))
        if total > 1e-12:
            self._health.update(sv)

        # Write state to disk (non-blocking, best-effort)
        try:
            self._write_state(state, label)
        except Exception:
            pass  # Non-invasive: never crash the sidecar

        if self._verbose and label is not None:
            print(
                f"[market] step={self._n_steps} verdict={verdict} "
                f"d_raw={state.d_raw:.2f} s_raw={state.s_raw:.2f} "
                f"label={label!r}",
                file=__import__('sys').stderr,
            )

        return label

    # -------------------------------------------------------------------
    # State access
    # -------------------------------------------------------------------

    def get_state(self) -> MarketSignalState:
        """Current market signal state (for logging, NOT for agent)."""
        return self._gate.get_state()

    def get_label(self) -> Optional[str]:
        """Last emitted label (None if last step was nominal)."""
        return self._last_label

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def n_alerts(self) -> int:
        return self._n_alerts

    @property
    def market_health_status(self) -> str:
        """Current market health: 'healthy', 'monopoly', or 'collapse'."""
        return self._health.status

    @property
    def audit_chain_length(self) -> int:
        return len(self._audit)

    def verify_chain(self):
        """Verify the audit chain integrity. Returns VerificationResult."""
        return self._audit.verify()

    # -------------------------------------------------------------------
    # Disk I/O
    # -------------------------------------------------------------------

    def _write_state(self, state: MarketSignalState, label: Optional[str]) -> None:
        """Write market state to disk (JSON). Best-effort, non-blocking."""
        # Qualitative-only output for agent consumption
        output = {
            "market_active": True,
            "any_elevated": state.any_elevated,
            "label": label,  # None or qualitative string
            "health_status": self._health.status,
            "n_evaluations": self._gate.n_evaluations,
            "n_alerts": self._n_alerts,
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime()
            ),
        }

        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(output, f, separators=(",", ":"))
        os.replace(tmp, self._state_path)

    # -------------------------------------------------------------------
    # Calibration data extraction
    # -------------------------------------------------------------------

    @staticmethod
    def _extract_benign_intervals(
        log_path: str,
        max_entries: int = 50000,
    ) -> List[float]:
        """
        Extract inter-action intervals from PASS-verdict entries in the
        proprioception log. Returns sorted list of intervals in seconds.
        """
        if not os.path.exists(log_path):
            return []

        pass_timestamps: List[float] = []
        try:
            with open(log_path) as f:
                for i, line in enumerate(f):
                    if i >= max_entries:
                        break
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("verdict") == "PASS":
                        pass_timestamps.append(float(entry["ts"]))
        except (OSError, KeyError):
            return []

        if len(pass_timestamps) < 11:
            return []

        pass_timestamps.sort()
        intervals = np.diff(pass_timestamps)
        intervals = intervals[(intervals > 0.01) & (intervals < 86400)]  # 10ms to 24h
        return sorted(intervals.tolist())
