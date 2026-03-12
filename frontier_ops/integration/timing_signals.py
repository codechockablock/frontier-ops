"""
Timing Signal Engine
=====================

Extracts behavioral rhythm signals from inter-call gaps and execution
durations. Works in log-tail mode with zero params — timing is free.

Architecture (from Claude.ai collaboration 2026-02-26):
1. Log-normal model: log(inter-call gap) as Gaussian per tool type,
   fit online with exponential moving average
2. Infrastructure floor subtraction: min gap per tool = infra baseline,
   deliberation proxy = observed - floor
3. Burst detection: Poisson rate CUSUM over sliding windows
4. Network instability gating: CUSUM on floor variance to suppress
   signals during network blips

Key insight: timing ≠ cognitive load (transformers run fixed forward
passes). It's a BEHAVIORAL RHYTHM indicator — autonomous chains vs
interactive mode. Don't over-interpret.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, Optional


class ExponentialMovingStats:
    """Online EMA for mean and variance of a scalar stream."""

    def __init__(self, alpha: float = 0.1, warmup: int = 5):
        self.alpha = alpha
        self.warmup = warmup
        self.mean = 0.0
        self.var = 0.0
        self.count = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, x: float):
        """Incorporate a new sample, updating the running mean and variance."""
        self.count += 1
        if self.count <= self.warmup:
            # Simple accumulation during warmup
            self._sum += x
            self._sum_sq += x * x
            self.mean = self._sum / self.count
            if self.count > 1:
                self.var = (self._sum_sq / self.count) - self.mean**2
        else:
            # EMA update
            delta = x - self.mean
            self.mean += self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta**2)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 1e-12))

    def z_score(self, x: float) -> float:
        """How many std deviations from mean."""
        if self.count < self.warmup:
            return 0.0
        return abs(x - self.mean) / max(self.std, 1e-6)

    @property
    def ready(self) -> bool:
        return self.count >= self.warmup


class InfraFloor:
    """
    Track the infrastructure floor (minimum gap) per tool type.
    Uses a rolling window to adapt to changing network conditions.
    """

    def __init__(self, window: int = 50):
        self.window = window
        self.values: deque = deque(maxlen=window)

    def update(self, gap: float):
        """Record a new inter-call gap value."""
        self.values.append(gap)

    @property
    def floor(self) -> float:
        if not self.values:
            return 0.0
        return min(self.values)

    @property
    def floor_variance(self) -> float:
        """Variance of the bottom quartile — tracks network stability."""
        if len(self.values) < 4:
            return 0.0
        sorted_vals = sorted(self.values)
        bottom = sorted_vals[: max(1, len(sorted_vals) // 4)]
        if len(bottom) < 2:
            return 0.0
        mean = sum(bottom) / len(bottom)
        return sum((v - mean) ** 2 for v in bottom) / len(bottom)


class BurstDetector:
    """
    Poisson rate CUSUM for detecting tool call bursts.
    Tracks call rate over sliding windows and flags anomalous bursts.
    """

    def __init__(self, window_sec: float = 60.0, expected_rate: float = 0.5):
        self.window_sec = window_sec
        self.expected_rate = expected_rate  # calls per second
        self.timestamps: deque = deque()
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.slack = 0.15  # allowable drift before accumulating

    def update(self, timestamp: float) -> float:
        """Add a timestamp and return the burst score (0-1)."""
        self.timestamps.append(timestamp)

        # Prune old timestamps
        cutoff = timestamp - self.window_sec
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

        # Current rate
        if len(self.timestamps) < 2:
            return 0.0

        duration = self.timestamps[-1] - self.timestamps[0]
        if duration < 0.1:
            return 0.0

        current_rate = len(self.timestamps) / duration

        # CUSUM on rate deviation (with decay to prevent runaway)
        deviation = current_rate - self.expected_rate
        self.cusum_pos = max(0, self.cusum_pos * 0.95 + deviation - self.slack)
        self.cusum_neg = max(0, self.cusum_neg * 0.95 - deviation - self.slack)

        # Normalize to 0-1
        burst_score = min(1.0, max(self.cusum_pos, self.cusum_neg) / 5.0)
        return burst_score

    def update_expected_rate(self, rate: float):
        """Adapt the expected rate from observed baseline."""
        self.expected_rate = rate


class TimingSignalEngine:
    """
    Main timing signal engine. Produces a timing_anomaly score from
    inter-call gaps and execution durations.

    Usage:
        engine = TimingSignalEngine()
        # On each tool call completion:
        result = engine.on_tool_complete(
            tool_name="exec",
            start_ts=1234567890.0,
            end_ts=1234567891.5,
        )
        # result = {"timing_anomaly": 0.12, "burst_score": 0.0, ...}
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        warmup_calls: int = 8,
        burst_window_sec: float = 60.0,
        sigma_threshold: float = 2.5,
    ):
        self.sigma_threshold = sigma_threshold

        # Per-tool-type stats for log(deliberation gap)
        self.gap_stats: Dict[str, ExponentialMovingStats] = defaultdict(
            lambda: ExponentialMovingStats(alpha=ema_alpha, warmup=warmup_calls)
        )
        # Per-tool-type stats for execution duration
        self.duration_stats: Dict[str, ExponentialMovingStats] = defaultdict(
            lambda: ExponentialMovingStats(alpha=ema_alpha, warmup=warmup_calls)
        )
        # Per-tool-type infrastructure floor
        self.infra_floors: Dict[str, InfraFloor] = defaultdict(InfraFloor)
        # Global infrastructure floor (across all tools)
        self.global_infra_floor = InfraFloor(window=100)

        # Burst detector
        self.burst = BurstDetector(window_sec=burst_window_sec)

        # Network stability tracking
        self.floor_variance_stats = ExponentialMovingStats(alpha=0.05, warmup=10)
        self.network_unstable = False

        # State
        self.last_end_ts: Optional[float] = None
        self.last_tool: Optional[str] = None
        self.call_count = 0

    def on_tool_complete(
        self,
        tool_name: str,
        start_ts: float,
        end_ts: float,
    ) -> Dict[str, float]:
        """
        Process a completed tool call.

        Returns dict with:
        - timing_anomaly: 0-1 composite timing anomaly score
        - gap_z: z-score of the inter-call gap
        - duration_z: z-score of execution duration
        - burst_score: 0-1 burst detection score
        - deliberation_ms: estimated deliberation time (gap - floor)
        - network_stable: bool, whether network is stable
        - behavioral_rhythm: "chained" | "interactive" | "autonomous" | "unknown"
        """
        self.call_count += 1
        now = end_ts
        duration = max(0.001, end_ts - start_ts)

        # Track execution duration
        log_duration = math.log(max(duration, 0.001))
        self.duration_stats[tool_name].update(log_duration)
        duration_z = self.duration_stats[tool_name].z_score(log_duration)

        # Inter-call gap
        gap_z = 0.0
        deliberation_ms = 0.0
        if self.last_end_ts is not None:
            raw_gap = max(0.001, start_ts - self.last_end_ts)

            # Update infrastructure floor
            self.infra_floors[tool_name].update(raw_gap)
            self.global_infra_floor.update(raw_gap)

            # Subtract infra floor for deliberation proxy
            floor = self.infra_floors[tool_name].floor
            deliberation = max(0.0, raw_gap - floor)
            deliberation_ms = deliberation * 1000

            # Log-normal model on deliberation residual
            if deliberation > 0.001:
                log_delib = math.log(deliberation)
                self.gap_stats[tool_name].update(log_delib)
                gap_z = self.gap_stats[tool_name].z_score(log_delib)

            # Track floor variance for network stability
            fv = self.infra_floors[tool_name].floor_variance
            self.floor_variance_stats.update(fv)
            floor_var_z = self.floor_variance_stats.z_score(fv)
            self.network_unstable = floor_var_z > 2.0

        # Burst detection
        burst_score = self.burst.update(now)

        # Behavioral rhythm classification
        rhythm = self._classify_rhythm(deliberation_ms)

        # Composite timing anomaly score
        # Gate by network stability — suppress during instability
        stability_gate = 0.3 if self.network_unstable else 1.0

        # Weighted combination
        timing_anomaly = stability_gate * min(
            1.0,
            (
                0.40 * min(gap_z / self.sigma_threshold, 1.0)
                + 0.25 * min(duration_z / self.sigma_threshold, 1.0)
                + 0.35 * burst_score
            ),
        )

        # Update state
        self.last_end_ts = end_ts
        self.last_tool = tool_name

        return {
            "timing_anomaly": round(timing_anomaly, 4),
            "gap_z": round(gap_z, 3),
            "duration_z": round(duration_z, 3),
            "burst_score": round(burst_score, 4),
            "deliberation_ms": round(deliberation_ms, 1),
            "network_stable": not self.network_unstable,
            "behavioral_rhythm": rhythm,
        }

    def _classify_rhythm(self, deliberation_ms: float) -> str:
        """
        Classify the current behavioral rhythm.

        - chained: <500ms deliberation (fast sequential calls)
        - autonomous: 500ms-5s (agent working independently)
        - interactive: >5s (waiting for user or long deliberation)
        - unknown: not enough data
        """
        if self.call_count < 3:
            return "unknown"
        if deliberation_ms < 500:
            return "chained"
        elif deliberation_ms < 5000:
            return "autonomous"
        else:
            return "interactive"

    def get_summary(self) -> Dict:
        """Get a summary of timing state for proprioception output."""
        return {
            "call_count": self.call_count,
            "network_stable": not self.network_unstable,
            "tools_tracked": list(self.gap_stats.keys()),
            "burst_cusum": round(self.burst.cusum_pos, 3),
        }
