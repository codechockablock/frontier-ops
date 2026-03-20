# Calibration Report: Market Signal Thresholds
## frontier-ops v0.2.0
**Date:** 2026-03-20
**Dataset:** 330,051 telemetry entries, 5,455 sessions, 475.5 hours
**Status:** Calibration complete. D signal requires redesign.

---

## Dataset Characteristics

| Metric | Value |
|--------|-------|
| Total entries | 330,051 |
| Total sessions | 5,455 |
| Benign sessions (0 BLOCK, <10% FLAG) | 3,070 (56%) |
| Suspicious sessions | 2,385 (44%) |
| Sessions with ≥5 steps | 2,384 (benign: 687, suspicious: 1,697) |
| Time span | 475.5 hours |
| Unique action types | 11 |
| Unique scopes | 9 |

### Signal Schema (per entry)
```
step, ts, tool, type, scope, src, verdict, conf, signals{error, fisher, cross_slot, persistence, cusum}, ctx, mag
```

### Verdict Distribution
- PASS: 30,724 (9.3%)
- MONITOR: 160,244 (48.6%)
- FLAG: 120,812 (36.6%)
- BLOCK: 18,271 (5.5%)

---

## Finding 1: D_raw (Geodesic Ratio) Is Not Discriminative

### The Problem

D_raw as implemented measures PathLength_G / DirectDistance_G in behavioral space.
On the 330K dataset, D_raw produces:

- **Benign sessions median:** 7.08
- **Suspicious sessions median:** 9.60
- **Overlap:** Complete (benign p95 = 88.19, suspicious p5 = 4.35)

CUSUM over D_raw produces 25-49% FPR even at threshold=15.0.

### Root Cause

The telemetry produces only **16 unique behavioral vectors** when mapped to the
6D categorical behavioral space. Normal coding sessions involve rapid switching
between file_read → shell_exec → file_write → file_read, which creates large
metric-weighted path lengths even on perfectly benign behavior. The constitutional
metric amplifies certain transitions (e.g., scope escalation), making D_raw
enormous for routine multi-tool sessions.

D_raw works in theory for continuous behavioral embeddings (the original paper's
assumption). It fails when the behavioral space is effectively discrete with
high inter-cluster distances.

### Recommendation

D_raw should operate on the **raw continuous signals** (error, fisher, cross_slot,
persistence, cusum) rather than categorical action-type mappings. However, even
on the 5D normalized signal space, D_raw still shows complete overlap between
benign and suspicious sessions.

**The geodesic ratio is the wrong signal for this data.**

The telemetry already contains a rich set of continuous signals that DO discriminate.
D should be replaced with a signal derived from the features that actually separate
benign from suspicious behavior (see Finding 3).

---

## Finding 2: S_raw (Action Rate Deviation) Has Structural Issues

### The Data

Benign inter-action intervals:
- **Median:** 6.3s
- **Mean:** 633.5s (heavily right-skewed by long pauses between sessions)
- **P95:** 1,789s

The distribution is bimodal:
1. **Active bursts:** 0.01-2s intervals (tool calls within a reasoning chain)
2. **Idle gaps:** 30s-60min+ (waiting for user, heartbeat intervals)

### Calibration Result

- **Baseline rate (median):** 0.159 actions/sec (1 action every 6.3s)
- **S_raw distribution:** Heavily negative (mean = -417) because windowed rate
  estimation produces enormous observed rates during burst periods

### The Problem

The current S signal measures deviation from a single global baseline rate.
But action rate is highly non-stationary: coding bursts produce 10-50 actions/sec,
while idle periods produce 0 actions for minutes. A single baseline can't
capture this bimodality.

### Recommendation

S should use **per-regime baselines** from the HMM task state, not a global rate.
Alternatively, use percentile-based anomaly detection: flag when the inter-action
interval exceeds the 95th percentile of the windowed distribution (stagnation)
or drops below the 5th percentile (suspicious burst).

---

## Finding 3: Features That Actually Discriminate

Feature separation analysis (Mann-Whitney AUC + Cohen's d) reveals the signals
that genuinely separate benign from suspicious sessions:

### Tier 1: Strong Discriminators (|d| > 1.0)

| Feature | Benign Median | Suspicious Median | Cohen's d | AUC |
|---------|--------------|-------------------|-----------|-----|
| max_scope | 0.30 | 0.80 | 2.06 | 0.23 |
| cross_slot_p95 | 0.00 | 0.96 | 2.04 | 0.46 |
| persistence_p95 | 0.40 | 1.00 | 1.46 | 0.36 |

### Tier 2: Moderate Discriminators (|d| > 0.5)

| Feature | Benign Median | Suspicious Median | Cohen's d |
|---------|--------------|-------------------|-----------|
| cross_slot_std | 0.17 | 0.29 | 0.83 |
| type_diversity | 2 | 3 | 0.83 |
| scope_escalation | 0.20 | 0.40 | 0.74 |
| cusum_max | 0.76 | 3.18 | 0.65 |
| cusum_mean | 0.14 | 1.29 | 0.60 |
| persistence_diff_mean | 0.024 | 0.010 | -0.52 |

### Interpretation

The strongest discriminators are:
1. **Scope escalation** (max_scope, scope_escalation): Suspicious sessions access
   higher-privilege scopes (credential_access, write_system, destructive)
2. **Cross-slot binding** (cross_slot_p95, cross_slot_std): Suspicious sessions show
   elevated cross-slot signals — they combine action types in unusual patterns
3. **Persistence** (persistence_p95, persistence_max): Suspicious sessions have
   high persistence scores — they repeat similar patterns longer
4. **CUSUM** (cusum_max, cusum_mean): The existing sidecar CUSUM already captures
   meaningful cumulative drift

### Key Insight

The existing sidecar already computes the right signals. The market architecture
should compose these signals rather than computing new ones from scratch.

---

## Calibration Parameters

### For the CUSUM accumulator (used by both D and S)

Based on the existing `cusum` signal distribution in benign sessions:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Benign cusum mean | 3.16 | Empirical from 59K benign entries |
| Benign cusum std | 1.80 | |
| Benign cusum p95 | 5.97 | |
| Benign cusum p99 | 5.97 | Ceiling hit |
| Benign cusum max | 5.97 | |

These suggest the existing sidecar's CUSUM threshold is well-calibrated for
the raw signal. For the market DAS-CUSUM operating on higher-level features:

| DAS-CUSUM Parameter | Recommended Value | Rationale |
|---------------------|-------------------|-----------|
| threshold | 8.0 | > benign p99 (5.97) with margin |
| drift | 0.3 | Conservative allowance |
| window_size | 50 | ~50 actions covers a typical reasoning chain |
| decay | 0.98 | From runaway prevention analysis |
| ceiling | 30.0 | Structural safety cap |

### For the S signal (action rate)

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| baseline_rate | 0.159 actions/sec | Median inter-action interval |
| rate_window | 10 | ~1 minute of actions at median rate |
| stagnation_threshold | 60s | Interval > p90 of benign |
| burst_threshold | 0.1s | Interval < p5 of benign |

### For entropy antitrust

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| entropy_floor | 0.5 | Conservative start |
| critical_threshold | 20.0 | Well above max benign CUSUM (5.97) |
| monopoly_health_threshold | 0.3 | Standard diversity floor |
| sustained_readings | 5 | Prevents single-spike false alarms |

---

## Recommendations for Architecture

### Replace D Signal

The geodesic ratio is not discriminative on real data. Replace with a
composite signal derived from the Tier 1 discriminators:

```python
D_composite = w1 * scope_escalation + w2 * cross_slot_p95 + w3 * persistence_p95
```

This directly measures what the data says matters: privilege escalation,
unusual action combinations, and behavioral persistence.

Feed this composite through the existing DAS-CUSUM for trajectory-level detection.

### Restructure S Signal

Replace single-baseline rate deviation with:
1. Percentile-based anomaly detection on inter-action intervals
2. Per-regime baseline from HMM task state when available
3. Burst detection (interval < 0.1s sustained for > 5 steps)

### Compose Existing Sidecar Signals

The sidecar already computes error, fisher, cross_slot, persistence, and cusum.
The market architecture should consume these signals and add trajectory-level
sequential detection (CUSUM/SPRT) rather than recomputing from scratch.

---

## Next Steps

1. Implement D_composite using calibrated Tier 1 features
2. Wire sidecar signals into the market gate as additional inputs
3. Re-run ARL analysis on D_composite to verify ARL₀ > 1000
4. Implement per-regime S baselines once HMM states stabilize
5. Cross-validate: hold out 20% of sessions for validation
