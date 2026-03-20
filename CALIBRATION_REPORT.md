# Calibration Report: Market Signal Thresholds
## frontier-ops v0.2.0
**Date:** 2026-03-20
**Status:** Calibration complete. Architecture pivot required.

---

## Data Sources

| Source | Entries | Sessions | Signals | Period |
|--------|---------|----------|---------|--------|
| Old sidecar log | 330,051 | 5,455 | 5 (error, fisher, cross_slot, persistence, cusum) | ~475h |
| frontier-ops observations | 538 | 2 | 6D behavioral + efference copy + detection + governance | 2 days |
| Deep suite (labeled) | 24 | 24 | spike, drift, health, alerts (12 benign, 12 adversarial) | 1 run |

---

## Finding 1: Geodesic Ratio (D_raw) Is Fundamentally Wrong for This Data

**D_raw = PathLength_G / DirectDistance_G** assumes continuous behavioral embeddings.

Reality: The old sidecar produces **16 unique behavioral vectors**. The frontier-ops
pipeline produces **94 unique vectors** from 538 entries. Both are effectively
categorical — each tool call maps to one of a finite set of behavioral archetypes.

Normal coding sessions rapidly switch between these archetypes (read → exec → write → read),
producing enormous path-to-geodesic ratios on purely benign behavior.

**Result on 330K dataset:** D_raw median = 7.08 (benign) vs 9.60 (suspicious). Complete overlap.
CUSUM over D_raw: 25-49% FPR even at threshold=15.

**Result on 538 frontier-ops entries:** ARL = 2-4 steps on PASS-verdict data at all thresholds tested.

**Verdict: D_raw as geodesic ratio is not recoverable. The signal definition must change.**

---

## Finding 2: The Existing Sidecar Already Computes Good Signals

The frontier-ops pipeline computes per-step:
- `fisher_divergence` — Fisher information divergence from baseline
- `baseline_error` — Efference copy prediction error (G-weighted)
- `intent_adjusted_error` — Error adjusted for user intent alignment
- `cusum` — Running CUSUM accumulator
- `mmd_window` — Maximum mean discrepancy over sliding window
- `e_value` — Sequential e-value for testing
- `conjunction_flags` — Binary flags for specific threat patterns (exfiltration, C2, etc.)

**Verdict-conditioned distributions (frontier-ops, N=538):**

| Signal | PASS mean | FLAG mean | BLOCK mean | Separation |
|--------|-----------|-----------|------------|------------|
| fisher_divergence | 0.014 | 0.632 | 0.728 | 45× between PASS/FLAG |
| baseline_error | 0.046 | 0.411 | 0.306 | 9× between PASS/FLAG |
| cusum (existing) | 0.122 | 0.600 | 0.170 | 5× between PASS/FLAG |
| mmd_window | 0.250 | 0.372 | 0.465 | 1.5× (weak) |

**Conjunction flags are perfect classifiers:** Zero false positives on PASS data,
all flags occur exclusively on BLOCK verdicts.

**Deep suite confirms (91.7% accuracy):**
- max_spike threshold=3.0 → 91.7% accuracy
- alerts threshold=14 → 91.7% accuracy
- health threshold=0.70 → 87.5% accuracy

---

## Finding 3: DAS-CUSUM Over Per-Step Signals Doesn't Work

Every candidate signal produces unacceptable FPR when fed raw into DAS-CUSUM:

| Signal | ARL₀ (thr=5) | FPR (PASS) | TPR (BLOCK) |
|--------|-------------|------------|-------------|
| fisher_divergence | 3 | 0.347 | 0.487 |
| baseline_error | 2 | 0.458 | 0.333 |
| intent_adjusted_error | 3 | 0.322 | 0.372 |
| cusum_existing | 4 | 0.263 | 0.436 |
| mmd_window | 1 | 0.898 | 0.282 |
| composite_v1 | 1 | 0.716 | 0.487 |

**Root cause:** Tool-type switching in normal coding produces spiky signals.
Every read→exec→write transition creates a prediction error spike that looks
identical to a genuine behavioral anomaly at the per-step level.

DAS-CUSUM is designed for shift detection in approximately stationary processes.
The per-step signals are categorically non-stationary — they're a Markov chain
jumping between discrete states.

---

## Finding 4: The Right Architecture

The market architecture should operate at a **different timescale** than the sidecar.

### What the sidecar does well (keep unchanged):
- Per-step spike detection (Mahalanobis step detector)
- Per-step efference copy prediction error
- Per-step conjunction flag detection
- Per-step verdict (PASS/MONITOR/FLAG/BLOCK)

### What the market adds (new value):
- **Trajectory-level pattern detection**: sustained elevation over N steps, not single spikes
- **Formal clearance**: SPRT can affirmatively ACCEPT benign behavior after enough evidence
- **Multi-signal integration**: entropy-based antitrust prevents single-signal dominance
- **Tamper-evident audit**: every evaluation signed into governance chain

### The D signal should be:

**D = DAS-CUSUM over windowed verdict severity score**

Map sidecar verdicts to a severity score:
```
PASS = 0, MONITOR = 1, FLAG = 2, BLOCK = 3
```

Compute a rolling severity over a window of N recent verdicts.
Feed the windowed severity into DAS-CUSUM to detect sustained elevation.

This works because:
1. The sidecar already makes good per-step decisions
2. A single FLAG is normal (task switching, tool burst)
3. *Sustained* FLAG or BLOCK is the real signal (prolonged anomalous behavior)
4. DAS-CUSUM on severity is approximately stationary within behavioral regimes

### The S signal should be:

**S = DAS-CUSUM over inter-action interval percentile**

Instead of raw rate deviation, compute where each interval falls in the
empirical CDF of benign intervals. Feed the percentile (0-1) into DAS-CUSUM.

This normalizes the bimodal distribution (burst vs idle) into a uniform [0,1] signal.
Sustained high percentiles (long gaps) = stagnation. Sustained low percentiles
(rapid fire) = suspicious burst.

---

## Calibrated Parameters

### D Signal (Verdict Severity)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| severity_map | {PASS:0, MONITOR:1, FLAG:2, BLOCK:3} | Ordinal verdict encoding |
| severity_window | 10 | ~10 actions ≈ 1 reasoning chain |
| DAS-CUSUM threshold | 5.0 | Tuned below |
| DAS-CUSUM drift | 0.5 | Standard |
| DAS-CUSUM window | 30 | 30-step reference |
| DAS-CUSUM decay | 0.98 | Runaway prevention |
| DAS-CUSUM ceiling | 30.0 | Structural safety |

Rationale for threshold=5.0: Benign sessions average severity ~0.5 (mix of PASS
and MONITOR). Sustained FLAG/BLOCK pushes windowed severity to 2-3. After
standardization in DAS-CUSUM, a sustained shift from 0.5 to 2.5 produces
z-scores of ~4-5σ, triggering the threshold.

### S Signal (Interval Percentile)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| benign_interval_cdf | Empirical from PASS verdicts | Non-parametric |
| baseline_rate | 0.164 actions/sec | Median interval = 6.1s |
| stagnation_p95 | 193s | Benign p95 interval |
| DAS-CUSUM threshold | 5.0 | Symmetric with D |
| DAS-CUSUM drift | 0.5 | Standard |
| rate_window | 10 | 10 recent intervals |

### Entropy Antitrust

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| entropy_floor | 0.5 | Standard diversity target |
| critical_threshold | 3.0 | Severity score of 3 = BLOCK |
| monopoly_health_threshold | 0.3 | Standard |
| sustained_readings | 5 | Prevent single-spike false alarms |

### SPRT Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| alpha (Type I) | 0.05 | 5% false alarm tolerance |
| beta (Type II) | 0.10 | 10% missed detection tolerance |
| LLR computation | (severity - benign_mean) / benign_std | Standardized severity as LLR proxy |

---

## Validation Plan

### Cross-validation (to be implemented):

1. **Train/test split:** 70% of PASS-verdict sessions for calibration, 30% for validation
2. **ARL₀ target:** > 100 steps on validation PASS data (not just training data)
3. **Detection delay target:** < 20 steps to detect sustained FLAG/BLOCK elevation
4. **SPRT clearance time:** < 50 steps to ACCEPT benign behavior
5. **End-to-end test:** Replay deep suite traces through market pipeline, verify ≥ 87.5% accuracy

### Integration test with live sidecar:

Run the market pipeline alongside the sidecar on the next Chocka session.
Compare market verdicts with sidecar verdicts. Log divergences for analysis.
