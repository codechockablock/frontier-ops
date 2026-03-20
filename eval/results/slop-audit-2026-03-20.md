# AI Slop Audit — frontier_ops

**Date:** 2026-03-20
**Scope:** 70 .py files, ~19,117 lines
**Codebase:** AI agent governance middleware

---

## Overall Slop Score: 3.5 / 10

This is not slop. The codebase has a coherent mathematical framework (VSA/phasor algebra, Riemannian geometry on concept spaces, CUSUM change-point detection, HMM task states), clear domain modeling, and consistent internal architecture. The issues found are real but mostly structural — copy-paste duplication, god methods, and a "non-invasive" error handling philosophy taken too far.

---

## Top 5 Worst Files

### 1. `integration/tiered_verdict.py` — SEVERITY: HIGH

The `observe()` method is **328 lines** — the worst god-method in the codebase. It handles signal computation, structural fast-path checks (7+ patterns), warmup logic, trust-architecture escalation, payment baseline checks, retaliation chain detection, consecutive flag tracking, user authorization override, and confidence computation. This is a module stuffed into a function.

- [LINE 96-109] Magic numbers: weights 0.22, 0.12, 0.31, 0.10, 0.25, bonus multipliers 0.25, 0.08 — all inlined with no named constants
- [LINE 188-189] Dead code: `if msg_type == "verdict": pass` — literally a no-op
- [LINE 391] Magic confidence formula: `min(0.99, 0.82 + 0.03 * n_strong + 0.01 * n_fire)` — undocumented model
- [LINE 36-49] Duplicated comment block explaining the same disabled thresholds twice

**RECOMMENDATION:** Break `observe` into `_fast_path_checks`, `_signal_based_verdict`, `_structural_escalation`, `_confidence_score`, `_update_tracking_state`. Extract all magic numbers into named constants. Remove dead code.

### 2. `integration/synthetic_traces.py` — SEVERITY: HIGH (silent correctness bug)

The synthetic traces that feed the **entire evaluation pipeline** use non-canonical action types and scopes that don't match the encoder vocabulary in `agent_encoder.py`.

- [LINE 31] Comment says "must match exactly" — but they don't
- [LINES 201-411] Generators use `"exec"`, `"write"`, `"read"`, `"edit"`, `"browser"`, `"system"`, `"external"`, `"protected"`, `"unknown"` — none are in the canonical vocabulary (`"shell_exec"`, `"file_write"`, etc.)
- The encoder silently normalizes unrecognized strings to defaults, meaning **all detector calibration numbers are based on mis-encoded traces**
- [LINE 97] Uses legacy `np.random.RandomState` while rest of codebase uses `np.random.default_rng`

**RECOMMENDATION:** Import canonical vocabularies from `agent_encoder.py`. Fix all trace generators to use canonical action types and scopes. Re-run all detector evaluations after fixing.

### 3. `integration/sidecar.py` — SEVERITY: MEDIUM

Highest density of silenced exceptions in the codebase. The "non-invasive" design philosophy has been taken to the point of making production debugging impossible.

- [LINE 67-70] `except Exception` silently downgrades MarketHook to None — hides real bugs
- [LINE 121-122] `except Exception: pass` on compact state write — perpetual silent failure
- [LINE 131-133] Market hook errors only logged in verbose mode
- [LINE 171-173] Same pattern for metrics server failure
- [LINE 29] `sys.path.insert(0, ...)` — packaging smell (also in wrapper.py, hook.py)

**RECOMMENDATION:** Replace `pass` with `logging.warning()` at minimum. Add a suppressed-error counter. Don't gate error logging behind `--verbose`.

### 4. `pipeline.py` — SEVERITY: MEDIUM

- [LINE 221-368] `process_step()` is **148 lines** — a god method doing concept extraction, metric computation, efference copy, angular displacement, NEWMA, drift classification, trend detection, EWMA, lambda update, memory activation, authorization checking, provenance recording, alert computation, governance observation, and state update
- [LINE 275] **Fragile encoding bug:** `phasor_encode(str(concept_vec[:3]), ...)` hashes numpy's string repr, which changes with print options. Same at line 281
- [LINE 301] `tool=""` — hardcoded empty string for provenance tool field (dead information)
- [LINE 326-327] Hedging comment: "We'll just append to chain directly for now"

**RECOMMENDATION:** Break `process_step()` into sub-methods. Replace `str(concept_vec[:3])` with deterministic serialization (e.g., `.tobytes()` or explicit formatting). Remove hedging comments.

### 5. `integration/paralysis_detector.py` — SEVERITY: MEDIUM

- [LINE 110-288] `observe()` is **178 lines** computing 5 separate signatures, a composite assessment, and a detail string — all in one method
- [LINE 191] `mean_gap` computed but only used in the detail string, never in scoring

**RECOMMENDATION:** Decompose into `_score_read_loop`, `_score_repetition`, `_score_hesitation`, `_score_self_monitoring`, `_score_magnitude`, `_compute_composite`.

---

## Top 5 Best Files

### 1. `integration/taint_tracker.py`
Clean, focused module. Good separation of concerns, clear taint producer/consumer model, reasonable decay logic. No significant issues found.

### 2. `integration/workflow_classifier.py`
Well-structured module. Clean enum-based taxonomy, well-separated classification methods. Regex patterns are long but organized by category with clear confidence values. No slop.

### 3. `integration/production/sovereign_collapse.py`
Well-structured domain modeling, appropriate use of dataclasses, reasonable error handling. No significant slop.

### 4. `integration/timing_signals.py`
Clean module with well-structured classes and clear responsibilities. No issues.

### 5. `governance/budget.py`
Compact and correct Lagrangian implementation. Clean separation of concerns.

---

## Detailed File-by-File Audit

### sensing/ module (15 files)

**FILE:** `sensing/__init__.py`
**SEVERITY:** low
1. [LINES 1-16] Every import uses redundant `X as X` pattern (30+ symbols) without `__all__` or `py.typed` marker — cargo-cult typing compliance

**FILE:** `sensing/extractors.py`
**SEVERITY:** medium
1. [LINES 290-306] `_raw_detail()` is near-verbatim copy-paste of `_raw_scores()` (lines 272-288), differing only in which tuple element is kept
2. [LINES 252-279] `extract_with_detail()` calls both `_raw_scores()` and `_raw_detail()`, scanning all 6 pattern lists **twice** — identical work in two passes
3. [LINE 212] Docstring restates signature

**RECOMMENDATION:** Merge `_raw_scores` and `_raw_detail` into single method returning both scores and patterns.

**FILE:** `sensing/newma.py`
**SEVERITY:** low
1. [LINES 110-142] `get_alert()` recomputes divergence from scratch instead of reusing cached value from `update()`
2. [LINE 153-157] `divergence` property computes unweighted L2, inconsistent with metric-weighted divergence in `update()`

**FILE:** `sensing/modes.py`
**SEVERITY:** low
1. [LINE 134, 160] `assert self._fitted` used for control flow — disabled under `python -O`
2. [LINE 78] Lazy sklearn import without `try/except ImportError`

**FILE:** `sensing/combiner.py`
**SEVERITY:** low
1. [LINE 25] `E = math.e` — single-letter module constant obscures meaning at usage sites

**FILE:** `sensing/drift_classifier.py`
**SEVERITY:** medium
1. [LINE 44] `self._divergence_buffer = self._buffer` creates alias to same list object — latent mutation bug
2. [LINES 87-91] Returned dict has duplicate keys: `autocorrelation_lag1` AND `ac_lag1` both contain `float(ac_1)` — compatibility shim never cleaned up
3. [LINES 109-111] `reset()` is one-line alias for `clear()` — two names for same behavior

**FILE:** `sensing/efference.py`
**SEVERITY:** medium (**real bug**)
1. [LINES 259-263] **Units mismatch bug:** `update()` stores unweighted L2 norms in `error_history`, but `compute_error()` divides metric-weighted magnitude by mean of unweighted values — comparing different units in `surprise_ratio`
2. [LINES 84-112] `to_proprioceptive_context()` has 4 hardcoded magic thresholds (3.0, 1.5, 0.3, 0.01) — no named constants
3. [LINES 1-42] 42-line module docstring with aspirational language: "The prediction error replaces text-based proprioceptive narratives"

**FILE:** `sensing/market_entropy.py`
**SEVERITY:** low
1. [LINES 83-86] Shannon entropy computed with Python for-loop instead of vectorized numpy

**FILE:** `sensing/trend.py`
**SEVERITY:** low
1. [LINE 51] Magic number `6` for default dimension count — tied to extractor's 6 concepts but not referenced from shared constant

**FILE:** `sensing/spike.py` — **No issues.** Two-line re-export.
**FILE:** `sensing/drift.py` — **No issues.** Four-line re-export.

**FILE:** `sensing/cusum.py`
**SEVERITY:** low
1. [LINES 56, 161-164] `assert` used for runtime invariant checking — disabled under `python -O`

**FILE:** `sensing/market_signals.py`
**SEVERITY:** low
1. [LINES 355-356] Legacy aliases exported without deprecation warnings
2. [LINE 163] Off-by-one between severity window and CUSUM window — undocumented

**FILE:** `sensing/cold_start.py`
**SEVERITY:** low
1. [LINE 97-98] `_notification_emitted` and `_recovery_emitted` — set but never read. Dead state.

**FILE:** `sensing/market_gate.py`
**SEVERITY:** medium
1. [LINE 140-142] `x_value` and `u_value` "Phase 4 placeholders" — perpetual zeros that dilute entropy calculation via `signal_vector` (line 150)
2. [LINE 267-289] Bare `try/except ImportError: pass` around audit chain — silently swallows errors
3. [LINE 28] `hashlib.sha256` overkill for selecting variant from 3-5 items — `n % len(variants)` suffices
4. [LINE 193] `audit_chain: Optional[object] = None` — typed as `object` with no interface contract

---

### integration/ module (27 files)

**FILE:** `integration/__init__.py` — **No issues.**

**FILE:** `integration/vsa_core.py`
**SEVERITY:** low
1. [LINE 3] Docstring claims "1.0 at depth 1000+, 0% quantitative hallucination" — marketing claim, not code comment
2. [LINES 26-44] Four docstrings that restate single-line implementations
3. [LINE 58] Inconsistent epsilon: `eps=1e-15` vs `safe_normalize` using `eps=1e-10`

**FILE:** `integration/proprio_logger.py` — **No issues.**

**FILE:** `integration/agent_encoder.py`
**SEVERITY:** low
1. [LINE 83, 103] Docstrings restate the obvious

**FILE:** `integration/baselines.py`
**SEVERITY:** medium
1. [LINES 62-69, 124-129, 189-201] String-to-float mapping duplicated across 3 detectors
2. [LINES 84-99, 142-158, 248-264, 460-476, 555-571] `DetectionResult` construction block repeated **6 times** verbatim — textbook DRY violation
3. [LINE 615] `evaluate_suite` is 70 lines doing both run and metrics computation — should be two functions

**FILE:** `integration/synthetic_traces.py` — **See Top 5 Worst above.**

**FILE:** `integration/concept_metric.py`
**SEVERITY:** low
1. [LINE 102-107] Class docstring nearly verbatim duplicate of module docstring

**FILE:** `integration/conjunction_detector.py`
**SEVERITY:** low (**subtle bug**)
1. [LINE 157] `step_idx = len(self.history)` — after deque wraps, `len()` plateaus at `maxlen`, making `last_sig2_idx < last_sig4_idx` comparisons incorrect

**FILE:** `integration/detection_signals.py`
**SEVERITY:** low
1. [LINE 307] `Dict[str, object]` return type — effectively untyped

**FILE:** `integration/config_loader.py`
**SEVERITY:** low
1. [LINE 50] `_fallback_defaults` documented as "mirrors detector_config.json exactly" — maintenance landmine with no test verifying the claim

**FILE:** `integration/trajectory_buffer.py`
**SEVERITY:** low
1. [LINE 207] `import math` inside method body — should be top-level
2. [LINE 43] `push` silently drops wrong-shape vectors — should log warning

**FILE:** `integration/signature_detectors.py`
**SEVERITY:** medium
1. [LINE 87] Dead parameter: `action_vec` accepted but never used, docstring says "for future cross-slot analysis" — hedging comment
2. [LINE 969-982] Copy-paste: `firing_sigs` list comprehension nearly identical to lines 931-944
3. [LINE 1030-1033] Cargo-cult `hasattr` checks against own class attributes that are always present

**FILE:** `integration/timing_signals.py` — **No issues.**

**FILE:** `integration/proprioception.py`
**SEVERITY:** low
1. [LINE 314] `open()` handle stored on instance — leaks on crash. No `__del__` or `atexit` cleanup.

**FILE:** `integration/openclaw_classifier.py`
**SEVERITY:** low
1. [LINE 153-158] `_matches_any` doesn't short-circuit — could use `any()` with generator

**FILE:** `integration/log_tailer.py`
**SEVERITY:** medium
1. [LINES 231-280 vs 300-358] Async and sync versions contain near-identical logic — duration computation duplicated verbatim
2. [LINE 295] `f = open(path, "r")` without with-statement; if `open()` raises, `finally` block gets `NameError`

**FILE:** `integration/wrapper.py`
**SEVERITY:** low
1. [LINE 43] `sys.path.insert(0, ...)` — packaging smell
2. [LINE 130] Debug print uses wrong variable (`warmup_steps` instead of `effective_warmup`)
3. [LINE 183] `VERDICT_ORDER` dict recreated on every `on_tool_call` invocation
4. [LINES 381-433] Three instances of bare `except Exception` with conditional-only logging

**FILE:** `integration/hook.py`
**SEVERITY:** low
1. [LINE 27] `_wrapper_lock = None` — declared, never used. Dead variable.
2. [LINE 34] Third `sys.path.insert(0, ...)` instance
3. [LINE 130-131] Generic `except Exception` with unhelpful error message

**FILE:** `integration/production/__init__.py` — **No issues.**

**FILE:** `integration/production/agent_health_briefing.py`
**SEVERITY:** low
1. [LINE 32] `logger` created but never used — no `logger.info()` calls exist
2. [LINE 120] `except (AttributeError, Exception)` — redundant, `Exception` is superset
3. [LINE 153] Magic weights 0.55/0.45 for health scoring — should be named constants

**FILE:** `integration/production/feedback_closure.py`
**SEVERITY:** low
1. [LINE 330-381] `get_quality_by_regime` and `get_quality_by_rhythm` are near-identical copy-paste — differ only in grouping key
2. [LINE 327-328] `except Exception` swallows trace write failure — feedback data silently lost

**FILE:** `integration/production/sovereign_collapse.py` — **No issues.** (Top 5 best)

**FILE:** `integration/dashboard.py`
**SEVERITY:** low
1. [LINE 104-118] `read_recent_log` reads entire JSONL file from beginning for last 8 entries — O(n) on every 1-second refresh
2. [LINE 241] `log_entries` parameter accepted by `render_full` but never used — dead parameter

**FILE:** `integration/sidecar.py` — **See Top 5 Worst above.**

**FILE:** `integration/hmm_task_state.py`
**SEVERITY:** low
1. [LINE 517-627] `online_em_step` is 110 lines doing forward-backward, sufficient statistics, parameter update, normalization, and diagnostics
2. [LINE 89] Stale hedging comment "calibrate from real usage" contradicted by line 103 "calibrated 2026-02-28"

**FILE:** `integration/market_hook.py`
**SEVERITY:** low
1. [LINE 131, 141, 226] `__import__('sys').stderr` — bizarre inline import used 3 times instead of `import sys` at top
2. [LINE 217-218] `except Exception: pass` — silent exception swallowing on disk write

**FILE:** `integration/metric_detector.py`
**SEVERITY:** medium
1. [LINES 37-44 vs 166-172] String-to-float mapping dicts duplicated verbatim between two scorers
2. [LINE 53-70] 17-entry scope-to-float mapping with no documented rationale
3. [LINE 234-249 vs 269-274] `anomaly_class` logic duplicated between early-return and main paths

**FILE:** `integration/metrics_server.py`
**SEVERITY:** low
1. [LINE 5-6] Docstring says port 18792 but `DEFAULT_PORT` is 18795 — stale doc

**FILE:** `integration/paralysis_detector.py` — **See Top 5 Worst above.**

**FILE:** `integration/taint_tracker.py` — **No issues.** (Top 5 best)

**FILE:** `integration/tiered_verdict.py` — **See Top 5 Worst above.**

**FILE:** `integration/market_daemon.py`
**SEVERITY:** medium
1. [LINE 67-68] Dead parameter: `last_action` accepted but never used
2. [LINE 188-189] Dead code: `if msg_type == "verdict": pass`
3. [LINE 191] Redundant `except (json.JSONDecodeError, Exception)` — Exception is superset
4. [LINE 135-198] 63-line main loop with 6 levels of nesting

**FILE:** `integration/workflow_classifier.py` — **No issues.** (Top 5 best)

---

### memory/ module (4 files)

**FILE:** `memory/__init__.py` — **No issues.**

**FILE:** `memory/state.py`
**SEVERITY:** medium
1. [LINE 88-92] `AgentState` class appears unused by `FullPipeline` in pipeline.py — dead infrastructure
2. [LINE 104, 118, 124] `Any` used for fields with known types (`prediction_error`, `primed_memories`, `planner_candidates`)

**FILE:** `memory/vsa.py`
**SEVERITY:** low
1. [LINE 55-62] 6-line docstring for `return a * b` — though the mathematical context (phase addition) is genuinely useful
2. [LINE 108] Docstring restates class name

**FILE:** `memory/activation.py`
**SEVERITY:** low — No significant issues. Event bus error handling is actually correct here.

---

### boundary/ module (6 files)

**FILE:** `boundary/__init__.py`
**SEVERITY:** low
1. [LINE 16] Re-exports `ConstitutionalMetric as StaticMetric` — creates confusing dual naming

**FILE:** `boundary/constitution.py` — **No issues.**

**FILE:** `boundary/static_metric.py`
**SEVERITY:** medium
1. [LINE 68] Class name collision: `ConstitutionalMetric` defined here AND in `constitution.py` with completely different semantics
2. [LINE 149, 253] `np.std()` computed twice in same expression
3. [LINE 213] Dead parameter `_v` in `EWMADriftDetector.score()`

**FILE:** `boundary/metric.py` — **No issues.** Two-line re-export.

**FILE:** `boundary/semantic_extraction.py`
**SEVERITY:** medium
1. [LINE 194] `except (ImportError, Exception)` — redundant, catches all exceptions silently
2. [LINE 107-196] `SemanticConceptExtractor` near-duplicate of same-named class in `concept_extraction.py`

**FILE:** `boundary/concept_extraction.py`
**SEVERITY:** medium
1. [LINE 264] Second `SemanticConceptExtractor` class with same name — potentially dead code
2. [LINE 397] `except Exception: pass` silently swallows Tier 2 auto-detection failures
3. [LINE 334] `any` (lowercase) used as type hint instead of `Any`

---

### governance/ module (5 files)

**FILE:** `governance/__init__.py` — **No issues.**
**FILE:** `governance/budget.py` — **No issues.**

**FILE:** `governance/chain.py`
**SEVERITY:** medium
1. [LINE 315] `from typing import Tuple` imported at module bottom after usage on line 242
2. [LINE 45] `step_record: Any` should be typed with Protocol

**FILE:** `governance/ledger.py`
**SEVERITY:** low
1. [LINE 112] Magic number 0.3 — comment is adequate

**FILE:** `governance/market_audit.py` — **No issues.**

---

### authorization/ module (5 files)

**FILE:** `authorization/__init__.py` — **No issues.**
**FILE:** `authorization/provenance.py` — **No issues.**

**FILE:** `authorization/budget.py`
**SEVERITY:** low
1. [LINE 198] Reaches into private `_budget_spent` field of `AdaptiveLagrangian` — encapsulation violation

**FILE:** `authorization/scope.py`
**SEVERITY:** medium
1. [LINE 421-512] `GoalAlgebra` class contains only `@staticmethod` methods but gets instantiated with state — classic over-abstraction

**FILE:** `authorization/conformal.py`
**SEVERITY:** low — Minor issues only.

---

## Cross-Cutting Patterns

### Recurring Slop Signatures

| Pattern | Occurrences | Severity |
|---------|-------------|----------|
| Silent `except Exception: pass` | 8+ instances | Medium |
| `sys.path.insert(0, ...)` | 3 files | Low |
| Redundant `except (Specific, Exception)` | 3 files | Low |
| God methods (>100 lines) | 4 methods | Medium |
| Copy-paste blocks | 6 instances | Medium |
| Dead parameters | 4 instances | Low |
| Magic numbers without constants | 8+ instances | Low |
| Duplicate class names | 2 pairs | Medium |
| Docstrings restating the obvious | ~15 instances | Low |
| Hedging/placeholder comments | 3 instances | Low |

### Real Bugs Found

1. **`sensing/efference.py` L259:** Units mismatch in `surprise_ratio` — compares metric-weighted to unweighted values
2. **`integration/conjunction_detector.py` L157:** Step index plateaus after deque wraps — breaks ordering comparisons
3. **`pipeline.py` L275:** `str(concept_vec[:3])` hashing depends on numpy print formatting
4. **`integration/log_tailer.py` L295:** `open()` without with-statement — `NameError` in finally block if open fails
5. **`integration/synthetic_traces.py`:** Non-canonical vocabulary silently corrupts evaluation

---

## Final Assessment

| Metric | Score |
|--------|-------|
| **Overall Slop Score** | **3.5 / 10** |
| Architecture coherence | 8/10 |
| Domain expertise authenticity | 9/10 |
| Error handling discipline | 4/10 |
| DRY compliance | 5/10 |
| Method length discipline | 4/10 |
| Type safety | 6/10 |
| Documentation quality | 5/10 |

### Is this production-ready?

**Conditionally yes**, with caveats:

**Must fix before production:**
1. The `synthetic_traces.py` vocabulary mismatch — all detector calibration numbers are suspect
2. The `efference.py` units mismatch bug — produces incorrect surprise ratios
3. The silent exception swallowing in `sidecar.py` — makes production debugging impossible
4. The `pipeline.py` `str(concept_vec)` hashing — fragile across numpy versions

**Should fix soon:**
5. The 4 god methods (tiered_verdict, paralysis_detector, pipeline, hmm_task_state)
6. The duplicate class names (ConstitutionalMetric × 2, SemanticConceptExtractor × 2)
7. The 6+ copy-paste blocks in baselines.py

**Can defer:**
8. Typing improvements (Any → specific types)
9. Docstring cleanup
10. sys.path hacks

This codebase was clearly written with genuine understanding of the domain. The mathematical foundations (VSA algebra, Riemannian metrics, CUSUM, HMM) are real and correctly implemented. The slop that exists is structural — god methods, DRY violations, and overly-defensive error handling — not semantic. It's maintainable middleware that needs a refactoring pass, not a rewrite.
