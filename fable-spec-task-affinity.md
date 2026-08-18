# Fable Spec — Task-Affinity Channel (Escalate-Only)

Self-contained handoff. Written 2026-07-03 by the session that produced the
goal-conditioning null + controls; executable by a fresh session. Everything
needed is below or on disk.

---

## 1. Problem statement

**Repo:** `frontier-ops`, local checkout `/Users/joseph/frontier-ops`. Python
3.10–3.14, `.venv` present, `sentence-transformers` installed locally (NOT
assumed in CI — the channel and its tests must degrade gracefully).

**Where the last layer left off.** Goal-conditioning (the relational layer) is
implemented as pure composition (`frontier_ops/authorization/goal_conditioning.py`)
and evaluated on real directive→action data. Findings, all in
`eval/results/goal-conditioning-2026-07-03.md` and
`goal-conditioning-controls-2026-07-03.md`:

- Concept-space (6-dim) goal→action distance carries **no task-identity
  signal**: pairwise own-vs-mismatched AUC ≈ 0.467, and — decisively — flat
  at chance in EVERY directive-similarity stratum (0.454 on the cleanest,
  least-similar negatives). The 6 dims encode risk surface; task identity is
  projected out. This is a representation ceiling, not a data or tuning issue.
- The identity signal **exists in the text**: raw MiniLM embedding cosine gives
  AUC ~0.73 on clean negatives (0.837 on long directives), and survives
  aggressive lexical masking (long band 0.837→0.712 with 13% of tokens
  masked) — mostly semantic affinity, not vocabulary matching.
- The goal-conditioned relaxation contributes ~nothing to pairwise ordering;
  the LOCKED/RELAXABLE machinery's role is risk placement, not directionality.

**What this spec builds.** A `TaskAffinityChannel`: an embedding-space
directive↔action affinity signal that supplies the directional information the
concept space cannot. It sits BESIDE the concept-space risk check, never inside
it.

**The one non-negotiable design principle — asymmetry (the mimicry answer).**
Embedding affinity is adversarially mimicable: a covert action can borrow the
directive's vocabulary/phrasing to inflate cosine similarity. Therefore:

> **Low affinity may only ADD suspicion (escalate). High affinity must never
> REMOVE suspicion (authorize, relax a boundary, expand a radius, or offset a
> concept-space verdict).**

Under this rule, mimicry can only buy an attacker back to the baseline they
would have had without the channel — it can never unlock anything. The
non-mimicable hard floor remains the concept-space risk placement (an action's
credential/self-mod surface is scored from what it DOES, not how it is worded)
plus LOCKED dims. State this reasoning in code docs and the report.

**What already exists (use, do not reinvent):**

- `GoalConditioningScorer` / `GoalConditioningResult`
  (`frontier_ops/authorization/goal_conditioning.py`) — the relational-layer
  entry point. The channel integrates here as an OPT-IN extra signal.
- `GoalVector.raw_message` (`authorization/scope.py`) — the directive text is
  already carried on the goal.
- `AuthorizationRadius.calibrate` (`authorization/scope.py` ~line 177) — split
  conformal quantile calibration. Mirror its style for the affinity threshold
  (lower tail: escalate when affinity < threshold).
- `SemanticConceptExtractor.create()` (`boundary/semantic_extraction.py`) —
  the graceful-degradation factory pattern for optional sentence-transformers.
- `eval/goal_conditioning_eval.py` — `load_records`, `sample_mismatches`,
  `cluster_bootstrap_ci`, `band_of`, `action_text`;
  `eval/goal_conditioning_controls.py` — `mask_lexical` (Q2 masking rule).
- Dataset: `data/2026-07-02/directive-dataset.jsonl` — GITIGNORED, local-only,
  real personal content, never commit. Regenerate:
  `python3 eval/extract_directive_dataset.py`.

---

## 2. Tasks (priority order)

**T1 — Channel.** `frontier_ops/authorization/task_affinity.py`:
`TaskAffinityChannel` with `affinity(directive_text, action_text) -> float`
(cosine in embedding space, unit-normalized, cached embeddings). Constructor
takes an injectable `embed_fn` for testability; `create()` classmethod returns
`None` when sentence-transformers is unavailable (mirror
`SemanticConceptExtractor.create`). `AffinityThreshold` dataclass with
`calibrate(benign_affinities, alpha)` — split-conformal LOWER-tail quantile:
uncalibrated ⇒ never flags. No default constant that flags anything.

**T2 — Scorer integration (opt-in, asymmetric).** `GoalConditioningScorer`
accepts an optional `affinity_channel`. `score`/`score_action` accept optional
`action_text`. When both are present and the goal has a directive text, the
result gains `affinity: Optional[float]` and `affinity_flagged: Optional[bool]`
(None when channel absent/uncalibrated/no text). `authorized` MUST remain
byte-identical to the channel-off behavior — asymmetry is structural, and a
unit test must prove it.

**T3 — Held-out eval.** `eval/task_affinity_eval.py`. Session-level split of
the dataset (deterministic hash of session id → calibration / held-out; state
the split). On HELD-OUT sessions only: pairwise own-vs-mismatched affinity AUC
(negatives sampled from other held-out sessions, seeded), with cluster
bootstrap CI, per-length bands, similarity strata (Q1 methodology), and the
masked-action robustness variant (Q2 methodology). Threshold-free — AUC needs
no constants.

**T4 — Calibration demo.** Calibrate `AffinityThreshold` (alpha=0.1) on
CALIBRATION-split own-pair affinities; report the held-out benign
false-escalation rate (should be ≲ alpha, distribution-free). This validates
the conformal mechanism without fitting anything to held-out data.

**T5 — Report + regression.** `eval/results/task-affinity-<date>.md` with the
on-goal-only limitation restated, the mimicry stance (asymmetry) explained, and
the single-user/single-workload caveat. `pytest tests/ -q` green with and
without sentence-transformers importable; `ruff check .` clean on new files.

---

## 3. Acceptance criteria

1. `TaskAffinityChannel` exists; `create()` returns None without
   sentence-transformers; embeddings cached; injectable `embed_fn`.
2. Asymmetry proven by test: for any inputs, `authorized` with the channel ==
   `authorized` without it; `affinity_flagged` can only be True when affinity
   is BELOW a calibrated threshold; an uncalibrated threshold never flags.
3. Held-out eval runs end-to-end, deterministic under a fixed seed, prints AUC
   + CI + strata + masked variant, and writes the report.
4. Calibration demo reports held-out false-escalation rate against alpha.
5. Tests pass in an environment WITHOUT sentence-transformers (skip or stub via
   `embed_fn`); full suite green locally; ruff clean on new/changed files.

## 4. Constraints

**Must:** asymmetric use (escalate-only) enforced structurally, not by
convention; all thresholds via conformal calibration, none hand-set; held-out
sessions never used for any decision during development; the frozen seed-7
goal-conditioning pairs stay untouched as the historical benchmark.

**Must-not:** no hand-authored off-goal actions (unchanged project rule); no
blending affinity into the geodesic distance or radius; no committing
`data/2026-07-02/`; no claiming mimicry robustness — it is UNTESTED (no
adversarial data exists); the report must say so.

**Escalation path:** if held-out AUC lands ≈ 0.5, the ~0.73 clean-negative
ceiling did not transfer across the session split — report it plainly; that
would mean the affinity signal is session-idiosyncratic, and the channel should
NOT ship. Do not chase it with model swaps or preprocessing tweaks.

## 5. Known limitations to restate in the report

All-on-goal data (directionality only, not misuse detection); one user, one
workload, 22 sessions; session split is not a time split; mimicry untested;
MiniLM is a fixed choice — model selection was not searched (deliberately, to
avoid tuning on this data).
