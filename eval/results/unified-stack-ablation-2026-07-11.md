# unified-stack L0-vs-L2 ablation — does the geometric core earn its keep? (2026-07-11)

Context: `~/unified-stack` is the diverged fork (its own `frontier_ops`
v2.0.0). Before deciding whether its geometric core (L0) can be retired in
favor of its session-semantic layer (L2) — the upstream Q8 move — we ran
the ablation on its own frozen corpora, using its own eval pipeline
unmodified except for an in-memory patch at `Pipeline._determine_verdict`
(the seam its own tests monkeypatch).

**Method.** Baseline run with per-trace fired-step attribution
(`eval/unified_stack_ablation/ablation_runner.py`), step census
(`ablation_analysis.py`), then a TRUE L2-only re-run
(`ablation_true_l2only.py`): conformal → ACT, cumulative e-value → 0,
conjunction off, displacement → 0, coherence violation off, curvature → 0,
quality e-value → OK, trajectory_cost → 0 (strict) or kept (sensitivity —
it is a Mahalanobis accumulator over *semantic* risk categories, so its
classification is contested). Semantic cascade left fully intact; the
cascade falls through the geometric steps and every action reaches the
semantic checks. Baseline reproduces every canonical number exactly.

## Results (TPR / FPR, %)

| corpus | combined (canonical) | L2-only strict | L2-only + trajectory_cost |
|---|---|---|---|
| internal (518) | 94.5 / 10.9 | **66.0 / 3.8** | 66.0 / 3.8 |
| thinking_augmented (400) | 99.0 / 13.0 | **74.5 / 8.0** | 74.5 / 8.0 |
| ATBench (500) | 100.0 / 12.0 | **78.4 / 9.6** | **99.2 / 12.0** |
| AgentDojo (357) | 93.2 / 0.8 | 93.2 / 0.8 | 93.2 / 0.8 |

Attribution census of GEO-unique catches (baseline, first-fired step):
internal 59/253 adversarial traces are caught by geometric steps alone —
69 fires from `step_4_displacement_*` (embedding drift from the task),
17 from e-value/conformal fusion, 7 curvature-interaction, 2 quality;
thinking_augmented mirrors it (73 displacement, ~20 e-value fusion).
ATBench's GEO-unique fires are ~all `step_5c_trajectory_*`
(the risk-category cost accumulator); AgentDojo has zero.

## Reading

1. **The fork's geometric core is NOT a shell.** Unlike upstream's Q8
   (where no ensemble subset beat prototype+NEWMA and the 12-signal
   pipeline collapsed), removing unified-stack's geometry costs
   **28.5 TPR points on internal** and **24.5 on thinking_augmented**.
   "Retire L0, keep L2" is refuted on the fork's own corpora.
2. **But the load-bearing geometry is exactly the family upstream kept.**
   What carries the 25–30 points: embedding displacement from the task
   (a drift direction — the same family as upstream's prototype/NEWMA),
   the e-value martingale + conformal statistics (upstream: conformal
   core), and a cumulative severity accumulator. What contributes almost
   nothing unique: curvature (7/253 internal, 2/500 ATBench — and only
   ever with corroboration), i.e. the same curvature-as-signal idea
   upstream killed in v2. The two repos' evidence CONVERGES: drift
   geometry + sequential statistics survive; curvature does not.
3. **ATBench's "geometry" is one accumulator.** Strict removal drops it
   to 78.4, but keeping only trajectory_cost recovers 99.2 at unchanged
   FPR. That accumulator is 30 lines of metric-weighted counting over
   semantic risk categories — trivially portable to any stack.
4. **AgentDojo never needed geometry at all** (93.2 / 0.8 unchanged).
5. **Caveats.** (a) Operating points aren't matched: L2-only also drops
   FPR (10.9 → 3.8 internal), and the cascade's thresholds were co-tuned
   with geometry present — a re-tuned L2-only would recover some TPR,
   so the deltas are upper bounds on the *tuned* gap. (b) These corpora
   were built and threshold-tuned in-sample with the full system; the
   usual in-domain inflation applies to all columns equally.

## Verdict for the "is unified-stack needed?" decision

The fork's detection value does not reduce to its semantic layer — but it
does reduce to *semantic layer + drift-displacement + sequential
statistics + one accumulator*, which is architecturally the same skeleton
as upstream v3 (prototype-drift detector + NEWMA + conformal + the
authorization/semantics layer). A merge is therefore cleaner than feared:
nothing unique to unified-stack's geometry survives EXCEPT components
with direct upstream equivalents; its genuinely unique assets remain the
L2 cascade, the corpora, the enforcement/MCP product surface, and the
papers.

## Provenance

Scripts in `eval/unified_stack_ablation/` (paths point at
`~/unified-stack`, which is untouched — in-memory patch only; its suite
still passes 2040/2040). Full per-trace records stayed in the session
scratchpad; the JSONs here carry the aggregate results
(`unified-stack-ablation-2026-07-11.json`) and the attribution census
(`unified-stack-ablation-attribution-2026-07-11.json`).
