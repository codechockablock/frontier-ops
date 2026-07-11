# Repo-wide open questions — answers (2026-07-10)

All nine questions from the post-v2 review, answered. New scripts:
`eval/combiner_diagnostic.py` (Q1/Q8), `eval/redteam_prototype.py` (Q6),
`eval/regime_benign_stability.py` (Q5), plus JSON artifacts under
`eval/results/`. Data: the drift-harness emergent records (646 actions, 50
off-goal) and the directive dataset (237 real sessions, 2212 actions).

---

## Q1 — Why does the evidence combiner invert signal? SOLVED: `surprise`.

`FullPipeline.alert_level` scores AUROC 0.594 on drift while its own
components score far higher: `boundary`-only 0.854, `ewma`-only 0.824,
`newma` 0.847, `geodesic` 0.824. The combiner takes good channels and drags
the combination below them. Two compounding mechanisms, both measured:

1. **The `surprise` channel is anti-informative and dominates.** It fires on
   16% of benign actions contributing **+1.31 log₁₀(BF) per benign action**,
   vs +0.16 on off-goal — i.e. it dumps "violation evidence" on *benign*
   actions. `surprise`-only AUROC is 0.429 (below chance). Ablation:
   dropping `surprise` lifts the recombined ensemble from 0.753 → **0.853**
   (+0.10). It is the single inverting channel.
2. **The authorization escalation bump flattens what's left.** The pure
   combiner output recombined from captured p-values is 0.753; the shipped
   `alert_level` is 0.594 because `needs_escalation` fires on ~98% of
   actions and overwrites the score with `max(alert_level, 0.6)`, destroying
   the ranking (this is the same 98% native-FPR seen in
   drift-harness-2026-07-03).

The Vovk-Sellke design multiplies Bayes factors from **hardcoded**
`p = exp(-k·signal)` transforms rather than from benign null distributions,
so any channel with a non-zero benign baseline (surprise, and to a lesser
extent boundary/ewma which are net-positive) contributes standing evidence
on benign traffic. The best 2-channel subset (`boundary`, `ewma`) reaches
0.902 — the combiner is salvageable by dropping surprise, removing the
escalation overwrite, and calibrating p-values against benign nulls, but
see Q8 for whether that's worth doing.

## Q8 — Change-of-mind: does any ensemble beat prototype+newma? NO. Verdict: shell.

Pre-registered threshold: if no subset of the 8 combiner channels beats
`prototype + newma` by ≥0.03 AUROC, the pipeline is a legacy shell.

| detector | AUROC |
|---|---|
| supervised prototype (grouped CV) | **0.930** |
| prototype + newma (rank-fused) | 0.902 |
| best-possible ensemble subset (`boundary`,`ewma`) | 0.902 |
| full ensemble `alert_level` (shipped) | 0.594 |

Margin (best ensemble − prototype+newma) = **−0.001**. The single supervised
prototype (0.930) beats the best achievable 8-channel subset by +0.028.
**Verdict fires: the 12-signal pipeline is a shell around two good signals.**
v3 detection is: calibrated prototype + newma_alarm; the constitution,
authorization graph, and governance chain are the *semantics/governance*
layer over it, not the detector.

## Q2 — How much is one `git clean` from unrecoverable? WAS: ~all of it. NOW: backed up.

The main checkout has 47 dirty entries; the irreplaceable ones existed on no
remote: the model-generated drift records (~1.8 MB of real API runs, not
byte-reproducible), the entire `drift_harness/`, production modules
`goal_conditioning.py` + `task_affinity.py`, `paper/`, and the unpushed
12-commit `v2-postmortem-refactor` branch (all v2 + this work). Tracked
history is safe (HEAD 51c6f46 is on origin/fable-spec-v2-encoder-drift), but
everything above was working-tree-only.

**Done (non-destructive):** created and verified
`~/frontier-ops-safety-backup-2026-07-10/` — a git bundle of the v2 branch
(complete history, clone-restorable) + a 84-file tar of all untracked/WIP
material. Data-loss risk eliminated.

**Still needs your decisions (reconciliation, not done autonomously):**
commit the untracked production/eval code; decide record data → committed vs
archived; merge/rebase the v2 branch (it conflicts with the WIP in
`pipeline.py`); push; refresh CLAUDE.md (514→743 tests, "private"→public).

## Q3 — Does the public repo make killed claims? YES.

`main`'s README (public) still states the v1 story the campaign refuted:
"makes boundary violations expensive to approach" (walls-as-metric — killed,
§1), a custom-constitution example that **crashed** the shipped pipeline
until Phase 1, and "Evidence combination weighs multiple signals into a
single confidence measure" (the combiner Q1 shows inverts signal). The
corrected README exists only on the unpushed v2 branch. This is now a
credibility exposure, not just hygiene: the repo is public.

## Q4 — Does the published PyPI package diverge from the repo? THERE IS NO PACKAGE.

`frontier-ops` and `frontier_ops` both 404 on PyPI (JSON API + `pip
install`). Nothing is published, so there is no divergence — but two
consequences: the README's `pip install frontier-ops` is a broken
instruction, and `pyproject` Homepage/Documentation point at
`github.com/openclaw/frontier-ops` while the real repo is
`codechockablock/frontier-ops` — both wrong and would 404 on any future
release. The handoff's "escalate if PyPI diverges" clause is resolved: it
doesn't exist; fix the metadata before any 0.3.0 publish.

## Q5 — Do drift/NEWMA/trend behave in their design regime? BENIGN SIDE: yes, stable. DETECTION: still blocked.

On the directive dataset's real long sessions (67 sessions ≥10 actions, 32
≥20, up to 60), the benign-side false-alarm behavior is exactly what the
"long-session" design promised:

- Overall benign NEWMA false-alarm rate **3.6%** (trend 1.9%) — vs the
  combined stack's 98% native FPR.
- **Flat with session length**: FPR by bucket 0.0 (len 1-4) / 3.7% (5-9) /
  5.4% (10-19) / 3.6% (20+); within long sessions, early positions (<5)
  3.0% vs late (≥20) 2.3% — **STABLE, does not climb**. This is the
  in-regime property that short-path testing could never show, and it's
  good. NEWMA is genuinely one of the two channels worth keeping (Q8).
- **Detection in-regime remains BLOCKED**: whether it catches drift that
  *emerges mid-session* needs a long-session drift-generation harness that
  does not exist. Not answerable without new data.

## Q6 — Can the supervised prototype survive the mimicry red-team? PARTIALLY — 3× harder to evade than affinity, but not immune.

Same three attack tiers as the affinity red-team (pad off-goal actions with
the benign directive's vocabulary), defender = grouped-CV prototype,
threshold at α=0.1:

| tier | evasion (off-goal not flagged) |
|---|---|
| raw (baseline miss) | 12% |
| keyword mimicry | 32% |
| verbatim mimicry | 32% |

Mimicry lifts evasion +20 points (12%→32%) — so the prototype is **not**
immune. But it is far more robust than cosine affinity (81-98% evasion): a
discriminative direction moves only slightly when you pad on-task words
(mean score 0.210→0.183), whereas a similarity-to-directive measure is
trivially inflated. Verdict: the prototype is a strong benign-drift detector
(Q1/Q8/domain-port) but a deliberate adversary still evades ~1 in 3 — it
cannot stand alone against the adversarial threat model, consistent with the
repo's own rule to keep benign-drift and deliberate-adversary separate.

## Q7 — Which packages have empirical support? Inventory.

| package | src modules | benchmark contact | status |
|---|---|---|---|
| **boundary** | 5 | Apollo battery, domain port (45 eval imports) | **validated** (detection) |
| **authorization** | 6 | drift harness: geodesic 0.86, radius transport, affinity red-team | **validated** (detection) |
| **sensing** | 14 | NEWMA/drift: benign-regime (Q5) + a top-2 channel (Q8) | **partial** — NEWMA/drift good; combiner *defective* (Q1); efference/Kalman, CUSUM/SPRT/entropy, trend untested on benchmarks |
| **integration** | 32 | market benchmark, ATBench live | **noisy** — ATBench ±7pt/run (single-draw), glue code |
| **governance** | 4 | none (unit tests only) | **infrastructure** — Ed25519 tamper-evidence is a crypto correctness claim, not an AUROC; no field/e2e deployment evidence |
| **memory** | 3 | 2 incidental eval imports | **speculative** — VSA phasor activation's contribution to detection has never been measured |

The repo distinguishes none of these tiers anywhere in its docs. The largest
package (integration, 32 modules) and the most-cited components (governance,
memory) have the least detection evidence. A v3 scope call should make this
explicit: validated core (boundary + authorization + NEWMA), infrastructure
(governance), and speculative (memory, most of sensing's detector zoo).

## Q9 — Does any of this hold against frontier-model deception? OUT OF REACH.

Confirmed: the only labeled data locally is llama-70b Apollo rollouts and
haiku-4.5/sonnet-5 drift-harness runs. On-policy deception from frontier
models does not exist here, and generating it is a budgeted red-team campaign
(model access, elicitation, grading), not a session. Every AUROC in this
project is conditional on that population. Genuinely out of reach — flagged,
not faked.

---

## Bottom line

The three load-bearing answers converge on one v3 thesis: **the detector is a
calibrated supervised prototype + NEWMA; everything else is
governance/semantics or dead weight.** Q1 shows the combiner inverts signal
(surprise + escalation overwrite), Q8 shows no ensemble subset beats the
prototype, Q5 shows NEWMA is the one legacy channel that behaves in-regime,
and Q6 shows even the prototype needs the authorization/governance layer
against a deliberate adversary. Q2/Q3/Q4 are now hygiene with the data-loss
risk removed; Q7 says label the tiers honestly; Q9 bounds every claim to the
tested population.
