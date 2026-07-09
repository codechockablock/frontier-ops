# Session artifacts — Apollo deception campaign (2026-07)

Verbatim working harnesses and reference outputs from the one-day empirical
campaign behind the v2 refactor (HANDOFF_frontier_ops_v2.md §7). These scripts
were written against module-mutation workarounds (`CE.CONCEPTS[:] = ...`) that
predate the Phase 1 constructor-param APIs; `eval/battery/` is the maintained
port. Do not "fix" these files — they are the provenance record the battery is
cross-checked against.

| file | role |
|---|---|
| `run_deception_eval.py` | RUN A (stock control) / RUN B (zero-shot encoding) / RUN C on roleplaying |
| `run_c_retry.py` | RUN C retry with Tier-2 extractor injected directly |
| `earned_anchors.py` | **single source of truth** for the 4 deception-dim anchors (written blind) |
| `encode_cache.py` | 4-D chart encodings for all four tasks |
| `loto_analysis.py` | metric comparison (identity / asserted / transfer / in-task / earned) + LOTO transfer |
| `earned_metric_loto.py` | superseded by `encode_cache.py` + `loto_analysis.py` |
| `curvature_kill_test.py` | pre-registered curvature kill test; F4 = step-mean champion |
| `encode_raw.py` | raw 384-D MiniLM encodings |
| `external_half.py` | 384-D prototype-direction transfer + cosine geometry |
| `deception_eval_scores.csv` | reference per-row scores (roleplaying) |
| `curvature_kill_scores.npz` | reference per-episode scores (insider trading) |
| `earned_metric.npz` | reference earned metric G + per-task risk directions |
| `external_transfer_384d.npz` | reference 384-D prototype directions + cosine matrix |

Missing from the delivered set: `metric_faceoff.py` (§6 "metric ordering"
check). The battery reconstructs that check from the estimator implementations
in `loto_analysis.py` (`identity` / `asserted_G` / `intask_cov` under the same
5-fold protocol); see `eval/battery/checks.py` for the reconstruction note.

Paths inside these scripts point at the original sandbox (`/home/claude/...`);
they are not runnable here as-is. Data comes from
`ApolloResearch/deception-detection` (disclosure level "Public", no LICENSE
file) — downloaded at runtime by `eval/battery/data.py`, never committed.
