# Q2 — Label budget for in-domain calibration (2026-07-10)

`python -m eval.label_budget_curve`. AUROC vs number of labeled in-domain
episodes for the candidate calibrators, 5-fold CV × 10 stratified draws per
budget (fold seed 11, Q1 encoding caches). This was the last open design
question for the v3 detection stack: Q1 demoted the chart at full labels,
but a 4×4 estimated metric might plausibly beat a 384-D prototype when
labels are scarce.

## It doesn't. Raw step-mean wins at every budget, including n=8.

Mean AUROC ± std (see label-budget-2026-07-10.json for full curves):

| labels | insider raw-sm | insider chart-sm | sandbag raw-sm | sandbag chart-sm | roleplay raw-sm |
|---|---|---|---|---|---|
| 8 | **0.715** | 0.640 | **0.933** | 0.549 | 0.607 |
| 16 | **0.770** | 0.686 | **0.961** | 0.547 | 0.632 |
| 32 | **0.790** | 0.726 | **0.973** | 0.585 | 0.716 |
| 64 | **0.821** | 0.766 | **0.980** | 0.586 | 0.757 |
| full | 0.837 | 0.794 | 0.985 | 0.650 | 0.755 |

(ai_liar, n=53, is too noisy to read: std ±0.15–0.26 at all budgets.)

Budget to reach within 0.02 of full-pool AUROC (raw step-mean): insider
**64**, sandbagging **32**, roleplaying **64**. The chart cells need 2–8×
more labels for worse ceilings and never overtake at any budget — the
prototype direction (two class means, no covariance) is exactly the
statistic that tiny label sets can estimate, while the chart pays its
compression loss everywhere and its 4×4 metric adds noise below ~64 labels
(chart-est trails chart-identity at n≤16 on two tasks).

Sound bite: **32 labeled insider episodes buy 0.790 — the entire v2
campaign champion's full-data score.** 8 labeled sandbagging episodes buy
0.933.

## Deployment guidance for the port

The in-domain calibration recipe is now fully specified and cheap:
split → embed (raw MiniLM, per step) → step-mean → prototype direction from
**~32–64 labeled in-domain episodes** (a stratified ~50 is a sensible
default ask). No covariance estimation, no anchors, no chart in the
detection path; `CalibratedMetric`/chart remain the reporting layer.
Labeling 50 episodes of real agent-session traffic is a feasible one-day
task — the binding input for the domain port is now quantified.
