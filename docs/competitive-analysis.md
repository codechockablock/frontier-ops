# Competitive Analysis: Runtime Agent Monitoring
## frontier-ops vs. existing approaches — 2026-03-20

### AMDM (Shukla 2025, arXiv:2509.00115)

**What they do:** EWMA per-axis normalization + Mahalanobis joint anomaly detection.
5-axis framework (capability, robustness, safety, human factors, economics).
FPR 0.9%, latency 5.6s. Published code (MIT, ~200 LOC).

**What we do differently:**
1. **DAS-CUSUM** (adaptive symmetric) vs their vanilla EWMA — handles variance shifts
2. **SPRT formal clearance** — they can flag, we can ACCEPT
3. **Entropy antitrust** — prevents single-metric monopoly (their Mahalanobis doesn't address this)
4. **Qualitative-only surfacing** — they expose raw metrics to agents (gameable per Skalse 2022)
5. **Ed25519 governance chain** — tamper-evident audit trail (they have no audit)
6. **Calibration from real data** — we validated on 330K entries + 24 labeled traces

**Their advantage:** Published, cited, peer-reviewed path. We're unpublished.

**Key diff for paper:** AMDM is stateless EWMA; we do trajectory-level sequential detection.
AMDM operates per-evaluation; we detect *sustained* anomalies across trajectories.
AMDM has no formal clearance mechanism; SPRT is our unique contribution here.

### OpenGuardrails (2025)

14B model quantized to 3.3B, P95 latency 274.6ms. Content safety.
**Stateless** — evaluates each request independently. No trajectory awareness.
No sequential detection, no behavioral drift, no governance chain.

### Guardrails AI (open source, 5.9K stars)

Output validation library. Per-output validators (PII, toxicity, JSON schema).
**Stateless** per-output. No cross-action memory. No metric tensor.
Not designed for agent behavioral monitoring — designed for output filtering.

### Datadog AI Guard / Straiker Defend

Runtime guardrails with sub-second detection. Prompt injection, data leakage, tool misuse.
Behavioral monitoring tracks tool call frequency and response lengths.
**No trajectory-level features** (geodesic efficiency, cusum accumulation).
**No formal detection theory** (CUSUM, SPRT, ARL guarantees).

### The gap we fill

Every existing system operates **per-evaluation**. None does:
- Sequential detection (CUSUM/SPRT) over behavioral trajectories
- Formal clearance (affirmative ACCEPT, not just reject)
- Entropy-based multi-signal antitrust
- Qualitative-only signal surfacing (provably harder to game)
- Tamper-evident audit of the monitoring system itself
