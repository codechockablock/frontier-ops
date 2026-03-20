# Market Architecture — Day 1 Data Report
## Date: 2026-03-20
## Generated: 2026-03-20T13:45:21.042150

---

## Data Sources Collected

| File | Size | Entries |
|------|------|---------|
| DAY1_REPORT.md | 0.0 KB | 1 |
| frontier-ops-observations.jsonl | 1.4 MB | 738 |
| governance-chain.jsonl | 234.3 KB | 333 |
| market-daemon.log | 15.7 KB | 171 |
| market-state-final.json | 0.2 KB | 1 |
| proprioception-log-330k.jsonl | 81.8 MB | 330051 |
| proprioception-state.json | 1.0 KB | 1 |
| sidecar.log | 121.1 KB | 1500 |

---

## Market Evaluation Summary

- **Total evaluations:** 172
- **Alerts fired:** 81 (47%)
- **Final status:** monopoly
- **Final elevated:** False
- **Final label:** None

### Verdict Distribution (market daemon steps)

- block: 70 (86%)
- pass: 4 (5%)
- flag: 4 (5%)
- monitor: 3 (4%)

### Signal Distributions

- **D_raw (verdict severity):** mean=2.68 std=0.57 min=0.90 max=3.00
- **S_raw (interval percentile):** mean=0.57 std=0.28 min=0.00 max=0.97

### Label Rotation
- **Unique labels seen:** 12
  - Action
  - Approach
  - Behavioral
  - Combined
  - Concurrent
  - Extended
  - Movement
  - Multiple
  - Observed
  - Operational
  - Prolonged
  - Simultaneous

---

## Frontier-Ops Observation Summary

- **Total observations:** 738
- **Sessions:** 2
  - **openclaw-20260319:** 276 entries
    - Verdicts: {'pass': 125, 'flag': 92, 'monitor': 38, 'block': 21}
    - Tools: {'exec': 141, 'read': 29, 'write': 18, 'memory_search': 13, 'process': 6, 'web_search': 1, 'web_fetch': 17, 'browser': 25, 'edit': 16, 'image': 7, 'cron': 2, 'sessions_spawn': 1}
  - **openclaw-20260320:** 462 entries
    - Verdicts: {'pass': 116, 'flag': 96, 'monitor': 11, 'block': 239}
    - Tools: {'exec': 248, 'write': 31, 'read': 81, 'edit': 22, 'memory_search': 8, 'web_fetch': 11, 'gateway': 1, 'image': 3, 'sessions_spawn': 1, 'browser': 5, 'process': 36, 'web_search': 2, 'message': 5, 'canvas': 2, 'nodes': 1, 'cron': 5}

---

## Governance Chain

- **Chain entries:** 333
- **First entry:** init at 2026-02-27T20:21:45.864265Z
- **Last entry:** export at 2026-03-01T01:28:43.354709Z

---

## Key Findings

1. **Cold-start BLOCK storm:** Sidecar restart produced wall-to-wall BLOCK verdicts
   (e-values in the quintillions). Market correctly detected sustained severity elevation.
   Alerts cleared around step ~95 as sidecar warmed up.

2. **D signal dominant, S nominal:** Health status = monopoly because D (severity)
   was the only elevated signal. S (interval percentile) stayed near 0.5 (median).
   This correctly identifies that the BLOCK verdicts were from the classifier,
   not from unusual timing patterns.

3. **Label rotation working:** All 4 D-only variants appeared across evaluations.
   SHA-256 hash of evaluation count produces deterministic but varied selection.

4. **Audit chain intact:** 172 market evaluations signed into Ed25519 chain.
   No tampering detected.

5. **Tool diversity:** 12+ distinct tool types generated across the session,
   including high-risk classes (message, SSH, credential-adjacent reads).

---

## Desktop GPU Server

| Component | Status |
|-----------|--------|
| Python 3.12.9 | ✅ Installed |
| PyTorch 2.6 + CUDA 12.4 | ✅ 16.11 TFLOPS verified |
| Ollama 0.18.2 | ✅ Serving on :11434 |
| phi4-mini (3.8B) | ✅ 38.8 t/s |
| qwen2.5:14b (14B) | ✅ 66.7 t/s |
| codestral:22b (22B) | ✅ 47.6 t/s |
