# Eval Environment Versions

Recorded 2026-07-09 per v2 handoff Phase 0.

## Battery environment

- Python 3.14.3 (darwin / arm64)
- numpy 2.5.1
- scipy 1.18.0
- scikit-learn 1.9.0
- cryptography 49.0.0
- sentence-transformers 5.6.0
- torch 2.13.0
- pytest 9.1.1
- Encoder: `all-MiniLM-L6-v2` (default Tier-2 model)

## Notes

- The v2 handoff §6 expected numbers were produced by the original campaign
  under an unrecorded sentence-transformers version. Per §6, if battery numbers
  miss the ±0.01 tolerance, suspect encoder version first and record new
  baselines in `expected.json` with a version stamp; the ordering invariants
  are unconditional and must hold regardless of encoder version.
- The repo's long-lived local `.venv` carries sentence-transformers 5.3.0;
  the battery environment above was built fresh for the v2 work.
