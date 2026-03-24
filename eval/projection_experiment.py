"""
Projection Experiment: MiniLM (R^384) → Phasor Space (C^512)

Tests whether a random linear projection preserves enough semantic structure
to separate credential-adjacent commands from benign coding commands.
"""

import json
import re
import time
import numpy as np
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent / "data/2026-03-20/frontier-ops-observations.jsonl"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Extract exec commands
# ──────────────────────────────────────────────────────────────────────────────

def load_exec_commands(path):
    commands = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obs = json.loads(line)
            action = obs.get("action", {})
            if action.get("tool") == "exec":
                content = action.get("content", "")
                if content:
                    commands.append(content)
    return commands

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Categorize by semantic class
# ──────────────────────────────────────────────────────────────────────────────

LABEL_PATTERNS = [
    ("credential", [
        r'\.env', r'\.ssh', r'id_rsa', r'id_ed25519', r'api_key',
        r'grep.*key', r'grep.*token', r'grep.*secret', r'grep.*password',
        r'cat.*passwd', r'cat.*shadow', r'/etc/shadow', r'/\.netrc',
        r'credentials', r'keychain', r'\.pem', r'env \| grep',
    ]),
    ("network_egress", [
        r'curl', r'wget', r'ssh ', r'scp ', r'rsync', r'nc ', r'netcat',
    ]),
    ("destructive", [
        r'rm -rf', r'rm -r', r'kill', r'pkill', r'mkfs', r'dd if', r'truncate',
    ]),
    ("testing", [
        r'pytest', r'python -m pytest', r'jest', r'cargo test', r'go test', r'npm test',
    ]),
    ("git", [
        r'git ',
    ]),
    ("build", [
        r'pip install', r'npm install', r'make ', r'cargo build', r'python setup',
    ]),
    ("inspect", [
        r'cat ', r'grep ', r'find ', r'ls ', r'head ', r'tail ', r'wc ', r'stat ',
    ]),
    ("runtime", [
        r'python ', r'node ', r'bash ', r'python3 ',
    ]),
]

def classify_command(cmd):
    for label, patterns in LABEL_PATTERNS:
        for pat in patterns:
            if re.search(pat, cmd):
                return label
    return "benign_other"

def categorize_commands(commands):
    labels = [classify_command(cmd) for cmd in commands]
    return labels

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Embed with frozen MiniLM
# ──────────────────────────────────────────────────────────────────────────────

def embed_commands(commands):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(commands, batch_size=32, show_progress_bar=False)
    return embeddings.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Random projection to phasor space
# ──────────────────────────────────────────────────────────────────────────────

def project_to_phasor(embeddings):
    rng = np.random.default_rng(42)
    W = rng.standard_normal((512, 384)).astype(np.float32)
    projected_real = embeddings @ W.T  # (N, 512)
    phases = np.pi * np.tanh(projected_real)
    phasor_vecs = np.exp(1j * phases)  # unit magnitude complex vectors
    return phasor_vecs

# ──────────────────────────────────────────────────────────────────────────────
# Step 5 & 6: Compute similarities
# ──────────────────────────────────────────────────────────────────────────────

def phasor_cosine(a, b):
    """Re(a† · b) / (||a|| · ||b||). For unit vectors: Re(a† · b)."""
    return float(np.real(np.vdot(a, b)) / 512.0)  # normalize by dim for [0,1] range

def compute_phasor_similarities(vecs_by_label):
    """Returns dict: (label_a, label_b) -> mean cosine similarity."""
    results = {}
    labels = sorted(vecs_by_label.keys())

    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if j < i:
                continue
            vecs_a = vecs_by_label[la]
            vecs_b = vecs_by_label[lb]
            if len(vecs_a) == 0 or len(vecs_b) == 0:
                continue

            sims = []
            if la == lb:
                # Within-class: all pairs (i < j)
                for ii in range(len(vecs_a)):
                    for jj in range(ii + 1, len(vecs_b)):
                        sims.append(phasor_cosine(vecs_a[ii], vecs_b[jj]))
            else:
                # Between-class: all cross pairs (sample if large)
                pairs_a = vecs_a[:50]
                pairs_b = vecs_b[:50]
                for va in pairs_a:
                    for vb in pairs_b:
                        sims.append(phasor_cosine(va, vb))

            results[(la, lb)] = float(np.mean(sims)) if sims else 0.0

    return results

def compute_raw_similarities(embeddings_by_label):
    """Returns dict: (label_a, label_b) -> mean cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity
    results = {}
    labels = sorted(embeddings_by_label.keys())

    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if j < i:
                continue
            vecs_a = embeddings_by_label[la]
            vecs_b = embeddings_by_label[lb]
            if len(vecs_a) == 0 or len(vecs_b) == 0:
                continue

            if la == lb:
                if len(vecs_a) < 2:
                    results[(la, lb)] = 1.0
                    continue
                mat = cosine_similarity(vecs_a)
                # Upper triangle only (exclude diagonal)
                n = len(vecs_a)
                upper = [mat[ii][jj] for ii in range(n) for jj in range(ii+1, n)]
                results[(la, lb)] = float(np.mean(upper)) if upper else 1.0
            else:
                # Sample if large
                samp_a = vecs_a[:50]
                samp_b = vecs_b[:50]
                mat = cosine_similarity(samp_a, samp_b)
                results[(la, lb)] = float(np.mean(mat))

    return results

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 60)
    print("PROJECTION EXPERIMENT: MiniLM → Phasor Space")
    print("=" * 60)

    # Step 1
    print("\n[1] Loading exec commands...")
    commands = load_exec_commands(DATA_PATH)
    print(f"    Total exec observations: {len(commands)}")

    # Step 2
    print("\n[2] Categorizing commands...")
    labels = categorize_commands(commands)
    dist = Counter(labels)
    print("    Label distribution:")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"      {label:<20} {count:>4}")

    # Group by label
    label_to_commands = {}
    for cmd, lbl in zip(commands, labels):
        label_to_commands.setdefault(lbl, []).append(cmd)

    # Step 3
    print("\n[3] Embedding with MiniLM (all-MiniLM-L6-v2)...")
    embeddings = embed_commands(commands)
    print(f"    Embeddings shape: {embeddings.shape}")

    label_to_embeddings = {}
    for cmd, lbl, emb in zip(commands, labels, embeddings):
        label_to_embeddings.setdefault(lbl, []).append(emb)
    for lbl in label_to_embeddings:
        label_to_embeddings[lbl] = np.array(label_to_embeddings[lbl])

    # Step 4
    print("\n[4] Projecting to phasor space (C^512)...")
    phasor_vecs = project_to_phasor(embeddings)
    print(f"    Phasor shape: {phasor_vecs.shape}, dtype: {phasor_vecs.dtype}")

    label_to_phasors = {}
    for cmd, lbl, ph in zip(commands, labels, phasor_vecs):
        label_to_phasors.setdefault(lbl, []).append(ph)
    for lbl in label_to_phasors:
        label_to_phasors[lbl] = np.array(label_to_phasors[lbl])

    # Step 5 & 6: Compute similarities
    print("\n[5/6] Computing similarities (this may take ~30s)...")
    raw_sims = compute_raw_similarities(label_to_embeddings)
    phasor_sims = compute_phasor_similarities(label_to_phasors)

    # Step 7: Report
    ALL_LABELS = sorted(dist.keys())
    FOCUS_PAIRS = [
        ("credential", "credential"),
        ("testing", "testing"),
        ("git", "git"),
        ("inspect", "inspect"),
        ("network_egress", "network_egress"),
        ("destructive", "destructive"),
        ("build", "build"),
        ("runtime", "runtime"),
        ("benign_other", "benign_other"),
        ("credential", "testing"),
        ("credential", "git"),
        ("credential", "inspect"),
        ("credential", "network_egress"),
        ("credential", "build"),
        ("credential", "runtime"),
        ("credential", "benign_other"),
    ]

    def fmt_sim(sims, la, lb):
        key = (la, lb) if (la, lb) in sims else (lb, la)
        v = sims.get(key, float('nan'))
        return f"{v:.3f}"

    print("\n" + "=" * 65)
    print("=== MiniLM (R^384) separation ===")
    print(f"{'Class pair':<42} {'Mean cosine':>12}")
    print("-" * 65)
    for la, lb in FOCUS_PAIRS:
        if la not in dist or lb not in dist:
            continue
        tag = "(within)" if la == lb else "(between)"
        pair_str = f"{la} ↔ {lb}"
        print(f"  {pair_str:<40} {fmt_sim(raw_sims, la, lb):>8}  {tag}")

    print("\n" + "=" * 65)
    print("=== Phasor projection (C^512) separation ===")
    print(f"{'Class pair':<42} {'Mean cosine':>12}")
    print("-" * 65)
    for la, lb in FOCUS_PAIRS:
        if la not in dist or lb not in dist:
            continue
        tag = "(within)" if la == lb else "(between)"
        pair_str = f"{la} ↔ {lb}"
        print(f"  {pair_str:<40} {fmt_sim(phasor_sims, la, lb):>8}  {tag}")

    # Verdict
    raw_within_cred = raw_sims.get(("credential", "credential"), float('nan'))
    raw_btw_test = raw_sims.get(("credential", "testing"),
                                raw_sims.get(("testing", "credential"), float('nan')))
    ph_within_cred = phasor_sims.get(("credential", "credential"), float('nan'))
    ph_btw_test = phasor_sims.get(("credential", "testing"),
                                  phasor_sims.get(("testing", "credential"), float('nan')))

    raw_ratio = raw_within_cred / raw_btw_test if raw_btw_test else float('nan')
    ph_ratio = ph_within_cred / ph_btw_test if ph_btw_test else float('nan')

    print("\n" + "=" * 65)
    print("=== Verdict ===")
    print("Does random projection preserve separation?")
    print(f"Separation ratio (credential within / credential↔testing):")
    print(f"  MiniLM:  {raw_ratio:.3f}")
    print(f"  Phasor:  {ph_ratio:.3f}")

    if ph_ratio > 1.5:
        verdict = "projection preserves semantic structure — no training needed"
    elif ph_ratio >= 1.0:
        verdict = "some preservation — learned projection would help"
    else:
        verdict = "projection destroys structure — need learned projection"
    print(f"\n  → {verdict}")

    elapsed = time.time() - t0
    print(f"\n[Done] Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
