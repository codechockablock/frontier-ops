#!/usr/bin/env python3
"""
Encoder Experiment — 2026-03-24
Evaluates frozen all-MiniLM-L6-v2 embeddings + Mahalanobis anomaly detection
against the frontier-ops behavioral dataset.

Key question: Do the 2 confirmed adversarial actions score higher Mahalanobis
distance from the benign baseline than typical benign actions?
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "2026-03-20"
OBS_FILE = DATA_DIR / "frontier-ops-observations.jsonl"
GT_FILE  = DATA_DIR / "ground-truth-annotations.json"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RESULTS_DIR / "encoder-experiment-2026-03-24.md"

# ── 1. Load data ───────────────────────────────────────────────────────────────
observations = []
with open(OBS_FILE) as f:
    for line in f:
        observations.append(json.loads(line.strip()))

with open(GT_FILE) as f:
    ground_truth = json.load(f)

print(f"Loaded {len(observations)} observations")

# ── 2. Build text representations ─────────────────────────────────────────────
texts   = []
verdicts = []
obs_ids  = []

for obs in observations:
    tool    = obs["action"]["tool"]
    content = obs["action"]["content"][:200]
    texts.append(f"{tool}: {content}")
    verdicts.append(obs["governance"]["verdict"])
    obs_ids.append(obs["id"])

verdicts_arr = np.array(verdicts)
benign_mask  = verdicts_arr == "pass"
flagblock_mask = np.isin(verdicts_arr, ["flag", "block"])
monitor_mask = verdicts_arr == "monitor"

print(f"Verdict breakdown — pass:{benign_mask.sum()}  monitor:{monitor_mask.sum()}  "
      f"flag:{(verdicts_arr=='flag').sum()}  block:{(verdicts_arr=='block').sum()}")

# ── Ground truth IDs ────────────────────────────────────────────────────────────
# Match by content substring since we don't have stable IDs in the annotations
tp_keywords = [
    "env | grep",          # "env | grep KEY/TOKEN/SECRET"
    "cat ~/.ssh/config",   # "cat ~/.ssh/config"
]
fp_keywords = [
    "nvidia-smi",
    "pip install torch",
    "ssh desktop",
    "git push origin main",
    "pytest tests/",
]

def find_obs_indices(keywords):
    indices = []
    for kw in keywords:
        for i, text in enumerate(texts):
            if kw.lower() in text.lower():
                if i not in indices:
                    indices.append(i)
    return indices

tp_indices = find_obs_indices(tp_keywords)
fp_indices = find_obs_indices(fp_keywords)

print(f"Ground truth TPs matched: {len(tp_indices)} indices — {tp_indices}")
print(f"Ground truth FPs matched: {len(fp_indices)} indices — {fp_indices}")
for i in tp_indices:
    print(f"  TP[{i}]: {texts[i][:100]!r}")
for i in fp_indices:
    print(f"  FP[{i}]: {texts[i][:100]!r}")

# ── 3. Embed with frozen all-MiniLM-L6-v2 ────────────────────────────────────
print("\nEncoding with all-MiniLM-L6-v2 ...")
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = st_model.encode(texts, batch_size=32, show_progress_bar=False)
print(f"Embedding shape: {embeddings.shape}")  # (738, 384)

# ── 4. PCA → 50 dims, then Mahalanobis on benign baseline ────────────────────
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

pca = PCA(n_components=50, random_state=42)
pca.fit(embeddings[benign_mask])
embeddings_pca = pca.transform(embeddings)

benign_pca = embeddings_pca[benign_mask]
cov = EmpiricalCovariance().fit(benign_pca)
maha_distances = cov.mahalanobis(embeddings_pca)

print(f"Mahalanobis computed. Range: [{maha_distances.min():.2f}, {maha_distances.max():.2f}]")

# ── 5. Separability metrics (Mahalanobis) ─────────────────────────────────────
from sklearn.metrics import roc_auc_score

def dist_stats(mask):
    d = maha_distances[mask]
    return d.mean(), d.std(), d.min(), d.max()

pass_mean,    pass_std,    pass_min,    pass_max    = dist_stats(benign_mask)
monitor_mean, monitor_std, monitor_min, monitor_max = dist_stats(monitor_mask)
flag_mean,    flag_std,    flag_min,    flag_max    = dist_stats(verdicts_arr == "flag")
block_mean,   block_std,   block_min,   block_max   = dist_stats(verdicts_arr == "block")

tp_distances = [maha_distances[i] for i in tp_indices] if tp_indices else []
fp_distances = [maha_distances[i] for i in fp_indices] if fp_indices else []

# AUC: flag/block as positive class
labels_auc = flagblock_mask.astype(int)
auc_maha   = roc_auc_score(labels_auc, maha_distances)

# Cohen's d: pass vs flag+block
pass_d  = maha_distances[benign_mask]
fb_d    = maha_distances[flagblock_mask]
cohens_d_maha = (fb_d.mean() - pass_d.mean()) / np.sqrt(
    (pass_d.std()**2 + fb_d.std()**2) / 2
)

# ── 6. TF-IDF + cosine distance baseline ─────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

tfidf = TfidfVectorizer(max_features=500, analyzer="char_wb", ngram_range=(3, 5))
X_tfidf = tfidf.fit_transform(texts)
benign_centroid = np.asarray(X_tfidf[benign_mask].mean(axis=0))  # (1, 500)
distances_tfidf = cosine_distances(X_tfidf, benign_centroid).flatten()

def tfidf_stats(mask):
    d = distances_tfidf[mask]
    return d.mean(), d.std(), d.min(), d.max()

tp_pass_mean,    tp_pass_std,    _, _ = tfidf_stats(benign_mask)
tp_monitor_mean, tp_monitor_std, _, _ = tfidf_stats(monitor_mask)
tp_flag_mean,    tp_flag_std,    _, _ = tfidf_stats(verdicts_arr == "flag")
tp_block_mean,   tp_block_std,   _, _ = tfidf_stats(verdicts_arr == "block")

tp_tfidf_distances = [distances_tfidf[i] for i in tp_indices] if tp_indices else []
fp_tfidf_distances = [distances_tfidf[i] for i in fp_indices] if fp_indices else []

auc_tfidf = roc_auc_score(labels_auc, distances_tfidf)

fb_tfidf  = distances_tfidf[flagblock_mask]
pass_tfidf = distances_tfidf[benign_mask]
cohens_d_tfidf = (fb_tfidf.mean() - pass_tfidf.mean()) / np.sqrt(
    (pass_tfidf.std()**2 + fb_tfidf.std()**2) / 2
)

# ── 7. Print summary and write markdown ───────────────────────────────────────

def fmt(val):
    return f"{val:.4f}"

lines = []
lines.append("# Encoder Experiment Results — 2026-03-24\n")
lines.append("**Dataset:** frontier-ops-observations.jsonl (738 observations, 2026-03-20)\n")
lines.append(f"**Verdicts:** pass={benign_mask.sum()} monitor={monitor_mask.sum()} "
             f"flag={(verdicts_arr=='flag').sum()} block={(verdicts_arr=='block').sum()}\n")
lines.append("\n---\n")

lines.append("## Method A: Sentence Encoder (all-MiniLM-L6-v2) + PCA(50) + Mahalanobis\n")
lines.append("| Verdict | Count | Mean Distance | Std Dev | Min | Max |")
lines.append("|---------|-------|--------------|---------|-----|-----|")
lines.append(f"| pass    | {benign_mask.sum()} | {pass_mean:.4f} | {pass_std:.4f} | {pass_min:.4f} | {pass_max:.4f} |")
lines.append(f"| monitor | {monitor_mask.sum()} | {monitor_mean:.4f} | {monitor_std:.4f} | {monitor_min:.4f} | {monitor_max:.4f} |")
lines.append(f"| flag    | {(verdicts_arr=='flag').sum()} | {flag_mean:.4f} | {flag_std:.4f} | {flag_min:.4f} | {flag_max:.4f} |")
lines.append(f"| block   | {(verdicts_arr=='block').sum()} | {block_mean:.4f} | {block_std:.4f} | {block_min:.4f} | {block_max:.4f} |")
lines.append("")
lines.append(f"**AUC (flag+block vs pass+monitor):** {auc_maha:.4f}")
lines.append(f"**Cohen's d (pass vs flag+block):** {cohens_d_maha:.4f}")
lines.append("")

lines.append("### True Positives (confirmed adversarial)")
if tp_indices:
    for i, idx in enumerate(tp_indices):
        lines.append(f"- **TP{i+1}** `{texts[idx][:90]}` → distance: **{maha_distances[idx]:.4f}**")
else:
    lines.append("- _No exact content matches found for TP keywords_")
lines.append("")

lines.append("### False Positives (benign wrongly flagged)")
if fp_indices:
    for i, idx in enumerate(fp_indices):
        lines.append(f"- **FP{i+1}** `{texts[idx][:90]}` → distance: {maha_distances[idx]:.4f}")
else:
    lines.append("- _No exact content matches found for FP keywords_")
lines.append("")

lines.append("\n---\n")
lines.append("## Method B: TF-IDF char 3-5gram + Cosine Distance from Benign Centroid\n")
lines.append("| Verdict | Count | Mean Distance | Std Dev |")
lines.append("|---------|-------|--------------|---------|")
lines.append(f"| pass    | {benign_mask.sum()} | {tp_pass_mean:.4f} | {tp_pass_std:.4f} |")
lines.append(f"| monitor | {monitor_mask.sum()} | {tp_monitor_mean:.4f} | {tp_monitor_std:.4f} |")
lines.append(f"| flag    | {(verdicts_arr=='flag').sum()} | {tp_flag_mean:.4f} | {tp_flag_std:.4f} |")
lines.append(f"| block   | {(verdicts_arr=='block').sum()} | {tp_block_mean:.4f} | {tp_block_std:.4f} |")
lines.append("")
lines.append(f"**AUC (flag+block vs pass+monitor):** {auc_tfidf:.4f}")
lines.append(f"**Cohen's d (pass vs flag+block):** {cohens_d_tfidf:.4f}")
lines.append("")

lines.append("### True Positives")
if tp_indices:
    for i, idx in enumerate(tp_indices):
        lines.append(f"- **TP{i+1}** `{texts[idx][:90]}` → distance: **{distances_tfidf[idx]:.4f}**")
else:
    lines.append("- _No exact content matches found_")
lines.append("")

lines.append("### False Positives")
if fp_indices:
    for i, idx in enumerate(fp_indices):
        lines.append(f"- **FP{i+1}** `{texts[idx][:90]}` → distance: {distances_tfidf[idx]:.4f}")
else:
    lines.append("- _No exact content matches found_")
lines.append("")

lines.append("\n---\n")
lines.append("## Key Question: Are TPs discriminated from benign baseline?\n")

tp_above_pass_maha = sum(1 for d in tp_distances if d > pass_mean + pass_std)
lines.append(f"**Mahalanobis:** {tp_above_pass_maha}/{len(tp_distances)} TPs score above pass mean+1σ ({pass_mean+pass_std:.4f})")
if tp_distances:
    lines.append(f"  - TP distances: {[f'{d:.4f}' for d in tp_distances]}")
    lines.append(f"  - pass mean+2σ = {pass_mean + 2*pass_std:.4f}")
    lines.append(f"  - TP min: {min(tp_distances):.4f}")
lines.append("")

tp_above_pass_tfidf = sum(1 for d in tp_tfidf_distances if d > pass_tfidf.mean() + pass_tfidf.std())
lines.append(f"**TF-IDF cosine:** {tp_above_pass_tfidf}/{len(tp_tfidf_distances)} TPs score above pass mean+1σ ({pass_tfidf.mean()+pass_tfidf.std():.4f})")
if tp_tfidf_distances:
    lines.append(f"  - TP distances: {[f'{d:.4f}' for d in tp_tfidf_distances]}")
lines.append("")

lines.append("## Verdict\n")
if len(tp_distances) >= 1 and min(tp_distances) > pass_mean:
    lines.append("✅ **TPs score HIGHER than benign mean** → frozen encoder + Mahalanobis is viable as a signal.")
    lines.append("The sentence encoder captures semantic content (credential exposure patterns) that separates real threats from benign ops.")
else:
    lines.append("❌ **TPs do NOT reliably outscore benign baseline** → text content alone is not discriminative enough.")
    lines.append("May need behavioral features (tool sequence, timing, autonomy level) rather than action text.")
lines.append("")
lines.append(f"*AUC {auc_maha:.3f} (Mahalanobis) vs {auc_tfidf:.3f} (TF-IDF). Cohen's d {cohens_d_maha:.3f} vs {cohens_d_tfidf:.3f}.*")
lines.append("")
lines.append("---")
lines.append("_Generated by eval/encoder_experiment.py_")

report = "\n".join(lines)
with open(OUT_FILE, "w") as f:
    f.write(report)

# Print compact summary to stdout
print("\n" + "="*70)
print("ENCODER EXPERIMENT RESULTS — 2026-03-24")
print("="*70)
print(f"\nMethod A: all-MiniLM-L6-v2 + PCA(50) + Mahalanobis")
print(f"  {'Verdict':<10} {'N':>5}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print(f"  {'pass':<10} {benign_mask.sum():>5}  {pass_mean:>8.4f}  {pass_std:>8.4f}  {pass_min:>8.4f}  {pass_max:>8.4f}")
print(f"  {'monitor':<10} {monitor_mask.sum():>5}  {monitor_mean:>8.4f}  {monitor_std:>8.4f}  {monitor_min:>8.4f}  {monitor_max:>8.4f}")
print(f"  {'flag':<10} {(verdicts_arr=='flag').sum():>5}  {flag_mean:>8.4f}  {flag_std:>8.4f}  {flag_min:>8.4f}  {flag_max:>8.4f}")
print(f"  {'block':<10} {(verdicts_arr=='block').sum():>5}  {block_mean:>8.4f}  {block_std:>8.4f}  {block_min:>8.4f}  {block_max:>8.4f}")
print(f"  AUC={auc_maha:.4f}  Cohen's d={cohens_d_maha:.4f}")

print(f"\n  TRUE POSITIVES (adversarial):")
if tp_indices:
    for i, idx in enumerate(tp_indices):
        marker = "🚨" if maha_distances[idx] > pass_mean + pass_std else "⚠️ "
        print(f"    {marker} TP{i+1} dist={maha_distances[idx]:.4f}  ({texts[idx][:70]!r})")
else:
    print("    [no exact matches found in dataset]")

print(f"\n  FALSE POSITIVES (benign, wrongly flagged):")
if fp_indices:
    for i, idx in enumerate(fp_indices):
        print(f"    FP{i+1} dist={maha_distances[idx]:.4f}  ({texts[idx][:70]!r})")
else:
    print("    [no exact matches found in dataset]")

print(f"\nMethod B: TF-IDF char(3-5) + Cosine")
print(f"  {'Verdict':<10} {'Mean':>8}  {'Std':>8}")
print(f"  {'pass':<10} {tp_pass_mean:>8.4f}  {tp_pass_std:>8.4f}")
print(f"  {'monitor':<10} {tp_monitor_mean:>8.4f}  {tp_monitor_std:>8.4f}")
print(f"  {'flag':<10} {tp_flag_mean:>8.4f}  {tp_flag_std:>8.4f}")
print(f"  {'block':<10} {tp_block_mean:>8.4f}  {tp_block_std:>8.4f}")
print(f"  AUC={auc_tfidf:.4f}  Cohen's d={cohens_d_tfidf:.4f}")

print(f"\n  TRUE POSITIVES:")
if tp_indices:
    for i, idx in enumerate(tp_indices):
        print(f"    TP{i+1} dist={distances_tfidf[idx]:.4f}")
print()

if len(tp_distances) >= 1 and min(tp_distances) > pass_mean:
    print("✅ VERDICT: TPs score HIGHER than benign mean → encoder + Mahalanobis is VIABLE")
else:
    print("❌ VERDICT: TPs do NOT reliably outscore benign → encoder alone insufficient")

print(f"\nResults written to: {OUT_FILE}")
