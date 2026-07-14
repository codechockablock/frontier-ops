# Empirical Validation Brainstorm

## Multi-Modal Understanding Geometry — Pushing on the Framework

*Generated 2026-03-28. This document is adversarial-constructive: it takes the paper's claims seriously enough to design experiments that could kill them.*

---

## 1. What Would Actually Falsify This Framework?

The framework makes several claims that go beyond "this is a nice metaphor." Each needs a corresponding null hypothesis.

### Sharp Predictions vs. Hand-Waving

**Sharp predictions (testable):**
- Adding a modality can only shrink the stable kernel (monotonicity). Falsified if adding a genuinely non-redundant modality *increases* the set of globally-consistent understandings.
- Phantom understanding exists: there are cases where all pairwise modality agreements are high but the triple-intersection agreement is low (Borromean structure). Falsified if pairwise consistency always implies global consistency in practice.
- Projection aliasing is real: distinct root causes can produce identical eval signatures. This is almost certainly true in general, but the framework claims the *specific* modality decomposition it proposes is the right one to break the aliasing. Falsified if a simpler decomposition (e.g., just "quantitative vs. qualitative") works equally well.

**Hand-wavy parts (not yet testable without more specification):**
- The curvature interpretation (flat/positive/negative) is metaphorical until you specify a metric on the understanding manifold. Currently unfalsifiable because you can always redefine the metric post-hoc.
- The claim that $\mathcal{M}$ is Riemannian (as opposed to, say, a discrete graph or a topological space without smooth structure) is a modeling choice with no proposed test.
- The sheaf axioms (gluing + locality) are *assumed* to hold. What if they don't? What if understanding is fundamentally non-local — knowing about theory and ops separately doesn't determine what you know about their intersection?

### The Critical Falsification Experiment

**Null hypothesis:** The sheaf/cohomology framework provides no predictive advantage over simpler disagreement measures.

**Simpler baselines to beat:**
1. **Pairwise correlation** — just compute correlations between modality-derived features. If $H^1 \neq 0$ always co-occurs with low pairwise correlation, the cohomological framing adds nothing.
2. **PCA disagreement** — project all modalities into a shared PCA space, measure reconstruction error. If this catches the same failures as the cohomology approach, Occam's razor kills the framework.
3. **Ensemble disagreement** — train separate models on each modality, measure prediction disagreement. This is basically what cross-modal prediction residuals do, minus the cohomological interpretation.
4. **Simple majority vote** — for each claim about agent behavior, check if modalities agree by majority. If majority-vote disagreement predicts the same failures as $H^1$, the topology is decorative.

**The framework survives only if:** there exist empirically observable failure modes that (a) the cohomological measure detects but the baselines miss, or (b) the cohomological measure detects *earlier* or *more precisely*. The three predicted failure modes are the candidates — especially phantom understanding, which by definition involves pairwise agreement masking global disagreement.

---

## 2. Concrete Measurement of $H^1$ — Operationalizing Gluing Obstructions

### What the Čech Cochain Actually Looks Like

Let's get concrete. You have four modalities for an AI agent governance system:

| Modality | Raw Data | Representation Space $\mathcal{V}_i$ |
|----------|----------|--------------------------------------|
| Theory | Design docs, architecture specs, safety cases | Concept graph embeddings (e.g., node2vec on a knowledge graph of claimed properties) |
| Memory | Reasoning traces, chain-of-thought logs, retrieval-augmented generation context | Sequence embeddings (e.g., mean-pooled transformer representations of trace segments) |
| Operations | Deployment logs, latency, error rates, resource usage, API call patterns | Time-series feature vectors (windowed statistics) |
| Evaluation | Benchmark scores, red-team results, human preference ratings | $\mathbb{R}^k$ directly |

**Step 1: Define overlaps.** The "open cover" is over some domain $X$ — let's say $X$ is the set of *agent capabilities* or *behavioral dimensions* (e.g., "truthfulness," "instruction following," "safety under adversarial input," "reasoning coherence"). Each modality covers a subset:
- Theory covers all claimed capabilities (broad, shallow)
- Memory covers capabilities exercised during logged sessions (sparse, deep)
- Ops covers capabilities that produce measurable deployment signals (dense, bounded)
- Eval covers capabilities with benchmarks (structured, finite)

The overlap $U_i \cap U_j$ is the set of capabilities addressed by both modalities $i$ and $j$.

**Step 2: Train cross-modal predictors.** For each overlap $(i,j)$, train $f_{ij}: \mathcal{V}_i|_{U_i \cap U_j} \to \mathcal{V}_j|_{U_i \cap U_j}$. Concretely:
- $f_{\text{theory,eval}}$: Given concept-graph features for a capability, predict its benchmark scores.
- $f_{\text{memory,ops}}$: Given reasoning trace embeddings, predict operational statistics.
- Etc.

**Step 3: Compute residuals.** The 1-cochain assigns to each overlap the residual $r_{ij} = f_{ij}(\sigma_i) - \sigma_j$. This is a vector in $\mathcal{V}_j$.

**Step 4: Check the cocycle condition.** On triple overlaps $U_i \cap U_j \cap U_k$, check whether $r_{ij} + r_{jk} + r_{ki} \approx 0$. If yes, the 1-cochain is a coboundary (trivially cohomologous to zero — the disagreements are "calibration errors" fixable by adjusting individual modality representations). If not, you have a genuine $H^1$ class.

### Practical Concerns

**Problem 1: The predictor $f_{ij}$ introduces its own error.** You can't distinguish "genuine gluing obstruction" from "my neural net is bad at this regression." 

*Mitigation:* Use held-out calibration. Split the overlap data. Train $f_{ij}$ on one half, evaluate on the other. Establish a baseline residual level for each pair. The $H^1$ signal is the *excess* residual on triple overlaps beyond what pairwise prediction error would produce.

**Problem 2: What's the topology of the overlap?** The paper assumes a nice open cover, but in practice the "capabilities" might not form a topological space with well-defined overlaps. 

*Mitigation:* Use a discrete nerve complex. Each capability is a vertex. Modalities define simplices (a modality covering capabilities $\{c_1, c_2, c_3\}$ gives a 2-simplex). The nerve complex is the simplicial complex of the cover. This is well-defined and computable.

**Problem 3: The representation spaces $\mathcal{V}_i$ are heterogeneous.** You can't directly subtract a concept-graph embedding from an $\mathbb{R}^k$ eval vector. 

*Mitigation:* Work in a shared embedding space. Map all modalities to a common latent space via learned encoders (contrastive learning, CCA, etc.), then compute residuals there. This is standard multi-modal fusion, but the cohomological interpretation adds the triple-overlap consistency check. Alternatively, use scalar-valued cochains: for each pair $(i,j)$, the cochain value is a scalar disagreement measure (e.g., mutual information deficit, prediction $R^2$ gap).

### Concrete Data Requirements

For a real agent governance system, you need:

1. **Theory corpus:** All design documents, safety cases, capability claims, architecture specs. Must be *versioned* — you need to track when theory changes.
2. **Memory traces:** Complete reasoning chains for a representative sample of interactions. Including retrieval context, intermediate steps, and final outputs.
3. **Operational logs:** Time-stamped deployment telemetry. Error rates, latency distributions, API patterns, resource usage. Windowed into epochs matching eval cadence.
4. **Evaluation results:** Benchmark suite results over time. Red-team findings. Human preference data.
5. **Ground truth labels:** For at least a subset of "capabilities," you need independent assessment of whether the agent actually has that capability. This is the hardest part — it requires expert annotation.

### What the Output Looks Like

The $H^1$ measurement produces a **set of capabilities** (or capability-clusters) where:
- Each pair of modalities agrees about the agent's behavior
- But the full set of modalities is globally inconsistent

Example output: "For the capability 'multi-step reasoning under adversarial input,' theory docs claim robustness, memory traces show successful chains, eval benchmarks pass — but operational logs show anomalous latency spikes and error patterns inconsistent with genuine robust reasoning. Any two modalities can be reconciled, but all three cannot."

---

## 3. Experiments for the Three Predicted Failure Modes

### Experiment 1: Curvature-Masked Drift

**Prediction:** There exist cases where theory undergoes significant revision (high curvature in theory-space) while evaluation metrics remain flat, and this divergence has operational consequences that *would* be visible in ops/memory modalities.

**Design:**

*Setup:* Track an AI agent through a period where its underlying model or system prompt is significantly updated (e.g., a major version bump, a safety patch, a fine-tuning run). Collect all four modalities before, during, and after.

*Measurement:*
- **Theory change:** Compute semantic distance between pre- and post-update design docs / safety cases using document embeddings. Measure the magnitude of concept-graph edit distance.
- **Eval trajectory:** Track benchmark scores through the update. Compute the derivative (should be ~0 for flat trajectory).
- **Ops divergence:** Compute distributional shift in operational logs (KL divergence, MMD, etc.) between pre- and post-update.
- **Memory shift:** Compare reasoning trace distributions before and after.

*Detection criterion:* Curvature-masked drift exists when:
$$\|\Delta_{\text{theory}}\| > \tau_1 \quad \text{AND} \quad \|\Delta_{\text{eval}}\| < \tau_2 \quad \text{AND} \quad \|\Delta_{\text{ops}}\| > \tau_3$$

*Control condition:* 
- **Positive control:** A known-benign update (e.g., prompt reformatting with no semantic change). All deltas should be small.
- **Negative control:** A known-significant update (e.g., switching base models). All deltas should be large.
- The interesting case is when theory and ops diverge but eval doesn't — that's the predicted failure mode.

*What would falsify it:* If every theory change that produces ops divergence also produces eval divergence (no masking), the prediction fails. Alternatively, if ops divergence without eval divergence never correlates with downstream problems (false positive), the prediction is uninteresting.

*Data needed:* At minimum 10-20 update events for statistical power. Ideally from multiple agent systems.

### Experiment 2: Projection Aliasing

**Prediction:** There exist distinct root causes of agent failure that produce identical (or near-identical) evaluation signatures but are distinguishable via other modalities.

**Design:**

*Setup:* Curate a dataset of known agent failures with documented root causes. Classify root causes into categories (e.g., knowledge gap, reasoning error, retrieval failure, safety bypass, hallucination, instruction misunderstanding).

*Measurement:*
- For each failure, compute the eval-space signature (which benchmarks it would affect, by how much).
- Cluster failures by eval signature similarity.
- Within each cluster, check whether root causes are homogeneous (one cause per cluster) or heterogeneous (multiple causes, same eval signature).

*The framework's claim:* Some clusters will be heterogeneous — failures with different causes that eval can't distinguish. Adding memory traces or ops logs will break these clusters into subclusters with homogeneous causes.

*Control condition:* 
- If eval-signature clusters are already homogeneous (each cluster = one root cause), projection aliasing doesn't exist in this system.
- If adding more modalities doesn't improve cluster purity, the framework's specific modality decomposition isn't useful.

*Baseline:* Compare the cluster-purity improvement from adding framework-specified modalities vs. adding arbitrary additional features (e.g., random projections of the same data, or a single "everything-else" feature vector). If the specific modality structure doesn't matter, the framework's contribution is minimal.

*What would falsify it:* (a) Eval signatures already uniquely identify root causes, or (b) any additional information (not specifically the paper's modalities) breaks the aliasing equally well.

*Data needed:* A failure taxonomy with ≥50 labeled failures, each with eval scores, reasoning traces, and operational logs.

### Experiment 3: Phantom Understanding

**Prediction:** There exist agent capabilities where any two modalities agree but all three (or four) disagree — the Borromean rings structure.

**Design:**

This is the most distinctive prediction and the hardest to test, because it requires showing that the *triple-wise* inconsistency isn't reducible to pairwise inconsistency.

*Setup:* For a set of capabilities $\{c_1, \ldots, c_n\}$, compute:
- Pairwise consistency scores: $s_{ij}(c) = 1 - \|r_{ij}(c)\| / \|r_{ij}\|_{\max}$ for each pair $(i,j)$ and capability $c$.
- Triple consistency scores: For each triple $(i,j,k)$, measure the cocycle residual $\|r_{ij}(c) + r_{jk}(c) + r_{ki}(c)\|$.

*Detection criterion:* Phantom understanding at capability $c$ iff:
$$\min_{(i,j)} s_{ij}(c) > \theta_{\text{high}} \quad \text{AND} \quad \max_{(i,j,k)} \|r_{ij} + r_{jk} + r_{ki}\|(c) > \theta_{\text{low}}$$

i.e., all pairs look consistent but the cycle doesn't close.

*The key difficulty:* This requires that your cross-modal predictors $f_{ij}$ are good enough that low pairwise residual actually means pairwise consistency, not just that both predictors are equally bad. You need to carefully calibrate prediction quality.

*Control condition:*
- **Synthetic phantom:** Construct an artificial agent where you *engineer* a Borromean failure. For example: theory says "model handles adversarial inputs safely," memory traces show safe handling in logged sessions, ops logs show safe metrics — but the safety is achieved by different mechanisms in each case (theory assumes alignment, memory shows avoidance, ops reflects rate-limiting), and these mechanisms are mutually inconsistent when the agent faces novel adversarial inputs.
- **Synthetic non-phantom:** Construct an agent with genuine pairwise disagreement (some pairs are inconsistent) — the standard case. Verify the method correctly identifies this as pairwise failure, not phantom.

*What would falsify it:* If phantom understanding (high pairwise, low triple) never occurs in practice — if all global inconsistencies are always visible as pairwise inconsistencies — then $H^1$ is always detectable from pairwise measures and the cohomological framing adds no value.

*Data needed:* Same as §2 above, but with enough capabilities (≥30) to have statistical power for detecting the Borromean pattern.

---

## 4. VSA Coherence Spectrum as Cohomology Proxy

### The Validation Problem

The paper claims that the VSA coherence spectrum $m_k = |\frac{1}{n}\sum_i v_{i,k}|$ serves as a proxy for section agreement, and that its distribution profile (unimodal near 1, bimodal, or uniform near $1/\sqrt{n}$) indicates the degree of cohomological obstruction.

### What Ground Truth Looks Like

To validate this proxy, you need cases where you *independently know* whether modalities agree or disagree, then check whether the VSA spectrum reflects this.

**Ground truth construction:**

1. **Synthetic agreement:** Create modality representations that are identical (perfect agreement). VSA spectrum should be all ~1.
2. **Synthetic disagreement:** Create representations that are randomly related. VSA spectrum should be all ~$1/\sqrt{n}$.
3. **Synthetic partial agreement:** Create representations where 3 of 4 modalities agree and 1 disagrees. VSA spectrum should be bimodal.
4. **Known real-world agreement:** Find an agent capability where all modalities demonstrably agree (e.g., a well-benchmarked, well-understood capability with consistent logs and accurate docs). Measure VSA spectrum.
5. **Known real-world disagreement:** Find a capability with known modality conflicts (e.g., a benchmark that was later shown to be gamed, producing high eval but low ops performance). Measure VSA spectrum.

### Specific Validation Protocol

**Phase 1: Synthetic calibration**
- Generate 1000 synthetic 4-modality bundles with controlled agreement levels (0%, 25%, 50%, 75%, 100%).
- Compute VSA coherence spectrum for each.
- Establish the mapping from agreement level to spectrum profile.
- Measure sensitivity: how much disagreement is needed before the spectrum reliably detects it?

**Phase 2: Semi-synthetic validation**
- Take real modality data from a known-good system.
- Inject controlled perturbations into one modality at a time.
- Verify that the VSA spectrum degrades proportionally to the perturbation magnitude.

**Phase 3: Real-world validation**
- Apply to the real agent system.
- Correlate VSA coherence values with independently assessed modality agreement (expert ratings).
- Compute ROC curve: does low VSA coherence predict expert-identified disagreements?

### Known Weaknesses

1. **VSA encoding is lossy.** The phasor encoding $v_{i,k} = e^{i\theta_{i,k}}$ discards magnitude information. Two modalities could agree on direction but disagree on confidence — the VSA spectrum wouldn't catch this. This is a genuine information loss, not just an approximation.

2. **Dimensionality mismatch.** The phasor vector has dimension $d$ (the VSA dimension). Each component $k$ is treated independently. But modality representations have internal structure — correlations between dimensions matter. The VSA spectrum treats components as independent, which may miss structured disagreements.

3. **No natural phasor encoding for all modalities.** Eval scores in $\mathbb{R}^k$ don't have a natural circular structure. The encoding from $\mathbb{R}^k$ to $S^1$ (the unit circle) requires an arbitrary choice of mapping. Different mappings could produce different coherence spectra.

4. **The "all ~1/√n" case is indistinguishable from noise.** If modalities genuinely contain independent information about the same capability (which is the *point* of multi-modal assessment), their VSA vectors might naturally have low coherence without any "disagreement." Independence ≠ inconsistency.

---

## 5. Stable Kernel Measurement — Operationalizing $\dim H^0$

### The Problem

$H^0 = \mathcal{K} = \bigcap_i \ker(\pi_i - \pi_i)$ — the set of understandings consistent across all modalities. But $\mathcal{M}$ is not directly observable, so you can't compute $\pi_i$ directly.

### Proposed Estimator: Consensus Dimensionality

**Idea:** The stable kernel is the subspace of "things all modalities agree on." Without access to $\mathcal{M}$, estimate it as the dimensionality of the space spanned by *consistent* cross-modal predictions.

**Concrete procedure:**

1. For each modality pair $(i,j)$, train bidirectional predictors $f_{ij}$ and $f_{ji}$.
2. Compute the "consistent subspace" for pair $(i,j)$: the subspace of $\mathcal{V}_i$ where $f_{ji}(f_{ij}(x)) \approx x$ (round-trip consistency). Measure this as the number of principal components with round-trip reconstruction $R^2 > \tau$.
3. The stable kernel dimension estimate is:
$$\widehat{\dim H^0} = \min_{(i,j)} \dim(\text{consistent subspace}_{ij})$$
or more precisely, the dimensionality of the intersection of all pairwise consistent subspaces projected into a common space.

**Alternative estimator: Canonical Correlation Analysis (CCA)**

CCA between all modality pairs gives canonical correlations. The number of significant canonical correlations (above a threshold) estimates the dimensionality of shared information. The minimum over all pairs gives a $\dim H^0$ estimate.

**Advantages of the CCA approach:**
- Well-understood statistical properties.
- Efficient to compute.
- Natural interpretation: canonical correlations = dimensions of agreement.

**Disadvantages:**
- CCA is linear. The stable kernel might require nonlinear relationships.
- CCA is pairwise. Extending to the full intersection requires multiway CCA, which is less standard.
- CCA measures *shared variance*, not *consistent understanding*. These are related but not identical.

**Better alternative: Shared latent space models**

Fit a shared latent variable model (e.g., multi-view VAE, or DCCA — deep CCA) where a latent vector $z$ generates all modalities. The dimensionality of $z$ that achieves good reconstruction across all modalities estimates $\dim H^0$.

**Validation:** 
- Synthetic test: Create a system with a *known* latent space of dimension $d$, with four noisy projections. Verify the estimator recovers $d$.
- Monotonicity test: Add a modality. The estimate should decrease or stay the same (never increase). Increase iff the new modality is redundant with existing ones.
- Redundancy test: Add a noisy copy of an existing modality. The estimate should stay the same.

---

## 6. The Minimal Viable Experiment

If I had to pick ONE experiment to demonstrate the framework's value:

### The Phantom Understanding Detector

**Why this one:** It's the most distinctive prediction. Pairwise disagreement measures are standard. The claim that $H^1 \neq 0$ can exist even when all pairwise measures look clean — that's the unique contribution. If you can demonstrate one real case of phantom understanding, the framework instantly justifies itself.

**Minimal setup:**

1. **System:** Pick one well-instrumented AI agent system with at least 3 modalities of documentation/monitoring (e.g., design docs + reasoning traces + eval scores, or safety case + deployment logs + benchmark results).

2. **Capabilities:** Identify 20-50 capabilities or behavioral dimensions that all three modalities address.

3. **Pairwise consistency:** For each pair of modalities, train a cross-modal predictor. Compute residuals. Identify capabilities where all pairwise residuals are low.

4. **Triple consistency:** For the "all-pairwise-consistent" capabilities, compute the cocycle residual $r_{12} + r_{23} + r_{31}$. If any have high cocycle residual, you've found a phantom.

5. **Validate the phantom:** Take the flagged capability. Do a deep-dive investigation. Is there *actually* a problem that the pairwise checks missed? If yes — if the phantom understanding corresponds to a real failure mode — the framework is validated.

6. **Compare to baselines:**
   - Simple average of pairwise residuals (should NOT detect the phantom).
   - PCA-based anomaly detection across all modalities (might or might not detect it).
   - Ensemble disagreement (might or might not detect it).
   - If the cohomological measure is the *only* one that detects the phantom, the framework wins.

**Sample size for significance:** You need enough capabilities that finding ≥1 phantom with high cocycle residual is unlikely under the null (no phantoms exist). Under the null, the cocycle residual should be distributed as the sum of three independent pairwise residuals — testable via permutation.

**Expected duration:** 2-4 weeks for a single system, assuming the monitoring infrastructure already exists.

**Cost:** Primarily in expert time for the deep-dive validation (step 5). The computational cost is modest — cross-modal predictors are just regression models.

---

## 7. Known Weaknesses and Gaps

### Method 1: Cross-Modal Prediction Residuals

**What could go wrong:**

1. **Predictor quality confound.** If $f_{ij}$ is a bad predictor, high residuals reflect model inadequacy, not modality disagreement. You'd detect "my regression is bad" and call it "$H^1 \neq 0$."
   - *Mitigation:* Calibrate against a null distribution. Use held-out data to establish baseline prediction quality. Only flag residuals that are significantly above baseline.

2. **Training data leakage.** If the modalities are generated by the same underlying system, the predictor might learn to exploit shared artifacts rather than genuine cross-modal relationships.
   - *Mitigation:* Use temporal splits. Train on historical data, test on future data.

3. **The cocycle condition is checked approximately.** In finite samples, $r_{ij} + r_{jk} + r_{ki} \approx 0$ is always only approximately satisfied. You need a principled threshold for "approximately zero."
   - *Mitigation:* Permutation testing. Shuffle modality assignments, recompute cocycle residuals, establish the null distribution.

4. **Non-uniqueness of $f_{ij}$.** Different predictor architectures could give different residuals. Is the $H^1$ measurement stable across architectures?
   - *Mitigation:* Use multiple predictor families (linear, MLP, kernel methods). Report $H^1$ estimates as a range.

5. **Curse of dimensionality.** If modality representations are high-dimensional and data is sparse, the residuals will be dominated by noise.
   - *Mitigation:* Dimensionality reduction first (PCA, UMAP), then cross-modal prediction. But this discards information — potential for missing real obstructions in the discarded dimensions.

### Method 2: VSA Coherence Spectrum

**What could go wrong:**

1. **Arbitrary encoding.** The mapping from modality representation to phasor vector is a choice with no canonical answer. Different encodings → different coherence spectra. The method is not encoding-invariant.
   - *Severity:* High. This is a fundamental issue. The paper doesn't address it.

2. **Dimensionality of the VSA.** The number of dimensions $d$ affects the spectrum. Too few → spectrum is noisy. Too many → everything looks incoherent because random high-dimensional vectors are nearly orthogonal.
   - *Mitigation:* The paper should specify how $d$ is chosen. Likely needs to be tuned to the problem.

3. **No theoretical guarantee connecting VSA coherence to sheaf cohomology.** The paper says it's a "proxy" but doesn't prove an equivalence or even a monotone relationship. It's possible that VSA coherence is high but $H^1 \neq 0$, or vice versa.
   - *Severity:* Medium-high. Without this guarantee, the VSA method is a heuristic, not a measurement of cohomology.

4. **Phase wrapping.** Phasor arithmetic is periodic. If the encoding maps semantically close representations to distant phases (near 0 and $2\pi$), the bundling will cancel when it shouldn't.
   - *Mitigation:* Use multi-resolution encodings or bipolar VSA (binary spatter codes) instead of circular phasors.

### Method 3: Persistent Cohomology via Filtration

**What could go wrong:**

1. **Computational cost.** Persistent homology is $O(n^3)$ in the number of simplices. For a fine-grained cover with many capabilities and modalities, the nerve complex could be large.
   - *Mitigation:* Approximate methods (e.g., Ripser). Or use a coarse cover and accept resolution loss.

2. **Threshold sensitivity.** The filtration parameter $\varepsilon$ determines which overlaps exist. The "persistence" is supposed to handle this (long-lived features are real), but in practice, the birth-death diagram might be hard to interpret.
   - *Mitigation:* Use standard persistence significance tests (bottleneck stability, confidence sets).

3. **The nerve theorem requires a good cover.** If the modality "open sets" aren't convex (or contractible) in some meaningful sense, the nerve complex doesn't faithfully represent the topology of the cover.
   - *Severity:* Theoretical concern. In practice, you're not actually computing topology of a space — you're computing combinatorial cohomology of a simplicial complex defined by consistency thresholds. The nerve theorem gives theoretical backing but isn't strictly necessary.

4. **H^1 vs. noise.** Short-lived $H^1$ classes are dismissed as "calibration issues." But what's the cutoff between short-lived and long-lived? This is the standard persistence threshold problem.
   - *Mitigation:* Use the persistence entropy or the gap in the persistence diagram as the cutoff. Compare to the persistence diagram of a random simplicial complex of the same size.

5. **Conflation of $H^1$ sources.** Persistent $H^1$ could come from genuinely different sources — measurement error vs. fundamental disagreement vs. stale data. The persistence diagram alone doesn't distinguish these.
   - *Mitigation:* Annotate each $H^1$ generator with the modalities and capabilities involved. Domain expert review is necessary.

### Cross-Cutting Weaknesses

1. **No existing ground truth for "understanding."** The entire framework rests on an unobservable latent variable ($\mathcal{M}$). Every empirical test is indirect. You can never *prove* the latent structure exists — only that modeling it as if it exists produces useful predictions.

2. **The modality decomposition is a choice, not a fact.** Why these four modalities? Why not five? Or three? The framework doesn't tell you the optimal decomposition. Different decompositions could give different cohomology. This is the "optimal cover design" open question from §7, and it's not just a technical detail — it's foundational.

3. **Stationarity assumption.** The framework treats $\mathcal{M}$ as a fixed manifold. In reality, the "understanding" of an agent system evolves over time. The paper acknowledges this ("dynamic cohomology") but doesn't solve it. All empirical methods need to account for temporal drift.

4. **Scalability to real systems.** A real agent governance system might have thousands of capabilities, dozens of evaluation benchmarks, millions of log entries, and hundreds of design documents. The cross-modal prediction approach requires $O(n^2)$ predictors (one per modality pair). The VSA approach requires encoding all of these into a common-dimensional phasor space. The persistent homology approach requires building and analyzing a potentially enormous simplicial complex.

5. **The framework might be right but vacuously so.** If $H^1 = 0$ in every real system you test (all modalities are consistent enough that phantom understanding never occurs), the framework is correct but adds nothing. The interesting case is $H^1 \neq 0$ with practical consequences.

---

## 8. Summary: Honesty Scorecard

| Aspect | Sharp or Hand-Wavy? | Testability |
|--------|---------------------|-------------|
| Stable kernel monotonicity | **Sharp** — mathematical theorem | Testable via CCA/shared-latent-space dimensionality |
| Phantom understanding prediction | **Sharp** — specific Borromean structure | Testable via cocycle residual measurement |
| Projection aliasing prediction | **Moderate** — generically true, specifics depend on modality choice | Testable with labeled failure datasets |
| Curvature-masked drift | **Moderate** — clear qualitative prediction, curvature itself is vague | Testable as theory-ops divergence with flat eval |
| VSA = cohomology proxy | **Hand-wavy** — no formal equivalence proven | Empirically checkable but theoretically unsupported |
| $\mathcal{M}$ is Riemannian | **Hand-wavy** — unfalsifiable modeling choice | Not directly testable |
| Curvature interpretation | **Hand-wavy** — requires metric specification | Requires metric, which the paper doesn't specify operationally |
| Sheaf axioms hold | **Assumed** — not tested | Would require testing locality & gluing, which is circular |

### The Bottom Line

The framework's strongest contribution is the **phantom understanding prediction** — the claim that $H^1$ captures global inconsistencies invisible to pairwise measures. This is sharp, testable, and non-obvious. If it holds empirically, the cohomological framing earns its keep.

The framework's weakest parts are the **curvature interpretation** (metaphorical without a specified metric) and the **VSA-cohomology correspondence** (asserted without proof). These need either formal theorems connecting them or honest demotion to "inspired analogy."

The **minimal viable experiment** is a phantom understanding detector applied to a real agent system with ≥3 monitored modalities. Budget 3-4 weeks. If you find even one genuine phantom — a capability where all pairs of modalities agree but the triple doesn't, corresponding to a real failure — the paper is validated. If you search hard and find none, the framework is theoretically elegant but empirically empty.
