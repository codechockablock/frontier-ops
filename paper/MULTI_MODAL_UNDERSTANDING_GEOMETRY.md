# The Geometry of Multi-Modal Understanding: Sheaves, Curvature, and the Stable Kernel of Agent Comprehension

**Joseph [Author]**

*Draft — March 27, 2026*

---

## Abstract

We propose a geometric framework for combining heterogeneous modalities of understanding about AI agent behavior — research papers, memory traces, operational logs, and evaluation metrics — into a unified representation. Each modality is formalized as a local section of a sheaf over a topological cover of the agent's *understanding space*, a Riemannian manifold whose curvature encodes the degree to which empirical observations have forced updates to the theoretical framework. Global understanding corresponds to consistent gluing of these local sections, and the obstructions to such gluing — the sheaf cohomology groups — provide a principled measure of inter-modal disagreement. We show that this framework recovers, as special cases, the information-geometric view of evaluation metrics as projection functionals, the cognitive science model of multi-store memory integration, and the phasor superposition mechanism of Vector Symbolic Architectures. We identify a *stable kernel* — the subspace surviving projection onto all modalities simultaneously — as the operationally meaningful core of understanding, and propose concrete empirical tests for measuring cohomological gaps between modalities. The framework predicts specific failure modes invisible to single-modality assessment, including *curvature-masked drift* (where a theory paper absorbs contradictory evidence through framework updates that leave evaluation metrics unchanged) and *projection aliasing* (where distinct failure modes collapse to the same signature in a single modality).

**Keywords:** sheaf cohomology, information geometry, multi-modal understanding, agent governance, Vector Symbolic Architecture, topological data analysis

---

## 1. Introduction

### 1.1 The Problem of Fragmented Understanding

Consider the state of knowledge about a deployed AI agent governance system. At any given moment, understanding is distributed across radically different representational substrates:

1. **Research papers and design documents** — these encode the *topology* of reasoning: which concepts connect to which, what the intended invariants are, why architectural decisions were made. They are the *shape* of the theory.

2. **Memory files and accumulated context** — these trace *geodesics* through reasoning space: the actual paths taken through problems, which alternatives were considered and rejected, how understanding evolved. They are the *dynamics* of comprehension.

3. **Operational logs and production data** — these sample the *local density* of the behavior manifold: what actually happened, at what frequency, in what configurations. They are the *measure* on the space.

4. **Evaluation metrics and benchmark scores** — these are *projection functionals*: linear (or nonlinear) maps from the high-dimensional behavior space to scalar or low-dimensional summary statistics. They compress the manifold to a shadow.

The central question of this paper: **what is the geometric structure of the object that all four modalities are partial views of?** And critically: what information lives in the *relationships between* modalities that is invisible from within any single one?

### 1.2 The CT Scan Analogy

The situation is directly analogous to computed tomography. A CT scanner recovers a 3D density field from a set of 2D projections taken at different angles. No single projection contains the full information, but the *consistency conditions* between projections — formalized by the Radon transform and its inverse — allow reconstruction of the underlying object.

Our claim is that modalities of understanding bear exactly this relationship to the underlying "object" of comprehension. Each modality is a projection operator $\pi_i: \mathcal{M} \to \mathcal{V}_i$ from a high-dimensional understanding manifold $\mathcal{M}$ onto a modality-specific subspace $\mathcal{V}_i$. The question is whether the collection $\{\pi_i(\mathcal{M})\}$ admits a consistent inverse — and if not, *where* it fails.

### 1.3 Why This Matters Now

This is not purely theoretical. In the context of AI agent governance, the gap between what a design document claims, what memory traces record, what logs show, and what evaluations measure is exactly where catastrophic failures hide. A system can pass evaluations (projection onto $\mathcal{V}_{\text{eval}}$ looks clean) while harboring failure modes visible only in the joint geometry of logs-and-theory (the cross-section $\mathcal{V}_{\text{log}} \cap \mathcal{V}_{\text{theory}}$).

The practical motivation comes from systems like constitutional metric governance frameworks, where agent behavior is embedded in a concept space equipped with a position-dependent Riemannian metric, and multiple orthogonal detection signals (efference copy prediction error, EWMA drift, scope creep regression, CUSUM change-point detection) each provide a different *section* over the behavior manifold. The question of how to combine these signals is, we argue, fundamentally a question of sheaf-theoretic consistency.

---

## 2. Theoretical Framework

### 2.1 The Understanding Manifold

**Definition 2.1.** Let $\mathcal{M}$ be a smooth manifold called the *understanding manifold*, whose points represent complete states of comprehension about an agent system. A point $p \in \mathcal{M}$ encodes everything that could in principle be known: the design intent, the implementation reality, the behavioral distribution, and the evaluative summary.

We do not assume $\mathcal{M}$ is directly observable. It is the latent space whose existence is implied by the fact that different modalities can *disagree* — if there were no common underlying object, the notion of disagreement would be incoherent.

**Definition 2.2.** A *modality* is a pair $(\mathcal{V}_i, \pi_i)$ where $\mathcal{V}_i$ is a representational space and $\pi_i: \mathcal{M} \to \mathcal{V}_i$ is a (possibly nonlinear) projection. The four canonical modalities are:

| Modality | Space $\mathcal{V}_i$ | Projection $\pi_i$ | Structure preserved |
|----------|----------------------|--------------------|--------------------|
| Theory (papers) | Simplicial complex | Nerve of concept relations | Topology (connectivity, holes) |
| Memory (geodesics) | Path space $P(\mathcal{M})$ | Trace of reasoning trajectory | Dynamics (ordering, transitions) |
| Operations (logs) | Measure space $(\Omega, \Sigma, \mu)$ | Empirical distribution of behaviors | Density (frequency, concentration) |
| Evaluation (metrics) | $\mathbb{R}^k$ | Summary statistics | Projections (means, rates, bounds) |

**Remark.** These modalities are not independent — they are coupled through $\mathcal{M}$. The theory constrains which log patterns are expected; the logs update the memory; the evaluations test predictions derived from theory. The coupling structure is precisely what the sheaf formalism captures.

### 2.2 The Sheaf of Understanding

We now formalize the multi-modal structure using sheaf theory. Let $X$ be a topological space representing the *domain of concern* — the space of situations, contexts, or behavioral regimes the agent might encounter.

**Definition 2.3.** An *understanding sheaf* $\mathcal{F}$ on $X$ assigns:
- To each open set $U \subseteq X$, a set $\mathcal{F}(U)$ of *local understandings* — consistent descriptions of agent behavior restricted to context $U$.
- To each inclusion $U \subseteq V$, a *restriction map* $\text{res}_{V,U}: \mathcal{F}(V) \to \mathcal{F}(U)$ that projects a broader understanding to its local consequences.

satisfying the sheaf axioms: (i) local understandings that agree on overlaps glue to a global understanding (gluing), and (ii) a global understanding is determined by its local restrictions (locality).

**Definition 2.4.** A *modality cover* is an open cover $\{U_i\}$ of $X$ where each $U_i$ represents the domain of applicability of modality $i$. For our four modalities:

- $U_{\text{theory}}$: contexts addressed by the theoretical framework (often broad but shallow)
- $U_{\text{memory}}$: contexts that have been encountered and reasoned about (sparse but deep)
- $U_{\text{ops}}$: contexts sampled in production (dense where deployed, absent elsewhere)
- $U_{\text{eval}}$: contexts covered by evaluation benchmarks (deliberately structured, finite)

**Key insight:** The covers have different *shapes*. Theory covers broadly but unevenly. Operations cover densely but only where deployed. Evaluation covers deliberately but sparsely. Memory covers the specific paths actually taken. The topology of the overlaps $U_i \cap U_j$ determines where inter-modal consistency can be checked.

### 2.3 Cohomology as Disagreement

The sheaf cohomology groups $H^n(X, \mathcal{F})$ measure obstructions to gluing local sections into global ones. For our purposes:

**$H^0(X, \mathcal{F})$: Global sections.** These are understandings that are simultaneously consistent across all modalities. $H^0$ is the *stable kernel* — the subspace of actual understanding that survives all projections. When $H^0$ is large, we have genuine comprehension. When $H^0$ is small relative to the local sections, most of what we "know" is modality-specific artifact.

**$H^1(X, \mathcal{F})$: Gluing obstructions.** A nonzero class in $H^1$ represents a situation where local understandings on each modality *look* consistent on pairwise overlaps but cannot be assembled into a global picture. This is the formal version of "the paper says X, the logs are consistent with X, the evals pass, but the system doesn't actually do X." Each modality locally confirms, but the global section doesn't exist.

**$H^2(X, \mathcal{F})$: Higher obstructions.** These capture situations where even the *disagreements* between modalities cannot be consistently described — the failure modes have their own failure modes. In practice, $H^2 \neq 0$ signals that the modality decomposition itself is inadequate: the cover needs refinement.

**Proposition 2.1.** *Let $\mathcal{F}$ be the understanding sheaf for a system with modalities $\{(\mathcal{V}_i, \pi_i)\}$. If $H^1(X, \mathcal{F}) = 0$, then every collection of pairwise-consistent local understandings extends to a global section. If $H^1 \neq 0$, there exist "phantom understandings" — locally consistent but globally incoherent states.*

This is not merely a restatement of the sheaf axioms. The content is that the *topology of the modality cover* determines whether phantom understandings exist. A cover with richer overlaps (more pairwise and triple intersections) provides more constraints and reduces $H^1$.

### 2.4 Curvature of the Understanding Manifold

We now equip $\mathcal{M}$ with a Riemannian metric that encodes the *informational cost* of moving between states of understanding.

**Definition 2.5.** The *understanding metric* $g$ on $\mathcal{M}$ is defined so that the distance $d(p, q)$ between two understanding-states reflects the minimal informational work required to update from $p$ to $q$. In the information-geometric sense, this is the Fisher-Rao metric on the space of beliefs about the agent.

The *curvature* of $(\mathcal{M}, g)$ has a direct interpretation:

**Flat regions** ($R = 0$): Understanding that is purely deductive. Given the axioms (design documents), everything follows by logical necessity. Logs confirm exactly what theory predicts. Evaluations return expected values. There is nothing to learn that isn't already implied.

**Positive curvature** ($R > 0$): Understanding that is *convergent*. Multiple independent lines of evidence point to the same conclusion. Different modalities "focus" toward agreement. This is the geometric signature of robust understanding — the analogue of a convex loss landscape.

**Negative curvature** ($R < 0$): Understanding that is *divergent*. Small perturbations in one modality amplify through others. A slight change in the theoretical framework requires large revisions to expected log patterns, which cascade into evaluation redesign. This is the geometric signature of *fragile* understanding — a saddle point in comprehension space.

**Definition 2.6.** A *curvature event* is a point $p \in \mathcal{M}$ where the sectional curvature $K_p(\sigma)$ is large in magnitude for some 2-plane $\sigma \subseteq T_p\mathcal{M}$. Curvature events correspond to moments where empirical observation forced a framework update — where the geodesic of expected understanding was deflected by contact with reality.

**Example.** In a concrete governance system, suppose a spectral concentration metric for task coherence is designed under the assumption that 7-dimensional composite slot vectors will separate semantically distinct action types. Empirical testing reveals that composite vectors conflate credential-access actions with legitimate file operations due to dimensional interference. The fix — projecting to semantic slot vectors via a learned embedding — is a high-curvature event: a point where the actual geometry of the representation forced a revision of the theoretical framework. The curvature is localized to the $(\text{theory}, \text{ops})$ 2-plane: the paper-to-logs transition required updating the theory, while the eval-to-memory transition was relatively flat (evaluation metrics detected the problem smoothly).

### 2.5 The Fisher-Rao Connection

The understanding metric has a natural connection to information geometry. If we view each modality as defining a family of probability distributions $\{p_\theta^{(i)}\}_{\theta \in \Theta}$ over observables, then the Fisher information matrix

$$G^{(i)}_{jk}(\theta) = \mathbb{E}_{p_\theta^{(i)}}\left[\frac{\partial \log p_\theta^{(i)}}{\partial \theta_j} \cdot \frac{\partial \log p_\theta^{(i)}}{\partial \theta_k}\right]$$

defines a Riemannian metric on the parameter space $\Theta$ for each modality $i$. The *joint* Fisher information from combining modalities is:

$$G^{\text{joint}}_{jk}(\theta) = \sum_i G^{(i)}_{jk}(\theta)$$

under independence, which is the information-geometric version of CT reconstruction: each modality contributes directional information, and their sum may be full-rank even when individual terms are not.

**Proposition 2.2.** *The rank deficiency of $G^{(i)}$ identifies the "blind directions" of modality $i$ — parameters that modality $i$ cannot distinguish. The stable kernel (Section 2.7) is the subspace where $G^{\text{joint}}$ is full-rank. Sheaf cohomology $H^1 \neq 0$ arises when the modalities' blind directions intersect nontrivially — there exist parameters that no modality can observe, yet whose variation affects the global understanding.*

This connects to the natural gradient: the gradient descent direction in the Fisher-Rao geometry is $\tilde{\nabla} = G^{-1} \nabla$, which automatically rescales updates by their informational content. Multi-modal understanding corresponds to computing the natural gradient with respect to the joint Fisher information rather than any single modality's.

### 2.6 VSA as Geometric Superposition

Vector Symbolic Architectures (VSAs), particularly phasor-based hyperdimensional computing, provide a concrete computational substrate for the sheaf-theoretic framework.

In a phasor VSA with dimension $d$, concepts are represented as vectors $\mathbf{v} \in \mathbb{C}^d$ with $|v_k| = 1$ for all $k$ — i.e., points on the $d$-dimensional torus $\mathbb{T}^d$. The key operations are:

- **Binding** (element-wise multiplication): $\mathbf{a} \odot \mathbf{b}$, where $(a \odot b)_k = a_k \cdot b_k = e^{i(\alpha_k + \beta_k)}$. This is phase addition — rotation in each component.
- **Bundling** (element-wise circular mean): $\bigoplus_i \mathbf{v}_i$, normalized back to the unit circle. This is superposition — the composite vector encodes all inputs.
- **Similarity** (cosine): $\text{sim}(\mathbf{a}, \mathbf{b}) = \text{Re}(\mathbf{a}^* \cdot \mathbf{b}) / d$.

**Claim 2.1.** *Phasor bundling is a concrete implementation of sheaf section gluing. Given local sections $s_i \in \mathcal{F}(U_i)$ encoded as phasor vectors $\mathbf{v}_i$, the bundled vector $\bigoplus_i \mathbf{v}_i$ represents the candidate global section. The magnitude of each component after bundling (before renormalization) measures the local consistency: components where all sections agree have magnitude $\approx n$ (constructive interference), while components where sections disagree have magnitude $\approx \sqrt{n}$ (destructive interference / random walk).*

This is profound: the bundling operation automatically encodes cohomological information in its *magnitude profile*. Define:

$$\text{coherence}(\mathbf{v}_1, \ldots, \mathbf{v}_n) = \frac{1}{d} \sum_{k=1}^{d} \left|\frac{1}{n} \sum_{i=1}^{n} v_{i,k}\right|^2$$

This is the mean resultant length squared — a standard circular statistics measure. It equals 1 when all sections agree perfectly (trivial cohomology) and $1/n$ when they are uniformly random (maximal cohomological obstruction).

**Proposition 2.3.** *The VSA coherence measure is a computationally efficient approximation to the Čech cohomology dimension $\dim H^0$, in the following sense: high coherence implies that the local sections lie in a small neighborhood of a global section (the circular mean), while low coherence implies that no consistent global section exists within the resolution of the representation.*

This connects the philosophical framework (sheaf cohomology) to a concrete engineering artifact (phasor superposition in a hyperdimensional memory system). The phasor vectors used in VSA-based agent memory are not merely an implementation convenience — they are the natural computational representation for multi-modal section gluing.

### 2.7 The Stable Kernel

**Definition 2.7.** The *stable kernel* $\mathcal{K} \subseteq \mathcal{M}$ is the submanifold

$$\mathcal{K} = \{p \in \mathcal{M} : \pi_i(p) \text{ is consistent across all } i\}$$

More precisely, $\mathcal{K}$ is the support of the global sections $H^0(X, \mathcal{F})$ — the set of understanding-states that survive projection onto every modality simultaneously.

**Properties of the Stable Kernel:**

1. **$\mathcal{K}$ is generically smaller than any $\pi_i(\mathcal{M})$.** Most of what appears in any single modality is not part of the stable kernel. A theoretical claim that has no operational evidence, a log pattern with no theoretical explanation, an evaluation metric with no memory trace of what it measures — all live outside $\mathcal{K}$.

2. **$\mathcal{K}$ is where actual understanding lives.** Understanding that survives all projections is robust in the sense that it cannot be an artifact of a single representational choice. This is the multi-modal analogue of the Bayesian posterior under multiple likelihoods.

3. **$\dim(\mathcal{K})$ is a measure of comprehension depth.** A system with large stable kernel is well-understood from multiple angles. A system with small stable kernel (even if individual modalities are information-rich) has *apparent* understanding that is actually modality-dependent illusion.

4. **$\mathcal{K}$ can shrink.** As new modalities are added (new evaluation benchmarks, new operational environments, new theoretical analyses), the stable kernel can only stay the same size or shrink. Genuine understanding must survive every new angle of examination.

**Theorem 2.1 (Stable Kernel Monotonicity).** *Let $\{(\mathcal{V}_i, \pi_i)\}_{i=1}^n$ and $\{(\mathcal{V}_i, \pi_i)\}_{i=1}^{n+1}$ be collections of modalities. Then $\mathcal{K}_{n+1} \subseteq \mathcal{K}_n$. Equality holds if and only if the new modality $(\mathcal{V}_{n+1}, \pi_{n+1})$ is informationally redundant given the existing modalities — i.e., $\pi_{n+1}$ factors through $(\pi_1, \ldots, \pi_n)$.*

This has an important practical consequence: **adding a modality that doesn't shrink the stable kernel teaches you nothing new.** If your new evaluation benchmark doesn't eliminate any understanding-states that were consistent with all previous modalities, it is measuring something already captured.

---

## 3. The Curvature Interpretation in Detail

### 3.1 Flat Theory, Curved Practice

In a purely deductive system — say, a mathematical proof that a sorting algorithm is correct — the understanding manifold is flat. Every modality agrees because the claims are logically necessary. The design document says "sorted output," the implementation produces sorted output, the tests confirm sorted output, and the complexity analysis matches observed runtime.

Agent governance systems are emphatically not flat. The curvature arises from the empirical character of the domain: no amount of theoretical analysis can predict all the behaviors an agent will exhibit in deployment. The design document says "scope-bounded behavior," but production reveals that the scope boundary interacts with the agent's planning horizon in unexpected ways.

### 3.2 Curvature as Framework Update

Formally, let $\gamma: [0,1] \to \mathcal{M}$ be a geodesic (shortest path) in the understanding manifold connecting state $p = \gamma(0)$ (initial theoretical framework) to state $q = \gamma(1)$ (framework after incorporating empirical evidence). The curvature along $\gamma$ measures how much the path deviates from what the initial framework predicted.

**Definition 3.1.** The *revision curvature* at point $p$ in the direction $(\xi, \eta)$ where $\xi$ represents the theory modality and $\eta$ represents the operations modality is:

$$K_p(\xi, \eta) = \frac{R_p(\xi, \eta, \eta, \xi)}{g_p(\xi, \xi) g_p(\eta, \eta) - g_p(\xi, \eta)^2}$$

where $R$ is the Riemann curvature tensor. This sectional curvature measures the tendency of nearby geodesics in the $(\text{theory}, \text{ops})$ plane to converge or diverge.

**Interpretation:**
- $K > 0$: Theory and operations are *synergistic* — each constrains the other, reducing the space of consistent understandings. This is the convergent regime where more data makes the theory sharper.
- $K = 0$: Theory and operations are *independent* — knowing one tells you nothing about the other. This is a failure of the theoretical framework to make operational predictions.
- $K < 0$: Theory and operations are *antagonistic* — each additional piece of evidence from one modality *widens* the uncertainty in the other. This happens during paradigm shifts: a surprising log pattern doesn't just falsify a prediction, it destabilizes the framework enough that previously settled questions reopen.

### 3.3 Worked Example: The Coherence Signal Curvature Event

Consider the development trajectory of a task coherence signal (Signal B) in a concrete agent governance system. The theoretical framework proceeds through several high-curvature events:

**Phase 1 (Flat).** Design document specifies four sub-signals: centroid drift, recurrence asymmetry, spectral concentration, and alternation index. Each has a clear theoretical motivation. The theory-to-theory region is flat — the sub-signals are derived from established concepts in time-series analysis and spectral methods.

**Phase 2 (Positive curvature).** Initial implementation tested against known adversarial traces. The spectral concentration sub-signal fires on goal-displacement traces as predicted. Theory and operations converge. $K(\text{theory}, \text{ops}) > 0$: empirical confirmation tightens the framework.

**Phase 3 (High negative curvature).** Empirical testing reveals that spectral concentration computed on 7-dimensional composite slot vectors fails to separate credential-access from legitimate file operations. The theory predicted separation; the operations showed conflation. Moreover, the failure mode is not a simple threshold adjustment — it requires changing the *representation* (from composite to semantic slot vectors via learned projection). This is a genuine curvature event: the framework update is not in the predicted direction. $K(\text{theory}, \text{ops}) \ll 0$ at this point.

**Phase 4 (Curvature relaxation).** The fix (MiniLM semantic projections) is implemented. The new theory-operations agreement restores positive curvature, but the understanding manifold now has a "scar" — a persistent topological feature recording that this region required a non-perturbative framework update.

The *curvature history* along this trajectory contains more information than the final state. A system that arrived at semantic slot vectors through a flat path (theoretical derivation without empirical falsification) would have the same endpoint but a fundamentally different understanding geometry — and would be more fragile to the next empirical surprise, because its flat-path confidence would be unjustified.

---

## 4. Connections to Cognitive Science

### 4.1 Multi-Store Memory Models

The modality decomposition mirrors the structure of human memory systems:

| AI Modality | Human Memory System | Encoding | Temporal Character |
|------------|-------------------|----------|-------------------|
| Theory (papers) | Semantic memory | Conceptual relations, rules, schemas | Atemporal, context-free |
| Memory (geodesics) | Episodic memory | Specific events, their context and sequence | Temporal, context-rich |
| Operations (logs) | Procedural memory | Habitual patterns, motor programs | Implicit, statistical |
| Evaluation (metrics) | Working memory | Current task-relevant summaries | Ephemeral, capacity-limited |

This is not merely an analogy. The cognitive science literature on *complementary learning systems* (McClelland et al., 1995; Kumaran et al., 2016) argues that biological memory uses multiple representational systems precisely *because* no single representation can simultaneously support fast learning, slow consolidation, and interference-free storage. The multi-modal structure is not a deficiency to be overcome but a fundamental architectural requirement.

**Insight from neuroscience:** Hippocampal-neocortical interaction (the process by which episodic memories are slowly consolidated into semantic knowledge) is exactly the sheaf-theoretic gluing operation. Each replay event is an attempt to extend a local section (episodic memory of a specific event) to a global section (semantic understanding of the pattern). The consolidation process *fails* (produces $H^1$ obstructions) when the episodic memory is inconsistent with the existing semantic framework — which triggers a framework update (curvature event).

### 4.2 The Binding Problem as Cohomology

The *binding problem* in cognitive science — how the brain combines features processed by different cortical areas into unified percepts — is a cohomology problem. Each cortical area computes a local section (color, shape, motion, spatial location). Conscious perception requires gluing these into a global section. Illusory conjunctions (incorrectly binding features from different objects) are exactly $H^1$ obstructions — locally consistent but globally incoherent bindings.

The VSA framework makes this precise: phasor binding ($\mathbf{a} \odot \mathbf{b}$) creates a composite representation that is *simultaneously* recoverable from the perspective of either component (via unbinding: $\mathbf{a} \odot \mathbf{b} \odot \mathbf{a}^* \approx \mathbf{b}$). This is the computational implementation of consistent restriction maps — a global section that restricts correctly to each local view.

---

## 5. Empirical Framework

### 5.1 Measuring Cohomology in Practice

The theoretical framework is only useful if cohomological gaps are measurable. We propose three concrete measurement approaches:

**Method 1: Cross-Modal Prediction Residuals.**

For each pair of modalities $(i, j)$, train a map $f_{ij}: \mathcal{V}_i \to \mathcal{V}_j$ that predicts modality $j$'s output from modality $i$'s. The residual $\|s_j - f_{ij}(s_i)\|$ on the overlap $U_i \cap U_j$ measures the local section mismatch. The pattern of residuals across all pairs defines a Čech 1-cochain, and its coboundary structure determines whether the mismatches can be attributed to systematic bias (exact cocycle, hence trivial in cohomology) or genuine obstruction (non-trivial cohomology class).

*Concrete instantiation:* Given a governance system's design document (theory), its memory traces (paths), its production logs (density), and its ATBench evaluation results (projections):

1. From the design document, extract predicted detection rates for each adversarial category.
2. From memory traces, extract the reasoning path that led to each detection threshold.
3. From production logs, compute empirical detection rates per category.
4. From ATBench results, extract TPR/FPR with confidence intervals.

The cross-prediction residuals — "theory predicts TPR=X, eval measures TPR=Y, but the memory trace explains the gap as Z" — form a Čech cochain. If the memory explanation Z accounts for the theory-eval gap, the cocycle is exact (trivially cohomologous to zero). If not, $H^1 \neq 0$: there is an unexplained inter-modal disagreement.

**Method 2: VSA Coherence Spectrum.**

Encode each modality's local section as a phasor vector (using the concept extraction mechanisms already present in VSA-based systems). Bundle all modality vectors and examine the *pre-normalization magnitude profile*:

$$m_k = \left|\frac{1}{n} \sum_{i=1}^{n} v_{i,k}\right|, \quad k = 1, \ldots, d$$

The distribution of $\{m_k\}$ encodes the cohomology:
- All $m_k \approx 1$: Perfect agreement. $H^0 = \mathcal{F}(X)$, $H^1 = 0$.
- Bimodal distribution (cluster near 1 and cluster near $1/\sqrt{n}$): Partial agreement. Some dimensions (concepts) are consistently understood; others are modality-dependent.
- All $m_k \approx 1/\sqrt{n}$: No agreement. The modalities are effectively random relative to each other. $H^0$ is trivial.

The *coherence spectrum* — the sorted magnitude profile — is a finite-dimensional proxy for the Betti numbers of the understanding sheaf.

**Method 3: Persistent Cohomology via Filtration.**

Construct a filtration of the overlap complex by *consistency threshold*. At threshold $\epsilon$, include the edge $(i,j)$ in the nerve complex if the cross-modal prediction residual on $U_i \cap U_j$ is below $\epsilon$. As $\epsilon$ increases, more edges appear (we become more tolerant of disagreement), and the cohomology groups change. The persistence diagram of this filtration identifies:

- *Short-lived $H^1$ classes*: Disagreements that disappear at low tolerance — minor calibration issues.
- *Long-lived $H^1$ classes*: Disagreements that persist across a wide range of tolerances — fundamental inter-modal inconsistencies.
- *Birth-death pairs*: Each pair identifies a specific inter-modal gap and the tolerance at which it resolves, providing a prioritized list of "understanding debts."

### 5.2 Predicted Failure Modes

The geometric framework predicts specific failure modes invisible to single-modality assessment:

**Failure Mode 1: Curvature-Masked Drift.**

A theoretical framework can absorb contradictory evidence through small, continuous updates that leave evaluation metrics unchanged. In the understanding manifold, this corresponds to a path that stays within the stable kernel's projection onto $\mathcal{V}_{\text{eval}}$ while actually leaving $\mathcal{K}$ when viewed from $\mathcal{V}_{\text{theory}}$.

*Detection signature:* Flat evaluation trajectory ($\Delta\text{TPR} \approx 0$, $\Delta\text{FPR} \approx 0$) combined with high curvature in the theory-operations plane ($K(\text{theory}, \text{ops}) \gg 0$). The evals say nothing changed; the theory-ops interaction says the framework underwent significant revision. Something real happened that the evals can't see.

*Concrete example:* An agent governance system's detection threshold is recalibrated to maintain TPR=91% while the underlying adversarial distribution shifts. The eval metric is constant, but the memory traces show fundamentally different reasoning about what constitutes adversarial behavior. The sheaf cohomology between the eval and memory modalities would be non-trivial.

**Failure Mode 2: Projection Aliasing.**

Distinct failure modes can collapse to the same signature in a single modality, just as distinct 3D objects can cast the same 2D shadow. A drop in TPR could indicate (a) a new adversarial strategy the detector hasn't seen, (b) a regression in the detector's sensitivity, or (c) a shift in the benign distribution that reduces the separability. These are geometrically distinct in $\mathcal{M}$ but project to the same region in $\mathcal{V}_{\text{eval}}$.

*Detection signature:* Multiple global sections project to the same evaluation summary but differ in their theory or operations components. Formally, the fiber $\pi_{\text{eval}}^{-1}(s_{\text{eval}})$ contains multiple points, distinguishable only by other modalities.

*Resolution:* Adding modalities (enriching the cover) separates the aliases. This is why a governance system needs multiple orthogonal signals — not for redundancy, but for *disambiguation*. Seven orthogonal detectors are not seven copies of the same signal; they are seven projections that jointly resolve aliases that any single projection conflates.

**Failure Mode 3: Phantom Understanding.**

Locally consistent but globally incoherent states ($H^1 \neq 0$). Each modality pair agrees, but there is no single ground truth consistent with all of them.

*Detection signature:* Low pairwise cross-prediction residuals but high triple-intersection residuals. The theory-ops pair agrees, the ops-eval pair agrees, the eval-theory pair agrees, but theory-ops-eval triple doesn't. This is the Borromean rings of understanding: remove any one link and the rest hold together; all three together are inconsistent.

*Concrete example:* A design document claims an agent will stay within a geodesic ball of radius $r$ around the authorized goal. Production logs show the agent always stays within radius $r$. Evaluations confirm boundary-respecting behavior. But the memory traces reveal that the agent's *reasoning path* routinely exits the ball and re-enters — the theory, ops, and eval each see compliance, but the memory shows a trajectory that, if theory were taken seriously as a *process* claim rather than an *outcome* claim, violates the specification. The cohomological obstruction lives in the theory-memory overlap, invisible to ops-eval.

### 5.3 A Measurement Protocol

We propose the following protocol for empirically assessing the multi-modal geometry of an agent governance system:

**Step 1: Section Extraction.** For each modality, extract the local section as a structured representation:
- Theory: Parse design documents into a concept graph (nodes = claims, edges = dependencies).
- Memory: Extract reasoning trajectories as sequences of concept-space positions.
- Operations: Compute empirical distributions over concept-space regions from production logs.
- Evaluation: Collect benchmark results as vectors of (metric, value, confidence interval).

**Step 2: Overlap Identification.** For each pair of modalities, identify the shared concepts — the regions where both modalities make claims about the same aspect of behavior. This defines the nerve of the cover.

**Step 3: Consistency Measurement.** On each overlap, compute cross-modal prediction residuals. Aggregate into a Čech cochain complex.

**Step 4: Cohomology Computation.** Compute $H^0$ (stable kernel dimension) and $H^1$ (gluing obstructions) of the resulting complex. Use persistent cohomology to identify the most significant obstructions.

**Step 5: Curvature Estimation.** From the time series of cohomology computations (repeated at each development milestone), estimate the curvature of the understanding manifold by computing how rapidly the stable kernel changes in response to new evidence.

---

## 6. The Geometric Signature of Understanding

### 6.1 Defining "Understanding" Geometrically

We can now propose a formal geometric signature for understanding:

**Definition 6.1.** A system's *understanding depth* at point $p \in \mathcal{M}$ is the tuple:

$$\mathcal{U}(p) = \left(\dim H^0(p),\ \dim H^1(p),\ \bar{K}(p),\ \text{coh}(p)\right)$$

where:
- $\dim H^0(p)$: Dimension of the stable kernel — how much survives all projections.
- $\dim H^1(p)$: Dimension of gluing obstructions — how many phantom understandings exist.
- $\bar{K}(p)$: Mean sectional curvature — how much empirical revision has occurred.
- $\text{coh}(p)$: VSA coherence — the computational proxy for section agreement.

**Interpretation Guide:**

| $H^0$ | $H^1$ | $\bar{K}$ | Interpretation |
|--------|--------|------------|----------------|
| High | Low | High | Deep understanding — examined from multiple angles, they agree, and empirical evidence has actively shaped the framework |
| High | Low | Low | Shallow understanding — modalities agree, but only because the system hasn't been empirically stressed |
| Low | High | High | Active confusion — many modality-specific claims, frequent framework revisions, no stable core |
| Low | Low | Low | Ignorance — little information in any modality, no disagreements because no claims are being made |
| High | High | High | Frontier understanding — large stable core coexists with significant unresolved inter-modal tensions. This is the signature of a system at the edge of a paradigm shift |

### 6.2 The $H^1$-as-Signal Principle

A key philosophical claim of this framework: **$H^1$ is signal, not noise.** Inter-modal disagreements are the most informative part of the understanding geometry. They point directly to:

1. **Theoretical claims lacking operational evidence.** (Theory section exists; ops section missing on the overlap.)
2. **Operational patterns lacking theoretical explanation.** (Ops section exists; theory section missing.)
3. **Evaluation metrics disconnected from both theory and practice.** (Eval section exists; neither theory nor ops confirms it.)
4. **Memory traces recording reasoning that contradicts current theory.** (The framework has been updated, but the memory of *why* it was updated has been lost.)

Each $H^1$ class is a *research question* — a specific, identifiable gap in understanding that can be addressed by gathering information in the deficient modality.

---

## 7. Open Questions

### 7.1 Computability and Approximation

The sheaf cohomology of continuous covers is in general not computable from finite data. What approximation guarantees can be provided? The persistent cohomology approach (Section 5.1, Method 3) gives stability guarantees via the bottleneck distance, but the relationship between the persistence diagram of the discrete approximation and the true cohomology of the underlying sheaf remains open.

### 7.2 Optimal Cover Design

Given $n$ modalities, what is the optimal topology of the cover $\{U_i\}$? Specifically: if we can choose *where* each modality concentrates its coverage (e.g., which scenarios an evaluation benchmark emphasizes), how should we design the overlaps to maximize the information content of the cohomology? This is a cover-design problem with connections to optimal experimental design in statistics.

### 7.3 Dynamic Cohomology

Understanding evolves. The sheaf, cover, and cohomology groups all change as the system develops. Is there a meaningful notion of *cohomological flow* — a differential equation governing the evolution of $H^*(t)$ as a function of the information received? The curvature interpretation (Section 3) suggests that $\dot{H}^1$ should be related to the Ricci flow on the understanding manifold, but this connection is speculative.

### 7.4 The Representation Problem

The phasor VSA representation (Section 2.6) provides a concrete computational substrate, but it flattens the sheaf structure into a single vector space. Can richer algebraic structures (e.g., graded rings, spectral sequences) be implemented in a VSA-like framework while maintaining the O(d) computational complexity of bundling and binding?

### 7.5 Cross-System Cohomology

Can the sheaf-theoretic framework be applied not just *within* a single agent governance system but *across* multiple systems? If two different agent monitors produce different verdicts on the same trace, is their disagreement usefully described by a cohomology class? This would connect to the broader question of consensus in multi-agent systems and the possibility of a "market of monitors" where inter-monitor cohomology replaces simple majority voting.

### 7.6 Empirical Calibration

What are the actual Betti numbers of real-world agent governance systems? We have hypothesized that $H^1 \neq 0$ generically, but this is an empirical question. A systematic study of cross-modal consistency across production governance systems would be the most direct validation of the framework.

---

## 8. Conclusion

We have proposed that multi-modal understanding of AI agent behavior has a natural geometric structure: a sheaf over a Riemannian manifold, where each modality provides a local section, curvature encodes the history of empirical revision, and cohomology measures the consistency of the global picture.

This framework is not merely descriptive. It makes concrete predictions: phantom understandings ($H^1 \neq 0$) should produce failures invisible to single-modality assessment; curvature-masked drift should allow significant framework revisions to hide behind stable evaluation metrics; and projection aliasing should cause distinct failure modes to appear identical in any single modality.

The framework also provides a concrete computational path through Vector Symbolic Architectures: phasor superposition is the natural computational implementation of section gluing, and the pre-normalization magnitude profile of bundled modality vectors provides a finite-dimensional proxy for sheaf cohomology.

Most importantly, the framework reframes *disagreement between modalities* as the primary signal rather than noise to be averaged away. The gaps, tensions, and inconsistencies between what the theory says, what the memory records, what the logs show, and what the evaluations measure — these are not failures of understanding but its most informative frontier. The stable kernel tells you what you know; the cohomology tells you what you need to learn next.

---

## References

1. Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
2. Bredon, G. (1997). *Sheaf Theory* (2nd ed.). Springer.
3. Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*, 46(2), 255–308.
4. Curry, J. (2014). Sheaves, cosheaves and applications. *arXiv:1303.3255v2*.
5. Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.
6. Gayler, R. W. (2003). Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience. *ICCS/ASCS Joint Conference*.
7. Ghrist, R. (2014). *Elementary Applied Topology*. Createspace.
8. Hansen, J., & Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. *Journal of Applied and Computational Topology*, 3(4), 315–358.
9. Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139–159.
10. Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences*, 20(7), 512–534.
11. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419–457.
12. Plate, T. A. (2003). *Holographic Reduced Representations*. CSLI Publications.
13. Robinson, M. (2014). *Topological Signal Processing*. Springer.
14. Schlegel, K., Neubert, P., & Protzel, P. (2022). A comparison of Vector Symbolic Architectures. *Artificial Intelligence Review*, 55, 4523–4555.
15. Shepard, G. (2020). Sheaves as a framework for understanding. *arXiv preprint*.

---

*This paper was developed in the context of frontier-ops, a geometric agent governance system that implements several of the theoretical constructs described here — including constitutional metric tensors, phasor VSA memory, and multi-signal detection architectures — providing concrete instantiation of the abstract framework.*
