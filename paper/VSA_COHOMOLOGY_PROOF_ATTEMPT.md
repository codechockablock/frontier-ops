# VSA–Cohomology Correspondence: Formal Analysis

## 1. Setup and Notation

### 1.1 Sheaf-Theoretic Setup

Let $X$ be a topological space with a finite open cover $\mathcal{U} = \{U_1, \ldots, U_n\}$. Let $\mathcal{F}$ be a sheaf of abelian groups on $X$. For each inclusion $U \subseteq V$, we have restriction maps $\rho_{V,U}: \mathcal{F}(V) \to \mathcal{F}(U)$.

The Čech complex for $\mathcal{F}$ with respect to $\mathcal{U}$ is:

$$C^0(\mathcal{U}, \mathcal{F}) = \prod_{i} \mathcal{F}(U_i), \qquad C^1(\mathcal{U}, \mathcal{F}) = \prod_{i < j} \mathcal{F}(U_{ij})$$

where $U_{ij} = U_i \cap U_j$. The coboundary map is:

$$(\delta^0 s)_{ij} = \rho_{U_j, U_{ij}}(s_j) - \rho_{U_i, U_{ij}}(s_i)$$

The zeroth cohomology is $\check{H}^0(\mathcal{U}, \mathcal{F}) = \ker \delta^0$, the space of global sections.

### 1.2 VSA / Phasor Setup

Let $d \in \mathbb{N}$ be the vector dimension. The phasor representation space is $\mathbb{T}^d = (S^1)^d$, where $S^1 = \{z \in \mathbb{C} : |z| = 1\}$. Elements are written $\mathbf{v} = (v_1, \ldots, v_d)$ with $v_k = e^{i\theta_k}$.

**Bundling** of $n$ vectors $\mathbf{v}_1, \ldots, \mathbf{v}_n \in \mathbb{T}^d$:

$$\mathbf{b} = \bigoplus_{i=1}^n \mathbf{v}_i, \qquad b_k = \frac{1}{n}\sum_{i=1}^n v_{i,k} \in \mathbb{D} \quad (\text{closed unit disk})$$

**Coherence** (mean squared resultant length):

$$\mathrm{coh}(\mathbf{v}_1, \ldots, \mathbf{v}_n) = \frac{1}{d}\sum_{k=1}^d \left|\frac{1}{n}\sum_{i=1}^n v_{i,k}\right|^2 = \frac{1}{d}\sum_{k=1}^d |b_k|^2$$

Key properties:
- $\mathrm{coh} = 1$ iff $\mathbf{v}_1 = \cdots = \mathbf{v}_n$ (all identical).
- $\mathbb{E}[\mathrm{coh}] = 1/n$ when phases are i.i.d. uniform on $S^1$.
- $\mathrm{coh} \in [0, 1]$ always.

### 1.3 The Encoding Map

We assume an encoding map $\phi: \bigsqcup_i \mathcal{F}(U_i) \to \mathbb{T}^d$ that sends each local section $s_i \in \mathcal{F}(U_i)$ to a phasor vector $\phi(s_i) = \mathbf{v}_i$.

**The central question:** Under what conditions on $\phi$ does coherence of $\{\phi(s_i)\}$ faithfully reflect the cohomological structure of $\{s_i\}$?

---

## 2. Analysis of the Original Claims

### 2.1 Claim 1: "Phasor bundling is a concrete implementation of sheaf section gluing"

**Verdict: False as stated, but contains a true core.**

Sheaf section gluing is a *predicate*: given local sections $\{s_i\}$ satisfying the cocycle condition $\rho_{U_j, U_{ij}}(s_j) = \rho_{U_i, U_{ij}}(s_i)$ for all $i, j$, there exists a *unique* global section $s \in \mathcal{F}(X)$ with $\rho_{X, U_i}(s) = s_i$. Gluing is an all-or-nothing algebraic property.

Phasor bundling is an *operation*: it always produces an output $\mathbf{b} \in \mathbb{D}^d$ regardless of whether the inputs are compatible. It is a *soft* aggregation, not a *hard* algebraic construction.

**The true core:** Bundling produces a candidate that is close to the inputs when they agree and far when they don't. This is *analogous* to gluing but is not an implementation of it.

### 2.2 Claim 2: Constructive/destructive interference magnitudes

**Verdict: True, as a probabilistic statement about generic encodings.**

**Proposition (Interference Dichotomy).** Let $\mathbf{v}_1, \ldots, \mathbf{v}_n \in \mathbb{T}^d$. Fix a coordinate $k$.

- If $v_{1,k} = v_{2,k} = \cdots = v_{n,k} = e^{i\alpha}$ (all agree), then $\left|\sum_{i=1}^n v_{i,k}\right| = n$.
- If $v_{1,k}, \ldots, v_{n,k}$ are i.i.d. uniform on $S^1$, then $\mathbb{E}\left[\left|\sum_{i=1}^n v_{i,k}\right|^2\right] = n$ (so the magnitude is $\approx \sqrt{n}$ in expectation).

*Proof.* The first claim is trivial. For the second:

$$\mathbb{E}\left[\left|\sum_i v_{i,k}\right|^2\right] = \sum_i \mathbb{E}[|v_{i,k}|^2] + \sum_{i \neq j} \mathbb{E}[v_{i,k}\overline{v_{j,k}}] = n + 0 = n$$

since $\mathbb{E}[v_{i,k}\overline{v_{j,k}}] = \mathbb{E}[v_{i,k}]\mathbb{E}[\overline{v_{j,k}}] = 0$ by independence and uniformity. $\square$

This is a clean result but note the dichotomy is *coordinate-wise* and requires the "disagreement" case to be modeled as *independent uniform*. Real disagreements between sections need not produce uniform phases.

---

## 3. The Exact Gap in Proposition 2.3

Proposition 2.3 claims:

> High coherence $\implies$ local sections lie near a global section.
> Low coherence $\implies$ no consistent global section exists.

**The gap has three layers:**

### Gap 1: Algebraic vs. Metric

Sheaf cohomology is algebraic. $H^0 = \ker \delta^0$ is defined by *exact* equality on overlaps: $\rho_{U_j, U_{ij}}(s_j) = \rho_{U_i, U_{ij}}(s_i)$. Coherence is a *metric* quantity measuring *approximate* agreement of encoded vectors. These are categorically different.

For the correspondence to work, we need $\phi$ to convert algebraic agreement into metric proximity. Specifically, we need:

$$s_i|_{U_{ij}} = s_j|_{U_{ij}} \quad \iff \quad \phi(s_i) \approx \phi(s_j) \text{ in some sense}$$

But the encoding $\phi$ maps from $\mathcal{F}(U_i)$ to $\mathbb{T}^d$ — it maps *entire* local sections, not their restrictions. The restriction structure is internal to the sheaf and invisible to $\phi$ unless $\phi$ is specifically constructed to respect it.

### Gap 2: The Dimensionality Mismatch

$\dim H^0$ is the dimension of a vector space (or rank of an abelian group). Coherence is a single real number in $[0,1]$. A single scalar cannot, in general, determine a dimension. At best, coherence can serve as a *proxy* for a binary question: is $\dim H^0 > 0$ or $= 0$?

Even this binary question requires strong assumptions on $\phi$.

### Gap 3: The Missing Overlap Structure

Coherence measures agreement of the *full vectors* $\phi(s_i)$. But gluing requires agreement only on *overlaps* $U_i \cap U_j$. High coherence of the full vectors is *sufficient* but not *necessary* for gluability: sections could agree on overlaps while being very different on their non-overlapping parts.

Conversely, low coherence of the full vectors does not imply a gluing obstruction — the disagreement might be entirely in non-overlapping regions.

---

## 4. What Can Be Formally Proved

We now state and prove the strongest true results connecting VSA coherence to sheaf-like structure.

### 4.1 Setup: Metric Sheaves with Faithful Encoding

**Definition 4.1 (Metric Sheaf).** A *metric sheaf* is a sheaf $\mathcal{F}$ where each $\mathcal{F}(U)$ carries a metric $d_U$, and restriction maps are 1-Lipschitz: $d_V(\rho_{U,V}(s), \rho_{U,V}(t)) \leq d_U(s, t)$.

**Definition 4.2 (Faithful Phasor Encoding).** An encoding $\phi: \bigsqcup_i \mathcal{F}(U_i) \to \mathbb{T}^d$ is *$(\alpha, \beta)$-faithful* if for all $s_i \in \mathcal{F}(U_i)$, $s_j \in \mathcal{F}(U_j)$ with $U_i \cap U_j \neq \emptyset$:

$$\alpha \cdot d_{U_{ij}}(\rho_{U_i, U_{ij}}(s_i),\, \rho_{U_j, U_{ij}}(s_j)) \;\leq\; d_{\mathbb{T}^d}(\phi(s_i), \phi(s_j)) \;\leq\; \beta \cdot d_{U_{ij}}(\rho_{U_i, U_{ij}}(s_i),\, \rho_{U_j, U_{ij}}(s_j)) + \gamma$$

where $d_{\mathbb{T}^d}$ is some metric on $\mathbb{T}^d$ (e.g., $\ell^2$ of angular differences), $\alpha, \beta > 0$, and $\gamma \geq 0$ accounts for disagreement on non-overlapping parts.

This is a strong condition. When $\gamma = 0$, it says the phasor distance between encoded sections is controlled entirely by their compatibility on overlaps.

### 4.2 Theorem: Coherence Bounds for Compatible Sections

**Theorem 4.3 (Coherence–Compatibility Bound).** Let $\mathcal{U} = \{U_1, \ldots, U_n\}$ be a cover of $X$ and let $s = (s_1, \ldots, s_n) \in C^0(\mathcal{U}, \mathcal{F})$ be a 0-cochain. Define:

$$\Delta(s) = \max_{i < j,\, U_{ij} \neq \emptyset} d_{U_{ij}}\!\left(\rho_{U_i, U_{ij}}(s_i),\, \rho_{U_j, U_{ij}}(s_j)\right)$$

as the *maximal compatibility defect*. Then:

**(a) High coherence from compatibility.** If $\Delta(s) = 0$ (i.e., $s \in \ker \delta^0 = H^0$) and $\phi$ satisfies $d_{\mathbb{T}^d}(\phi(s_i), \phi(s_j)) \leq \gamma$ for all $i, j$, then:

$$\mathrm{coh}(\phi(s_1), \ldots, \phi(s_n)) \geq 1 - \frac{(n-1)\gamma^2}{4}$$

**(b) Low coherence from incompatibility (probabilistic).** Let $d \to \infty$. If $\phi$ is a random encoding such that for each coordinate $k$ independently:
- when $\rho_{U_i, U_{ij}}(s_i) = \rho_{U_j, U_{ij}}(s_j)$, then $\phi(s_i)_k = \phi(s_j)_k$ (phases agree on compatible components), and
- when $\rho_{U_i, U_{ij}}(s_i) \neq \rho_{U_j, U_{ij}}(s_j)$, then $\phi(s_i)_k$ and $\phi(s_j)_k$ are independent uniform on $S^1$,

then $\mathrm{coh} \to 1/n$ in probability as $d \to \infty$.

*Proof of (a).* When all $\phi(s_i)$ are within angular distance $\gamma$ of each other, we can bound each coordinate. Write $\phi(s_i)_k = e^{i\theta_{i,k}}$. The condition $d_{\mathbb{T}^d}(\phi(s_i), \phi(s_j)) \leq \gamma$ implies, in the $\ell^\infty$-angular metric, that $|\theta_{i,k} - \theta_{j,k}| \leq \gamma$ for all $k$ (modulo $2\pi$). Fix coordinate $k$. All phases lie in an arc of length $\gamma$. Then:

$$\left|\frac{1}{n}\sum_i e^{i\theta_{i,k}}\right| \geq \cos(\gamma/2)$$

Squaring and averaging over $k$:

$$\mathrm{coh} \geq \cos^2(\gamma/2) \geq 1 - \gamma^2/4$$

(The $(n-1)$ factor in the theorem statement was conservative; the tighter bound is $1 - \gamma^2/4$, independent of $n$.) $\square$

*Proof of (b).* Under the random encoding assumption, for each coordinate $k$, the random variable $S_k = \frac{1}{n}\sum_i \phi(s_i)_k$ has $\mathbb{E}[|S_k|^2] = 1/n$ (by the same calculation as in Section 2.2). By the law of large numbers applied to $\mathrm{coh} = \frac{1}{d}\sum_k |S_k|^2$, we get $\mathrm{coh} \to 1/n$ in probability. $\square$

### 4.3 Theorem: The Detectable Distinction

**Theorem 4.4 (Coherence Detects Compatibility, Probabilistically).** Under the random encoding model of Theorem 4.3(b), for $d$ sufficiently large:

$$\Pr\left[\mathrm{coh} > \frac{1}{n} + \frac{c}{\sqrt{d}} \;\middle|\; s \notin H^0\right] \leq e^{-c^2/4}$$

and

$$\Pr\left[\mathrm{coh} < 1 - \gamma^2/4 \;\middle|\; s \in H^0,\, \gamma\text{-faithful encoding}\right] = 0$$

In other words: with high probability over random encodings of sufficient dimension, coherence cleanly separates global sections from non-global sections.

*Proof sketch.* The upper tail bound follows from sub-Gaussian concentration of the average of bounded i.i.d. random variables $|S_k|^2 \in [0,1]$. The lower bound is deterministic from Theorem 4.3(a). $\square$

### 4.4 The Correctly Stated Correspondence

**Theorem 4.5 (VSA–Cohomology Correspondence — Corrected).** Let $\mathcal{F}$ be a metric sheaf on $(X, \mathcal{U})$. Let $\phi$ be a $(\alpha, \beta)$-faithful phasor encoding into $\mathbb{T}^d$ with $\gamma = 0$ (meaning phasor distance is controlled by overlap compatibility). Then:

**(i)** If $s \in H^0(\mathcal{U}, \mathcal{F})$ (a global section), then all $\phi(s_i)$ are identical and $\mathrm{coh} = 1$.

**(ii)** If $\phi$ is additionally a *random* faithful encoding in the sense of Theorem 4.3(b), and $s \notin H^0$, then $\mathrm{coh} \leq 1/n + O(1/\sqrt{d})$ with high probability.

**(iii)** Therefore, coherence is a *test statistic* for membership in $H^0$: it separates the cases $\dim H^0 > 0$ vs. $\dim H^0 = 0$ with probability $\to 1$ as $d \to \infty$.

This is the strongest true statement. Note what it does *not* say: it does not recover $\dim H^0$ as a number, and it requires both faithfulness and high dimension.

---

## 5. Where the Original Proposition 2.3 Breaks Down

### 5.1 "Computationally efficient approximation to $\dim H^0$"

**Problem:** Coherence is a single scalar. $\dim H^0$ is an integer that can be arbitrarily large. Coherence cannot approximate an arbitrary integer — it can only detect whether $H^0$ is trivial or non-trivial (binary detection).

**Fix:** Replace "approximation to $\dim H^0$" with "test statistic for $H^0 \neq 0$."

### 5.2 "High coherence implies sections lie in a small neighborhood of a global section"

**Problem:** This is only true if the encoding $\phi$ is faithful in the sense of Definition 4.2. An arbitrary encoding could map incompatible sections to identical phasors (making coherence high despite gluing failure) or compatible sections to distant phasors (making coherence low despite gluability).

**Fix:** State the faithfulness assumption explicitly.

### 5.3 "Low coherence implies no consistent global section exists"

**Problem:** Low coherence of the *full* phasor vectors could arise from disagreement on non-overlapping parts, while the sections agree perfectly on overlaps (and thus a global section exists). The coherence measure is *over-counting* — it penalizes disagreement everywhere, not just on overlaps.

**Fix:** Either (a) require $\gamma = 0$ in the faithfulness condition (so phasor distance reflects only overlap compatibility), or (b) define an *overlap-restricted coherence* that only measures agreement on dimensions corresponding to overlaps.

### 5.4 The Missing Assumption Inventory

For the correspondence to hold, one needs:

| Assumption | Why Needed | How Restrictive |
|---|---|---|
| Metric sheaf structure | To define "approximate agreement" on sections | Mild — most concrete sheaves carry natural metrics |
| Faithful encoding ($\gamma = 0$) | Phasor distance must reflect overlap compatibility | **Strong** — requires the encoding to "know" the overlap structure |
| High dimension ($d \gg n$) | For concentration of coherence around its expectation | Mild — standard in VSA (typically $d \sim 10^4$) |
| Random encoding or i.i.d. coordinates | For the probabilistic separation result | Moderate — satisfied by standard holographic reduced representations |

---

## 6. Proposed Fix: Modified Proposition 2.3

**Proposition 2.3′ (VSA Coherence as a Cohomological Test).** *Let $\mathcal{F}$ be a sheaf on $(X, \mathcal{U})$ equipped with a metric structure, and let $\phi: \bigsqcup_i \mathcal{F}(U_i) \to \mathbb{T}^d$ be a phasor encoding satisfying:*

1. *(Overlap-faithfulness) For all $i, j$ with $U_i \cap U_j \neq \emptyset$:*
$$s_i|_{U_{ij}} = s_j|_{U_{ij}} \implies \phi(s_i)_k = \phi(s_j)_k \text{ for a fraction } \geq \lambda \text{ of coordinates } k$$
$$s_i|_{U_{ij}} \neq s_j|_{U_{ij}} \implies \phi(s_i)_k, \phi(s_j)_k \text{ are approximately independent for a fraction } \geq \mu \text{ of coordinates}$$

2. *(Sufficient dimension) $d \gg n/\mu$.*

*Then the VSA coherence measure serves as a computationally efficient binary test for the existence of global sections:*

$$\mathrm{coh}(\phi(s_1), \ldots, \phi(s_n)) \approx \begin{cases} 1 & \text{if } (s_1, \ldots, s_n) \in H^0(\mathcal{U}, \mathcal{F}) \\ 1/n & \text{if } (s_1, \ldots, s_n) \text{ is maximally incompatible} \end{cases}$$

*More precisely:*
- *If $(s_1, \ldots, s_n) \in H^0$, then $\mathrm{coh} \geq \lambda^2$.*
- *If the sections are pairwise incompatible on all overlaps, then $\mathrm{coh} \leq 1/n + O(1/\sqrt{d})$ w.h.p.*
- *Intermediate coherence values interpolate between these extremes, with coherence monotonically related to the fraction of overlaps satisfying the cocycle condition.*

*In this sense, coherence provides a computationally efficient ($O(nd)$ time) proxy for the global-section existence problem, without explicitly computing the Čech complex ($O(n^2 |\mathcal{F}|)$ time).*

---

## 7. Supplementary Result: The Harmonic Analysis Connection

There is a deeper connection via harmonic analysis on $\mathbb{T}^d$ that partially redeems the original claim.

**Proposition 7.1 (Fourier-Theoretic Interpretation).** The coherence measure can be written as:

$$\mathrm{coh}(\mathbf{v}_1, \ldots, \mathbf{v}_n) = \frac{1}{n^2}\left(n + 2\sum_{i < j} \frac{1}{d}\sum_{k=1}^d \cos(\theta_{i,k} - \theta_{j,k})\right)$$

where $v_{i,k} = e^{i\theta_{i,k}}$. The inner sum $\frac{1}{d}\sum_k \cos(\theta_{i,k} - \theta_{j,k})$ is the *empirical characteristic function* of the angular difference distribution between $\phi(s_i)$ and $\phi(s_j)$, evaluated at frequency 1.

*Proof.* Expand:

$$\mathrm{coh} = \frac{1}{d}\sum_k \left|\frac{1}{n}\sum_i e^{i\theta_{i,k}}\right|^2 = \frac{1}{dn^2}\sum_k \sum_{i,j} e^{i(\theta_{i,k} - \theta_{j,k})}$$

$$= \frac{1}{n^2}\left(\sum_i 1 + \sum_{i \neq j} \frac{1}{d}\sum_k e^{i(\theta_{i,k} - \theta_{j,k})}\right) = \frac{1}{n^2}\left(n + 2\sum_{i<j} \frac{1}{d}\sum_k \cos(\theta_{i,k} - \theta_{j,k})\right) \quad \square$$

**Corollary 7.2.** Coherence decomposes as:

$$\mathrm{coh} = \frac{1}{n} + \frac{2}{n^2}\sum_{i<j} \mathrm{sim}(\phi(s_i), \phi(s_j))$$

where $\mathrm{sim}(\mathbf{u}, \mathbf{v}) = \frac{1}{d}\sum_k \mathrm{Re}(u_k \overline{v_k})$ is the *circular cosine similarity*.

**Interpretation:** Coherence is $1/n$ (the baseline) plus the average pairwise similarity. If $\phi$ preserves the sheaf's compatibility structure in pairwise similarities, then coherence reflects the *average* cocycle condition satisfaction — not $\dim H^0$ itself, but a scalar summary of how well the local sections cohere pairwise.

This is the link to the Čech complex: the coboundary $\delta^0$ is computed from *pairwise* comparisons on overlaps, and coherence is a *scalar aggregation* of pairwise similarities. They probe the same structure at different levels of resolution.

---

## 8. What This Means for the Paper

### 8.1 What Survives

1. **The interference analogy is correct** (Section 2.2): bundling of agreeing phasors gives constructive interference ($|b_k| \approx 1$), disagreeing phasors give destructive ($|b_k| \approx 1/\sqrt{n}$). This is a clean, provable fact.

2. **Coherence is a valid test for section compatibility**, provided the encoding is faithful. Under reasonable assumptions (high dimension, overlap-faithful encoding), it cleanly separates globally consistent from inconsistent local data.

3. **The computational efficiency claim is valid**: coherence is $O(nd)$ to compute, versus $O(n^2 \cdot |\mathcal{F}|)$ for explicit Čech computation. This is a genuine advantage.

4. **The Fourier/harmonic analysis connection** (Proposition 7.1) provides a principled mathematical link between coherence and the pairwise compatibility checks that define $\delta^0$.

### 8.2 What Needs Modification

1. **"Approximation to $\dim H^0$"** should be weakened to **"binary test for $H^0 \neq 0$"** or **"scalar proxy for cocycle-condition satisfaction."** Coherence cannot recover a dimension.

2. **The faithfulness assumption must be stated explicitly.** The correspondence is *not* a property of VSAs alone — it requires the encoding to respect the sheaf's overlap structure. This is an assumption about the *learned representations*, not a theorem about VSAs.

3. **The claim should be probabilistic, not deterministic.** The clean separation requires $d \to \infty$; for finite $d$, there is a gap of width $O(1/\sqrt{d})$ where the test is inconclusive.

### 8.3 Recommended Rewording

Replace the current Proposition 2.3 with the corrected Proposition 2.3′ from Section 6. In the surrounding text, add:

> *Remark.* The correspondence between VSA coherence and sheaf cohomology is a *conditional* result: it holds when the phasor encoding respects the overlap structure of the cover (overlap-faithfulness). In learned representations, this condition is not guaranteed a priori but can be encouraged through training objectives that penalize pairwise inconsistency on overlapping features. The correspondence is also *approximate* — coherence serves as a scalar proxy for the binary question of global-section existence, not as a numerical approximation to the cohomological dimension. The approximation quality improves with the representation dimension $d$ via concentration-of-measure effects standard in high-dimensional probability.

### 8.4 Strength of the Result

Despite the necessary corrections, the core insight is substantive and novel:

> **Phasor coherence provides a $O(nd)$-computable scalar test for sheaf-cohomological consistency, under a faithfulness assumption on the encoding that is natural for learned representations.**

This is worth stating precisely because it connects two seemingly distant fields (algebraic topology and hyperdimensional computing) through a concrete, implementable mechanism. The corrections make the claim *stronger* by being honest about what it requires, making it a proper theorem rather than a handwave.

---

## Appendix A: Concentration Inequality for Coherence

**Lemma A.1.** Let $X_1, \ldots, X_d$ be i.i.d. random variables with $X_k = |S_k|^2$ where $S_k = \frac{1}{n}\sum_{i=1}^n v_{i,k}$ and $v_{i,k}$ are i.i.d. uniform on $S^1$. Then:

$$\Pr\left[\left|\mathrm{coh} - \frac{1}{n}\right| > t\right] \leq 2\exp\left(-\frac{dt^2}{2}\right)$$

*Proof.* Each $X_k \in [0,1]$, so $X_k - \mathbb{E}[X_k]$ is bounded in $[-1, 1]$. By Hoeffding's inequality applied to $\mathrm{coh} = \frac{1}{d}\sum_k X_k$:

$$\Pr\left[\left|\frac{1}{d}\sum_k X_k - \frac{1}{n}\right| > t\right] \leq 2\exp\left(-\frac{2d t^2}{4}\right) = 2\exp\left(-\frac{dt^2}{2}\right) \quad \square$$

**Corollary A.2.** For the coherence test to distinguish $H^0 \neq 0$ (coherence $\geq 1 - \epsilon$) from $H^0 = 0$ (coherence $\approx 1/n$) with failure probability $\leq \delta$, it suffices to have:

$$d \geq \frac{2\ln(2/\delta)}{(1 - 1/n - \epsilon)^2}$$

For $n = 5$, $\epsilon = 0.1$, $\delta = 0.01$: $d \geq 17$. In practice, VSA dimensions of $d \sim 10^4$ provide overwhelming statistical power.
