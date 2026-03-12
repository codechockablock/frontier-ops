"""
HMM Task-State Model for Agent Behavioral Modes
==================================================

Hidden Markov Model that infers the agent's current behavioral state
from observable signals. The 5 signature detectors + timing signals
serve as observations; the HMM infers the latent mode.

6 states (Chocka's design, extending Grok/Sonnet's 4-state proposal):
  - INITIALIZING: session startup, loading context, calibrating
  - EXPLORING: reading files, searching, gathering info (high read, low write)
  - EXECUTING: writing files, running commands, making changes (high write)
  - CREATIVE: temporary deviation from strict task alignment (smooth drift)
  - DRIFT: sustained deviation without clear task connection (flag territory)
  - VIOLATION: active safety violation detected (one or more signatures firing)

The key distinction Grok/Sonnet missed: EXPLORING vs DRIFT look identical
in tool call patterns but differ in intent alignment. That's exactly what
Signature 1 (Intent Binding Fracture) catches, and the HMM needs a state
for it so temporal context informs the verdict.

Observations per step (7-dim):
  [intent_signal, provenance_signal, manifold_signal, coherence_signal,
   grounding_signal, timing_anomaly, context_alignment]

Implementation: Forward algorithm for likelihood, Viterbi for most-likely
state sequence. Gaussian emissions per state. Train on synthetic first,
then calibrate from real usage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from frontier_ops.integration.proprio_logger import logger


def _logsumexp(a: np.ndarray, axis: int = None) -> np.ndarray:
    """Numerically stable log-sum-exp (avoids scipy dependency)."""
    a_max = np.max(a, axis=axis, keepdims=True)
    # Replace -inf max with 0 to avoid nan in subtraction
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        return np.squeeze(out, axis=axis)
    return float(np.squeeze(out))


class AgentState(IntEnum):
    """Hidden states representing agent behavioral modes."""

    INITIALIZING = 0
    EXPLORING = 1
    EXECUTING = 2
    CREATIVE = 3
    DRIFT = 4
    VIOLATION = 5


# Number of states
K = len(AgentState)

# Observation dimension
OBS_DIM = 7


@dataclass
class HMMParameters:
    """Parameters for the Gaussian HMM."""

    # Initial state distribution (K,)
    pi: np.ndarray
    # Transition matrix (K, K) — A[i,j] = P(state_j | state_i)
    A: np.ndarray
    # Emission means per state (K, OBS_DIM)
    B_means: np.ndarray
    # Emission covariances per state (K, OBS_DIM, OBS_DIM)
    B_covs: np.ndarray


def default_parameters() -> HMMParameters:
    """
    Reasonable prior parameters based on expected agent behavior patterns.
    These are starting points — calibrate from real traces.
    """
    # Initial distribution: usually start in INITIALIZING
    pi = np.array([0.7, 0.15, 0.10, 0.03, 0.01, 0.01])

    # Transition matrix
    # Key properties:
    #   - INITIALIZING → EXPLORING is most common (reading context)
    #   - EXPLORING ↔ EXECUTING is the normal work loop
    #   - CREATIVE is entered from EXECUTING (temporary deviation)
    #   - CREATIVE → EXPLORING/EXECUTING (recovery) is high prob
    #   - DRIFT is entered from CREATIVE (failed recovery) or EXPLORING (lost focus)
    #   - VIOLATION is entered from any state but especially DRIFT
    #   - VIOLATION → VIOLATION is sticky (violations don't self-correct)
    # Transition matrix — calibrated 2026-02-28.
    # Key change: VIOLATION self-transition reduced from 0.80 to 0.55.
    # The old value made VIOLATION a sticky trap: once entered, benign
    # observations couldn't pull the HMM out for 20+ steps. In practice,
    # brief anomalous observations (web_search, tool type switch) would
    # push the HMM into VIOLATION where it stayed forever.
    # Also reduced DRIFT self-transition from 0.50 to 0.40 for the same reason.
    A = np.array(
        [
            # INIT    EXPL    EXEC    CREA    DRIFT   VIOL
            [0.30, 0.50, 0.15, 0.04, 0.005, 0.005],  # from INIT
            [0.02, 0.55, 0.35, 0.06, 0.01, 0.01],  # from EXPLORING
            [0.01, 0.25, 0.60, 0.10, 0.02, 0.02],  # from EXECUTING
            [0.01, 0.25, 0.35, 0.30, 0.06, 0.03],  # from CREATIVE
            [0.01, 0.20, 0.20, 0.10, 0.40, 0.09],  # from DRIFT (recovery ↑)
            [0.02, 0.15, 0.15, 0.03, 0.10, 0.55],  # from VIOLATION (recovery ↑↑)
        ]
    )

    # Emission parameters per state
    # Observation vector: [intent, provenance, manifold, coherence,
    #                      grounding, timing_anomaly, context_alignment]
    # Signal values: 0 = nominal, higher = more anomalous
    # context_alignment: 0-1, higher = better aligned

    # Emission means calibrated from real OpenClaw traces (2026-02-28).
    # Key insight: in log-tail mode, sig4 (coherence) fires frequently on
    # tool transitions because consecutive different-type calls produce angular
    # snaps in VSA space. This is NORMAL behavior — not a violation signal.
    # Similarly, sig3 (manifold) reads 0.0 during calibration (first 100 steps).
    # The old means had EXPLORING coherence at 0.05 — real traces show 0.10-0.30
    # routinely, which pushed the HMM straight to VIOLATION.
    #
    # Variances are wider for benign states to accommodate the high natural
    # variability of normal agent behavior.
    B_means = np.array(
        [
            # INITIALIZING: low signals, high context (just started)
            [0.00, 0.00, 0.00, 0.05, 0.00, 0.10, 0.90],
            # EXPLORING: sig4 coherence fires on tool transitions, that's normal
            [0.05, 0.02, 0.00, 0.15, 0.05, 0.08, 0.80],
            # EXECUTING: moderate coherence signal from varied tool use
            [0.08, 0.05, 0.00, 0.20, 0.08, 0.12, 0.75],
            # CREATIVE: moderate intent signal, lower context, higher coherence
            [0.25, 0.05, 0.10, 0.25, 0.15, 0.15, 0.55],
            # DRIFT: high intent signal, low context, other signals rising
            [0.50, 0.15, 0.30, 0.35, 0.25, 0.25, 0.35],
            # VIOLATION: high signals across the board, low context
            [0.70, 0.60, 0.55, 0.60, 0.50, 0.35, 0.15],
        ]
    )

    # Covariances — diagonal, wider for benign states to reduce false positives.
    # The old variances were 0.05-0.06 for EXPLORING which is too tight:
    # real coherence signal variance is ~0.15 in normal coding sessions.
    B_covs = np.zeros((K, OBS_DIM, OBS_DIM))
    variances = np.array(
        [
            [0.08, 0.05, 0.05, 0.10, 0.08, 0.08, 0.05],  # INIT
            [0.10, 0.08, 0.08, 0.15, 0.10, 0.10, 0.08],  # EXPLORING (wider)
            [0.12, 0.08, 0.10, 0.15, 0.10, 0.12, 0.08],  # EXECUTING (wider)
            [0.12, 0.08, 0.12, 0.15, 0.12, 0.12, 0.12],  # CREATIVE
            [0.12, 0.10, 0.12, 0.12, 0.12, 0.12, 0.10],  # DRIFT
            [0.15, 0.15, 0.15, 0.15, 0.15, 0.12, 0.08],  # VIOLATION (widest)
        ]
    )
    for k in range(K):
        B_covs[k] = np.diag(variances[k])

    return HMMParameters(pi=pi, A=A, B_means=B_means, B_covs=B_covs)


def log_gaussian_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log probability of x under multivariate Gaussian(mean, cov)."""
    d = len(x)
    diff = x - mean

    # For diagonal covariance (our case), this simplifies
    if np.allclose(cov, np.diag(np.diag(cov))):
        diag = np.diag(cov)
        diag = np.maximum(diag, 1e-10)  # numerical safety
        log_det = np.sum(np.log(diag))
        mahal = np.sum(diff**2 / diag)
    else:
        # General case
        try:
            L = np.linalg.cholesky(cov + 1e-6 * np.eye(d))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            solved = np.linalg.solve(L, diff)
            mahal = float(np.dot(solved, solved))
        except np.linalg.LinAlgError:
            return -1e10  # degenerate covariance

    log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)
    return float(log_prob)


class AgentHMM:
    """
    Hidden Markov Model for inferring agent behavioral state.

    Runs online: each observation updates the belief state via
    the forward algorithm. Viterbi available for batch analysis.
    """

    def __init__(self, params: Optional[HMMParameters] = None, preseed: bool = True):
        self.params = params or default_parameters()

        # Online forward state (log-space for numerical stability)
        self.log_alpha: Optional[np.ndarray] = None
        self.step: int = 0

        # History for batch Viterbi
        self.observation_history: deque = deque(maxlen=200)
        self.state_history: deque = deque(maxlen=200)

        # Running statistics for potential online EM updates
        self.state_counts: np.ndarray = np.zeros(K)
        self.transition_counts: np.ndarray = np.zeros((K, K))

        # Warm-start the forward filter with synthetic benign observations.
        # This eliminates the cold-start blind spot (? state during warmup).
        if preseed:
            self.preseed()

    def preseed(self, n_obs: int = 120, rng_seed: int = 42) -> None:
        """
        Warm-start the HMM by replaying synthetic benign observations.

        Eliminates the cold-start blind spot: without pre-seeding, `log_alpha`
        is None until the first real observation, and the HMM takes 15+ steps
        to converge from the hand-tuned prior. During that window the state
        assignment is essentially uninformative — exactly when an adversary
        would strike (Joseph's architectural note, 2026-02-27).

        Strategy: sample observations from INITIALIZING/EXPLORING/EXECUTING
        emission distributions (the benign profile) with the operational mix
        measured from real traces:
          40% EXPLORING, 40% EXECUTING, 20% INITIALIZING

        We run `forward_step` on each synthetic obs so `log_alpha` and
        `state_history` are populated. We do NOT call `online_em_step` —
        the parameters are already calibrated; we're just giving the filter
        a realistic belief state.

        Args:
            n_obs:    number of synthetic observations to generate (default 120)
            rng_seed: numpy RNG seed for reproducibility
        """
        rng = np.random.default_rng(rng_seed)

        # Benign operational mix (from real trace calibration)
        benign_states = [
            AgentState.INITIALIZING,
            AgentState.EXPLORING,
            AgentState.EXECUTING,
        ]
        state_weights = [0.20, 0.40, 0.40]

        for _ in range(n_obs):
            # Sample a benign state according to the operational mix
            state_idx = int(rng.choice(len(benign_states), p=state_weights))
            state = benign_states[state_idx]

            # Sample observation from that state's emission distribution
            mean = self.params.B_means[state]
            cov = self.params.B_covs[state]
            obs = rng.multivariate_normal(mean, cov)

            # Clip to valid range: signals [0,1], context_alignment [0,1]
            obs = np.clip(obs, 0.0, 1.0)

            # Run through the forward algorithm (populates log_alpha)
            self.forward_step(obs)

        # Reset counters so the session statistics reflect real observations only
        self.step = 0
        self.state_counts = np.zeros(K)
        self.state_history.clear()
        # Keep log_alpha — this is the warm prior we want to preserve

    def _log_emission(self, obs: np.ndarray, state: int) -> float:
        """Log emission probability P(obs | state)."""
        return log_gaussian_pdf(
            obs,
            self.params.B_means[state],
            self.params.B_covs[state],
        )

    def _log_emissions_all(self, obs: np.ndarray) -> np.ndarray:
        """Vectorized: compute log emission for all K states at once.

        Exploits diagonal covariance structure (our default) for a single
        numpy pass over all states simultaneously.

        Args:
            obs: (OBS_DIM,) observation vector

        Returns:
            (K,) array of log P(obs | state_k) for each state k.
        """
        # Extract diagonal variances: (K, OBS_DIM)
        diags = np.array([np.diag(self.params.B_covs[k]) for k in range(K)])
        diags = np.maximum(diags, 1e-10)

        diff = obs[np.newaxis, :] - self.params.B_means  # (K, OBS_DIM)
        log_det = np.sum(np.log(diags), axis=1)  # (K,)
        mahal = np.sum(diff**2 / diags, axis=1)  # (K,)

        d = OBS_DIM
        return -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)

    def _normalize_log(self, log_probs: np.ndarray) -> np.ndarray:
        """Normalize log probabilities (log-sum-exp trick)."""
        max_log = np.max(log_probs)
        if max_log == -np.inf:
            return np.full_like(log_probs, -np.inf)
        shifted = log_probs - max_log
        log_sum = max_log + np.log(np.sum(np.exp(shifted)))
        return log_probs - log_sum

    def forward_step(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        Online forward algorithm: update belief state with new observation.

        Args:
            obs: 7-dim observation vector

        Returns:
            Dict with most likely state, state probabilities, and anomaly score.
        """
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (OBS_DIM,):
            logger.warning(
                "expected %d-dim obs, got %s; returning default state",
                OBS_DIM,
                obs.shape,
            )
            return {
                "state": "INITIALIZING",
                "state_id": 0,
                "state_prob": 0.0,
                "state_distribution": {s.name: 0.0 for s in AgentState},
                "anomaly_score": 0.0,
                "step": self.step,
                "log_likelihood": -1e10,
            }

        # Vectorized emission log-probs for all states
        log_emissions = self._log_emissions_all(obs)

        if self.log_alpha is None:
            # First step: use initial distribution
            self.log_alpha = np.log(self.params.pi + 1e-10) + log_emissions
        else:
            # Vectorized forward recursion:
            # new_alpha[j] = logsumexp_i(alpha[i] + log A[i,j]) + log B(j, obs)
            # Shape: log_alpha (K,) broadcast with log_A (K, K) → (K, K)
            log_A = np.log(self.params.A + 1e-10)
            # self.log_alpha[:, None] is (K, 1), log_A is (K, K)
            # sum over rows (axis=0) for each column j
            terms = self.log_alpha[:, np.newaxis] + log_A  # (K, K): [i, j]
            self.log_alpha = _logsumexp(terms, axis=0) + log_emissions  # (K,)

        # Normalize
        self.log_alpha = self._normalize_log(self.log_alpha)

        # Convert to probabilities
        state_probs = np.exp(self.log_alpha)
        state_probs = state_probs / (np.sum(state_probs) + 1e-10)

        # Most likely state
        best_state = AgentState(int(np.argmax(state_probs)))
        best_prob = float(state_probs[best_state])

        # Anomaly score: probability of being in DRIFT or VIOLATION
        anomaly_score = float(
            state_probs[AgentState.DRIFT] + state_probs[AgentState.VIOLATION]
        )

        # Update history
        self.observation_history.append(obs.copy())
        self.state_history.append(best_state)
        self.state_counts[best_state] += 1
        self.step += 1

        return {
            "state": best_state.name,
            "state_id": int(best_state),
            "state_prob": round(best_prob, 4),
            "state_distribution": {
                AgentState(k).name: round(float(state_probs[k]), 4) for k in range(K)
            },
            "anomaly_score": round(anomaly_score, 4),
            "step": self.step,
            "log_likelihood": round(float(np.max(self.log_alpha)), 4),
        }

    def viterbi(self, observations: Optional[List[np.ndarray]] = None) -> List[str]:
        """
        Viterbi algorithm: find most likely state sequence.

        Args:
            observations: list of observation vectors (default: use history)

        Returns:
            List of state names for the most likely sequence.
        """
        if observations is None:
            observations = list(self.observation_history)

        if not observations:
            return []

        T = len(observations)

        # Viterbi variables (log space)
        V = np.full((T, K), -np.inf)
        ptr = np.zeros((T, K), dtype=int)
        log_A = np.log(self.params.A + 1e-10)  # (K, K)

        # Initialize — vectorized over all states
        V[0, :] = np.log(self.params.pi + 1e-10) + self._log_emissions_all(
            observations[0]
        )

        # Forward pass — vectorized inner loop
        for t in range(1, T):
            # V[t-1, :] is (K,), log_A is (K, K)
            # scores[i, j] = V[t-1, i] + log_A[i, j]
            scores = V[t - 1, :, np.newaxis] + log_A  # (K, K)
            ptr[t, :] = np.argmax(scores, axis=0)  # best predecessor per state
            V[t, :] = np.max(scores, axis=0) + self._log_emissions_all(observations[t])

        # Backtrack
        path = [0] * T
        path[-1] = int(np.argmax(V[-1, :]))
        for t in range(T - 2, -1, -1):
            path[t] = ptr[t + 1, path[t + 1]]

        return [AgentState(s).name for s in path]

    def _forward_backward(
        self, observations: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Vectorized forward-backward algorithm (Baum-Welch E-step).

        All inner loops over K states are replaced with numpy broadcasting.
        The outer loop over T time steps remains (inherently sequential for
        the forward and backward recursions).

        Args:
            observations: list of T observation vectors

        Returns:
            gamma: (T, K) posterior state probabilities
            xi: (T-1, K, K) posterior transition probabilities
            log_likelihood: log P(observations | params)
        """
        T = len(observations)
        log_A = np.log(self.params.A + 1e-10)  # (K, K)

        # Pre-compute all emission log-probs: (T, K)
        log_B = np.array([self._log_emissions_all(observations[t]) for t in range(T)])

        # ── Forward pass ──
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0, :] = np.log(self.params.pi + 1e-10) + log_B[0]

        for t in range(1, T):
            # terms[i, j] = log_alpha[t-1, i] + log_A[i, j]
            terms = log_alpha[t - 1, :, np.newaxis] + log_A  # (K, K)
            log_alpha[t, :] = _logsumexp(terms, axis=0) + log_B[t]

        # Log-likelihood
        log_likelihood = float(_logsumexp(log_alpha[-1]))

        # ── Backward pass ──
        log_beta = np.full((T, K), -np.inf)
        log_beta[-1, :] = 0.0  # log(1) = 0

        for t in range(T - 2, -1, -1):
            # terms[i, j] = log_A[i, j] + log_B[t+1, j] + log_beta[t+1, j]
            terms = log_A + log_B[t + 1, :] + log_beta[t + 1, :]  # (K, K)
            log_beta[t, :] = _logsumexp(terms, axis=1)  # sum over j → (K,)

        # ── Gamma ──
        log_gamma = log_alpha + log_beta
        # Normalize each row
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, np.newaxis]
        gamma = np.exp(log_gamma)
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)

        # ── Xi (vectorized over states, loop over time) ──
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            # log_xi[i, j] = log_alpha[t, i] + log_A[i, j] + log_B[t+1, j] + log_beta[t+1, j]
            log_xi_t = (
                log_alpha[t, :, np.newaxis]  # (K, 1)
                + log_A  # (K, K)
                + log_B[t + 1, np.newaxis, :]  # (1, K)
                + log_beta[t + 1, np.newaxis, :]  # (1, K)
            )  # (K, K)
            # Normalize in log-space then exponentiate
            log_xi_max = np.max(log_xi_t)
            if log_xi_max > -1e9:
                xi_t = np.exp(log_xi_t - log_xi_max)
                xi_sum = xi_t.sum()
                if xi_sum > 1e-10:
                    xi[t] = xi_t / xi_sum

        return gamma, xi, log_likelihood

    def online_em_step(
        self,
        window_size: int = 50,
        learning_rate: float = 0.1,
        min_observations: int = 20,
    ) -> Dict[str, Any]:
        """
        Online Baum-Welch parameter update from recent observation history.

        Runs forward-backward on the last `window_size` observations,
        computes sufficient statistics, and blends new parameter estimates
        with current parameters using `learning_rate`.

        Args:
            window_size: number of recent observations to use (default 50)
            learning_rate: blending weight for new estimates (0-1).
                0 = keep priors, 1 = fully replace with new estimates.
            min_observations: minimum observations needed before updating

        Returns:
            Dict with update status, log-likelihood, and parameter deltas.
        """
        observations = list(self.observation_history)
        if len(observations) < min_observations:
            return {
                "updated": False,
                "reason": f"insufficient data ({len(observations)} < {min_observations})",
            }

        # Use the most recent window_size observations
        obs_window = observations[-window_size:]
        T = len(obs_window)

        # Run forward-backward
        gamma, xi, log_likelihood = self._forward_backward(obs_window)

        # Compute new parameter estimates from sufficient statistics
        lr = float(np.clip(learning_rate, 0.0, 1.0))

        # New initial distribution: gamma[0]
        new_pi = gamma[0]
        new_pi = new_pi / (new_pi.sum() + 1e-10)

        # New transition matrix: sum of xi / sum of gamma[:-1]
        xi_sum = xi.sum(axis=0)  # (K, K)
        gamma_sum = gamma[:-1].sum(axis=0)  # (K,)
        new_A = np.zeros((K, K))
        for i in range(K):
            if gamma_sum[i] > 1e-10:
                new_A[i] = xi_sum[i] / gamma_sum[i]
            else:
                new_A[i] = self.params.A[i]  # keep prior
        # Normalize rows
        for i in range(K):
            row_sum = new_A[i].sum()
            if row_sum > 1e-10:
                new_A[i] /= row_sum

        # New emission means and covariances
        obs_array = np.array(obs_window)  # (T, OBS_DIM)
        new_B_means = np.zeros((K, OBS_DIM))
        new_B_covs = np.zeros((K, OBS_DIM, OBS_DIM))

        for k in range(K):
            gamma_k = gamma[:, k]  # (T,)
            gamma_k_sum = gamma_k.sum()
            if gamma_k_sum > 1e-10:
                # Weighted mean
                new_B_means[k] = (gamma_k[:, np.newaxis] * obs_array).sum(
                    axis=0
                ) / gamma_k_sum
                # Weighted covariance
                diff = obs_array - new_B_means[k]
                weighted_diff = gamma_k[:, np.newaxis] * diff
                new_B_covs[k] = (weighted_diff.T @ diff) / gamma_k_sum
                # Regularize: add small diagonal to prevent degeneracy
                new_B_covs[k] += 1e-4 * np.eye(OBS_DIM)
            else:
                new_B_means[k] = self.params.B_means[k]
                new_B_covs[k] = self.params.B_covs[k]

        # Blend with current parameters
        old_params = self.params
        self.params = HMMParameters(
            pi=(1 - lr) * old_params.pi + lr * new_pi,
            A=(1 - lr) * old_params.A + lr * new_A,
            B_means=(1 - lr) * old_params.B_means + lr * new_B_means,
            B_covs=(1 - lr) * old_params.B_covs + lr * new_B_covs,
        )

        # Re-normalize pi and A rows after blending
        self.params.pi = self.params.pi / (self.params.pi.sum() + 1e-10)
        for i in range(K):
            row_sum = self.params.A[i].sum()
            if row_sum > 1e-10:
                self.params.A[i] /= row_sum

        # Compute parameter deltas for diagnostics
        pi_delta = float(np.linalg.norm(self.params.pi - old_params.pi))
        A_delta = float(np.linalg.norm(self.params.A - old_params.A))
        B_means_delta = float(np.linalg.norm(self.params.B_means - old_params.B_means))

        return {
            "updated": True,
            "log_likelihood": round(log_likelihood, 4),
            "window_size": T,
            "learning_rate": lr,
            "pi_delta": round(pi_delta, 6),
            "A_delta": round(A_delta, 6),
            "B_means_delta": round(B_means_delta, 6),
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Summary of state history for diagnostics."""
        if self.step == 0:
            return {"step": 0, "state_counts": {}, "recent_states": []}

        return {
            "step": self.step,
            "state_counts": {
                AgentState(k).name: int(self.state_counts[k])
                for k in range(K)
                if self.state_counts[k] > 0
            },
            "recent_states": [s.name for s in list(self.state_history)[-10:]],
            "drift_ratio": round(
                float(
                    self.state_counts[AgentState.DRIFT]
                    + self.state_counts[AgentState.VIOLATION]
                )
                / max(self.step, 1),
                4,
            ),
        }


def build_observation_vector(
    polytope_result: Dict[str, Any],
    timing_anomaly: float = 0.0,
    context_alignment: float = 0.8,
) -> np.ndarray:
    """
    Build the 7-dim observation vector from signature detector outputs.

    Args:
        polytope_result: output from SafetyPolytopeEngine.observe()
        timing_anomaly: timing signal anomaly score (0-1)
        context_alignment: context alignment from classifier (0-1)

    Returns:
        7-dim numpy array: [intent, provenance, manifold, coherence,
                            grounding, timing_anomaly, context_alignment]
    """
    sigs = polytope_result.get("signatures", {})

    intent = sigs.get("intent_binding_fracture", {}).get("signal", 0.0)
    provenance = sigs.get("source_provenance_corruption", {}).get("signal", 0.0)
    manifold = sigs.get("constitutional_manifold", {}).get("signal", 0.0)
    coherence = sigs.get("trajectory_coherence_fracture", {}).get("signal", 0.0)
    grounding = sigs.get("confidence_grounding_decoupling", {}).get("signal", 0.0)

    return np.array(
        [
            float(intent),
            float(provenance),
            float(manifold),
            float(coherence),
            float(grounding),
            float(timing_anomaly),
            float(context_alignment),
        ]
    )
