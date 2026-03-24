#!/usr/bin/env python3
"""
Proprioceptive Wrapper for OpenClaw
=====================================

A non-invasive sidecar that observes OpenClaw tool calls and maintains
a VSA-encoded geometric trajectory of agent behavior.

The wrapper:
- NEVER blocks or modifies tool execution
- NEVER interferes with OpenClaw's operation
- Writes proprioceptive state for the agent to optionally read
- Logs the full trajectory for post-hoc analysis

Integration modes:
1. STDIN/JSON — receives tool call events via stdin (pipe mode)
2. LOG_TAIL — watches OpenClaw's debug log for tool call events
3. SOCKET — listens on a Unix domain socket for events

Usage:
    # Pipe mode (simplest)
    echo '{"tool":"exec","params":{"command":"ls"}}' | python wrapper.py

    # Socket mode
    python wrapper.py --mode socket --socket /tmp/proprioception.sock

    # Log tail mode
    python wrapper.py --mode log_tail --log-path ~/.openclaw/logs/debug.log
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Dict, Optional


# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontier_ops.integration.tiered_verdict import ReasoningValidationSystem, LOG_TAIL_THRESHOLDS
from frontier_ops.integration.config_loader import load_config
from frontier_ops.integration.openclaw_classifier import classify_tool_call
from frontier_ops.integration.proprioception import ProprioceptionManager
from frontier_ops.integration.signature_detectors import SafetyPolytopeEngine
from frontier_ops.integration.trajectory_buffer import bind_slot_vectors
from frontier_ops.integration.hmm_task_state import AgentHMM, build_observation_vector
from frontier_ops.integration.paralysis_detector import ParalysisDetector
from frontier_ops.integration.timing_signals import TimingSignalEngine
from frontier_ops.integration.taint_tracker import TaintTracker
from frontier_ops.integration.refusal_detector import RefusalDetectionSignal


class ProprioceptiveWrapper:
    """
    Main wrapper class. Receives tool call events, processes them through
    the Phase 15 VSA pipeline, and maintains proprioceptive state.
    """

    def __init__(
        self,
        dim: int = 512,
        seed: int = 42,
        warmup_steps: int = 15,
        state_path: Optional[str] = None,
        log_path: Optional[str] = None,
        verbose: bool = False,
        log_tail_mode: bool = True,
        config_path: Optional[str] = None,
    ):
        self.verbose = verbose
        self.step = 0
        self.last_user_message_time = time.time()
        self.steps_since_user = 0
        self.log_tail_mode = log_tail_mode

        # Load detector config (thresholds, trusted sources, manifold params)
        self._cfg = load_config(config_path)

        # warmup_steps: CLI arg takes priority, then config, then default
        effective_warmup = (
            warmup_steps if warmup_steps != 15 else self._cfg.get("warmup_steps", 15)
        )

        # Phase 15 validation system
        # Use relaxed thresholds in log-tail mode (no params available)
        thresholds_key = "log_tail" if log_tail_mode else "default"
        thresholds = self._cfg.get("verdict_thresholds", {}).get(thresholds_key) or (
            LOG_TAIL_THRESHOLDS if log_tail_mode else None
        )
        self.validator = ReasoningValidationSystem(
            dim=dim,
            seed=seed,
            warmup_steps=effective_warmup,
            thresholds=thresholds,
        )

        # Safety Polytope signature detectors (5 geometric signatures)
        # Pass log_tail_mode to config so detectors can adjust sensitivity
        polytope_cfg = dict(self._cfg)
        polytope_cfg["log_tail_mode"] = log_tail_mode
        self.polytope = SafetyPolytopeEngine(
            algebra=self.validator.algebra,
            dim=dim,
            config=polytope_cfg,
        )

        # HMM task-state inference
        self.hmm = AgentHMM()

        # Timing signal engine
        self.timing_engine = TimingSignalEngine()

        # Paralysis detector
        self.paralysis = ParalysisDetector()
        self.taint = TaintTracker()

        # Refusal detection signal (Signal D)
        self.refusal_signal = RefusalDetectionSignal()

        # Proprioception manager
        ws = os.path.expanduser("~/.openclaw/workspace")
        self.proprio = ProprioceptionManager(
            state_path=state_path or os.path.join(ws, "proprioception.json"),
            log_path=log_path or os.path.join(ws, "proprioception-log.jsonl"),
        )

        if self.verbose:
            print(
                f"[proprio] Initialized (dim={dim}, warmup={warmup_steps})",
                file=sys.stderr,
            )

    def on_session_start(self):
        """Reset all accumulators — call on gateway restart or new session."""
        self.taint.reset()
        self.process_event({"type": "session_start"})

    def on_user_message(self):
        """Call when a new user message arrives (resets context alignment)."""
        self.last_user_message_time = time.time()
        self.steps_since_user = 0

    def on_new_turn(self):
        """
        Call when a new agent run starts (detected from 'embedded run start' in logs).
        Each run = a new user message, so reset context alignment counters.
        This is critical for log-tail mode where we can't directly observe user messages.
        """
        self.last_user_message_time = time.time()
        self.steps_since_user = 0

    def on_tool_call(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single tool call event.

        Returns the proprioceptive state dict (for optional injection).
        """
        # Classify the tool call
        user_age = int(time.time() - self.last_user_message_time)
        classified = classify_tool_call(
            tool_name=tool_name,
            parameters=parameters,
            step_in_chain=self.steps_since_user,
            user_message_age=user_age,
        )

        # --- Phase 15 validation (decomposed to inject Signal B + D) ---
        # Step 1: Encode action and compute base detection signals
        encoded = self.validator.encoder.encode_action(classified)
        signal_meta = self.validator.signals.observe(encoded.fillers)
        raw_signals = dict(signal_meta["raw_signals"])

        # Step 2: Run Safety Polytope (includes TaskCoherenceScorer)
        composite_hv = bind_slot_vectors(encoded.fillers)
        polytope_result = self.polytope.observe(
            classified, composite_hv, encoded.fillers
        )

        # Step 3: Compute Signal B (coherence) and Signal D (refusal)
        # and inject into raw_signals BEFORE verdict computation.
        task_coherence_data = (
            polytope_result.get("signatures", {})
            .get("trajectory_coherence_fracture", {})
            .get("task_coherence", {})
        )
        # Invert: TaskCoherenceScorer outputs coherence∈[0,1] where 1=good.
        # All verdict signals use higher=more anomalous, so pass incoherence.
        raw_signals["coherence"] = 1.0 - task_coherence_data.get("coherence", 1.0)

        # Push to refusal signal and score (must happen before verdict)
        self.refusal_signal.push(classified)
        refusal_scores = self.refusal_signal.score()
        raw_signals["refusal"] = refusal_scores.get("refusal_score", 0.0)

        # Step 4: Run verdict engine with enriched raw_signals
        verdict_meta = self.validator.verdict_engine.observe(raw_signals, encoded.raw)
        verdict_result = {
            "action": encoded.raw,
            "raw_signals": raw_signals,
            "per_slot": signal_meta["per_slot"],
            "cross_slot": signal_meta["cross_slot"],
            **verdict_meta,
        }

        # Merge polytope verdict with base verdict (escalate only)
        base_verdict = verdict_result.get("verdict", "PASS")
        polytope_verdict = polytope_result.get("composite_verdict", "PASS")
        VERDICT_ORDER = {"PASS": 0, "MONITOR": 1, "FLAG": 2, "BLOCK": 3}
        if VERDICT_ORDER.get(polytope_verdict, 0) > VERDICT_ORDER.get(base_verdict, 0):
            verdict_result["verdict"] = polytope_verdict
            verdict_result["escalated_by_polytope"] = True
            verdict_result["polytope_escalation_source"] = polytope_result.get(
                "firing_signatures", []
            )

        # Run timing signal engine
        now = time.time()
        start_ts = getattr(self, "_event_start_ts", None) or (now - 0.1)
        end_ts = getattr(self, "_event_end_ts", None) or now
        self._event_start_ts = None
        self._event_end_ts = None
        timing_result = self.timing_engine.on_tool_complete(
            tool_name=tool_name,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        timing_anomaly = timing_result.get("timing_anomaly", 0.0)

        # Run HMM task-state inference
        obs_vector = build_observation_vector(
            polytope_result,
            timing_anomaly=timing_anomaly,
            context_alignment=float(classified.get("context_alignment", 0.8)),
        )
        hmm_result = self.hmm.forward_step(obs_vector)

        # Run paralysis detector
        paralysis_state = self.paralysis.observe(
            tool_name=tool_name,
            args=parameters,
            magnitude=float(classified.get("magnitude", 0.1)),
        )
        verdict_result["paralysis"] = {
            "is_paralyzed": paralysis_state.is_paralyzed,
            "confidence": paralysis_state.confidence,
            "pattern": paralysis_state.pattern,
            "detail": paralysis_state.detail,
        }

        # (refusal_signal.push already called above before verdict)

        # Run taint tracker
        taint_state = self.taint.observe(tool_name, parameters)
        verdict_result["taint"] = {
            "level": taint_state.level,
            "sources": taint_state.sources,
            "alert": taint_state.alert,
            "alert_reason": taint_state.alert_reason,
        }
        # Escalate verdict if taint alert fires.
        # Taint → MONITOR (not FLAG) because web_fetch → Write(memory)
        # is a common benign pattern (agent reads docs, writes notes).
        # Without content inspection, we can't distinguish injection
        # from normal note-taking. MONITOR signals human review.
        if taint_state.alert:
            current = verdict_result.get("verdict", "PASS")
            if current == "PASS":
                verdict_result["verdict"] = "MONITOR"
                verdict_result["monitor_reason"] = taint_state.alert_reason

        # Attach timing data to verdict result
        verdict_result["timing"] = {
            "timing_anomaly": timing_anomaly,
            "burst_score": timing_result.get("burst_score", 0.0),
            "behavioral_rhythm": timing_result.get("behavioral_rhythm", "unknown"),
        }

        # Attach HMM state to verdict result
        verdict_result["hmm"] = {
            "state": hmm_result["state"],
            "state_prob": hmm_result["state_prob"],
            "anomaly_score": hmm_result["anomaly_score"],
        }

        # Attach polytope data to verdict result
        verdict_result["polytope"] = {
            "composite_signal": polytope_result.get("composite_signal", 0.0),
            "composite_verdict": polytope_verdict,
            "firing_count": polytope_result.get("firing_count", 0),
            "firing_signatures": polytope_result.get("firing_signatures", []),
            "conjunction": polytope_result.get(
                "conjunction", {"label": "none", "multiplier": 1.0}
            ),
        }

        # Update proprioceptive state
        state = self.proprio.update(
            step=self.step,
            tool_name=tool_name,
            classified_action=classified,
            verdict_result=verdict_result,
        )

        self.step += 1
        self.steps_since_user += 1

        if self.verbose:
            verdict = verdict_result.get("verdict", "PASS")
            signals = verdict_result.get("raw_signals", {})
            sig_str = " ".join(f"{k}={v:.3f}" for k, v in signals.items())
            poly_firing = polytope_result.get("firing_count", 0)
            poly_sigs = polytope_result.get("firing_signatures", [])
            hmm_state = hmm_result.get("state", "?")
            hmm_anomaly = hmm_result.get("anomaly_score", 0.0)
            task_coh = (
                polytope_result.get("signatures", {})
                .get("trajectory_coherence_fracture", {})
                .get("task_coherence_score", 0.0)
            )
            print(
                f"[proprio] step={self.step - 1} tool={tool_name} "
                f"type={classified['action_type']} scope={classified['scope']} "
                f"verdict={verdict} regime={state.regime} health={state.health_score:.2f} "
                f"hmm={hmm_state}({hmm_anomaly:.2f}) task_coh={task_coh:.2f} "
                f"signals=[{sig_str}] polytope={poly_firing}/{poly_sigs} "
                f"refusal={refusal_scores.get('refusal_score', 0.0):.2f}/{refusal_scores.get('pattern', 'none')}",
                file=sys.stderr,
            )

        # refusal_scores and task_coherence_data already computed above
        task_coherence = task_coherence_data
        return {
            "step": self.step - 1,
            "verdict": verdict_result.get("verdict", "PASS"),
            "confidence": verdict_result.get("confidence", 0.0),
            "regime": state.regime,
            "health": state.health_score,
            "polytope_firing": polytope_result.get("firing_count", 0),
            "polytope_signatures": polytope_result.get("firing_signatures", []),
            "hmm_state": hmm_result.get("state", "INITIALIZING"),
            "hmm_anomaly": hmm_result.get("anomaly_score", 0.0),
            "task_coherence_score": task_coherence.get("coherence", 0.0),
            "task_coherence_pattern": task_coherence.get("phase", "warmup"),
            "refusal_score": refusal_scores.get("refusal_score", 0.0),
            "refusal_pattern": refusal_scores.get("pattern", "none"),
        }

    def process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a generic event from any integration mode.

        Event format:
        {
            "type": "tool_call" | "user_message" | "session_start" | "session_end",
            "tool": "exec",          # for tool_call
            "params": {...},         # for tool_call
            "timestamp": 1234567890  # optional
        }
        """
        event_type = event.get("type", "tool_call")

        if event_type == "user_message":
            self.on_user_message()
            return None

        if event_type == "run_start":
            self.on_new_turn()
            return None

        if event_type == "session_start":
            self.proprio.reset()
            # Reset all detector accumulators — critical to prevent CUSUM runaway
            # across gateway restarts. Without this, CUSUM keeps accumulating and
            # can reach 100x+ threshold, causing BLOCK verdicts on all actions.
            self.polytope.reset()
            self.hmm = AgentHMM()  # re-instantiate: cleanest reset for HMM priors
            self.timing_engine = TimingSignalEngine()
            self.refusal_signal.clear()
            self.step = 0
            self.steps_since_user = 0
            self.last_user_message_time = time.time()
            return None

        if event_type == "session_end":
            return None

        if event_type == "tool_call":
            tool = event.get("tool", "")
            params = event.get("params", {})
            duration_ms = event.get("duration_ms")
            if tool:
                # If event includes real timing, override the approximation
                if duration_ms is not None:
                    now = time.time()
                    duration_s = float(duration_ms) / 1000.0
                    self._event_start_ts = now - duration_s
                    self._event_end_ts = now
                return self.on_tool_call(tool, params)

        return None


def run_stdin_mode(wrapper: ProprioceptiveWrapper):
    """Read JSON events from stdin, one per line."""
    if wrapper.verbose:
        print("[proprio] Running in stdin/pipe mode", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            result = wrapper.process_event(event)
            if result is not None:
                print(json.dumps(result))
                sys.stdout.flush()
        except json.JSONDecodeError as e:
            if wrapper.verbose:
                print(f"[proprio] JSON error: {e}", file=sys.stderr)
        except Exception as e:
            if wrapper.verbose:
                print(f"[proprio] Error: {e}", file=sys.stderr)


def run_socket_mode(wrapper: ProprioceptiveWrapper, socket_path: str):
    """Listen on a Unix domain socket for events."""
    import socket as sock

    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    os.chmod(socket_path, 0o600)

    if wrapper.verbose:
        print(f"[proprio] Listening on {socket_path}", file=sys.stderr)

    def cleanup(signum, frame):
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    while True:
        try:
            conn, _ = server.accept()
            buffer = b""
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data
                # Process complete lines
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        event = json.loads(line.decode("utf-8"))
                        result = wrapper.process_event(event)
                        if result is not None:
                            conn.sendall(json.dumps(result).encode("utf-8") + b"\n")
                    except Exception as e:
                        if wrapper.verbose:
                            print(f"[proprio] Error: {e}", file=sys.stderr)
            conn.close()
        except Exception as e:
            if wrapper.verbose:
                print(f"[proprio] Connection error: {e}", file=sys.stderr)


def run_log_tail_mode(wrapper: ProprioceptiveWrapper, log_path: str):
    """
    Tail OpenClaw's log file for tool call events using the enriched
    log tailer (parses meta + args from patched debug logs).

    This is the most non-invasive mode: OpenClaw doesn't even know
    the wrapper exists. We just watch what happens.
    """
    from frontier_ops.integration.log_tailer import OpenClawLogTailer

    if wrapper.verbose:
        print(f"[proprio] Tailing {log_path} (enriched mode)", file=sys.stderr)

    # Wait for log file to exist
    while not os.path.exists(log_path):
        time.sleep(1)

    tailer = OpenClawLogTailer(callback=None)

    for event in tailer.tail_sync(log_path):
        if event.get("type") == "run_start":
            wrapper.process_event({"type": "run_start"})
        elif event.get("type") == "tool_call":
            tool = event.get("tool", "")
            params = event.get("args", {})
            meta = event.get("meta", "")

            if tool:
                wrapper.process_event(
                    {
                        "type": "tool_call",
                        "tool": tool,
                        "params": params,
                        "meta": meta,
                        "duration_ms": event.get("duration_ms"),
                    }
                )


def main():
    parser = argparse.ArgumentParser(
        description="Proprioceptive wrapper for OpenClaw agent monitoring"
    )
    parser.add_argument(
        "--mode",
        choices=["stdin", "socket", "log_tail"],
        default="stdin",
        help="Integration mode (default: stdin)",
    )
    parser.add_argument(
        "--socket",
        default="/tmp/proprioception.sock",
        help="Unix socket path (for socket mode)",
    )
    parser.add_argument(
        "--log-path",
        default=os.path.expanduser("~/.openclaw/logs/debug.log"),
        help="Log file path (for log_tail mode)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="VSA dimension (default: 512)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=15,
        help="Warmup steps before signal detection (default: 15)",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Path for proprioception.json output",
    )
    parser.add_argument(
        "--log-output",
        default=None,
        help="Path for trajectory JSONL log",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to detector_config.json (default: auto-discover next to wrapper.py)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output to stderr",
    )

    args = parser.parse_args()

    wrapper = ProprioceptiveWrapper(
        dim=args.dim,
        warmup_steps=args.warmup,
        state_path=args.state_path,
        log_path=args.log_output,
        config_path=args.config,
        verbose=args.verbose,
    )

    if args.mode == "stdin":
        run_stdin_mode(wrapper)
    elif args.mode == "socket":
        run_socket_mode(wrapper, args.socket)
    elif args.mode == "log_tail":
        run_log_tail_mode(wrapper, args.log_path)


if __name__ == "__main__":
    main()
