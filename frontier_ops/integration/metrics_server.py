#!/usr/bin/env python3
"""
Metrics HTTP Server — Lightweight health endpoint for the proprioceptive sidecar.

Serves /metrics (Prometheus-style text) and /health (JSON) on port 18792.
No dependencies beyond stdlib.

Usage:
    # Standalone (reads from proprioception.json)
    python metrics_server.py

    # Or import and start in background from sidecar:
    from metrics_server import start_metrics_server
    start_metrics_server(port=18792)
"""

from __future__ import annotations

import json
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional


STATE_PATH = os.path.expanduser("~/.openclaw/workspace/proprioception.json")
DEFAULT_PORT = 18795


def _read_state() -> Optional[dict]:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


class MetricsHandler(BaseHTTPRequestHandler):
    """Handle /metrics and /health requests."""

    def log_message(self, format, *args):
        pass  # suppress request logging

    def do_GET(self):
        if self.path == "/health":
            self._serve_health()
        elif self.path == "/metrics":
            self._serve_prometheus()
        elif self.path == "/":
            self._serve_health()
        else:
            self.send_error(404)

    def _serve_health(self):
        state = _read_state()
        if state is None:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"unavailable"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        compact = {
            "status": "ok",
            "verdict": state.get("verdict", "PASS"),
            "health_score": state.get("health_score", 1.0),
            "regime": state.get("regime", "warmup"),
            "hmm_state": state.get("hmm_state", "INITIALIZING"),
            "hmm_anomaly": state.get("hmm_anomaly", 0.0),
            "total_steps": state.get("total_steps", 0),
            "total_flags": state.get("total_flags", 0),
            "total_blocks": state.get("total_blocks", 0),
            "warmup_complete": state.get("warmup_complete", False),
            "context_alignment_trend": state.get("context_alignment_trend", "stable"),
            "signals": state.get("signals", {}),
        }
        self.wfile.write(json.dumps(compact, indent=2).encode())

    def _serve_prometheus(self):
        state = _read_state()
        lines = []
        lines.append("# HELP proprio_health_score Agent health score 0-1")
        lines.append("# TYPE proprio_health_score gauge")

        if state is None:
            lines.append("proprio_health_score -1")
        else:
            lines.append(f"proprio_health_score {state.get('health_score', -1)}")

            lines.append("# HELP proprio_total_steps Total tool calls observed")
            lines.append("# TYPE proprio_total_steps counter")
            lines.append(f"proprio_total_steps {state.get('total_steps', 0)}")

            lines.append("# HELP proprio_total_flags Total FLAG verdicts")
            lines.append("# TYPE proprio_total_flags counter")
            lines.append(f"proprio_total_flags {state.get('total_flags', 0)}")

            lines.append("# HELP proprio_total_blocks Total BLOCK verdicts")
            lines.append("# TYPE proprio_total_blocks counter")
            lines.append(f"proprio_total_blocks {state.get('total_blocks', 0)}")

            lines.append("# HELP proprio_hmm_anomaly HMM anomaly score")
            lines.append("# TYPE proprio_hmm_anomaly gauge")
            lines.append(f"proprio_hmm_anomaly {state.get('hmm_anomaly', 0)}")

            signals = state.get("signals", {})
            for name, val in signals.items():
                safe_name = name.replace("-", "_")
                lines.append(f"# HELP proprio_signal_{safe_name} Signal: {name}")
                lines.append(f"# TYPE proprio_signal_{safe_name} gauge")
                lines.append(f"proprio_signal_{safe_name} {val}")

            # Verdict as a labeled metric
            verdict = state.get("verdict", "PASS")
            for v in ["PASS", "MONITOR", "FLAG", "BLOCK"]:
                lines.append(
                    f'proprio_verdict{{level="{v}"}} {1 if verdict == v else 0}'
                )

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.end_headers()
        self.wfile.write(("\n".join(lines) + "\n").encode())


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def start_metrics_server(port: int = DEFAULT_PORT) -> threading.Thread:
    """Start the metrics HTTP server in a daemon thread."""
    server = ReusableHTTPServer(("127.0.0.1", port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Proprioceptive metrics server")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    print(f"Serving metrics on http://127.0.0.1:{args.port}/")
    print(f"  /health   — JSON health status")
    print(f"  /metrics  — Prometheus text format")

    server = HTTPServer(("127.0.0.1", args.port), MetricsHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
