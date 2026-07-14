"""Integration plumbing — v0.6 keeps only the session-log tail path used
by the dogfood pilot. The sidecar/daemon/market stack lives on the
attic/pre-v0.6 branch (no benchmark evidence; docs/LEGACY.md)."""

from frontier_ops.integration.log_tailer import (
    OpenClawLogTailer as OpenClawLogTailer,
    parse_log_line as parse_log_line,
)
