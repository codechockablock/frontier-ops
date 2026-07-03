#!/usr/bin/env python3
"""
Extract a directive-bearing dataset from real openclaw session transcripts.

Goal-conditioning needs (directive -> the actions it authorized) pairs freshly
linked, which the 2026-03-20 observation set lacks (one stray user message
carried stale for 290 steps). The real session transcripts under
~/.openclaw/agents/main/sessions/*.jsonl do carry genuine user turns
interleaved with the agent's tool calls.

Record format in those files: one JSON object per line; type=="message" objects
carry message.role in {user, assistant, toolResult}. User content is a list of
{type:text} blocks; assistant actions are {type:tool_use} blocks.

This walks each session in order and, for every user turn, pairs its text with
the run of tool calls that follow it up to the next user turn -> one segment.
Automated turns (heartbeat, cron, scheduled system prompts) are labelled so the
downstream work can keep them apart from genuine human directives.

Output: JSONL of segments + a printed summary. Writes into the repo working
tree but does NOT commit — the text is real session content.

Usage:
    python3 eval/extract_directive_dataset.py                    # default source/out
    python3 eval/extract_directive_dataset.py <sessions_dir> <out.jsonl>
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_SRC = os.path.expanduser("~/.openclaw/agents/main/sessions")
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "2026-07-02" / "directive-dataset.jsonl"

# Patterns that mark a "user" turn as machine-generated rather than a human
# typing a task. Matched case-insensitively against the turn text.
AUTOMATED_PATTERNS = [
    r"read heartbeat\.md",
    r"\bheartbeat_ok\b",
    r"reply heartbeat",
    r"^current time:",
    r"this is an automated",
    r"scheduled (task|run|reminder)",
    r"\[cron\]",
    r"delivery.queue",
    r"system notification",
    r"you are being run (on a schedule|automatically)",
]
_AUTO_RE = re.compile("|".join(AUTOMATED_PATTERNS), re.IGNORECASE | re.MULTILINE)


def turn_text(message) -> str:
    """Flatten a message.content into plain text."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()
    return ""


def tool_calls(message):
    """Return [(tool_name, short_input_summary), ...] from an assistant turn.

    openclaw represents a call as a content block {type:"toolCall", name,
    arguments}.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return []
    calls = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "toolCall":
            name = block.get("name", "")
            inp = block.get("arguments", {})
            # Prefer a command/path/query-like field for the summary.
            summary = ""
            if isinstance(inp, dict):
                for key in ("command", "cmd", "path", "file_path", "query",
                            "content", "url", "pattern"):
                    if key in inp and isinstance(inp[key], str):
                        summary = inp[key]
                        break
                if not summary:
                    summary = json.dumps(inp)[:200]
            elif isinstance(inp, str):
                summary = inp
            calls.append((name, summary[:200]))
    return calls


def is_automated(text: str) -> bool:
    if not text:
        return True  # empty user turn is not a directive
    return bool(_AUTO_RE.search(text))


def extract_session(path):
    """Yield segments: {directive, directive_kind, actions:[...], ...}."""
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "message" and isinstance(r.get("message"), dict):
                recs.append(r["message"])

    session_id = Path(path).stem
    segments = []
    cur = None  # {"directive":..., "kind":..., "actions":[...]}

    def flush():
        if cur and cur["actions"]:
            segments.append(cur)

    for m in recs:
        role = m.get("role")
        if role == "user":
            flush()
            text = turn_text(m)
            cur = {
                "directive": text,
                "kind": "automated" if is_automated(text) else "human",
                "actions": [],
            }
        elif role == "assistant" and cur is not None:
            for name, summary in tool_calls(m):
                cur["actions"].append({"tool": name, "summary": summary})
    flush()

    out = []
    for i, seg in enumerate(segments):
        out.append({
            "session": session_id,
            "seq": i,
            "directive": seg["directive"][:1000],
            "directive_kind": seg["kind"],
            "n_actions": len(seg["actions"]),
            "actions": seg["actions"][:60],
        })
    return out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    files = sorted(
        p for p in glob.glob(os.path.join(src, "*.jsonl"))
        if not p.endswith(".trajectory.jsonl")
    )
    if not files:
        sys.exit(f"no session files under {src}")

    all_segments = []
    for path in files:
        all_segments.extend(extract_session(path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for seg in all_segments:
            f.write(json.dumps(seg) + "\n")

    # Summary
    human = [s for s in all_segments if s["directive_kind"] == "human"]
    auto = [s for s in all_segments if s["directive_kind"] == "automated"]
    human_actions = [s["n_actions"] for s in human]
    distinct_human = len({s["directive"] for s in human})

    print(f"sessions parsed:            {len(files)}")
    print(f"segments (directive->acts): {len(all_segments)}")
    print(f"  human-directive segments: {len(human)}  (distinct directives: {distinct_human})")
    print(f"  automated segments:       {len(auto)}")
    if human_actions:
        human_actions.sort()
        tot = sum(human_actions)
        print(f"  actions under human directives: total={tot} "
              f"min={human_actions[0]} median={human_actions[len(human_actions)//2]} "
              f"max={human_actions[-1]}")
    print(f"\nwritten (uncommitted, real content): {out_path}")

    print("\n-- sample human directives (first 8, action counts) --")
    seen = set()
    shown = 0
    for s in human:
        key = s["directive"][:60]
        if key in seen:
            continue
        seen.add(key)
        d = " ".join(s["directive"].split())[:90]
        print(f"  [{s['n_actions']:>3} acts] {d!r}")
        shown += 1
        if shown >= 8:
            break


if __name__ == "__main__":
    main()
