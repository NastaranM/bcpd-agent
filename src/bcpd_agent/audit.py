"""Audit log: one line of JSON per question, appended to logs/audit.jsonl.

Why: if anyone later asks "what did the agent say, based on what, and who
approved it?", the answer is in this file. Each record holds the question,
the model, every tool call with its input and output, every guardrail result,
the final answer, whether a person approved it, tokens and latency.

JSON Lines (one JSON object per line) is easy to append to, easy to read with
pandas (`pd.read_json("logs/audit.jsonl", lines=True)`), and easy to monitor.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict
from typing import Any

from .agent import AgentResult

DEFAULT_LOG = os.path.join("logs", "audit.jsonl")


def write_audit(result: AgentResult, approved: bool | None, path: str = DEFAULT_LOG) -> dict[str, Any]:
    record = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "approved": approved}
    record.update(asdict(result))
    record["latency_s"] = round(record["latency_s"], 2)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record
