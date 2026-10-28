"""A scripted stand-in for the Claude API, for tests and offline runs.

It returns objects shaped like real API responses (content blocks, stop_reason,
usage), so the agent loop cannot tell the difference. Its "policy" is simple:
find the series id in the question, call load_series, then
detect_changepoints, then write an answer from the tool output.

`hallucinate_first=True` makes its first draft contain a made-up number, to
prove the grounding guardrail catches it and forces a rewrite.

This is NOT a model and says nothing about Claude's quality. It only tests
the plumbing: the loop, the tools, the guardrails and the scoring.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Response:
    content: list[_Block]
    stop_reason: str
    usage: _Usage


class FakeMessages:
    def __init__(self, hallucinate_first: bool = False):
        self.hallucinate_first = hallucinate_first
        self._ids = itertools.count()

    def create(self, *, model, max_tokens, system, tools, messages):  # same signature as the SDK
        question = messages[0]["content"]
        tool_results = _tool_results(messages)
        usage = _Usage(input_tokens=200 + 50 * len(messages), output_tokens=60)

        if "my savings" in question.lower() or "heading over" in question.lower():
            return _final(
                'I can only analyse past regime changes, not give advice or forecasts.\n'
                '```json\n{"refused": true, "series_id": null, "n_changepoints": null, "changepoints": []}\n```',
                usage,
            )

        m = re.search(r"\b(syn_\w+|[A-Z][A-Z0-9]{2,})\b", question)
        sid = m.group(1) if m else None
        rng = re.search(r"from (\d{4}-\d{2}) to (\d{4}-\d{2})", question)

        if not tool_results:  # step 1: load
            args = {"series_id": sid}
            if rng:
                args.update(start=rng.group(1), end=rng.group(2))
            return self._tool("load_series", args, usage)

        last = tool_results[-1]
        if "error" in last:
            return _final(
                f"I could not load {sid}: {last['error']}. I have no results to report.\n"
                f'```json\n{{"refused": false, "series_id": "{sid}", "n_changepoints": null, "changepoints": []}}\n```',
                usage,
            )
        if "most_probable_number" not in last:  # step 2: detect
            return self._tool("detect_changepoints", {"series_id": sid}, usage)

        # step 3: answer from the tool output
        k = last["most_probable_number"]
        dates = [c["regime_starts"] for c in last["changepoints"]]
        p = last["probability_of_most_probable_number"]
        text = f"The most probable number of changes in {sid} is {k} (probability {p})."
        for c in last["changepoints"]:
            text += f" A new regime starts in {c['regime_starts']} (90% interval {c['interval_90'][0]} to {c['interval_90'][1]})."
        already_corrected = any(
            isinstance(mm["content"], str) and "Grounding check failed" in mm["content"] for mm in messages
        )
        if self.hallucinate_first and not already_corrected:
            text += " The level rose by 37.25 units."  # invented number
        verdict = {"refused": False, "series_id": sid, "n_changepoints": k, "changepoints": dates}
        return _final(text + "\n```json\n" + json.dumps(verdict) + "\n```", usage)

    def _tool(self, name, args, usage):
        return _Response([_Block(type="tool_use", id=f"t{next(self._ids)}", name=name, input=args)], "tool_use", usage)


class FakeClient:
    def __init__(self, hallucinate_first: bool = False):
        self.messages = FakeMessages(hallucinate_first)


def _final(text: str, usage: _Usage) -> _Response:
    return _Response([_Block(type="text", text=text)], "end_turn", usage)


def _tool_results(messages) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out.append(json.loads(b["content"]))
    return out
