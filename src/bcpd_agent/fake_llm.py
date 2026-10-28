"""A scripted stand-in for the Claude API, for tests and offline runs.

It returns objects shaped like real API responses (content blocks, stop_reason,
usage), so the agent loop cannot tell the difference. Its "policy" is simple
keyword rules:

    several syn_mv_* / "together"      -> detect_joint_changepoints
    "right now" / "currently" / ...    -> load_series, detect_recent_change
    otherwise                          -> load_series, detect_changepoints

then it writes an answer from the tool output.

`hallucinate_first=True` adds an invented number to the first draft, and
`wrong_direction_first=True` says "rose" where the tool says the mean fell,
to prove the guardrails catch both and force a rewrite.

This is NOT a model and says nothing about Claude's quality. It only tests
the plumbing: the loop, the tools, the guardrails and the scoring.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any

NOW_WORDS = ("right now", "currently", "at the moment", "latest data", "recently", "is something changing")


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


def _verdict(**kw) -> str:
    v = {"refused": False, "method": None, "series_id": None, "n_changepoints": None,
         "changepoints": [], "recent_change": None}
    v.update(kw)
    return "\n```json\n" + json.dumps(v) + "\n```"


class FakeMessages:
    def __init__(self, hallucinate_first: bool = False, wrong_direction_first: bool = False):
        self.hallucinate_first = hallucinate_first
        self.wrong_direction_first = wrong_direction_first
        self._ids = itertools.count()

    def create(self, *, model, max_tokens, system, tools, messages):  # same signature as the SDK
        question = messages[0]["content"]
        ql = question.lower()
        tool_results = _tool_results(messages)
        usage = _Usage(input_tokens=200 + 50 * len(messages), output_tokens=60)
        corrected = any(
            isinstance(m["content"], str) and "Guardrail check failed" in m["content"] for m in messages
        )

        if "my savings" in ql or "heading over" in ql:
            return _final(
                "I can only analyse past regime changes, not give advice or forecasts."
                + _verdict(refused=True), usage,
            )

        ids = re.findall(r"\b(syn_\w+|[A-Z][A-Z0-9]{2,})\b", question)
        rng = re.search(r"from (\d{4}-\d{2}) to (\d{4}-\d{2})", question)

        # --- multivariate
        if len(ids) >= 2 or "together" in ql:
            if not tool_results:
                return self._tool("detect_joint_changepoints", {"series_ids": ids}, usage)
            out = tool_results[-1]
            if "error" in out:
                return _final(f"The joint analysis failed: {out['error']}." + _verdict(method="multivariate"), usage)
            dates = [c["regime_starts"] for c in out["changepoints"]]
            text = (f"Analysed jointly, the most probable number of shared changes is "
                    f"{out['most_probable_number']} (probability {out['probability_of_most_probable_number']}).")
            for c in out["changepoints"]:
                text += f" A shared change starts in {c['regime_starts']}."
            return _final(text + _verdict(method="multivariate", series_id=",".join(ids),
                                          n_changepoints=out["most_probable_number"], changepoints=dates), usage)

        sid = ids[0] if ids else None
        if not tool_results:  # step 1: load
            args = {"series_id": sid}
            if rng:
                args.update(start=rng.group(1), end=rng.group(2))
            return self._tool("load_series", args, usage)

        last = tool_results[-1]
        if "error" in last:
            return _final(
                f"I could not load {sid}: {last['error']}. I have no results to report."
                + _verdict(series_id=sid), usage,
            )

        # --- online
        if any(w in ql for w in NOW_WORDS):
            if "current_regime_started" not in last:
                return self._tool("detect_recent_change", {"series_id": sid}, usage)
            p = last["probability_regime_started_within_window"]
            recent = p >= 0.5
            text = (f"The current regime in {sid} most likely started in {last['current_regime_started']}; "
                    f"the probability that it started within the last {last['window_observations']} "
                    f"observations is {p}.")
            return _final(text + _verdict(method="online", series_id=sid, recent_change=recent,
                                          changepoints=[last["current_regime_started"]] if recent else []), usage)

        # --- offline
        if "most_probable_number" not in last:
            return self._tool("detect_changepoints", {"series_id": sid}, usage)
        k = last["most_probable_number"]
        dates = [c["regime_starts"] for c in last["changepoints"]]
        p = last["probability_of_most_probable_number"]
        text = f"The most probable number of changes in {sid} is {k} (probability {p})."
        for c in last["changepoints"]:
            word = "rose" if c["change_in_mean"] > 0 else "fell"
            if self.wrong_direction_first and not corrected:
                word = "rose" if word == "fell" else "fell"
            text += (f"\nA new regime starts in {c['regime_starts']} (90% interval {c['interval_90'][0]} "
                     f"to {c['interval_90'][1]}): the mean {word} from {c['mean_before']} to {c['mean_after']}.")
        if self.hallucinate_first and not corrected:
            text += " The level moved by 37.25 units."  # invented number
        return _final(text + _verdict(method="offline", series_id=sid, n_changepoints=k, changepoints=dates), usage)

    def _tool(self, name, args, usage):
        return _Response([_Block(type="tool_use", id=f"t{next(self._ids)}", name=name, input=args)], "tool_use", usage)


class FakeClient:
    def __init__(self, hallucinate_first: bool = False, wrong_direction_first: bool = False):
        self.messages = FakeMessages(hallucinate_first, wrong_direction_first)


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
