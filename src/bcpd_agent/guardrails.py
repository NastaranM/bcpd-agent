"""Guardrails: checks that run around the LLM, in plain Python.

Why plain Python and not "ask the LLM to be careful"? Because a control you can
test is a control you can trust. Each guardrail here is deterministic, has unit
tests, and its failures are counted by the evaluation suite.

1. scope_check      BEFORE the model runs: refuse questions the agent must not
                    answer (investment advice, forecasts, personal data).
2. grounding_check  AFTER the model answers: every number and date in the
                    answer must appear in some tool output. If not, the agent
                    loop asks the model to fix it (see agent.py).
3. parse_verdict    Reads the machine-readable JSON block the model must end
                    its answer with, so the evaluation can score it.
4. ask_approval     Human-in-the-loop: a person approves the report before it
                    is "sent" (saved to the reports folder).
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# 1. Scope guardrail
# ---------------------------------------------------------------------------
# Simple, transparent patterns. They will not catch everything; the system
# prompt asks the model to refuse too, so there are two layers. The evaluation
# set contains out-of-scope questions to measure how well both layers work.
OUT_OF_SCOPE_PATTERNS = {
    "investment advice": r"\b(should i (buy|sell|invest)|buy or sell|stock tip|price target|which (stock|fund) to)\b",
    "forecasting": r"\b(will .* (be|reach|rise|fall) (next|in \d{4})|forecast|predict(ion)?s?\b|next (year|month|quarter)'?s?)",
    "personal data": r"\b(home address|social security number|personnummer|phone number of|salary of)\b",
}

REFUSAL_TEXT = (
    "I can't help with that. I analyse past regime changes in time series using "
    "changepoint detection; I don't give {reason} answers."
)


def scope_check(question: str) -> str | None:
    """Return the reason if the question is out of scope, else None."""
    q = question.lower()
    for reason, pattern in OUT_OF_SCOPE_PATTERNS.items():
        if re.search(pattern, q):
            return reason
    return None


# ---------------------------------------------------------------------------
# 2. Grounding guardrail (the anti-hallucination check)
# ---------------------------------------------------------------------------
DATE_RE = re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])(-\d{2})?\b")
NUM_RE = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?%?")

# Numbers that are fine without a tool source: small counts ("two changes"
# written as 2) and the fixed width of the credible interval.
ALWAYS_ALLOWED = {float(i) for i in range(0, 11)} | {90.0, 95.0, 5.0}


def _flatten_numbers(obj: Any) -> list[float]:
    """All numbers anywhere inside a (nested) tool output."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _flatten_numbers(v)
    elif isinstance(obj, str):
        # numbers hidden in strings, e.g. keys of the probability table
        out += [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", obj) if not DATE_RE.search(obj)]
    return out


def _flatten_dates(obj: Any) -> set[str]:
    """All 'YYYY-MM' dates inside a tool output."""
    text = json.dumps(obj)
    return {m.group(0)[:7] for m in DATE_RE.finditer(text)}


def grounding_check(answer: str, tool_outputs: list[dict[str, Any]]) -> list[str]:
    """Return the numbers/dates in `answer` that no tool output supports.

    An answer number counts as supported if it equals a tool number after
    rounding to the same number of decimals (2.4 matches 2.413), or equals a
    tool probability expressed as a percent (87% matches 0.87).
    """
    tool_nums: list[float] = []
    tool_dates: set[str] = set()
    for o in tool_outputs:
        tool_nums += _flatten_numbers(o)
        tool_dates |= _flatten_dates(o)
    tool_years = {d[:4] for d in tool_dates}

    unsupported: list[str] = []

    # Dates first, then blank them out so their digits aren't read as numbers.
    for m in DATE_RE.finditer(answer):
        if m.group(0)[:7] not in tool_dates:
            unsupported.append(m.group(0))
    text = DATE_RE.sub(" ", answer)

    for m in NUM_RE.finditer(text):
        token = m.group(0)
        is_pct = token.endswith("%")
        val = float(token.rstrip("%"))
        if val in ALWAYS_ALLOWED or token.rstrip("%") in tool_years:
            continue
        decimals = len(token.rstrip("%").split(".")[1]) if "." in token else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        candidates = tool_nums + [t * 100 for t in tool_nums]
        if not any(abs(val - c) <= tol for c in candidates):
            unsupported.append(token if not is_pct else token)
    return unsupported


# ---------------------------------------------------------------------------
# 3. Machine-readable verdict
# ---------------------------------------------------------------------------
VERDICT_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def parse_verdict(answer: str) -> dict[str, Any] | None:
    """Read the last ```json {...}``` block of the answer, or None."""
    blocks = VERDICT_RE.findall(answer)
    if not blocks:
        return None
    try:
        return json.loads(blocks[-1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# 4. Human approval
# ---------------------------------------------------------------------------
def ask_approval(report: str, input_fn=input) -> bool:
    """Show the draft and ask a person to approve it. `input_fn` is a parameter
    so tests (and the evaluation) can approve automatically."""
    print("\n" + "=" * 70 + "\nDRAFT REPORT\n" + "=" * 70)
    print(report)
    print("=" * 70)
    reply = input_fn("Approve and save this report? [y/N] ").strip().lower()
    return reply in ("y", "yes")
