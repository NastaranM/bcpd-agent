"""Guardrails: checks that run around the LLM, in plain Python.

Why plain Python and not "ask the LLM to be careful"? Because a control you can
test is a control you can trust. Each guardrail here is deterministic, has unit
tests, and its failures are counted by the evaluation suite.

1. scope_check      BEFORE the model runs: refuse questions the agent must not
                    answer (investment advice, forecasts, personal data).
2. grounding_check  AFTER the model answers: every number and date in the
                    answer must appear in some tool output. If not, the agent
                    loop asks the model to fix it (see agent.py).
2b. direction_check AFTER the model answers: "rose"/"fell" must agree with the
                    sign of the change the tool computed. A first, rule-based
                    step towards an LLM-as-a-judge for wording.
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


MONTHS = {
    m: i + 1
    for i, names in enumerate(
        [("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
         ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
         ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"), ("december", "dec")]
    )
    for m in names
}
MONTH_NAME_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s+((?:19|20)\d{2})\b", re.IGNORECASE)


def _find_dates(text: str) -> list[tuple[str, str]]:
    """All dates in free text as (as written, 'YYYY-MM'). Recognises '2006-10',
    '2006-10-01', 'October 2006' and 'Oct 2006'."""
    found = [(m.group(0), m.group(0)[:7]) for m in DATE_RE.finditer(text)]
    for m in MONTH_NAME_RE.finditer(text):
        found.append((m.group(0), f"{m.group(2)}-{MONTHS[m.group(1).lower()]:02d}"))
    return found


_MINUS_RE = re.compile(r"[\u2212\u2013\u2012\u2010](?=\d)")


def normalise_minus(text: str) -> str:
    """Replace typographic minus signs and dashes directly before a digit with '-'."""
    return _MINUS_RE.sub("-", text)


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


def grounding_check(answer: str, tool_outputs: list[dict[str, Any]], question: str = "") -> list[str]:
    """Return the numbers/dates in `answer` that no tool output supports.

    An answer number counts as supported if it equals a tool number after
    rounding to the same number of decimals (2.4 matches 2.413), or equals a
    tool probability expressed as a percent (87% matches 0.87).

    Numbers and dates the USER wrote in the question also count as supported:
    repeating "the coming 12 months" back is not a hallucination. (Found by
    the evaluation: both Haiku and Sonnet were flagged for exactly this when
    refusing a forecast question - a false alarm of the guardrail itself.)
    """
    # Models often write a typographic minus (U+2212) or an en dash before a
    # number. Normalise them to "-" so -1.668 isn't read as +1.668. (Found by
    # the evaluation: a correct "change of −1.668" was flagged.)
    answer = normalise_minus(answer)
    question = normalise_minus(question)
    tool_nums: list[float] = []
    tool_dates: set[str] = set()
    for o in tool_outputs:
        tool_nums += _flatten_numbers(o)
        tool_dates |= _flatten_dates(o)
    if question:
        tool_dates |= {ym for _, ym in _find_dates(question)}
        q_text = MONTH_NAME_RE.sub(" ", DATE_RE.sub(" ", question))
        tool_nums += [float(m.group(0).rstrip("%")) for m in NUM_RE.finditer(q_text)]
    tool_years = {d[:4] for d in tool_dates}

    unsupported: list[str] = []

    # Dates first, then blank them out so their digits aren't read as numbers.
    # Both "2006-10" and "October 2006" / "Oct 2006" are checked.
    for raw, ym in _find_dates(answer):
        if ym not in tool_dates:
            unsupported.append(raw)
    text = MONTH_NAME_RE.sub(" ", DATE_RE.sub(" ", answer))

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
# 2b. Direction guardrail
# ---------------------------------------------------------------------------
# The grounding check verifies numbers, not the words around them: "the mean
# rose from 2.27 to 0.60" uses only correct numbers but says the wrong thing.
# This check compares rise/fall words with the SIGN of change_in_mean that the
# tool reported for that change date.
#
# How: split the answer into sentences/lines; a sentence is checked only if it
# mentions exactly one changepoint date and uses only "up" words or only "down"
# words. Anything ambiguous is skipped rather than guessed (documented limit:
# it can miss errors, but it should not raise false alarms).
UP_WORDS = re.compile(r"\b(rose|rise|rises|rising|increased?|increasing|jumped|climbed|grew|higher|upward)\b", re.I)
DOWN_WORDS = re.compile(r"\b(fell|falls?|falling|decreased?|decreasing|dropped|drops?|declined?|declining|lower|downward)\b", re.I)


def _change_signs(tool_outputs: list[dict[str, Any]]) -> dict[str, float]:
    """Map 'YYYY-MM' change date -> change_in_mean, from every tool output."""
    signs: dict[str, float] = {}
    for o in tool_outputs:
        for c in o.get("changepoints", []) or []:
            if isinstance(c, dict) and "change_in_mean" in c:
                signs[c["regime_starts"]] = c["change_in_mean"]
        if "current_regime_started" in o and "change_in_mean" in o:
            signs[o["current_regime_started"]] = o["change_in_mean"]
    return signs


def direction_check(answer: str, tool_outputs: list[dict[str, Any]]) -> list[str]:
    """Return sentences whose rise/fall wording contradicts the tool's sign."""
    signs = _change_signs(tool_outputs)
    if not signs:
        return []
    errors = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        dates = {ym for _, ym in _find_dates(sentence) if ym in signs}
        if len(dates) != 1:
            continue
        up, down = bool(UP_WORDS.search(sentence)), bool(DOWN_WORDS.search(sentence))
        if up == down:  # neither, or both: ambiguous
            continue
        sign = signs[dates.pop()]
        if (up and sign < 0) or (down and sign > 0):
            errors.append(sentence.strip())
    return errors


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
