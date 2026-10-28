"""The agent loop: how the LLM, the tools and the guardrails work together.

One call to `MonitoringAgent.ask(question)` does this:

    question
       │
       ▼
  [scope_check] ── out of scope ──► fixed refusal (the LLM is never called)
       │
       ▼
  ┌─► LLM ── wants a tool? ──yes──► run tool in Python ──► result back to LLM ─┐
  │    │                                                                       │
  │    no (final answer)                                                       │
  │    ▼                                                                       │
  │ [grounding_check + direction_check] ── problems? ──yes── ask LLM to fix ───┘
  │    │                                                    (max 2 times)
  │    ok
  │    ▼
  └ AgentResult(answer, verdict, tool calls, guardrail log, tokens, latency)

The LLM's main decision is WHICH detection method fits the question:
offline (history), online (is it changing now?) or multivariate (did several
series change together?). The evaluation checks that choice.

The loop is written by hand (no agent framework) so every step is visible and
can be logged, tested and evaluated.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .guardrails import REFUSAL_TEXT, direction_check, grounding_check, parse_verdict, scope_check
from .tools import TOOL_SCHEMAS, Toolbox

DEFAULT_MODEL = os.environ.get("BCPD_AGENT_MODEL", "claude-haiku-5-5")

SYSTEM_PROMPT = """You are a monitoring analyst. You answer questions about whether and when \
time series changed regime, using ONLY the tools provided.

Choose the method that fits the question:
- History ("did it change, when, how many times?"): load_series, then detect_changepoints (offline).
- Now ("is something changing at the moment, in the latest data, recently?"): load_series, then \
detect_recent_change (online).
- Several series together ("did these change together / at the same time?"): \
detect_joint_changepoints with all the series ids (it loads them itself).

Rules:
1. Never guess. Every number and date you write must be copied from a tool result. Do not compute \
new numbers (no differences, ratios or averages of your own). Probabilities may be written as \
percentages.
2. Describe uncertainty: give the 90% interval for each date and the relevant probability.
3. When you say a level rose or fell, it must match the sign of change_in_mean in the tool result.
4. If a tool returns an error, say so plainly and do not invent results.
5. Refuse questions outside this scope: investment advice, forecasts of the future, personal data.
6. Keep the answer short: 3-6 sentences for a manager who is not a statistician.
7. End EVERY answer with a JSON block in exactly this format:
```json
{"refused": false, "method": "offline|online|multivariate|null", "series_id": "<id, comma-separated ids, or null>", "n_changepoints": <int or null>, "changepoints": ["YYYY-MM", ...], "recent_change": <true, false or null>}
```
- offline / multivariate: n_changepoints = most probable number, changepoints = their dates, recent_change = null.
- online: recent_change = true if the current regime most likely started within the window, else false; \
changepoints = [the current regime start date] if recent_change is true, else []; n_changepoints = null.
- refusal: {"refused": true, "method": null, "series_id": null, "n_changepoints": null, "changepoints": [], "recent_change": null}."""

REFUSAL_VERDICT = (
    '\n```json\n{"refused": true, "method": null, "series_id": null, "n_changepoints": null, '
    '"changepoints": [], "recent_change": null}\n```'
)


@dataclass
class AgentResult:
    question: str
    answer: str
    verdict: dict[str, Any] | None
    refused_by_guardrail: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Guardrail results on the FIRST draft (raw model behaviour) and the FINAL answer.
    first_draft_unsupported: list[str] = field(default_factory=list)
    final_unsupported: list[str] = field(default_factory=list)
    first_draft_direction_errors: list[str] = field(default_factory=list)
    final_direction_errors: list[str] = field(default_factory=list)
    regenerations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""


class MonitoringAgent:
    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        toolbox: Toolbox | None = None,
        max_steps: int = 8,
        max_regenerations: int = 2,
        max_tokens: int = 1024,
    ):
        if client is None:
            # Imported here so tests can run without the SDK or an API key.
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.client = client
        self.model = model
        self.toolbox = toolbox or Toolbox()
        self.max_steps = max_steps  # safety limit: stops runaway tool loops
        self.max_regenerations = max_regenerations
        self.max_tokens = max_tokens

    def ask(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        result = AgentResult(question=question, answer="", verdict=None, model=self.model)

        # ---- Guardrail 1: scope, before spending any tokens.
        reason = scope_check(question)
        if reason:
            result.answer = REFUSAL_TEXT.format(reason=reason) + REFUSAL_VERDICT
            result.verdict = parse_verdict(result.answer)
            result.refused_by_guardrail = True
            result.latency_s = time.perf_counter() - t0
            return result

        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        tool_outputs: list[dict[str, Any]] = []

        for _ in range(self.max_steps):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            result.input_tokens += resp.usage.input_tokens
            result.output_tokens += resp.usage.output_tokens

            # The assistant's reply must be added to the history exactly as returned.
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "tool_use":
                # ---- The model asked for one or more tools: run them, send results back.
                tool_results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    output = self.toolbox.run(block.name, dict(block.input))
                    tool_outputs.append(output)
                    result.tool_calls.append({"tool": block.name, "input": dict(block.input), "output": output})
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)}
                    )
                messages.append({"role": "user", "content": tool_results})
                continue

            # ---- No tool requested: this is a (draft) final answer.
            answer = "".join(b.text for b in resp.content if b.type == "text")

            # ---- Guardrails 2 and 2b: numbers come from tools; rise/fall matches the sign.
            unsupported = grounding_check(answer, tool_outputs, question)
            wrong_direction = direction_check(answer, tool_outputs)
            if result.regenerations == 0:  # first draft: the "raw" error rates
                result.first_draft_unsupported = unsupported
                result.first_draft_direction_errors = wrong_direction
            if (unsupported or wrong_direction) and result.regenerations < self.max_regenerations:
                result.regenerations += 1
                problems = []
                if unsupported:
                    problems.append(
                        f"These numbers or dates do not appear in any tool result: {unsupported}."
                    )
                if wrong_direction:
                    problems.append(
                        "These sentences say rose/fell in the opposite direction to change_in_mean "
                        f"in the tool result: {wrong_direction}."
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": "Guardrail check failed. " + " ".join(problems)
                        + " Rewrite the answer using only values copied from tool results, "
                        "keeping the JSON block at the end.",
                    }
                )
                continue

            result.answer = answer
            result.final_unsupported = unsupported
            result.final_direction_errors = wrong_direction
            break
        else:
            result.answer = "Stopped: the agent reached its step limit without a final answer."

        result.verdict = parse_verdict(result.answer)
        result.latency_s = time.perf_counter() - t0
        return result
