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
  │ [grounding_check] ── unsupported numbers? ──yes── ask LLM to fix (max 2x) ─┘
  │    │
  │    ok
  │    ▼
  └ AgentResult(answer, verdict, tool calls, tokens, latency, guardrail log)

The loop is written by hand (no agent framework) so every step is visible and
can be logged, tested and evaluated.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .guardrails import REFUSAL_TEXT, grounding_check, parse_verdict, scope_check
from .tools import TOOL_SCHEMAS, Toolbox

DEFAULT_MODEL = os.environ.get("BCPD_AGENT_MODEL", "claude-haiku-5-5")

SYSTEM_PROMPT = """You are a monitoring analyst. You answer questions about whether and when a \
time series changed regime, using ONLY the tools provided.

Rules:
1. Always call load_series first, then detect_changepoints. Never guess.
2. Every number and date you write must be copied from a tool result. Do not compute new numbers \
(no differences, ratios or averages of your own). Probabilities may be written as percentages.
3. Describe uncertainty: give the 90% interval for each change and the probability of the number \
of changes.
4. If a tool returns an error, say so plainly and do not invent results.
5. Refuse questions outside this scope: investment advice, forecasts of the future, personal data.
6. Keep the answer short: 3-6 sentences for a manager who is not a statistician.
7. End EVERY answer with a JSON block in exactly this format:
```json
{"refused": false, "series_id": "<id or null>", "n_changepoints": <int or null>, "changepoints": ["YYYY-MM", ...]}
```
For a refusal use {"refused": true, "series_id": null, "n_changepoints": null, "changepoints": []}."""


@dataclass
class AgentResult:
    question: str
    answer: str
    verdict: dict[str, Any] | None
    refused_by_guardrail: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Grounding results: unsupported numbers in the FIRST draft and in the FINAL answer.
    first_draft_unsupported: list[str] = field(default_factory=list)
    final_unsupported: list[str] = field(default_factory=list)
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
            result.answer = REFUSAL_TEXT.format(reason=reason) + (
                '\n```json\n{"refused": true, "series_id": null, "n_changepoints": null, "changepoints": []}\n```'
            )
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

            # ---- Guardrail 2: grounding. Every number must come from a tool.
            unsupported = grounding_check(answer, tool_outputs)
            if result.regenerations == 0:  # first draft: this is the "raw" hallucination rate
                result.first_draft_unsupported = unsupported
            if unsupported and result.regenerations < self.max_regenerations:
                result.regenerations += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Grounding check failed. These numbers or dates in your answer do not "
                            f"appear in any tool result: {unsupported}. Rewrite the answer using only "
                            "values copied from tool results, keeping the JSON block at the end."
                        ),
                    }
                )
                continue

            result.answer = answer
            result.final_unsupported = unsupported
            break
        else:
            result.answer = "Stopped: the agent reached its step limit without a final answer."

        result.verdict = parse_verdict(result.answer)
        result.latency_s = time.perf_counter() - t0
        return result
