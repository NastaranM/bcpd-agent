# bcpd-agent

A monitoring-analyst agent that answers questions like *"Did this series change regime, when, and how sure are you?"* It calls exact Bayesian changepoint detection ([bcpd](https://github.com/NastaranM/bcpd-repo)) as a tool, and it's wrapped in guardrails and an evaluation suite that measure whether it can be trusted.

```
question ─► scope guardrail ─► Claude ⇄ tools (load data · detect changepoints · plot)
                                  │
                                  ▼
                        grounding guardrail ─► human approval ─► report
```

## Why

LLMs are good at explaining and bad at arithmetic. So this agent never does statistics itself. It only reports what the tools computed, and a deterministic check rejects any number or date that no tool produced. The same control pattern applies to any model-monitoring workflow in a regulated setting.

## What's inside

| Part | File | What it does |
|---|---|---|
| Tools | `src/bcpd_agent/tools.py` | Load a series (FRED or synthetic), run changepoint detection, plot |
| Agent loop | `src/bcpd_agent/agent.py` | Hand-written tool-calling loop with step limits and token/latency logging |
| Guardrails | `src/bcpd_agent/guardrails.py` | Scope check, grounding check, structured verdict, human approval |
| Evaluation | `evals/` | 15 cases with known truth: planted changes, no-change, missing data, out-of-scope questions |
| Tests | `tests/` | Run offline with a scripted fake model, so no API key is needed |

## Evaluation design

Each answer is scored on three separate questions, so an error can be traced to where it came from:

- **Detector accuracy:** did the statistics find the planted change? (tool vs truth)
- **Faithfulness:** did the agent report what the tool said? (answer vs tool)
- **End-to-end accuracy:** is the final answer right? (answer vs truth)

It also measures 90% interval coverage (calibration), refusal accuracy, fabrication on missing data, hallucination rate before and after the grounding guardrail, latency and tokens.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                  # offline, no key needed
python -m evals.run_eval --fake                         # offline smoke test of the eval

export ANTHROPIC_API_KEY=...                            # Windows: set ANTHROPIC_API_KEY=...
bcpd-agent "Did US unemployment (UNRATE) change regime since 2000?"
python -m evals.run_eval                                # real evaluation
python -m evals.run_eval --model claude-sonnet-5-5      # compare models
```

## Limitations

- The scope guardrail uses keyword patterns, so paraphrases can slip past. The model's own refusal is the second layer, and the evaluation includes paraphrased cases to measure both.
- The grounding check verifies that numbers come from tools, not that the wording around them is right. A wrong sentence built from correct numbers would pass.
- The evaluation set is small (15 cases), so treat the rates as indicative.
- Changepoint dates are posterior summaries. Read them with their 90% intervals.

MIT licence. Built with assistance from Claude (Anthropic).
