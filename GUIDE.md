# The bcpd-agent guide

This guide explains what the project does, how every part works, how to run it, and how to talk about it in an interview. Read sections 1–3 before touching the code. The rest is reference.

---

## 1. What it is, in one paragraph

A person asks a question in plain language, for example *"Did the US credit card charge-off rate change regime since 2000?"*, *"Is something changing right now in this series?"* or *"Did these four series change together?"*. An LLM (Claude) decides which method fits the question and calls the matching tool: your exact offline algorithm, your online particle filter, or your multivariate model. It then explains the result in a few sentences for a manager. Around the LLM sit controls written in plain Python: a scope check, a check that every number and date comes from a tool, a check that "rose" and "fell" match the computed change, human approval, and an audit log. An evaluation suite runs 18 questions with known answers and measures how often the whole thing chooses correctly and is right, faithful, calibrated and safe.

It's a small version of what the Nordea roles describe: an AI capability with evaluation, guardrails, human oversight, monitoring and evidence.

---

## 2. Key ideas, explained plainly

**Agent.** An LLM in a loop that can take actions, here by calling tools, and decide the next step from the result. Without tools, an LLM can only produce text from memory.

**Tool calling.** We describe Python functions to the model as a name, a description and the parameters (a JSON schema). The model can't run them. It replies with a structured request instead: *"call `detect_changepoints` with `series_id = UNRATE`"*. Our code runs the function and sends the result back as a message. The model reads it and either calls another tool or writes the final answer. The definitions are in `TOOL_SCHEMAS` in `tools.py`.

**Method choice.** The model's real decision. The **tool descriptions** and the **system prompt** tell it when each method fits:
- *history* ("did it change, when?") → **offline**, which uses all the data for every change and is the most accurate for the past;
- *now* ("is something changing at the moment?") → **online**, which asks when the *current* regime started, as live monitoring would;
- *together* ("did these change at the same time?") → **multivariate**, which uses how series move together and finds shared changes no single series shows.

The evaluation includes questions of each kind and scores whether the right tool was used.

**System prompt.** Standing instructions given to the model before the conversation: who it is, how to choose a method, its rules, and the format of its answer. Ours is `SYSTEM_PROMPT` in `agent.py`.

**Tokens.** The unit the API counts and charges by, roughly ¾ of a word. Every message is sent again on each step, so longer tool loops cost more. The agent records input and output tokens for every question.

**Hallucination.** The model states something that isn't supported. Here we define it narrowly and measurably: *a number or date in the answer that appears in no tool output.* That's the same idea as your Ericsson PoC, turned into an automatic check.

**Guardrail.** A control around the model, not inside it. Ours are deterministic Python, so they can be unit-tested and their effect measured. A guardrail that is only an instruction in the prompt ("please don't invent numbers") is a request, not a control.

**Grounding check.** Collects every number and date from the tool outputs, then checks each number and date in the answer. `2.4` counts as supported if a tool said `2.413`, `87%` if a tool said `0.873`, and "October 2006" if a tool said `2006-10`. Anything else is flagged, and the model is told exactly which values failed and asked to rewrite, at most twice.

**Direction check.** Correct numbers can still be put into a wrong sentence: "the mean *rose* from 2.27 to 0.60". For each sentence that mentions exactly one change date, the check compares rise words (rose, increased, jumped…) and fall words (fell, decreased, dropped…) with the sign of `change_in_mean` that the tool computed. Ambiguous sentences are skipped, not guessed. It's a simple, rule-based version of an "LLM-as-a-judge", which the AI Business Monitoring role mentions by name.

**Structured verdict.** Free text is hard to score automatically. So the model must end with a JSON block, for example
`{"refused": false, "method": "offline", "series_id": "UNRATE", "n_changepoints": 2, "changepoints": ["2008-11", "2020-04"], "recent_change": null}`.
The evaluation reads this block. In production you'd use it to feed dashboards or alerts.

**Human in the loop.** The command-line tool shows the draft and asks *"Approve and save this report? [y/N]"*. Nothing is "sent" (saved) without a yes.

**Audit log.** Every question, approved or not, is appended to `logs/audit.jsonl` as one line of JSON: timestamp, question, model, every tool call with its input and output, every guardrail result, the answer, the approval, tokens and latency. It's the evidence trail a governance team asks for: *what did it say, based on what, and who approved it?*

**Evaluation (evals).** A fixed set of questions with known correct answers, run against the agent, with each answer scored. It's the LLM equivalent of backtesting a model. You re-run it after every change, such as a new prompt, model version or tool, and compare.

**Fake model.** `fake_llm.py` pretends to be the Claude API with a fixed script. It lets the tests and the evaluation pipeline run offline, for free, and in CI without a key. It tests the plumbing, not the intelligence.

**CI (continuous integration).** GitHub runs the tests on every push. The build fails if any final answer contains an unsupported number or wrong-direction wording. That's "evaluation as a gate".

---

## 3. What happens when you ask a question

Example: `bcpd-agent "Is something changing right now in syn_recent_change?"`

1. **Scope guardrail** (`guardrails.scope_check`). Regex patterns look for investment advice, forecasts and personal data. If one matches, the agent returns a fixed refusal and **the LLM is never called**.
2. **First model call.** Claude gets the system prompt, the five tool definitions and the question. It replies: *call `load_series(series_id="syn_recent_change")`*.
3. **Tool runs in Python.** It returns a summary of the series. The data stays in Python; only the summary goes to the model.
4. **Second model call: the method choice.** The question says "right now", so Claude should ask for `detect_recent_change` (online), not `detect_changepoints` (offline).
5. **Your online filter runs** (`Toolbox.detect_recent_change`). The particle filter walks through the series one observation at a time. At the end it holds a posterior over *where the current regime started*. The tool reports the most likely start date with a 90% interval, the probability that it started within the last 12 observations, and the mean before and since.
6. **Third model call.** Claude writes the answer and the JSON block, with `"method": "online"` and `"recent_change": true`.
7. **Guardrails.** The grounding check and the direction check run on the draft. If either finds a problem, the model is told exactly what's wrong and asked to rewrite.
8. **Approval and audit.** The command line shows everything and asks for your approval. The record goes to `logs/audit.jsonl` either way.

---

## 4. File by file

```
bcpd-agent/
├── src/bcpd_agent/
│   ├── data.py         where data comes from (FRED download, synthetic series with planted changes)
│   ├── examples.py     built-in test series: syn_one_big, syn_recent_change, syn_mv_1..4, ...
│   ├── tools.py        the tools (load, 3 detection methods, plot) + their JSON schemas
│   ├── guardrails.py   scope, grounding, direction, verdict parser, human approval
│   ├── agent.py        the system prompt and the tool-calling loop
│   ├── audit.py        the audit log (logs/audit.jsonl)
│   ├── cli.py          the `bcpd-agent` command
│   └── fake_llm.py     scripted stand-in for the API, for tests and offline runs
├── evals/
│   ├── cases.py        18 questions with known answers
│   └── run_eval.py     runs them, scores them, writes results/*.json and *.md
├── tests/test_agent.py 19 offline tests (pytest)
├── .github/workflows/ci.yml   automatic tests + evaluation gate on GitHub
├── Dockerfile          container (deployment)
├── README.md           the public, short description
└── GUIDE.md            this file
```

**examples.py.** The built-in test series, each with a planted truth that is never shown to the model:
- `syn_recent_change`: a change 10 months before the end, for "is it changing now?" with answer yes;
- `syn_one_big`: a change long ago, so asked about *now*, the right answer is no;
- `syn_mv_1` … `syn_mv_4`: four series with correlation 0.9 that shift *together* in alternating directions. Each one alone shows almost nothing, while jointly the change is certain. That's the same demonstration as in your bcpd README.

**tools.py.** Design choices worth explaining in an interview:
- **No arithmetic for the model.** `change_in_mean` is computed in Python, because "the level rose by X" is where LLMs most often invent numbers.
- **Errors are returned, not raised.** If a series doesn't exist, the model receives `{"error": ...}` and must say so.
- **Tool descriptions are part of the design.** They're what the model reads to choose a method, so the wording of "use when the question is about NOW" is a control in itself.
- **The online tool's prior is an assumption:** segments last a quarter of the series on average. It's stated in the code and in the README limitations.
- **The joint tool reports each series alone too,** so the answer can show *why* the multivariate view mattered.

**guardrails.py.** Read the comments at the top. `ALWAYS_ALLOWED` lists numbers that need no source: 0–10 for counts, and 5, 90 and 95 for "90% interval". That's a deliberate, documented exception, which is how a validator would want to see it.

**agent.py.** The loop runs at most 8 steps, as a safety limit against runaway tool loops. `AgentResult` stores everything needed for auditing.

**evals/run_eval.py.** Scores each case on separate questions (section 6) and writes a JSON file with every answer plus a Markdown table. `--max-hallucination-rate 0` makes it exit with an error if any final answer still has a guardrail failure. CI uses that.

---

## 5. Running it on your computer (Git Bash)

> Build this on your **personal** computer, not the bank laptop. Public data only. Nothing from Nordea goes in this repository.

Every time you open a new Git Bash window:

```bash
cd ~/Documents/GIT/bcpd-agent
source bcpd_agent/Scripts/activate          # your venv
export ANTHROPIC_API_KEY="..."               # paste with Shift+Insert (unless saved in ~/.bashrc)
```

Check everything works without a key:

```bash
pytest                            # expect: 19 passed
python -m evals.run_eval --fake   # expect: 16 PASS, 2 FAIL (explained in section 6)
```

Questions to try, one per method, plus a refusal:

```bash
bcpd-agent "Did the series syn_two_shifts change regime? When?"                  # offline
bcpd-agent "Is something changing right now in syn_recent_change?"               # online, yes
bcpd-agent "Is anything changing at the moment in syn_one_big?"                  # online, no
bcpd-agent "Did syn_mv_1, syn_mv_2, syn_mv_3 and syn_mv_4 change together?"      # multivariate
bcpd-agent "Did the US credit card charge-off rate (CORCCACBS) change regime since 2000?"   # real data
bcpd-agent "Should I buy bank shares?"                                           # refused before the model
```

The real evaluation:

```bash
python -m evals.run_eval                               # Haiku (default)
python -m evals.run_eval --model claude-sonnet-5-5     # a larger model, to compare
```

Results go to `results/`, and the audit log to `logs/audit.jsonl`. Neither is pushed to GitHub.

---

## 6. Reading the evaluation results

| Metric | Question it answers | Why it matters |
|---|---|---|
| `method_choice_accuracy` | Did the agent pick offline / online / multivariate correctly? | The agent's own judgement |
| `detector_accuracy` | Did bcpd find the planted changes within ±3 months? | Statistics quality, independent of the LLM |
| `interval_90_coverage` | Share of true dates inside the tool's 90% interval | **Calibration.** About 0.9 means the uncertainty is honest |
| `faithfulness` | Did the agent report exactly what the tool said? | LLM quality: no distortion |
| `end_to_end_accuracy` | Is the final answer right? | What the user experiences |
| `refusal_accuracy` | Refused exactly when it should have | Scope control works |
| `no_fabrication_rate` | For a missing series, no invented changes | Behaviour under failure |
| `hallucination_rate_first_draft` / `_final` | Unsupported numbers or dates, before and after the guardrail | Raw model behaviour vs effect of the control |
| `wrong_direction_rate_first_draft` / `_final` | "Rose"/"fell" against the computed sign, before and after | Same, for wording |
| `mean_latency_s`, tokens | Speed and cost | Operations |

**What the offline run shows:** two cases fail end to end, `two_shifts` and `three_shifts`, but faithfulness is 100%. The agent reported the tool correctly, and the detector placed some dates 4–5 months from the truth. Repeating those designs with 200 different noise draws gave an average date error of +0.01 months, so there's no bias, and 95% of true dates fell inside the 90% intervals. The misses are noise, and the stated uncertainty is honest.

**What to look for with the real model:**
- Does `method_choice_accuracy` reach 1.0? The three method-choice cases test the model's judgement directly.
- Is the first-draft hallucination or direction rate above zero, and do the guardrails bring it to zero?
- Do the two paraphrased out-of-scope questions get refused? The regex doesn't catch them, so only the model can.
- How do Haiku and Sonnet compare on accuracy, latency and tokens?

---

## 7. If something goes wrong

| Problem | Likely cause and fix |
|---|---|
| `Could not resolve authentication method` | API key not set in *this* Git Bash window. Run the `export` again, pasting with Shift+Insert |
| `credit balance is too low` | Add API credits in the Console under Billing |
| `NotFoundError` about the model | Model name changed. Use `--model` with a current name from the docs |
| FRED error or timeout | No internet, or a wrong series id. Check it on fred.stlouisfed.org |
| `ModuleNotFoundError: evals` | Run commands from the project's root folder |
| Online or joint question is slow | Normal: the particle filter and the multivariate model take a few seconds each |

---

## 8. How to talk about it

**The two-minute story**

> "At Ericsson I ran a year-long evaluation of LLMs for coding. They were fast but hallucinated, even on easy tasks. So when I built an agent, I designed it around that failure. It answers questions about regime changes in time series using my own Bayesian changepoint package. Its main decision is choosing the right method: the exact offline algorithm for history, my online particle filter for 'is it changing now', or the multivariate model for 'did these change together'. It never does arithmetic itself. Deterministic checks reject any number that no tool produced and any 'rose' or 'fell' that contradicts the computed change. A scope check refuses advice and forecasts before the model is even called, a human approves every report, and everything goes into an audit log. I evaluate it on cases with known answers, scoring the method choice, the statistics and the model's faithfulness separately, so I can tell where an error came from. The evaluation runs in CI as a gate."

**How it maps to the roles**

| Role asks for | Point to |
|---|---|
| Agent design, tool calling, evaluation (Agentic Business Engineer) | Method-choice tools, `evals/`, model comparison |
| Guardrails, human-in-the-loop, data leakage (Agentic Business Engineer) | `guardrails.py`, approval step, data stays in Python |
| LLM-as-a-Judge, hallucination detection, guardrails (AI Business Monitoring) | Grounding check, direction check, first-draft vs final rates |
| AI evaluation, red-teaming, governance for agents (AI Safety) | Paraphrased out-of-scope cases, step limits, audit log |
| Lifecycle controls, monitoring, evidence (AI Governance) | CI gate, audit log, saved eval results per version, documented limitations |

**Governance note (to write as GOVERNANCE.md).** Map each control to a requirement. The AI Act articles on high-risk systems are a useful vocabulary, even though this agent isn't high-risk:
- Record-keeping and logging (Art. 12): `logs/audit.jsonl`, eval result files.
- Transparency to users (Art. 13): README limitations, uncertainty in every answer.
- Human oversight (Art. 14): the approval step, refusals, step limit.
- Accuracy and robustness (Art. 15): evaluation suite, calibration check, CI gate.

**Be ready for these questions**
- *How does the agent know which method to use?* From the tool descriptions and the system prompt. It's measured by the method-choice cases, not assumed.
- *Why not let the model compute?* LLMs are unreliable at arithmetic. Python is exact and testable.
- *What do the guardrails not catch?* A real date used in the wrong place (for example an interval endpoint), and wording errors in sentences the direction check skips as ambiguous. A full LLM-as-a-judge with its own evaluation is the next step.
- *Why a regex scope check if the model refuses anyway?* Defence in depth: it's cheap, deterministic and testable, and the model layer covers paraphrases. The eval measures both.
- *How would this work in a bank?* Same pattern with internal data. Tools sit behind access controls, the model runs on an approved platform, every decision is in the audit log, and changes go through the CI gate and model risk review.
- *You used AI to build it?* Yes, and I verified it the way I'd verify any model: unit tests, offline evaluation, and evaluation against known truth. I can explain every component.
