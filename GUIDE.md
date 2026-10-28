# The bcpd-agent guide

This guide explains what the project does, how every part works, how to run it on your computer, and how to talk about it in an interview. Read sections 1–3 before touching the code. The rest is reference.

---

## 1. What you are building, in one paragraph

A person asks a question in plain language, for example *"Did the US credit card charge-off rate change regime since 2000?"* An LLM (Claude) decides which tools to call: load the data, run your Bayesian changepoint detection, maybe draw a chart. It then explains the result in a few sentences for a manager. Around the LLM sit controls written in plain Python. One refuses questions outside the agent's job. One checks that every number in the answer really came from a tool. One asks a human to approve the report before it's "sent". An evaluation suite runs 15 questions with known answers and measures how often the whole thing is right, faithful, calibrated and safe.

It's a small version of exactly what the Nordea roles describe: an AI capability with evaluation, guardrails, human oversight and monitoring.

---

## 2. Key ideas, explained plainly

**Agent.** An LLM in a loop that can take actions, here by calling tools, and decide the next step from the result. Without tools, an LLM can only produce text from memory.

**Tool calling.** We describe Python functions to the model as a name, a description and the parameters (a JSON schema). The model can't run them. It replies with a structured request instead: *"call `detect_changepoints` with `series_id = UNRATE`"*. Our code runs the function and sends the result back as a message. The model reads it and either calls another tool or writes the final answer. The definitions are in `TOOL_SCHEMAS` in `tools.py`.

**System prompt.** Standing instructions given to the model before the conversation: who it is, its rules, and the format of its answer. Ours is `SYSTEM_PROMPT` in `agent.py`. Two rules matter most: *only copy numbers from tool results*, and *end every answer with a JSON block*.

**Tokens.** The unit the API counts and charges by, roughly ¾ of a word. Every message in the conversation is sent again on each step, so longer tool loops cost more. The agent records input and output tokens for every question.

**Hallucination.** The model states something that isn't supported. Here we define it narrowly and measurably: *a number or date in the answer that appears in no tool output.* That's the same idea as your Ericsson PoC, turned into an automatic check.

**Guardrail.** A control around the model, not inside it. Ours are deterministic Python, so they can be unit-tested, and their effect can be measured. A guardrail that is only an instruction in the prompt ("please don't invent numbers") is a request, not a control.

**Grounding.** Tying every claim to a source. The grounding check collects every number and date from the tool outputs, then looks for each number and date in the answer. `2.4` counts as supported if a tool said `2.413`, and `87%` counts if a tool said `0.873`. Anything else is flagged. The model is then told exactly which values failed and asked to rewrite, at most twice.

**Structured verdict.** Free text is hard to score automatically. So the model must end with
`{"refused": false, "series_id": "UNRATE", "n_changepoints": 2, "changepoints": ["2008-11", "2020-04"]}`.
The evaluation reads this block. In production you'd use it to feed dashboards or alerts.

**Human in the loop.** The command-line tool shows the draft and asks *"Approve and save this report? [y/N]"*. Nothing is "sent" (saved) without a yes. That's human oversight in its simplest form.

**Evaluation (evals).** A fixed set of questions with known correct answers, run against the agent, with each answer scored. It's the LLM equivalent of backtesting a model. You re-run it after every change, such as a new prompt, a new model version or a new tool, and compare.

**Fake model.** `fake_llm.py` pretends to be the Claude API with a fixed script. It lets the tests and the evaluation pipeline run offline, for free, and in CI without a key. It tests the plumbing, not the intelligence.

**CI (continuous integration).** GitHub runs the tests automatically on every push. In `.github/workflows/ci.yml` the build fails if the hallucination rate is above zero, so a change that makes the agent less trustworthy can't slip in unnoticed. That's "evaluation as a gate".

---

## 3. What happens when you ask a question

Example: `bcpd-agent "Did the series syn_one_big change regime?"`

1. **Scope guardrail** (`guardrails.scope_check`). Regex patterns look for investment advice, forecasts and personal data. If one matches, the agent returns a fixed refusal and **the LLM is never called**, costing zero tokens.
2. **First model call.** Claude gets the system prompt, the three tool definitions and the question. It replies: *call `load_series(series_id="syn_one_big")`*.
3. **Tool runs in Python** (`Toolbox.load_series`). It returns a summary: number of observations, dates, mean, standard deviation and so on. The data stays in Python; only the summary goes to the model.
4. **Second model call.** Claude reads the summary and asks for `detect_changepoints`.
5. **Your package runs** (`Toolbox.detect_changepoints`). The series is standardised, `bcpd.detect` runs, and the output is condensed into the probability of each number of changes, plus, for the most likely number, the start date of each new regime with a 90% interval and the means before and after.
6. **Third model call.** Claude writes the answer and the JSON block.
7. **Grounding guardrail.** Every number and date is checked against the tool outputs. If something's unsupported, the model gets a message listing the values and is asked to rewrite.
8. **Result.** The answer, the verdict, every tool call, the guardrail results, tokens and latency. The command line shows all of it and asks for your approval.

---

## 4. File by file

```
bcpd-agent/
├── src/bcpd_agent/
│   ├── data.py         where data comes from (FRED download, synthetic series with planted changes)
│   ├── tools.py        the 3 tools + their JSON schemas + Toolbox that runs them
│   ├── guardrails.py   scope check, grounding check, verdict parser, human approval
│   ├── agent.py        the system prompt and the tool-calling loop
│   ├── cli.py          the `bcpd-agent` command
│   └── fake_llm.py     scripted stand-in for the API, for tests and offline runs
├── evals/
│   ├── cases.py        15 questions with known answers
│   └── run_eval.py     runs them, scores them, writes results/*.json and *.md
├── tests/test_agent.py 10 offline tests (pytest)
├── .github/workflows/ci.yml   automatic tests + evaluation gate on GitHub
├── Dockerfile          container for stage 2 (deployment)
├── README.md           the public, short description
└── GUIDE.md            this file
```

**data.py.** `SeriesStore` keeps loaded series for one conversation, so tools pass an id instead of hundreds of numbers. `fetch_fred` downloads public series (no key needed). `make_synthetic` builds monthly series from level segments, for example `[90, 70]` months with means `[0, 2]`, and records the true change dates. The truth is never shown to the model.

**tools.py.** Three tools, each returning a small dict. Two design choices are worth explaining in an interview:
- **No arithmetic for the model.** `change_in_mean` is computed in Python, because "the level rose by X" is where LLMs most often invent numbers.
- **Errors are returned, not raised.** If a series doesn't exist, the model receives `{"error": ...}` and must say so. The `missing_series` eval case checks that it doesn't invent results instead.

**guardrails.py.** Read the comments at the top. `ALWAYS_ALLOWED` lists numbers that need no source: 0–10 for counts, and 5, 90 and 95 for "90% interval". That's a deliberate, documented exception, which is how a validator would want to see it.

**agent.py.** The loop runs at most 8 steps, as a safety limit against runaway tool loops. `AgentResult` stores everything needed for auditing: tool calls with inputs and outputs, unsupported numbers in the first draft and the final answer, regenerations, tokens and latency. That record is the evidence a governance team would ask for.

**evals/run_eval.py.** Scores each case three ways (see section 6) and writes a JSON file with every answer plus a Markdown table. `--max-hallucination-rate 0` makes it exit with an error if the final hallucination rate is above zero. CI uses that.

---

## 5. Running it on your computer

> Build this on your **personal** computer, not the bank laptop. Public data only. Nothing from Nordea goes in this repository.

**One-time setup**

1. Install **Python 3.11** (python.org) and **Git** (git-scm.com).
2. On GitHub, create a new public repository called `bcpd-agent`, empty with no README.
3. Unzip the project folder, open a terminal in it and run:

   ```bash
   git init && git add . && git commit -m "Initial agent, guardrails and evaluation"
   git branch -M main
   git remote add origin https://github.com/NastaranM/bcpd-agent.git
   git push -u origin main
   ```

4. Create a virtual environment and install:

   **Mac/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
   **Windows (PowerShell)**
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e ".[dev]"
   ```

5. Check that everything works **without** a key:

   ```bash
   pytest                            # expect: 10 passed
   python -m evals.run_eval --fake   # expect: 13 PASS, 2 FAIL (explained in section 6)
   ```

**Add your API key** (from console.anthropic.com, with a spending limit set)

- Mac/Linux: `export ANTHROPIC_API_KEY="sk-ant-..."`
- Windows PowerShell: `$env:ANTHROPIC_API_KEY="sk-ant-..."`

This lasts for the current terminal window. Never put the key in a file you commit.

**First real questions**

```bash
bcpd-agent "Did the series syn_two_shifts change regime? When?"
bcpd-agent "Did the US credit card charge-off rate (CORCCACBS) change regime since 2000?"
bcpd-agent "Did US unemployment (UNRATE) change regime between 2015-01 and 2023-12?"
bcpd-agent "Should I buy bank shares?"        # should be refused without calling the model
```

**The real evaluation**

```bash
python -m evals.run_eval                               # Haiku (default)
python -m evals.run_eval --model claude-sonnet-5-5     # a larger model, to compare
```

Results go to `results/`. Open the `.md` file. If a model name is rejected, check the current model list in the Anthropic documentation and use `--model` with the right name.

---

## 6. Reading the evaluation results

| Metric | Question it answers | Why it matters |
|---|---|---|
| `detector_accuracy` | Did bcpd find the planted changes within ±3 months? | Statistics quality, independent of the LLM |
| `interval_90_coverage` | Share of true dates inside the tool's 90% interval | **Calibration.** About 0.9 means the uncertainty is honest |
| `faithfulness` | Did the agent report exactly what the tool said? | LLM quality: no distortion |
| `end_to_end_accuracy` | Is the final answer right? | What the user experiences |
| `refusal_accuracy` | Refused exactly when it should have | Scope control works |
| `no_fabrication_rate` | For a missing series, no invented changes | Behaviour under failure |
| `hallucination_rate_first_draft` | Answers with unsupported numbers **before** the guardrail | Raw model behaviour |
| `hallucination_rate_final` | The same **after** the guardrail | Effect of the control |
| `mean_latency_s`, tokens | Speed and cost | Operations |

**What the offline run already shows:** two cases fail end to end, `two_shifts` and `three_shifts`, but faithfulness is 100%. The agent reported the tool correctly, and the detector placed some dates 4–5 months from the truth. 9 of the 10 true dates still fell inside the 90% intervals. That's an attribution story: *the error was in the statistics, not the LLM, and the uncertainty statement was honest.* Being able to locate an error like that is the point of scoring three ways.

**What to look for with the real model:**
- Is the first-draft hallucination rate above zero, and does the guardrail bring it down to zero? That's your control working, and your Ericsson finding reproduced.
- Do the two paraphrased out-of-scope questions get refused? The regex doesn't catch them, so only the model's own refusal can.
- How does Haiku compare with a larger model on accuracy, latency and tokens?

---

## 7. Plan for today and tomorrow

**Today**
1. Setup, push to GitHub, run `pytest` and the fake eval (1 hour).
2. API key, try the four example questions, read the tool calls it prints (1 hour).
3. Real evaluation on Haiku, then a larger model. Save both result files (1 hour).
4. Read the answers in `results/*.json`. Where did it go wrong? Write down 3 observations.
5. One improvement, then re-run the eval and compare. Ideas:
   - Add 3 evaluation cases of your own (`evals/cases.py`), for example a trend break or a very small shift.
   - Tighten the system prompt if refusals or format fail.
   - Add an "always pass the 90% interval" rule if answers skip uncertainty.

**Tomorrow**
1. Push to GitHub, check that the CI run is green (Actions tab). Add the `ANTHROPIC_API_KEY` secret so the real gate runs.
2. Docker: `docker build -t bcpd-agent .` and run one question in the container.
3. Write the results section of the README: one table comparing models, plus 3 sentences of findings.
4. Write the governance note (section 9) as `GOVERNANCE.md`.
5. Thursday: tailored resumes and applications. Swedbank and the M365 role close Friday.

---

## 8. If something goes wrong

| Problem | Likely cause and fix |
|---|---|
| `pip install` fails on `bcpd` | Git isn't installed or not on PATH. Install Git, open a new terminal |
| `AuthenticationError` | API key not set in *this* terminal window, or typed wrong |
| `NotFoundError` about the model | Model name changed. Use `--model` with a current name from the docs |
| FRED error / timeout | No internet, or a wrong series id. Check it on fred.stlouisfed.org |
| `ModuleNotFoundError: evals` | Run commands from the project's root folder |
| Windows: "running scripts is disabled" | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |

---

## 9. How to talk about it

**The two-minute story**

> "At Ericsson I ran a year-long evaluation of LLMs for coding. They were fast but hallucinated, even on easy tasks. So when I built an agent, I designed it around that failure. It answers questions about regime changes in time series by calling my own Bayesian changepoint package as a tool, and it never does arithmetic itself. A deterministic grounding check rejects any number that no tool produced, a scope check refuses advice and forecasts before the model is even called, and a human approves every report. I evaluate it on cases with known answers, scored three ways, so I can tell whether an error came from the statistics or from the LLM. The evaluation runs in CI as a gate, so a change that raises the hallucination rate fails the build."

**How it maps to the roles**

| Role asks for | Point to |
|---|---|
| Evaluation strategies, test sets (Agentic Business Engineer) | `evals/`, three-way scoring, model comparison |
| Guardrails, human-in-the-loop, data leakage (Agentic Business Engineer) | `guardrails.py`, approval step, data stays in Python |
| LLM-as-a-Judge, hallucination detection, guardrails (AI Business Monitoring) | Grounding check, first-draft vs final rate. Next step: add an LLM judge for wording |
| AI evaluation, red-teaming, governance for agents (AI Safety) | Paraphrased out-of-scope cases, step limits, audit record in `AgentResult` |
| Lifecycle controls, monitoring, evidence (AI Governance) | CI gate, saved eval results per version, documented limitations |

**Governance note (to write tomorrow as GOVERNANCE.md).** Map each control to a requirement. The AI Act articles on high-risk systems are a useful vocabulary, even though this agent isn't high-risk:
- Record-keeping and logging (Art. 12): `AgentResult`, eval result files.
- Transparency to users (Art. 13): README limitations, uncertainty in every answer.
- Human oversight (Art. 14): the approval step, refusals, step limit.
- Accuracy and robustness (Art. 15): evaluation suite, calibration check, CI gate.

**Be ready for these questions**
- *Why not let the model compute?* LLMs are unreliable at arithmetic. Python is exact and testable.
- *What doesn't the grounding check catch?* Correct numbers in a wrong sentence, such as "rose" when the level fell. The fix is an LLM-as-a-judge or rule check on direction, with its own evaluation.
- *Why a regex scope check if the model refuses anyway?* Defence in depth: it's cheap, deterministic and testable, and the model layer covers paraphrases. The eval measures both.
- *How would this work in a bank?* Same pattern with internal data. Tools sit behind access controls, the model runs on an approved platform, results are logged, and changes go through the CI gate and model risk review.
- *You used AI to build it?* Yes, and I verified it the way I'd verify any model: unit tests, offline evaluation, and evaluation against known truth. I can explain every component.
