"""Run the evaluation set against the agent and report the results.

    python -m evals.run_eval                          # default model
    python -m evals.run_eval --model claude-sonnet-5-5
    python -m evals.run_eval --max-hallucination-rate 0.0   # exit 1 if exceeded (for CI)

What is measured, per case and in total
---------------------------------------
Three different questions, so an error can be traced to where it came from:

  detector_correct   Did the TOOL (bcpd) find the planted changes?
                     (tool output vs truth: a statistics question)
  faithful           Did the AGENT report what the tool said?
                     (answer vs tool output: an LLM question)
  end_to_end_correct Is the final answer right?
                     (answer vs truth: what the user experiences)

Plus:
  refusal_correct    Refused exactly when it should have.
  no_fabrication     For a missing series, did not invent changepoints.
  hallucination      Any number/date in the answer not found in tool output.
                     Measured on the first draft (raw model behaviour) and on
                     the final answer (after the grounding guardrail).
  latency, tokens    Speed and cost.

Results are written to results/eval-<model>-<timestamp>.json and .md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import asdict
from typing import Any

from bcpd_agent.agent import DEFAULT_MODEL, MonitoringAgent
from bcpd_agent.tools import Toolbox

from .cases import DATE_TOLERANCE_MONTHS, Case, build_cases


def _months(d: str) -> int:
    y, m = int(d[:4]), int(d[5:7])
    return y * 12 + m


def dates_match(found: list[str], truth: list[str], tol: int = DATE_TOLERANCE_MONTHS) -> bool:
    """Same number of changes and each one within `tol` months (in order)."""
    if len(found) != len(truth):
        return False
    return all(abs(_months(f) - _months(t)) <= tol for f, t in zip(sorted(found), sorted(truth)))


def score_case(case: Case, res: Any) -> dict[str, Any]:
    v = res.verdict or {}
    refused = bool(v.get("refused"))
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "difficulty": case.difficulty,
        "verdict_parsed": res.verdict is not None,
        "refused": refused,
        "refused_by_guardrail": res.refused_by_guardrail,
        "first_draft_hallucination": bool(res.first_draft_unsupported),
        "final_hallucination": bool(res.final_unsupported),
        "unsupported_first_draft": res.first_draft_unsupported,
        "regenerations": res.regenerations,
        "n_tool_calls": len(res.tool_calls),
        "latency_s": round(res.latency_s, 2),
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
    }

    if case.expect_refusal:
        row["refusal_correct"] = refused
        return row
    row["refusal_correct"] = not refused

    if case.expect_error:
        row["no_fabrication"] = not v.get("changepoints") and not v.get("n_changepoints")
        return row

    # Last detect_changepoints call = what the tool actually said.
    det = [c["output"] for c in res.tool_calls if c["tool"] == "detect_changepoints" and "error" not in c["output"]]
    tool_dates = [c["regime_starts"] for c in det[-1]["changepoints"]] if det else None
    answer_dates = [d[:7] for d in v.get("changepoints", [])]
    truth = [d[:7] for d in case.changepoints]

    row["detector_correct"] = tool_dates is not None and dates_match(tool_dates, truth)
    # Calibration: does each true date fall inside the tool's 90% interval?
    # If the method is well calibrated, this should happen about 90% of the time.
    if det and tool_dates is not None and len(tool_dates) == len(truth) and truth:
        covered = [
            _months(c["interval_90"][0]) <= _months(t) <= _months(c["interval_90"][1])
            for c, t in zip(det[-1]["changepoints"], sorted(truth))
        ]
        row["interval_hits"], row["interval_total"] = sum(covered), len(covered)
    row["faithful"] = tool_dates is not None and sorted(answer_dates) == sorted(tool_dates) and v.get("n_changepoints") == len(tool_dates)
    row["end_to_end_correct"] = v.get("n_changepoints") == case.n_changepoints and dates_match(answer_dates, truth)
    row["truth"] = truth
    row["tool_said"] = tool_dates
    row["agent_said"] = answer_dates
    return row


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if key in r]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarise(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "n_cases": len(rows),
        "end_to_end_accuracy": _rate(rows, "end_to_end_correct"),
        "detector_accuracy": _rate(rows, "detector_correct"),
        "interval_90_coverage": (
            round(sum(r.get("interval_hits", 0) for r in rows) / t, 3)
            if (t := sum(r.get("interval_total", 0) for r in rows)) else None
        ),
        "faithfulness": _rate(rows, "faithful"),
        "refusal_accuracy": _rate(rows, "refusal_correct"),
        "no_fabrication_rate": _rate(rows, "no_fabrication"),
        "hallucination_rate_first_draft": _rate(rows, "first_draft_hallucination"),
        "hallucination_rate_final": _rate(rows, "final_hallucination"),
        "verdict_parse_rate": _rate(rows, "verdict_parsed"),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
    }


def to_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [f"# Evaluation: {summary['model']}", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items() if k != "model"]
    lines += ["", "| Case | Difficulty | End-to-end | Faithful | Detector | Refusal ok | Halluc. (1st draft) | Truth | Tool | Agent |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['case_id']} | {r['difficulty']} | {r.get('end_to_end_correct', '')} | {r.get('faithful', '')} | "
            f"{r.get('detector_correct', '')} | {r['refusal_correct']} | {r['first_draft_hallucination']} | "
            f"{r.get('truth', '')} | {r.get('tool_said', '')} | {r.get('agent_said', '')} |"
        )
    return "\n".join(lines) + "\n"


def run(model: str, client: Any = None, out_dir: str = "results", verbose: bool = True) -> dict[str, Any]:
    rows = []
    for case in build_cases():
        # A fresh toolbox per case: no state leaks between questions.
        tb = Toolbox(plot_dir=os.path.join(out_dir, "plots"))
        for c in build_cases():
            if c.series is not None:
                tb.store.register_synthetic(c.series)
        agent = MonitoringAgent(client=client, model=model, toolbox=tb)
        res = agent.ask(case.question)
        row = score_case(case, res)
        row["answer"] = res.answer
        rows.append(row)
        if verbose:
            ok = row.get("end_to_end_correct", row.get("no_fabrication", row["refusal_correct"]))
            print(f"{'PASS' if ok else 'FAIL'}  {case.case_id:24s} {row['latency_s']:>5}s")

    summary = summarise(rows, model)
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(out_dir, f"eval-{model}-{stamp}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": rows}, f, indent=2, default=str)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(summary, rows))
    if verbose:
        print("\n" + json.dumps(summary, indent=2))
        print(f"\nSaved {base}.json and .md")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", default="results")
    p.add_argument(
        "--max-hallucination-rate", type=float, default=None,
        help="Fail (exit code 1) if the FINAL hallucination rate is above this. Used as a CI gate.",
    )
    p.add_argument(
        "--fake", action="store_true",
        help="Use the scripted fake model (no API key, no cost). Tests the plumbing only.",
    )
    args = p.parse_args(argv)
    client = None
    model = args.model
    if args.fake:
        from bcpd_agent.fake_llm import FakeClient

        client, model = FakeClient(), "fake"
    summary = run(model, client=client, out_dir=args.out_dir)
    if args.max_hallucination_rate is not None:
        rate = summary["hallucination_rate_final"] or 0.0
        if rate > args.max_hallucination_rate:
            print(f"GATE FAILED: hallucination rate {rate} > {args.max_hallucination_rate}")
            return 1
        print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
