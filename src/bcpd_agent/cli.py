"""Command line entry point.

    bcpd-agent "Did US unemployment change regime since 2000?"
    bcpd-agent "Is something changing right now in syn_recent_change?"
    bcpd-agent "Did syn_mv_1, syn_mv_2, syn_mv_3 and syn_mv_4 change together?"
    bcpd-agent --yes "..."          # skip the approval prompt
    bcpd-agent --model claude-sonnet-5-5 "..."

Flow: ask the agent -> show tool calls and guardrail results -> a person
approves -> the report is saved to reports/ (that is the "send" step).
Every question is recorded in logs/audit.jsonl, approved or not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

from .agent import DEFAULT_MODEL, MonitoringAgent
from .audit import DEFAULT_LOG, write_audit
from .examples import register_examples
from .guardrails import ask_approval
from .tools import Toolbox


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ask the monitoring analyst agent a question.")
    p.add_argument("question")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--yes", action="store_true", help="Approve automatically (no human in the loop).")
    p.add_argument("--plot-dir", default="reports")
    p.add_argument("--audit-log", default=DEFAULT_LOG)
    args = p.parse_args(argv)

    toolbox = Toolbox(plot_dir=args.plot_dir)
    register_examples(toolbox.store)  # so questions about syn_* test series work
    agent = MonitoringAgent(model=args.model, toolbox=toolbox)
    res = agent.ask(args.question)

    print("\nTool calls:")
    for c in res.tool_calls:
        print(f"  - {c['tool']}({json.dumps(c['input'])})")
    print(
        f"\nGuardrails: scope refusal={res.refused_by_guardrail}, "
        f"first-draft unsupported={res.first_draft_unsupported}, "
        f"first-draft wrong direction={len(res.first_draft_direction_errors)}, "
        f"regenerations={res.regenerations}, "
        f"final unsupported={res.final_unsupported}, "
        f"final wrong direction={len(res.final_direction_errors)}"
    )
    print(f"Tokens in/out: {res.input_tokens}/{res.output_tokens}   Latency: {res.latency_s:.1f}s")

    approved = args.yes or ask_approval(res.answer)
    write_audit(res, approved=approved, path=args.audit_log)
    if not approved:
        print(f"Not approved. Nothing saved (recorded in {args.audit_log}).")
        return 1
    if args.yes:
        print("\n" + res.answer)

    os.makedirs(args.plot_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.plot_dir, f"report-{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Question\n\n{res.question}\n\n# Answer\n\n{res.answer}\n")
    print(f"Saved {path} (recorded in {args.audit_log})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
