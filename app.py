"""Web interface for the monitoring-analyst agent.

    pip install -e ".[ui]"
    streamlit run app.py

A page opens in your browser. Type a question; the agent chooses a method,
runs it, and the page shows the chart, the answer, what the agent did, and
Approve / Reject buttons (the human-in-the-loop step, written to the audit log).

The page is only a front end: it uses exactly the same agent, tools and
guardrails as the `bcpd-agent` command and the evaluation.
"""

from __future__ import annotations

import datetime as dt
import os

import streamlit as st

from bcpd_agent.agent import MonitoringAgent
from bcpd_agent.audit import write_audit
from bcpd_agent.examples import register_examples
from bcpd_agent.guardrails import VERDICT_RE
from bcpd_agent.plots import figure_for
from bcpd_agent.tools import Toolbox

st.set_page_config(page_title="Changepoint monitoring agent", layout="wide")

EXAMPLES = {
    "History (offline)": "Did the series syn_two_shifts change regime? When, and how sure are you?",
    "Now (online)": "Is something changing right now in syn_recent_change?",
    "Together (multivariate)": "Did syn_mv_1, syn_mv_2, syn_mv_3 and syn_mv_4 change together?",
    "Real data (FRED)": "Did the US credit card charge-off rate (CORCCACBS) change regime since 2000?",
    "Out of scope": "Should I buy bank shares based on syn_one_big?",
}
METHOD_LABEL = {
    "detect_changepoints": "Offline: exact inference over the whole history",
    "detect_recent_change": "Online: particle filter, when did the current regime start?",
    "detect_joint_changepoints": "Multivariate: shared changes across series",
}

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    typed_key = st.text_input(
        "Anthropic API key", type="password", value="",
        help="Leave empty to use ANTHROPIC_API_KEY from your terminal. Kept only in this browser session.",
    )
    have_key = bool(typed_key or env_key)
    offline_demo = st.checkbox(
        "Offline demo (scripted fake model, no key, no cost)", value=not have_key,
        help="Uses the same scripted stand-in as the tests. Good for showing the interface; "
             "it says nothing about the real model's quality.",
    )
    model = st.selectbox("Model", ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
                         disabled=offline_demo)
    st.divider()
    st.caption("Try an example")
    for label, q in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["question"] = q

# ------------------------------------------------------------------- main
st.title("Changepoint monitoring agent")
st.caption(
    "Ask whether a time series changed regime, whether something is changing now, or whether several "
    "series changed together. The agent chooses the method; the statistics come from exact Bayesian "
    "changepoint detection (bcpd); guardrails check every number before you approve the answer."
)

question = st.text_input("Your question", key="question",
                         placeholder="e.g. Did US unemployment (UNRATE) change regime since 2000?")
ask = st.button("Ask", type="primary", disabled=not question)

if ask and question:
    if offline_demo:
        from bcpd_agent.fake_llm import FakeClient

        client, model_name = FakeClient(), "fake (offline demo)"
    elif not have_key:
        st.error("No API key. Paste one in the sidebar, set ANTHROPIC_API_KEY, or tick Offline demo.")
        st.stop()
    else:
        import anthropic

        client = anthropic.Anthropic(api_key=typed_key or env_key)
        model_name = model

    toolbox = Toolbox(plot_dir="reports")
    register_examples(toolbox.store)
    agent = MonitoringAgent(client=client, model=model_name, toolbox=toolbox)
    with st.spinner("The agent is choosing a method and running it..."):
        try:
            result = agent.ask(question)
        except Exception as exc:  # e.g. authentication or network problems
            st.error(f"{type(exc).__name__}: {exc}")
            st.stop()
    st.session_state.update(result=result, toolbox=toolbox, decision=None)

result = st.session_state.get("result")
if result is not None:
    toolbox = st.session_state["toolbox"]
    verdict = result.verdict or {}
    used = [c["tool"] for c in result.tool_calls]
    method_tool = next((t for t in reversed(used) if t in METHOD_LABEL), None)

    left, right = st.columns([3, 2], gap="large")

    with left:
        fig = figure_for(result, toolbox)
        if fig is not None:
            st.pyplot(fig, clear_figure=True)
        elif verdict.get("refused"):
            st.info("Refused: this question is outside the agent's scope.")
        else:
            st.info("No chart: no detection result to draw.")

    with right:
        st.subheader("Answer")
        st.markdown(VERDICT_RE.sub("", result.answer).strip())

        clean = not (result.final_unsupported or result.final_direction_errors)
        method_name = {"detect_changepoints": "offline", "detect_recent_change": "online",
                       "detect_joint_changepoints": "multivariate"}.get(method_tool, "none")
        st.markdown(
            f"**Method:** {method_name} &nbsp;·&nbsp; "
            f"**Guardrails:** {':green[passed]' if clean else ':red[failed]'} &nbsp;·&nbsp; "
            f"**Rewrites:** {result.regenerations}"
        )

        decision = st.session_state.get("decision")
        if decision is None:
            a, r = st.columns(2)
            if a.button("Approve and save", type="primary", use_container_width=True):
                write_audit(result, approved=True)
                os.makedirs("reports", exist_ok=True)
                path = os.path.join("reports", f"report-{dt.datetime.now():%Y%m%d-%H%M%S}.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# Question\n\n{result.question}\n\n# Answer\n\n{result.answer}\n")
                st.session_state["decision"] = f"Approved. Saved {path} and recorded in logs/audit.jsonl."
                st.rerun()
            if r.button("Reject", use_container_width=True):
                write_audit(result, approved=False)
                st.session_state["decision"] = "Rejected. Nothing saved; recorded in logs/audit.jsonl."
                st.rerun()
        else:
            st.success(decision)

    with st.expander("What the agent did", expanded=False):
        if result.refused_by_guardrail:
            st.write("Stopped by the scope guardrail **before** the model was called (0 tokens).")
        if method_tool:
            st.write(f"**Method chosen:** {METHOD_LABEL[method_tool]}")
        st.write("**Tool calls**")
        for i, c in enumerate(result.tool_calls, 1):
            st.markdown(f"{i}. `{c['tool']}` with `{c['input']}`")
            st.json(c["output"], expanded=False)
        st.write("**Guardrails**")
        st.table({
            "Check": ["Unsupported numbers/dates", "Wrong rise/fall wording"],
            "First draft": [", ".join(result.first_draft_unsupported) or "none",
                            str(len(result.first_draft_direction_errors)) or "0"],
            "Final answer": [", ".join(result.final_unsupported) or "none",
                             str(len(result.final_direction_errors))],
        })
        st.write(f"**Model:** {result.model} · **Tokens:** {result.input_tokens:,} in / "
                 f"{result.output_tokens:,} out · **Latency:** {result.latency_s:.1f} s")
        st.write("**Structured verdict**")
        st.json(verdict)
