"""Tests that run without an API key. Run with:  pytest

They check the plumbing and the guardrails, not the quality of a real model
(that is what the evaluation suite is for).
"""

from bcpd_agent.agent import MonitoringAgent
from bcpd_agent.data import make_synthetic
from bcpd_agent.fake_llm import FakeClient
from bcpd_agent.guardrails import grounding_check, parse_verdict, scope_check
from bcpd_agent.tools import Toolbox


def _toolbox():
    tb = Toolbox(plot_dir="/tmp/bcpd-agent-test")
    tb.store.register_synthetic(make_synthetic("syn_one_big", [90, 70], [0.0, 2.0], seed=2))
    return tb


# ----------------------------------------------------------------- tools
def test_detect_finds_planted_change():
    tb = _toolbox()
    tb.load_series("syn_one_big")
    out = tb.detect_changepoints("syn_one_big")
    assert out["most_probable_number"] == 1
    assert out["changepoints"][0]["regime_starts"] == "2007-07"  # planted at month 90


def test_tool_errors_are_returned_not_raised():
    tb = _toolbox()
    out = tb.run("detect_changepoints", {"series_id": "never_loaded"})
    assert "error" in out


def test_plot_is_saved():
    tb = _toolbox()
    tb.load_series("syn_one_big")
    tb.detect_changepoints("syn_one_big")
    assert tb.make_plot("syn_one_big")["path"].endswith(".png")


# ------------------------------------------------------------ guardrails
def test_scope_blocks_investment_and_forecasts():
    assert scope_check("Should I buy Nordea shares?") == "investment advice"
    assert scope_check("What will unemployment be next year?") == "forecasting"
    assert scope_check("Did UNRATE change regime?") is None


def test_grounding_accepts_rounded_and_percent_values():
    tool = [{"mean_before": 4.213, "probability_of_most_probable_number": 0.873, "regime_starts": "2008-09"}]
    assert grounding_check("Mean was 4.2 and we are 87% sure, from 2008-09.", tool) == []


def test_grounding_flags_invented_number_and_date():
    tool = [{"mean_before": 4.213, "regime_starts": "2008-09"}]
    bad = grounding_check("It rose by 37.25 starting 2009-01.", tool)
    assert "37.25" in bad and "2009-01" in bad


def test_parse_verdict():
    v = parse_verdict('text\n```json\n{"refused": false, "n_changepoints": 1, "changepoints": ["2007-07"]}\n```')
    assert v["n_changepoints"] == 1


# ---------------------------------------------------------- agent loop
def test_loop_end_to_end_with_fake_model():
    agent = MonitoringAgent(client=FakeClient(), model="fake", toolbox=_toolbox())
    res = agent.ask("Did the series syn_one_big change regime?")
    assert [c["tool"] for c in res.tool_calls] == ["load_series", "detect_changepoints"]
    assert res.verdict["changepoints"] == ["2007-07"]
    assert res.final_unsupported == []


def test_grounding_guardrail_forces_rewrite():
    agent = MonitoringAgent(client=FakeClient(hallucinate_first=True), model="fake", toolbox=_toolbox())
    res = agent.ask("Did the series syn_one_big change regime?")
    assert "37.25" in res.first_draft_unsupported  # the invented number was caught
    assert res.regenerations == 1                 # the model was asked to fix it
    assert res.final_unsupported == []           # and the final answer is clean


def test_scope_guardrail_skips_the_model():
    agent = MonitoringAgent(client=FakeClient(), model="fake", toolbox=_toolbox())
    res = agent.ask("Should I buy Nordea shares based on syn_one_big?")
    assert res.refused_by_guardrail and res.input_tokens == 0
