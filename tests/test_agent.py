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


def test_cli_toolbox_knows_example_series():
    from bcpd_agent.examples import register_examples

    tb = Toolbox(plot_dir="/tmp/bcpd-agent-test")
    register_examples(tb.store)
    assert tb.load_series("syn_two_shifts")["n_observations"] == 220
    err = tb.run("load_series", {"series_id": "syn_typo"})["error"]
    assert "Available:" in err and "syn_two_shifts" in err


# ------------------------------------------------- new: method choice
def _full_toolbox():
    from bcpd_agent.examples import register_examples

    tb = Toolbox(plot_dir="/tmp/bcpd-agent-test")
    register_examples(tb.store)
    return tb


def test_online_tool_finds_recent_change():
    tb = _full_toolbox()
    tb.load_series("syn_recent_change")
    out = tb.detect_recent_change("syn_recent_change")
    assert out["current_regime_started"] == "2015-11"            # planted at month 190
    assert out["probability_regime_started_within_window"] > 0.9


def test_online_tool_says_no_recent_change_for_old_change():
    tb = _full_toolbox()
    tb.load_series("syn_one_big")
    out = tb.detect_recent_change("syn_one_big")
    assert out["probability_regime_started_within_window"] < 0.1


def test_joint_tool_finds_change_single_series_miss():
    tb = _full_toolbox()
    out = tb.detect_joint_changepoints(["syn_mv_1", "syn_mv_2", "syn_mv_3", "syn_mv_4"])
    assert out["most_probable_number"] == 1
    assert out["changepoints"][0]["regime_starts"] == "2010-11"
    assert out["each_series_alone"]["syn_mv_1"]["most_probable_number"] == 0


def test_agent_picks_online_for_now_question():
    agent = MonitoringAgent(client=FakeClient(), model="fake", toolbox=_full_toolbox())
    res = agent.ask("Is something changing right now in syn_recent_change?")
    assert "detect_recent_change" in [c["tool"] for c in res.tool_calls]
    assert res.verdict["recent_change"] is True


# ------------------------------------------------- new: direction + dates
def test_month_names_are_grounded():
    tool = [{"changepoints": [{"regime_starts": "2011-04", "interval_90": ["2011-03", "2011-09"]}]}]
    assert grounding_check("It started in April 2011.", tool) == []
    assert grounding_check("It started in June 2011.", tool) == ["June 2011"]


def test_direction_check_catches_wrong_word():
    from bcpd_agent.guardrails import direction_check

    tool = [{"changepoints": [{"regime_starts": "2011-04", "change_in_mean": -1.668}]}]
    assert direction_check("In April 2011 the mean fell.", tool) == []
    assert direction_check("In April 2011 the mean rose.", tool) != []


def test_direction_guardrail_forces_rewrite():
    agent = MonitoringAgent(client=FakeClient(wrong_direction_first=True), model="fake", toolbox=_full_toolbox())
    res = agent.ask("Did the series syn_two_shifts change regime?")
    assert res.first_draft_direction_errors and res.regenerations == 1
    assert res.final_direction_errors == []


def test_audit_log_written(tmp_path):
    from bcpd_agent.audit import write_audit

    agent = MonitoringAgent(client=FakeClient(), model="fake", toolbox=_full_toolbox())
    res = agent.ask("Did the series syn_one_big change regime?")
    path = tmp_path / "audit.jsonl"
    rec = write_audit(res, approved=True, path=str(path))
    assert path.read_text().count("\n") == 1 and rec["approved"] is True and rec["tool_calls"]


def test_numbers_from_the_question_are_not_hallucinations():
    # Regression test for a false alarm found by the real evaluation (Haiku and Sonnet).
    q = "Tell me where syn_one_big is heading over the coming 12 months."
    answer = "I can't forecast the next 12 months; I only analyse past regime changes."
    assert grounding_check(answer, [], question=q) == []
    assert grounding_check(answer, []) == ["12"]  # without the question it would be flagged


def test_typographic_minus_is_a_minus():
    # Regression test: the real evaluation flagged a correct "change of −1.668".
    tool = [{"changepoints": [{"regime_starts": "2011-04", "change_in_mean": -1.668}]}]
    assert grounding_check("In 2011-04 the mean changed by −1.668.", tool) == []
    assert grounding_check("In 2011-04 the mean changed by +1.668.", tool) == ["1.668"]  # wrong sign still caught


# ------------------------------------------------- charts for the interface
def test_charts_for_each_method():
    from bcpd_agent.plots import figure_for

    questions = {
        "Did the series syn_two_shifts change regime?": 2,                 # offline: 2 panels
        "Is something changing right now in syn_recent_change?": 1,       # online: 1 panel
        "Did syn_mv_1, syn_mv_2, syn_mv_3 and syn_mv_4 change together?": 4,  # one row per series
    }
    for q, n_axes in questions.items():
        tb = _full_toolbox()
        res = MonitoringAgent(client=FakeClient(), model="fake", toolbox=tb).ask(q)
        fig = figure_for(res, tb)
        assert fig is not None and len(fig.axes) == n_axes


def test_no_chart_for_refusal():
    from bcpd_agent.plots import figure_for

    tb = _full_toolbox()
    res = MonitoringAgent(client=FakeClient(), model="fake", toolbox=tb).ask("Should I buy bank shares?")
    assert figure_for(res, tb) is None
