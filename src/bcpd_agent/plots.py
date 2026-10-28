"""Charts for the interface: one figure per detection method.

The chart is drawn from what the TOOLS computed (never from the model's text),
so what you see is exactly what the guardrails checked the answer against.

    offline       the series, a dashed line at each detected change, the 90%
                  interval shaded, and the changepoint probability underneath
    online        the series with the current regime shaded from its start
    multivariate  each series (standardised) on one chart, joint changes marked
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # draw without a screen
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .tools import Toolbox  # noqa: E402

LINE = "#1f4e5a"
CHANGE = "#c0504d"
BAND = "#f2c4c3"


def _index_of(dates: list[str], ym: str) -> int:
    """Index of the first date starting with 'YYYY-MM'."""
    for i, d in enumerate(dates):
        if d.startswith(ym):
            return i
    return len(dates) - 1


def _date_ticks(ax, dates: list[str], n: int = 8) -> None:
    x = np.arange(len(dates))
    ticks = x[:: max(1, len(x) // n)]
    ax.set_xticks(ticks, [dates[i][:7] for i in ticks], rotation=30, ha="right", fontsize=8)


def figure_for(result: Any, toolbox: Toolbox):
    """Return a matplotlib Figure for the last detection tool the agent used,
    or None if there is nothing to draw (refusal, error, no detection)."""
    detections = [
        c for c in result.tool_calls
        if c["tool"] in ("detect_changepoints", "detect_recent_change", "detect_joint_changepoints")
        and "error" not in c["output"]
    ]
    if not detections:
        return None
    call = detections[-1]
    if call["tool"] == "detect_changepoints":
        return _offline(call["output"], toolbox)
    if call["tool"] == "detect_recent_change":
        return _online(call["output"], toolbox)
    return _joint(call["output"], toolbox)


def _offline(out: dict[str, Any], toolbox: Toolbox):
    s = toolbox.store.get(out["series_id"])
    x = np.arange(len(s.values))
    det = toolbox._last_detection.get(out["series_id"])
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, height_ratios=[3, 1])
    ax = axes[0]
    ax.plot(x, s.values, lw=1, color=LINE)
    for c in out["changepoints"]:
        lo, hi = _index_of(s.dates, c["interval_90"][0]), _index_of(s.dates, c["interval_90"][1])
        ax.axvspan(lo, hi + 0.99, color=BAND, alpha=0.7, lw=0)
        ax.axvline(_index_of(s.dates, c["regime_starts"]), color=CHANGE, ls="--", lw=1.2)
    # Segment means as horizontal steps.
    bounds = [0] + [_index_of(s.dates, c["regime_starts"]) for c in out["changepoints"]] + [len(x)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        ax.hlines(s.values[a:b].mean(), a, b - 1, color=CHANGE, lw=1.5, alpha=0.8)
    ax.set_title(f"{out['series_id']}: offline, {out['most_probable_number']} change(s) "
                 f"(P = {out['probability_of_most_probable_number']})", fontsize=10)
    if det is not None:
        axes[1].fill_between(x, det["result"].location_prob, color=CHANGE, alpha=0.6, step="mid", lw=0)
    axes[1].set_ylabel("P(change)", fontsize=8)
    axes[1].set_ylim(0, 1)
    _date_ticks(axes[1], s.dates)
    fig.tight_layout()
    return fig


def _online(out: dict[str, Any], toolbox: Toolbox):
    s = toolbox.store.get(out["series_id"])
    x = np.arange(len(s.values))
    start = _index_of(s.dates, out["current_regime_started"])
    lo = _index_of(s.dates, out["current_regime_interval_90"][0])
    window_start = len(x) - out["window_observations"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axvspan(window_start, len(x), color="#e8eef0", lw=0, label="recent window")
    ax.plot(x, s.values, lw=1, color=LINE)
    ax.axvspan(lo, start + 0.99, color=BAND, alpha=0.7, lw=0)
    ax.axvline(start, color=CHANGE, ls="--", lw=1.2, label="current regime starts")
    ax.hlines(s.values[start:].mean(), start, len(x) - 1, color=CHANGE, lw=1.5)
    if start > 0:
        ax.hlines(s.values[:start].mean(), 0, start - 1, color=CHANGE, lw=1.5, alpha=0.5)
    ax.set_title(f"{out['series_id']}: online, current regime since {out['current_regime_started']} "
                 f"(P recent = {out['probability_regime_started_within_window']})", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    _date_ticks(ax, s.dates)
    fig.tight_layout()
    return fig


def _joint(out: dict[str, Any], toolbox: Toolbox):
    """One row per series, each with its mean before/after every shared change,
    so the (often small, opposite-direction) shifts are visible."""
    ids = out["series_ids"]
    fig, axes = plt.subplots(len(ids), 1, figsize=(9, 1.25 * len(ids) + 0.8), sharex=True, squeeze=False)
    dates = toolbox.store.get(ids[0]).dates
    for ax, sid in zip(axes[:, 0], ids):
        s = toolbox.store.get(sid)
        x = np.arange(len(s.values))
        ax.plot(x, s.values, lw=0.8, color=LINE)
        bounds = [0]
        for c in out["changepoints"]:
            lo, hi = _index_of(s.dates, c["interval_90"][0]), _index_of(s.dates, c["interval_90"][1])
            ax.axvspan(lo, hi + 0.99, color=BAND, alpha=0.7, lw=0)
            i = _index_of(s.dates, c["regime_starts"])
            ax.axvline(i, color=CHANGE, ls="--", lw=1.1)
            bounds.append(i)
        bounds.append(len(x))
        for a, b in zip(bounds[:-1], bounds[1:]):
            ax.hlines(s.values[a:b].mean(), a, b - 1, color=CHANGE, lw=1.6)
        alone = out["each_series_alone"].get(sid, {})
        ax.set_ylabel(sid, fontsize=8, rotation=0, ha="right", va="center")
        ax.text(0.995, 0.92, f"alone: P(no change) = {alone.get('probability_of_no_change', '?')}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#555")
        ax.tick_params(labelsize=7)
    axes[0, 0].set_title(f"Joint: {out['most_probable_number']} shared change(s) "
                         f"(P = {out['probability_of_most_probable_number']}). Red steps: each series' mean "
                         "before and after.", fontsize=9.5)
    _date_ticks(axes[-1, 0], dates)
    fig.tight_layout()
    return fig
