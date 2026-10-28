"""The tools the agent can call.

An LLM cannot run code by itself. With "tool calling", we describe some Python
functions to the model (name, purpose, parameters as a JSON schema). When the
model decides it needs one, it replies with a structured request like
`{"name": "detect_changepoints", "input": {"series_id": "UNRATE"}}`. Our code
runs the function and sends the result back. The model then continues.

Design rules for these tools:
- Each tool does ONE thing and returns a small JSON-friendly dict.
- Numbers are rounded, so the model can quote them exactly and the grounding
  guardrail can check them.
- Errors are returned as {"error": "..."} instead of crashing, so the model
  can tell the user what went wrong instead of inventing an answer.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from bcpd import constant_design, detect, trend_design

from .data import SeriesStore

# ---------------------------------------------------------------------------
# Tool definitions sent to the model (JSON schema). The descriptions matter:
# they are the only thing the model knows about each tool.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "load_series",
        "description": (
            "Load a time series by id and return summary statistics. Ids are FRED "
            "series ids (for example UNRATE = US unemployment rate, CPIAUCSL = US "
            "consumer prices, DRCCLACBS = credit card delinquency rate) or synthetic "
            "test ids starting with 'syn_'. Must be called before any other tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string", "description": "FRED id or synthetic id."},
                "start": {"type": "string", "description": "Optional start, 'YYYY-MM' or 'YYYY-MM-DD'."},
                "end": {"type": "string", "description": "Optional end, 'YYYY-MM' or 'YYYY-MM-DD'."},
            },
            "required": ["series_id"],
        },
    },
    {
        "name": "detect_changepoints",
        "description": (
            "Run exact Bayesian changepoint detection on a loaded series. Returns the "
            "posterior probability of each number of changepoints, and for the most "
            "probable number, the date each new regime starts with a 90% credible "
            "interval and the mean before and after. design='level' detects shifts in "
            "level (default); design='trend' detects changes in slope."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string"},
                "design": {"type": "string", "enum": ["level", "trend"]},
                "min_distance": {
                    "type": "integer",
                    "description": "Minimum observations between changes. Default: series length / 25, at least 6.",
                },
                "k_max": {"type": "integer", "description": "Maximum number of changepoints. Default 6."},
            },
            "required": ["series_id"],
        },
    },
    {
        "name": "make_plot",
        "description": "Save a PNG chart of a loaded series with the changepoint probability underneath. Returns the file path.",
        "input_schema": {
            "type": "object",
            "properties": {"series_id": {"type": "string"}},
            "required": ["series_id"],
        },
    },
]


def _r(x: float, nd: int = 3) -> float:
    """Round for display. Plain float so it serialises to JSON."""
    return float(round(float(x), nd))


class Toolbox:
    """Holds the per-conversation state (loaded series, last detection results)
    and runs tool calls by name."""

    def __init__(self, store: SeriesStore | None = None, plot_dir: str = "reports", seed: int = 0):
        self.store = store or SeriesStore()
        self.plot_dir = plot_dir
        self.seed = seed  # fixed seed so results are reproducible run to run
        self._last_detection: dict[str, Any] = {}

    # Dispatcher: the agent loop calls this with the model's requested tool.
    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "load_series":
                return self.load_series(**args)
            if name == "detect_changepoints":
                return self.detect_changepoints(**args)
            if name == "make_plot":
                return self.make_plot(**args)
            return {"error": f"Unknown tool '{name}'."}
        except Exception as exc:  # report, don't crash: the model must see the failure
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------ tools
    def load_series(self, series_id: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        s = self.store.load(series_id, start, end)
        v = s.values
        return {
            "series_id": s.series_id,
            "source": s.source,
            "n_observations": int(len(v)),
            "first_date": s.dates[0],
            "last_date": s.dates[-1],
            "mean": _r(v.mean()),
            "std": _r(v.std()),
            "min": _r(v.min()),
            "max": _r(v.max()),
            "last_value": _r(v[-1]),
        }

    def detect_changepoints(
        self,
        series_id: str,
        design: str = "level",
        min_distance: int | None = None,
        k_max: int = 6,
    ) -> dict[str, Any]:
        s = self.store.get(series_id)
        y = s.values
        n = len(y)
        if min_distance is None:
            min_distance = max(6, n // 25)
        # Standardise so the default priors in bcpd behave the same for any units
        # (unemployment in %, prices in index points, ...).
        z = (y - y.mean()) / (y.std() or 1.0)
        X = trend_design(n) if design == "trend" else constant_design(n)
        res = detect(z, X, min_distance=min_distance, k_max=k_max, rng=np.random.default_rng(self.seed))
        post = res.posterior

        count_probs = {str(k): _r(p) for k, p in enumerate(post.n_changepoints) if p >= 0.001}
        k_map = post.map_count()

        # Summarise locations using only the draws that have the most probable
        # number of changepoints, so "the 1st change", "the 2nd change", ... are
        # well defined. bcpd marks the LAST observation of the old regime, so the
        # new regime starts at index + 1.
        changes = []
        draws = [sorted(d) for d in post.draws if len(d) == k_map]
        if k_map > 0 and draws:
            arr = np.array(draws)  # shape (n_draws, k_map)
            med = np.median(arr, axis=0).round().astype(int)
            lo = np.percentile(arr, 5, axis=0).round().astype(int)
            hi = np.percentile(arr, 95, axis=0).round().astype(int)
            bounds = [0] + [int(i) + 1 for i in med] + [n]
            for j in range(k_map):
                start_new = min(int(med[j]) + 1, n - 1)
                before = y[bounds[j] : bounds[j + 1]]
                after = y[bounds[j + 1] : bounds[j + 2]]
                changes.append(
                    {
                        "regime_starts": s.dates[start_new][:7],
                        "interval_90": [s.dates[min(int(lo[j]) + 1, n - 1)][:7], s.dates[min(int(hi[j]) + 1, n - 1)][:7]],
                        "mean_before": _r(before.mean()),
                        "mean_after": _r(after.mean()),
                        # Given explicitly so the model never has to do arithmetic
                        # (arithmetic is where LLMs most often invent numbers).
                        "change_in_mean": _r(after.mean() - before.mean()),
                    }
                )

        out = {
            "series_id": series_id,
            "design": design,
            "min_distance": int(min_distance),
            "probability_by_number_of_changepoints": count_probs,
            "most_probable_number": int(k_map),
            "probability_of_most_probable_number": _r(post.n_changepoints[k_map]),
            "probability_of_no_change": _r(post.n_changepoints[0]),
            "changepoints": changes,
        }
        self._last_detection[series_id] = {"result": res, "summary": out}
        return out

    def make_plot(self, series_id: str) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")  # no screen needed
        import matplotlib.pyplot as plt

        s = self.store.get(series_id)
        det = self._last_detection.get(series_id)
        os.makedirs(self.plot_dir, exist_ok=True)
        path = os.path.join(self.plot_dir, f"{series_id}.png")

        fig, axes = plt.subplots(2 if det else 1, 1, figsize=(10, 5), sharex=True, squeeze=False)
        x = np.arange(len(s.values))
        axes[0, 0].plot(x, s.values, lw=1, color="#1f4e5a")
        axes[0, 0].set_title(f"{series_id}")
        if det:
            p = det["result"].location_prob
            axes[1, 0].fill_between(x, p, color="#c0504d", alpha=0.6, step="mid")
            axes[1, 0].set_ylabel("P(change)")
            for c in det["summary"]["changepoints"]:
                i = next(k for k, d in enumerate(s.dates) if d.startswith(c["regime_starts"]))
                axes[0, 0].axvline(i, color="#c0504d", ls="--", lw=1)
        ticks = x[:: max(1, len(x) // 8)]
        axes[-1, 0].set_xticks(ticks, [s.dates[i][:7] for i in ticks], rotation=30)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return {"path": path}
