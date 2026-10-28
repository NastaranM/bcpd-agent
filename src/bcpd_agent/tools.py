"""The tools the agent can call.

An LLM cannot run code by itself. With "tool calling", we describe some Python
functions to the model (name, purpose, parameters as a JSON schema). When the
model decides it needs one, it replies with a structured request like
`{"name": "detect_changepoints", "input": {"series_id": "UNRATE"}}`. Our code
runs the function and sends the result back. The model then continues.

The agent has THREE detection methods and must choose the right one:

    detect_changepoints        OFFLINE (exact). "Did it change, and when?" about
                               the history. Uses all the data for every change.
    detect_recent_change       ONLINE (particle filter). "Is something changing
                               NOW?" Looks at the latest observations, the way
                               live monitoring would.
    detect_joint_changepoints  MULTIVARIATE. "Did these series change TOGETHER?"
                               Finds shared changes that no single series shows.

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
from bcpd import (
    MVPriorSpec,
    OnlineChangepointFilter,
    PriorSpec,
    SegmentLengthPrior,
    SegmentLikelihood,
    constant_design,
    detect,
    detect_multivariate,
    trend_design,
)

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
        "name": "detect_recent_change",
        "description": (
            "ONLINE monitoring view. Use when the question is about NOW: is something "
            "changing in the latest data, has the series recently broken from its past, "
            "is the current regime new. Runs a sequential (online) Bayesian particle "
            "filter through the series and reports when the CURRENT regime most likely "
            "started (with a 90% interval), the probability that it started within the "
            "last `window` observations, and the mean before and since. Not for listing "
            "all historical changes: use detect_changepoints for that."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string"},
                "window": {"type": "integer", "description": "How many latest observations count as 'recent'. Default 12."},
            },
            "required": ["series_id"],
        },
    },
    {
        "name": "detect_joint_changepoints",
        "description": (
            "MULTIVARIATE. Use when the question is about SEVERAL series changing together "
            "(at the same time). Loads the series if needed, aligns them on common dates, "
            "and runs a multivariate Bayesian changepoint model that uses how the series "
            "move together, so it can find shared changes that no single series shows. "
            "Returns the joint result and, for comparison, each series analysed alone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "k_max": {"type": "integer", "description": "Maximum number of changepoints. Default 4."},
            },
            "required": ["series_ids"],
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
            if name == "detect_recent_change":
                return self.detect_recent_change(**args)
            if name == "detect_joint_changepoints":
                return self.detect_joint_changepoints(**args)
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

    def detect_recent_change(self, series_id: str, window: int = 12) -> dict[str, Any]:
        """ONLINE view: when did the CURRENT regime start, and is that recent?

        The particle filter (Fearnhead & Liu 2007, in bcpd.online) walks through
        the series one observation at a time. After the last observation it
        holds a posterior over "where did the current segment start". That is
        exactly the monitoring question: how long has the present regime lasted?
        """
        s = self.store.get(series_id)
        y = s.values
        n = len(y)
        z = (y - y.mean()) / (y.std() or 1.0)
        likelihood = SegmentLikelihood(z, constant_design(n), PriorSpec())
        # Prior on segment length: on average a quarter of the series. A
        # documented assumption, not a fact about the data.
        prior = SegmentLengthPrior(mean_length=max(12.0, n / 4))
        filt = OnlineChangepointFilter(likelihood, prior, max_particles=100)
        res = filt.run(rng=np.random.default_rng(self.seed), store_history=False)

        # Posterior over the start index of the current regime, after the last step.
        last = res.run_length_prob[-1]
        starts = np.array(sorted(last))
        probs = np.array([last[k] for k in starts], dtype=float)
        probs = probs / probs.sum()
        cdf = np.cumsum(probs)
        map_start = int(starts[np.argmax(probs)])
        lo = int(starts[np.searchsorted(cdf, 0.05)])
        hi = int(starts[min(np.searchsorted(cdf, 0.95), len(starts) - 1)])

        out: dict[str, Any] = {
            "series_id": series_id,
            "method": "online particle filter",
            "last_date": s.dates[-1][:7],
            "window_observations": int(window),
            "current_regime_started": s.dates[map_start][:7],
            "current_regime_interval_90": [s.dates[lo][:7], s.dates[hi][:7]],
            "current_regime_length_observations": int(n - map_start),
            "probability_regime_started_within_window": _r(probs[starts >= n - window].sum()),
            "probability_no_change_in_whole_series": _r(probs[starts == 0].sum()),
            "mean_since_regime_start": _r(y[map_start:].mean()),
        }
        if map_start > 0:
            out["mean_before_regime_start"] = _r(y[:map_start].mean())
            out["change_in_mean"] = _r(y[map_start:].mean() - y[:map_start].mean())
        return out

    def detect_joint_changepoints(self, series_ids: list[str], k_max: int = 4) -> dict[str, Any]:
        """MULTIVARIATE view: changes shared by several series at the same time."""
        if len(series_ids) < 2:
            return {"error": "Give at least two series ids."}
        loaded = []
        for sid in series_ids:
            try:
                loaded.append(self.store.get(sid))
            except KeyError:
                loaded.append(self.store.load(sid))
        # Align on the dates every series has.
        common = sorted(set.intersection(*(set(s.dates) for s in loaded)))
        if len(common) < 30:
            return {"error": f"Only {len(common)} common dates; need at least 30."}
        idx = [{d: i for i, d in enumerate(s.dates)} for s in loaded]
        Y = np.column_stack([s.values[[ix[d] for d in common]] for s, ix in zip(loaded, idx)])
        n = len(common)
        Z = (Y - Y.mean(0)) / np.where(Y.std(0) == 0, 1.0, Y.std(0))
        min_distance = max(6, n // 25)

        res = detect_multivariate(
            Z, constant_design(n), min_distance=min_distance, k_max=k_max,
            prior=MVPriorSpec.vague_for(Z), rng=np.random.default_rng(self.seed),
        )
        post = res.posterior
        k_map = post.map_count()
        changes = []
        draws = [sorted(d) for d in post.draws if len(d) == k_map]
        if k_map > 0 and draws:
            arr = np.array(draws)
            med = np.median(arr, axis=0).round().astype(int)
            lo = np.percentile(arr, 5, axis=0).round().astype(int)
            hi = np.percentile(arr, 95, axis=0).round().astype(int)
            for j in range(k_map):
                changes.append({
                    "regime_starts": common[min(int(med[j]) + 1, n - 1)][:7],
                    "interval_90": [common[min(int(lo[j]) + 1, n - 1)][:7], common[min(int(hi[j]) + 1, n - 1)][:7]],
                })

        # For comparison: what each series shows on its own (offline, univariate).
        alone = {}
        for j, sid in enumerate(series_ids):
            r = detect(Z[:, j], constant_design(n), min_distance=min_distance, k_max=k_max,
                       rng=np.random.default_rng(self.seed))
            alone[sid] = {
                "most_probable_number": int(r.posterior.map_count()),
                "probability_of_no_change": _r(r.n_changepoints[0]),
            }

        return {
            "series_ids": list(series_ids),
            "method": "multivariate (joint) offline",
            "n_common_observations": int(n),
            "first_date": common[0][:7],
            "last_date": common[-1][:7],
            "most_probable_number": int(k_map),
            "probability_of_most_probable_number": _r(post.n_changepoints[k_map]),
            "probability_of_no_change": _r(post.n_changepoints[0]),
            "changepoints": changes,
            "each_series_alone": alone,
        }

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
