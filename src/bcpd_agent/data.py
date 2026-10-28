"""Where the data comes from.

Two sources:

1. FRED (Federal Reserve Economic Data): free public economic series such as
   US unemployment (id "UNRATE"). Downloaded as CSV, no API key needed.
2. Synthetic series: generated here with changepoints we planted ourselves.
   Because we know the true answer, they are what the evaluation suite uses
   to measure whether the agent is right.

Every loaded series is kept in a `SeriesStore` for the duration of one
conversation, so the tools can refer to a series by its id instead of passing
hundreds of numbers back and forth through the LLM (which would be slow,
expensive and error-prone).
"""

from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass, field
from datetime import date

import numpy as np

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


@dataclass
class Series:
    """One time series: dates (as 'YYYY-MM-DD' strings) and values."""

    series_id: str
    dates: list[str]
    values: np.ndarray
    source: str  # "fred" or "synthetic"
    # Only synthetic series have a known truth. Kept out of tool outputs so
    # the agent can never "peek" at the answer.
    true_changepoints: list[str] = field(default_factory=list)


class SeriesStore:
    """In-memory store of the series loaded in one conversation."""

    def __init__(self) -> None:
        self._series: dict[str, Series] = {}
        # Synthetic series registered here can be "loaded" by id like FRED ones.
        self._synthetic: dict[str, Series] = {}

    def register_synthetic(self, s: Series) -> None:
        self._synthetic[s.series_id] = s

    def put(self, s: Series) -> None:
        self._series[s.series_id] = s

    def get(self, series_id: str) -> Series:
        if series_id not in self._series:
            raise KeyError(
                f"Series '{series_id}' is not loaded. Call load_series first."
            )
        return self._series[series_id]

    def load(self, series_id: str, start: str | None = None, end: str | None = None) -> Series:
        """Load a series by id (synthetic first, then FRED) and store it."""
        if series_id in self._synthetic:
            s = self._synthetic[series_id]
        elif series_id.startswith("syn_"):
            # Never send test ids to FRED: fail clearly instead.
            raise KeyError(f"Unknown synthetic series '{series_id}'.")
        else:
            s = fetch_fred(series_id)
        s = _slice(s, start, end)
        self.put(s)
        return s


def fetch_fred(series_id: str, timeout: float = 20.0) -> Series:
    """Download one FRED series as CSV. Missing values ('.') are skipped."""
    url = FRED_URL.format(series_id=series_id)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed host)
        text = resp.read().decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)  # e.g. ["observation_date", "UNRATE"]
    if len(header) < 2:
        raise ValueError(f"Unexpected FRED response for '{series_id}'.")
    dates, values = [], []
    for row in reader:
        if len(row) < 2 or row[1] in ("", "."):
            continue
        dates.append(row[0])
        values.append(float(row[1]))
    if not values:
        raise ValueError(f"FRED returned no data for '{series_id}'.")
    return Series(series_id, dates, np.asarray(values), source="fred")


def _slice(s: Series, start: str | None, end: str | None) -> Series:
    """Keep only observations between start and end (inclusive, 'YYYY-MM-DD' or 'YYYY-MM')."""
    if not start and not end:
        return s
    lo = start or "0000"
    hi = (end or "9999") + "~"  # '~' sorts after digits, so '2010-05' includes all of May
    keep = [i for i, d in enumerate(s.dates) if lo <= d <= hi]
    if not keep:
        raise ValueError(f"No observations of '{s.series_id}' between {start} and {end}.")
    return Series(
        s.series_id,
        [s.dates[i] for i in keep],
        s.values[keep],
        s.source,
        s.true_changepoints,
    )


def monthly_dates(n: int, start_year: int = 2000) -> list[str]:
    """n consecutive month-start dates: 2000-01-01, 2000-02-01, ..."""
    out = []
    y, m = start_year, 1
    for _ in range(n):
        out.append(date(y, m, 1).isoformat())
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def make_synthetic(
    series_id: str,
    segment_lengths: list[int],
    segment_means: list[float],
    noise_sd: float = 1.0,
    trend: float = 0.0,
    seed: int = 0,
    start_year: int = 2000,
) -> Series:
    """A monthly series made of level segments, with changepoints we planted.

    Example: segment_lengths=[80, 60], segment_means=[0, 2] gives 140 months
    with one level shift at month 80. `true_changepoints` holds the date of the
    first observation of each new segment.
    """
    if len(segment_lengths) != len(segment_means):
        raise ValueError("Need one mean per segment.")
    rng = np.random.default_rng(seed)
    y = np.concatenate(
        [rng.normal(mu, noise_sd, n) for n, mu in zip(segment_lengths, segment_means)]
    )
    y = y + trend * np.arange(len(y))
    dates = monthly_dates(len(y), start_year)
    starts = np.cumsum(segment_lengths)[:-1]
    truth = [dates[i] for i in starts]
    return Series(series_id, dates, y, source="synthetic", true_changepoints=truth)
