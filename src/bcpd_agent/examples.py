"""Built-in synthetic test series with known (planted) changepoints.

They are registered automatically by the command-line tool and by the
evaluation, so you can ask about e.g. "syn_two_shifts" anywhere.

    syn_no_change       150 months, no change
    syn_one_big         one large shift (0 -> 2) at month 90
    syn_one_small       one small shift (0 -> 0.6) at month 100   (hard)
    syn_two_shifts      shifts at months 80 and 140 (0 -> 2 -> 0.5)
    syn_three_shifts    shifts at months 60, 120, 180
    syn_noisy           200 months of noisy data, no change
    syn_late_change     one shift at month 170 of 200             (hard)
    syn_recent_change   one shift at month 190 of 200: for "is something changing NOW?"
    syn_mv_1 .. syn_mv_4  four correlated series (correlation 0.9) that shift
                        TOGETHER at month 130, in alternating directions.
                        Each one alone shows almost nothing; jointly the
                        change is obvious. For the multivariate tool.
"""

from __future__ import annotations

import numpy as np

from .data import Series, SeriesStore, make_synthetic, monthly_dates


def _correlated_group(prefix: str, n: int = 240, cp: int = 130, d: int = 4,
                      rho: float = 0.9, shift: float = 0.5, seed: int = 11) -> list[Series]:
    """d correlated series with one shared change at index `cp`."""
    rng = np.random.default_rng(seed)
    cov = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    Y = rng.multivariate_normal(np.zeros(d), cov, size=n)
    signs = np.array([1 if j % 2 == 0 else -1 for j in range(d)])
    Y[cp:] += signs * shift
    dates = monthly_dates(n)
    return [
        Series(f"{prefix}_{j + 1}", dates, Y[:, j], source="synthetic", true_changepoints=[dates[cp]])
        for j in range(d)
    ]


def example_series() -> dict[str, Series]:
    series = [
        make_synthetic("syn_no_change", [150], [0.0], seed=1),
        make_synthetic("syn_one_big", [90, 70], [0.0, 2.0], seed=2),
        make_synthetic("syn_one_small", [100, 100], [0.0, 0.6], seed=3),
        make_synthetic("syn_two_shifts", [80, 60, 80], [0.0, 2.0, 0.5], seed=4),
        make_synthetic("syn_three_shifts", [60, 60, 60, 60], [0.0, 1.5, 3.0, 1.0], seed=5),
        make_synthetic("syn_noisy", [200], [5.0], noise_sd=2.0, seed=6),
        make_synthetic("syn_late_change", [170, 30], [0.0, 2.5], seed=7),
        make_synthetic("syn_recent_change", [190, 10], [0.0, 2.5], seed=8),
        *_correlated_group("syn_mv"),
    ]
    return {s.series_id: s for s in series}


def register_examples(store: SeriesStore) -> None:
    for s in example_series().values():
        store.register_synthetic(s)
