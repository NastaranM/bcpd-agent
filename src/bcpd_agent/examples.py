"""Built-in synthetic test series with known (planted) changepoints.

They are registered automatically by the command-line tool and by the
evaluation, so you can ask about e.g. "syn_two_shifts" anywhere.

    syn_no_change     150 months, no change
    syn_one_big       one large shift (0 -> 2) at month 90
    syn_one_small     one small shift (0 -> 0.6) at month 100   (hard)
    syn_two_shifts    shifts at months 80 and 140 (0 -> 2 -> 0.5)
    syn_three_shifts  shifts at months 60, 120, 180
    syn_noisy         200 months of noisy data, no change
    syn_late_change   one shift at month 170 of 200             (hard)
"""

from __future__ import annotations

from .data import Series, SeriesStore, make_synthetic


def example_series() -> dict[str, Series]:
    series = [
        make_synthetic("syn_no_change", [150], [0.0], seed=1),
        make_synthetic("syn_one_big", [90, 70], [0.0, 2.0], seed=2),
        make_synthetic("syn_one_small", [100, 100], [0.0, 0.6], seed=3),
        make_synthetic("syn_two_shifts", [80, 60, 80], [0.0, 2.0, 0.5], seed=4),
        make_synthetic("syn_three_shifts", [60, 60, 60, 60], [0.0, 1.5, 3.0, 1.0], seed=5),
        make_synthetic("syn_noisy", [200], [5.0], noise_sd=2.0, seed=6),
        make_synthetic("syn_late_change", [170, 30], [0.0, 2.5], seed=7),
    ]
    return {s.series_id: s for s in series}


def register_examples(store: SeriesStore) -> None:
    for s in example_series().values():
        store.register_synthetic(s)
