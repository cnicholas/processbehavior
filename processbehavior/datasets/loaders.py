"""Loaders for bundled demo datasets shipped with processbehavior."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..process_behavior import ProcessBehavior

_DATA_DIR = Path(__file__).parent / 'data'


def load_coffee_shop() -> ProcessBehavior:
    """Coffee-shop wait times — a full SPC story, ready to formulate.

    A barista station's drink wait time (seconds), sampled four times a day at
    08:00, 10:00, 13:00 and 16:00 across 16 weeks, Mon-Sat (96 days x 4 readings
    = 384 rows). The two rush hours (08:00, 13:00) are the ``Peak`` daypart and
    the two quieter hours (10:00, 16:00) are ``Off-peak`` — Peak is genuinely
    slower, a real main effect. Formulate with ``daypart`` as the factor and
    ``date`` as the time axis for a full-replication (SDS 1) study.

    The series has three deliberate events:

    * **Week 8 (2026-02-23)** — a new espresso machine permanently lowers wait
      time (~240s -> ~205s), improving Peak a touch more than Off-peak (a mild
      interaction). A favorable shift on the Xbar chart.
    * **Week 12** — a new POS is installed; a one-week learning curve elevates
      and destabilises wait time, decaying back to normal by week 13 (a run
      above the center line, not a lone outlier).
    * **Weeks 14-16** — several new baristas are hired; the mean holds but the
      within-subgroup spread roughly doubles — the dispersion (S) chart breaks
      while the location (Xbar) chart looks fine. Averages hide variation.

    Columns
    -------
    date : str (ISO date)   day of the reading (use as the date time axis)
    year_week : str         ``2026-Www`` label for the week
    week : int              1..16 (an integer time axis; handy for capability windows)
    day_of_week : str       Mon..Sat
    hour : int              hour of the reading (8, 10, 13, 16)
    daypart : str           Peak / Off-peak (the process-design factor)
    wait_sec : int          wait time in seconds (the response)

    Returns
    -------
    ProcessBehavior
        Wrapping the demo data; call ``.formulate(...)`` to build a Study.

    Examples
    --------
    >>> from processbehavior import load_coffee_shop
    >>> pb = load_coffee_shop()
    >>> study = pb.formulate(response='wait_sec', factors=['daypart'], time='date')
    >>> study.observed_design_state.sds  # 1 — full replication
    1
    >>> # before/after the week-8 change along the integer week axis
    >>> study_w = pb.formulate(response='wait_sec', factors=['daypart'], time='week')
    >>> study_w.capability(usl=240, target=180, window=(8, None)).ppk  # after the change
    """
    # Lazy import keeps the datasets package import light and avoids any
    # circular-import risk at package load.
    from ..process_behavior import ProcessBehavior

    return ProcessBehavior.read_csv(str(_DATA_DIR / 'coffee_shop_demo_long.csv'))
