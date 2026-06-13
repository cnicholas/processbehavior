"""Loaders for bundled demo datasets shipped with processbehavior."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..process_behavior import ProcessBehavior

_DATA_DIR = Path(__file__).parent / 'data'


def load_coffee_shop() -> ProcessBehavior:
    """Coffee-shop wait times with a deliberate process change — ready to formulate.

    A barista station's wait time (seconds) measured four times a day across 16
    weeks (96 days x 4 readings = 384 rows). A new espresso machine is installed
    at the start of week 8 (2026-02-23): the baseline era averages ~240s and the
    post-change era ~200s, so it is ideal for a before/after process-capability
    comparison along the study's time axis.

    Columns
    -------
    date : str (ISO date)   day of the reading (use as a date time axis)
    day_of_week : str       Mon..Sat (a rational-subgroup factor)
    week : int              1..16 (an integer time axis)
    subgroup_id : str       one subgroup per day
    reading_no : int        1..4 within the day
    wait_sec : int          wait time in seconds (the response)

    Returns
    -------
    ProcessBehavior
        Wrapping the demo data; call ``.formulate(...)`` to build a Study.

    Examples
    --------
    >>> from processbehavior import load_coffee_shop
    >>> pb = load_coffee_shop()
    >>> study = pb.formulate(response='wait_sec', factors=['day_of_week'], time='week')
    >>> study.capability(usl=240, target=180, window=(8, None)).ppk  # after the change
    """
    # Lazy import keeps the datasets package import light and avoids any
    # circular-import risk at package load.
    from ..process_behavior import ProcessBehavior

    return ProcessBehavior.read_csv(str(_DATA_DIR / 'coffee_shop_demo_long.csv'))
