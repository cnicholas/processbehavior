"""Deterministic generator for the coffee-shop demo dataset.

Produces one fixed, seeded dataset used by ``pb.load_coffee_shop()`` and the
tutorials. Re-run to regenerate the bundled CSV and its test-fixture copy
byte-for-byte.

The story (a barista station's drink wait time, in seconds), Mon-Sat over 16
weeks, 4 timed readings per day at hours 8/10/13/16:

* Factor ``daypart`` (Peak = 08:00 & 13:00 rush; Off-peak = 10:00 & 16:00).
  Peak is genuinely slower than Off-peak -> a real main effect.
* Weeks 1-7  : stable baseline (daily mean ~240s).
* Week 8+    : a new espresso machine drops wait time (~203s), permanently
  (Peak improves a touch more than Off-peak -> a mild interaction).
* Week 12    : a new POS is installed; a one-week learning curve elevates and
  destabilises wait time, decaying back to normal by week 13 (a run, not a
  lone outlier).
* Weeks 14-16: several new baristas are hired; the mean holds but the
  within-subgroup spread roughly doubles -> the dispersion (S/Range) chart
  breaks while the location (Xbar) chart looks fine.

Design state: factor x time grid -> SDS 1 (full replication).

Run:  python scripts/generate_coffee_shop_demo.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260105  # fixed -> reproducible "one stable dataset"
START = date(2026, 1, 5)  # Monday of week 1
N_WEEKS = 16
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']  # 6 trading days (no Sunday)

# Reading schedule: 4 timed samples/day. Hour -> daypart.
HOURS = [8, 10, 13, 16]
DAYPART = {8: 'Peak', 13: 'Peak', 10: 'Off-peak', 16: 'Off-peak'}

# Location (mean wait_sec) by era and daypart. Daily mean = mean(Peak, Off):
#   baseline (258+222)/2 = 240 ; post (220+185)/2 = 202.5
BASELINE = {'Peak': 258.0, 'Off-peak': 222.0}
POST = {'Peak': 220.0, 'Off-peak': 185.0}  # Peak drops 38, Off 37 -> mild interaction

SD_STABLE = 14.0  # within-reading noise, stable eras
SD_POS = 20.0  # week 12 is noisier during POS learning
SD_NEWHIRE = 28.0  # weeks 14-16: spread ~doubles, mean unchanged

# Week-12 POS learning curve: additive bump by weekday index (Mon..Sat), decaying.
POS_BUMP = [55, 45, 35, 25, 15, 5]


def _era_mean(week: int, daypart: str) -> float:
    return BASELINE[daypart] if week <= 7 else POST[daypart]


def _sd(week: int) -> float:
    if week == 12:
        return SD_POS
    if week >= 14:
        return SD_NEWHIRE
    return SD_STABLE


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for w in range(1, N_WEEKS + 1):
        for d_idx, dow in enumerate(DOW):
            day = START + timedelta(days=(w - 1) * 7 + d_idx)
            for hour in HOURS:
                daypart = DAYPART[hour]
                mean = _era_mean(w, daypart)
                if w == 12:
                    mean += POS_BUMP[d_idx]
                wait = int(round(rng.normal(mean, _sd(w))))
                rows.append(
                    {
                        'date': day.isoformat(),
                        'year_week': f'2026-W{w:02d}',
                        'week': w,
                        'day_of_week': dow,
                        'hour': hour,
                        'daypart': daypart,
                        'wait_sec': max(wait, 1),
                    }
                )
    df = pd.DataFrame(rows)
    # Canonical order: chronological by date then by reading hour.
    return df.sort_values(['date', 'hour']).reset_index(drop=True)


def main() -> None:
    df = build()
    repo = Path(__file__).resolve().parents[1]
    targets = [
        repo / 'processbehavior' / 'datasets' / 'data' / 'coffee_shop_demo_long.csv',
        repo / 'tests' / 'fixtures' / 'data' / 'coffee_shop_demo_long.csv',
    ]
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(t, index=False)
        print(f'wrote {t}  ({len(df)} rows, {df.shape[1]} cols)')


if __name__ == '__main__':
    main()
