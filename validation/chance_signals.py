"""How often charts signal on pure noise. A report, not a gate.

    python validation/chance_signals.py

Two tables, both from data with no signal in it: readings are independent N(0, 1), with no
condition, time or interaction effects. Each design gets its own random stream,
default_rng([SEED, ...]), so every row reproduces on its own.

1. Signal rules. One stream of T readings, charted as X/mR. The share of charts with at least
   one signal under the default rules (X: Rules 1-8; mR: Rule 1), and on mR with all eight.
   Successive moving ranges share a reading and their distribution is skewed, which is why the
   zone and run rules are off by default on mR.

2. Effect charts. The R5 (condition), R4 (period) and R3 (interaction) Xbar charts on a K x T
   grid with n readings per cell. The plotted effects sum to zero and their limits use sigma from
   R2, so their limits do not sit at 3 standard deviations of the plotted points. The effective
   multiple is the limit half-width divided by the observed standard deviation of the plotted
   points across replicates (3 for a textbook chart), with the share of charts showing a point
   beyond the limits.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import processbehavior as pb

SEED = 20261102
SIGNAL_REPS = 500
SIGNAL_LENGTHS = (25, 50, 100)
EFFECT_REPS = 300
EFFECT_DESIGNS = ((2, 20, 2), (2, 20, 5), (5, 20, 5), (8, 20, 2), (8, 20, 5), (8, 20, 1))
EFFECT_CHARTS = {'R5 by condition': ('R5', ['cond']), 'R4 by period': ('R4', ['period']),
                 'R3 by cell': ('R3', ['cond', 'period'])}


def signal_rates(T: int) -> dict[str, float]:
    rng = np.random.default_rng([SEED, T])
    hits = {'X': 0, 'mR': 0, 'mR, all eight': 0}
    for _ in range(SIGNAL_REPS):
        df = pd.DataFrame({'t': range(T), 'y': rng.standard_normal(T)})
        result = pb.formulate(df, response='y', time='t').execute(chart='X', companion=True)
        hits['X'] += result.detect_signals(chart='X').has_signals
        hits['mR'] += result.detect_signals(chart='mR').has_signals
        hits['mR, all eight'] += result.detect_signals(chart='mR', rules='all').has_signals
    return {k: v / SIGNAL_REPS for k, v in hits.items()}


def effect_chart(K: int, T: int, n: int) -> dict[str, tuple[float, float] | None]:
    rng = np.random.default_rng([SEED, K, T, n])
    cells = pd.DataFrame([(f'C{k}', t) for k in range(K) for t in range(1, T + 1) for _ in range(n)],
                         columns=['cond', 'period'])
    points: dict[str, list] = {c: [] for c in EFFECT_CHARTS}
    halves: dict[str, list] = {c: [] for c in EFFECT_CHARTS}
    beyond = dict.fromkeys(EFFECT_CHARTS, 0)
    for _ in range(EFFECT_REPS):
        study = pb.formulate(cells.assign(y=rng.standard_normal(len(cells))), response='y',
                             factors=['cond'], time='period')
        for name, (value, by) in EFFECT_CHARTS.items():
            if value == 'R3' and n == 1:
                continue  # one reading per cell: the interaction chart is not available
            table = study.execute(chart='Xbar', value=value, by=by).chart_table('Xbar')
            v, lo, hi = (table[c].to_numpy(float) for c in ('value', 'lpl', 'upl'))
            points[name].append(v - table['center'].to_numpy(float))
            halves[name].append((hi - lo) / 2)
            beyond[name] += bool(((v > hi) | (v < lo)).any())
    out: dict[str, tuple[float, float] | None] = {}
    for name in EFFECT_CHARTS:
        if not points[name]:
            out[name] = None
            continue
        multiple = np.mean(np.concatenate(halves[name])) / np.std(np.concatenate(points[name]), ddof=1)
        out[name] = (float(multiple), beyond[name] / EFFECT_REPS)
    return out


def render() -> str:
    lines = ['## Signals on pure noise', '',
             f'Share of {SIGNAL_REPS} pure-noise X/mR charts with at least one signal.', '',
             '| Readings | X (Rules 1-8) | mR (Rule 1, the default) | mR (all eight rules) |', '|---|---|---|---|']
    for T in SIGNAL_LENGTHS:
        r = signal_rates(T)
        lines.append(f"| {T} | {r['X']:.0%} | {r['mR']:.0%} | {r['mR, all eight']:.0%} |")
    lines += ['', '## Effect charts on pure noise', '',
              f'Effective sigma multiple of the limits (3 for a textbook chart), and the share of {EFFECT_REPS}',
              'charts with a point beyond them.', '',
              '| K | T | n | ' + ' | '.join(EFFECT_CHARTS) + ' |', '|---|---|---|' + '---|' * len(EFFECT_CHARTS)]
    for K, T, n in EFFECT_DESIGNS:
        r = effect_chart(K, T, n)
        cells = ['not available' if r[c] is None else f'{r[c][0]:.1f} ({r[c][1]:.0%})' for c in EFFECT_CHARTS]
        lines.append(f'| {K} | {T} | {n} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print(render())
