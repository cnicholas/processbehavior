#!/usr/bin/env python3
"""
Fingerprint every Xbar/S request on Tom Bishop's reference data.

One line per request: the request, then a SHA-256 over the chart frames (every value at 17
significant digits), the statistics and the metadata, or the exception a refused request
raises. Run it on two revisions and diff the outputs: a refactor that changes nothing prints
identical files. Written for issue #144, whose first PR extracts the per-group Xbar/S
estimator and must leave every existing chart bit-identical.

Grid: response PM SDS 1 (ADS 1) and PM SDS 3 (ADS 3) x chart {Xbar, S} x companion {off, on}
x by {None, [], one factor, both factors, time, one factor + time, both factors + time}
x value {response, R5, R2} x n_mode {actual, average}.

Usage, from the repo root:
    python validation/xbar_s_sweep.py > sweep.txt
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import processbehavior as pb
from processbehavior.exceptions import ProcessBehaviorError

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'PBTESTDATABASE_T100.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']
TIME = 'PRODUCTION TIME'
RESPONSES = ('PM SDS 1', 'PM SDS 3')
BY_SHAPES = (None, [], ['FACTOR 1'], FACTORS, [TIME], ['FACTOR 1', TIME], FACTORS + [TIME])
VALUES = (None, 'R5', 'R2')
N_MODES = ('actual', 'average')


def _normalise(obj):
    """JSON-ready copy: numpy scalars to Python, NaN to the string 'nan', dict keys to str."""
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return 'nan'
    return obj


def fingerprint(result) -> str:
    digest = hashlib.sha256()
    for name in sorted(result.charts):
        payload = result.charts[name]
        digest.update(name.encode())
        digest.update(payload['data'].to_csv(index=True, float_format='%.17g').encode())
        digest.update(json.dumps(_normalise(payload['statistics']), sort_keys=True).encode())
        digest.update(json.dumps(_normalise(payload['metadata']), sort_keys=True).encode())
        digest.update(json.dumps(_normalise(payload.get('strata')), sort_keys=True).encode())
    return digest.hexdigest()


def main() -> int:
    df = pd.read_csv(REFERENCE, na_values=['*'])
    for response in RESPONSES:
        study = pb.formulate(df, response=response, factors=FACTORS, time=TIME)
        for chart in ('Xbar', 'S'):
            for companion in (False, True):
                for by in BY_SHAPES:
                    for value in VALUES:
                        for n_mode in N_MODES:
                            request = (
                                f'{response} | chart={chart} companion={companion} by={by} '
                                f'value={value} n_mode={n_mode}'
                            )
                            try:
                                result = study.execute(
                                    chart=chart, by=by, value=value, companion=companion, n_mode=n_mode
                                )
                            except ProcessBehaviorError as exc:
                                print(f'{request} -> refused {type(exc).__name__}: {exc}')
                                continue
                            print(f'{request} -> {fingerprint(result)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
