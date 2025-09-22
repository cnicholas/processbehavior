from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .spec import AnalysisSpec


def _data_signature(df: pd.DataFrame) -> str:
    # Stable-ish signature based on the first 10 rows and columns names
    head_bytes = pd.util.hash_pandas_object(df.head(10), index=True).values.tobytes()
    cols_bytes = '|'.join(df.columns).encode()
    return 'sha256:' + hashlib.sha256(head_bytes + cols_bytes).hexdigest()


def analyze(df: pd.DataFrame, spec: AnalysisSpec) -> dict[str, Any]:
    """Entry point: pandas in, tidy dict out. Minimal stub today."""
    if spec.response_var not in df.columns:
        raise KeyError(f"response_var '{spec.response_var}' not found in DataFrame")
    if spec.time_var and spec.time_var not in df.columns:
        raise KeyError(f"time_var '{spec.time_var}' not found in DataFrame")

    # Minimal "Xbar" & "Imr" stubs so the shape is correct; replace with real math.
    y = df[spec.response_var]
    xbar_frame = pd.DataFrame(
        {
            'center': [float(y.mean())],
            'count': [int(y.notna().sum())],
        }
    )
    imr_frame = pd.DataFrame(
        {'i': y.reset_index(drop=True), 'mr': y.diff().abs().reset_index(drop=True)}
    )

    charts = {
        'Xbar': {'data': xbar_frame, 'summary': {'note': 'stub'}, 'signals': pd.DataFrame()},
        'Imr': {'data': imr_frame, 'summary': {'note': 'stub'}, 'signals': pd.DataFrame()},
    }
    return {
        'charts': charts,
        'meta': {
            'spec': spec.__dict__,
            'engine_version': '0.1.0',
            'constants_version': 'placeholder',
            'data_signature': _data_signature(df),
        },
    }
