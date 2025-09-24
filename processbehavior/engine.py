from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from .analysis_dataset import (
    calculate_statistics_R,
    calculate_statistics_XbarS,
)
from .charts.imr import calculate_statistics_Imr
from .data_prep import prepare_dataset
from .spec import AnalysisSpec, AnalysisSpecification


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


def perform_analysis(df: pd.DataFrame, specification: dict) -> pd.DataFrame:
    """Perform SPC analysis based on specification.

    Args:
        df: Input dataframe
        specification: Analysis specification dictionary

    Returns:
        DataFrame with analysis results

    Raises:
        ValueError: If analysis type is not supported
    """
    analysis_type = specification['analysis_type']

    # Create specification and prepare dataset
    spec = AnalysisSpecification.from_dict(
        analysis_type=analysis_type, analysis_specification=specification
    )
    prepared_df = prepare_dataset(df=df, analysis_specification=spec)

    # Direct mapping to calculation functions
    if analysis_type == 'Xbar' or analysis_type == 'S':
        return calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)
    elif analysis_type == 'Imr':
        return calculate_statistics_Imr(df=prepared_df, analysis_specification=spec)
    elif analysis_type == 'R':
        return calculate_statistics_R(df=prepared_df, analysis_specification=spec)
    else:
        raise ValueError(
            f'Analysis type {analysis_type} not supported. '
            f'Available types: ["Xbar", "S", "Imr", "R"]'
        )
