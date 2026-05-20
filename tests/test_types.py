"""Structural tests for the chart-payload TypedDicts.

These tests don't exercise runtime behavior so much as pin the
``processbehavior.types`` contract: the required keys on
:class:`ChartPayload` and :class:`ChartStatistics`, and the
end-to-end shape of ``AnalysisResult.charts[name]`` against the
declared type alias.

The TypedDicts themselves give edit-time mypy enforcement; these
tests make sure the runtime payloads keep matching the declared shape
when a producer site is touched.
"""

from __future__ import annotations

import pandas as pd

import processbehavior as pb
from processbehavior.types import (
    ChartMetadata,
    ChartPayload,
    ChartStatistics,
)


def test_chart_payload_constructor_accepts_required_keys() -> None:
    """ChartPayload(...) builds a dict with the four expected keys."""
    payload = ChartPayload(
        data=pd.DataFrame({'y': [1.0, 2.0, 3.0]}),
        statistics={'N': 3, 'center': 2.0, 'lpl': 1.0, 'upl': 3.0},
        metadata=ChartMetadata(chart_type='Xbar', value_col='y'),
    )
    assert set(payload.keys()) == {'data', 'statistics', 'metadata'}
    assert payload['metadata']['chart_type'] == 'Xbar'


def test_chart_payload_supports_optional_strata() -> None:
    """`strata` is NotRequired but accepted when supplied."""
    payload = ChartPayload(
        data=pd.DataFrame({'y': [1.0]}),
        statistics={'N': 1, 'center': 1.0, 'lpl': None, 'upl': None},
        metadata=ChartMetadata(chart_type='Xbar'),
        strata=['stratum_1'],
    )
    assert payload['strata'] == ['stratum_1']


def test_chart_statistics_required_keys() -> None:
    """ChartStatistics(...) requires all four keys."""
    stats = ChartStatistics(N=10, center=50.0, lpl=45.0, upl=55.0)
    assert stats['N'] == 10
    assert stats['center'] == 50.0
    assert stats['lpl'] == 45.0
    assert stats['upl'] == 55.0


def test_chart_metadata_total_false() -> None:
    """ChartMetadata accepts an empty dict (every key is optional)."""
    meta = ChartMetadata()
    assert meta == {}
    # Adding a real key works.
    meta = ChartMetadata(chart_type='Xbar', stratified=True)
    assert meta['stratified'] is True


def test_analysis_result_charts_match_payload_shape() -> None:
    """End-to-end: an executed result's charts dict matches ChartPayload."""
    df = pb.make_design(state=1, seed=42)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        factors=['factor 1', 'factor 2'],
        time='time',
    )
    result = study.execute(chart='Xbar', companion=True)

    for chart_name, payload in result.charts.items():
        # Required keys present
        assert 'data' in payload, f"{chart_name} missing 'data'"
        assert 'statistics' in payload, f"{chart_name} missing 'statistics'"
        assert 'metadata' in payload, f"{chart_name} missing 'metadata'"
        # Shape of inner values
        assert isinstance(payload['data'], pd.DataFrame)
        assert isinstance(payload['statistics'], dict)
        assert isinstance(payload['metadata'], dict)
        assert payload['metadata'].get('chart_type') in {chart_name, 'X', 'mR'}


def test_histogram_statistics_carry_unified_shape() -> None:
    """Histogram now exposes N/center/lpl/upl plus mean/std/n extras."""
    df = pb.make_design(state=1, seed=42)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        factors=['factor 1', 'factor 2'],
        time='time',
    )
    result = study.execute(chart='Histogram')
    stats = result.get_statistics('Histogram')

    # Unified ChartStatistics keys
    assert 'N' in stats
    assert 'center' in stats
    assert 'lpl' in stats
    assert 'upl' in stats

    # Histogram extras
    assert 'mean' in stats
    assert 'std' in stats
    assert 'n' in stats

    # Aliasing: mean == center, n == N
    assert stats['mean'] == stats['center']
    assert stats['n'] == stats['N']

    # Histogram has no control limits
    assert stats['lpl'] is None
    assert stats['upl'] is None
