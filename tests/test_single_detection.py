"""Design-state detection runs exactly once, during ``formulate()``.

CLAUDE.md states it as an invariant — "Detected once during formulate() and passed through
the system; no class re-detects state" — and it holds today, but nothing enforced it. That is
the shape of trap this suite exists to remove: an invariant that is true by accident of the
current call graph, with no test to notice when it stops being.

Two things break if a second detection creeps in:

- **Correctness.** ODS must be detected on *raw* data, before NA response rows are dropped,
  because cells whose responses are all NA are what reveal an incomplete grid (ODS 4/5/6). A
  re-detection downstream would run against tidied data and silently collapse those to 1/2/3.
- **Performance.** Detection was 73% of ``formulate()`` runtime before it was vectorized. A
  duplicate call is a large, silent regression.
"""

from pathlib import Path

import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.sds_detector import SDSRegistry

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']
TIME = 'PRODUCTION TIME'


@pytest.fixture
def counted(monkeypatch):
    """Count calls to both detection entry points for the duration of a test."""
    calls = {'detect_sds_from_structure': 0, 'detect_sds': 0}

    for name in calls:
        original = getattr(SDSRegistry, name)

        def make(name=name, original=original):
            def counting(self, *args, **kwargs):
                calls[name] += 1
                return original(self, *args, **kwargs)

            return counting

        monkeypatch.setattr(SDSRegistry, name, make())

    return calls


@pytest.fixture
def df():
    return pd.read_csv(REFERENCE_CSV, na_values=['*'])


def test_formulate_detects_exactly_once(df, counted):
    pb.formulate(df, response='PM SDS 1', factors=FACTORS, time=TIME)
    total = sum(counted.values())
    assert total == 1, f'expected one detection during formulate(), got {counted}'


def test_execute_never_re_detects(df, counted):
    """`execute()` is the cheap step; it consumes the design state, never re-derives it."""
    study = pb.formulate(df, response='PM SDS 1', factors=FACTORS, time=TIME)
    after_formulate = sum(counted.values())

    study.execute(chart='Xbar')
    study.execute(chart='S')
    study.execute(chart='X', by=[])
    study.execute(chart='Xbar', value='R5', by=['FACTOR 1'])
    study.execute(chart='Xbar', by=[TIME])

    assert sum(counted.values()) == after_formulate, (
        f'execute() re-detected the design state: {counted}'
    )


def test_result_accessors_never_re_detect(df, counted):
    """Reaching into a result — its design states, diagnostics, or a focused stratum —
    must not trigger detection either."""
    study = pb.formulate(df, response='PM SDS 1', factors=FACTORS, time=TIME)
    result = study.execute(chart='X', by=['FACTOR 1'])
    baseline = sum(counted.values())

    _ = study.observed_design_state
    _ = study.analytical_design_state
    _ = study.valid_charts
    _ = study.residual_charts
    _ = study.why_not('S')
    _ = result.summary
    _ = result.focus(result.strata[0]).get_chart('X')

    assert sum(counted.values()) == baseline, f'an accessor re-detected: {counted}'


@pytest.mark.parametrize('response', ['PM SDS 1', 'PM SDS 2', 'PM SDS 3'])
def test_holds_across_design_states(df, counted, response):
    """The count must not depend on which design state the data turns out to be.

    ADS 2/3 route to X/mR, which needs an explicit ``by``; ``by=[]`` is the one call
    that is valid in all three states.
    """
    study = pb.formulate(df, response=response, factors=FACTORS, time=TIME)
    study.execute(chart='X', by=[])
    assert sum(counted.values()) == 1, f'{response}: {counted}'


def test_each_formulate_detects_once(df, counted):
    """Two studies means two detections — the invariant is once *per formulate*, not once
    per process. Guards against a cache that would make a re-formulated study stale."""
    pb.formulate(df, response='PM SDS 1', factors=FACTORS, time=TIME)
    pb.formulate(df, response='PM SDS 2', factors=FACTORS, time=TIME)
    assert sum(counted.values()) == 2, counted


def test_the_counter_would_notice_a_second_call(df, counted):
    """Guards the guard: if the monkeypatch stopped intercepting, every test above would
    pass vacuously at zero. Calling detection directly must move the count."""
    study = pb.formulate(df, response='PM SDS 1', factors=FACTORS, time=TIME)
    before = sum(counted.values())

    SDSRegistry().detect_sds_from_structure(
        study.dataset, study._spec, response_col='PM SDS 1',
    )

    assert sum(counted.values()) == before + 1, 'the counter is not intercepting'
