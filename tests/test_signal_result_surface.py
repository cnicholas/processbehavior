"""SignalResult's drill-down and export surface (#113).

The class computed signals correctly but its user-facing surface — summary,
per-rule/per-observation drill-down, to_excel/to_json — was almost entirely
untested (worst branch coverage in the repo at 20%). These tests use the
literal-DataFrame construction pattern from test_signals.py.
"""

import json

import pandas as pd
import pytest

from processbehavior.signals import SignalResult


def _violations(n=7):
    """A violations frame with enough rows to trip the '... and N more' tail."""
    return pd.DataFrame(
        {
            'obs_id': list(range(1, n + 1)),
            'rule_name': (['rule_1'] * 3 + ['rule_4'] * (n - 3)),
            'rule_number': ([1] * 3 + [4] * (n - 3)),
            'description': [f'violation {i}' for i in range(1, n + 1)],
            'value': [120.0 + i for i in range(n)],
            'center': [100.0] * n,
            'upl': [115.0] * n,
            'lpl': [85.0] * n,
        }
    )


@pytest.fixture
def signals():
    data = pd.DataFrame({'mean': [100.0] * 10})
    stats = {'center': 100.0, 'upl': 115.0, 'lpl': 85.0}
    return SignalResult(_violations(), 'Test Chart', data, stats)


@pytest.fixture
def clean():
    data = pd.DataFrame({'mean': [100.0] * 10})
    stats = {'center': 100.0, 'upl': 115.0, 'lpl': 85.0}
    return SignalResult(pd.DataFrame(), 'Clean Chart', data, stats)


class TestDrillDown:
    def test_by_rule_groups_violations(self, signals):
        by_rule = signals.by_rule
        assert set(by_rule) == {'rule_1', 'rule_4'}
        assert len(by_rule['rule_1']) == 3

    def test_by_rule_empty(self, clean):
        assert clean.by_rule == {}

    def test_by_observation_groups(self, signals):
        assert len(signals.by_observation) == 7

    def test_by_observation_empty(self, clean):
        assert clean.by_observation == {}

    def test_get_rule_violations_filters(self, signals):
        rows = signals.get_rule_violations('rule_4')
        assert len(rows) == 4
        assert set(rows['rule_name']) == {'rule_4'}

    def test_get_rule_violations_empty(self, clean):
        assert len(clean.get_rule_violations('rule_1')) == 0

    def test_get_observation_violations_filters(self, signals):
        obs = sorted(signals.flagged_observations)[0]
        rows = signals.get_observation_violations(obs)
        assert len(rows) >= 1
        assert set(rows['obs_id']) == {obs}

    def test_get_observation_violations_empty(self, clean):
        assert len(clean.get_observation_violations(1)) == 0


class TestSummary:
    def test_summary_with_signals(self, signals):
        text = signals.summary
        assert 'Test Chart' in text
        assert 'rule_1' in text and 'rule_4' in text
        assert 'more' in text, "expected the '... and N more' tail with 7 violations"

    def test_summary_clean(self, clean):
        assert 'No signals' in clean.summary

    def test_repr_and_str(self, signals, clean):
        assert 'SignalResult' in repr(signals)
        assert isinstance(str(signals), str) and str(signals)
        assert isinstance(str(clean), str) and str(clean)


class TestExports:
    def test_to_dataframe_returns_copy(self, signals):
        df = signals.to_dataframe()
        assert len(df) == 7
        df.loc[0, 'value'] = -1
        assert signals.to_dataframe().loc[0, 'value'] != -1

    def test_to_dataframe_empty(self, clean):
        assert clean.to_dataframe().empty

    @pytest.mark.io
    def test_to_excel_writes_violations_and_summary(self, signals, tmp_path):
        path = str(tmp_path / 'signals.xlsx')
        signals.to_excel(path)
        sheets = pd.ExcelFile(path, engine='openpyxl').sheet_names
        assert 'Violations' in sheets
        assert any('Summary' in s for s in sheets)
        assert len(pd.read_excel(path, sheet_name='Violations')) == 7

    @pytest.mark.io
    def test_to_excel_empty_warns_and_writes_nothing(self, clean, tmp_path, caplog):
        path = tmp_path / 'nothing.xlsx'
        with caplog.at_level('WARNING'):
            clean.to_excel(str(path))
        assert not path.exists()
        assert any('No violations' in r.message for r in caplog.records)

    @pytest.mark.io
    def test_to_json_round_trips(self, signals, tmp_path):
        path = tmp_path / 'signals.json'
        signals.to_json(str(path))
        records = json.loads(path.read_text())
        assert len(records) == 7
        assert records[0]['rule_name'] == 'rule_1'
