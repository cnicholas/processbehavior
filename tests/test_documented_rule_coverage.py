"""The README's signal-detection claim matches what the detector actually runs.

The README said "Rule 1 (3-sigma) point classification" while ``signals/config.py``
configured all eight Western Electric rules for X/mR. Understating a feature is the
cheap direction of a docs/code gap, but it is the same gap: nothing tied the sentence
to the code, so either could move.

These tests assert the configuration against a real run rather than against the config
comment — ``by_rule`` reports only rules that *fired*, so the check is "every configured
rule is evaluated", proven by exercising data that trips them.
"""

from pathlib import Path

import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.signals.config import CHART_TYPE_RULES

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
README = Path(__file__).resolve().parent.parent / 'README.md'
ALL_EIGHT = [f'rule_{n}' for n in range(1, 9)]


@pytest.fixture(scope='module')
def study():
    df = pd.read_csv(REFERENCE_CSV, na_values=['*'])
    return pb.formulate(
        df, response='PM SDS 1', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME',
    )


def test_sequential_charts_run_all_eight_rules():
    assert CHART_TYPE_RULES['X'] == ALL_EIGHT
    assert CHART_TYPE_RULES['mR'] == ALL_EIGHT


def test_subgroup_charts_run_rule_1_only():
    """Rules 2-8 read runs and zones off a time order; Xbar/S compare subgroups."""
    assert CHART_TYPE_RULES['Xbar'] == ['rule_1']
    assert CHART_TYPE_RULES['S'] == ['rule_1']


def test_configuration_is_what_actually_runs(study):
    """Config is a claim; this executes it.

    ``by_rule`` only contains rules with at least one violation, so a rule missing here
    means either "not evaluated" or "evaluated, found nothing". Asserting no *unexpected*
    rule appears catches the direction that matters — a chart evaluating rules it should
    not.
    """
    for chart in ('Xbar', 'S', 'X', 'mR'):
        signals = study.execute(chart=chart, by=[]).detect_signals(chart=chart)
        unexpected = set(signals.by_rule) - set(CHART_TYPE_RULES[chart])
        assert not unexpected, f'{chart} evaluated unconfigured rules: {sorted(unexpected)}'


def test_x_chart_actually_fires_the_run_rules(study):
    """Proof the eight rules are wired up, not merely listed in a dict."""
    signals = study.execute(chart='X', by=[]).detect_signals(chart='X')
    assert set(signals.by_rule) == set(ALL_EIGHT), (
        f'X evaluated all eight but only {sorted(signals.by_rule)} fired — if the '
        'reference data changed, this test needs different data, not a weaker assertion'
    )


def test_xbar_never_fires_a_run_rule(study):
    signals = study.execute(chart='Xbar', by=[]).detect_signals(chart='Xbar')
    assert set(signals.by_rule) <= {'rule_1'}


def test_readme_does_not_claim_rule_1_only():
    """The exact sentence that was wrong, pinned so it cannot come back."""
    text = README.read_text()
    assert '**Signal detection**' in text, 'the Features bullet was renamed; update this test'
    line = next(ln for ln in text.splitlines() if '**Signal detection**' in ln)
    assert 'eight' in line.lower(), f'README understates rule coverage: {line}'
