"""The residual alias table means what it says, and there is only one of it.

``RESIDUAL_ALIASES`` used to carry its own ``label``/``description`` per entry — a third
residual vocabulary after ``plotting.plotter._RESIDUAL_LABELS`` and the app's dropdown —
and it had drifted:

    'noise': {'id': 'R5', 'label': 'Noise', 'description': 'Unexplained variation'}

Unexplained variation is **R2**; R5 is design-condition main effects. The table
contradicted itself, since it also mapped ``within_cell`` to R2. Nothing read the labels,
so the only person the wrong answer reached was a user who typed ``noise`` and got
pointed at factor effects — and anyone reading the source.

These tests pin the two properties that would have caught it: aliases agree with the
canonical labels, and there is exactly one label table.
"""

from pathlib import Path

import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.exceptions import ValidationError
from processbehavior.spc_constants import (
    ALL_RESIDUALS,
    RESIDUAL_ALIASES,
    RESIDUAL_LABELS,
)

# ---------------------------------------------------------------------------
# One vocabulary
# ---------------------------------------------------------------------------


def test_plotter_reuses_the_canonical_labels_rather_than_copying_them():
    """Identity, not equality — a copy that happens to match today is the defect."""
    from processbehavior.plotting.plotter import _RESIDUAL_LABELS

    assert _RESIDUAL_LABELS is RESIDUAL_LABELS


def test_every_residual_has_a_label():
    assert set(RESIDUAL_LABELS) == set(ALL_RESIDUALS)


def test_labels_are_distinct():
    """Two residuals sharing a label makes a chart title ambiguous."""
    labels = list(RESIDUAL_LABELS.values())
    assert len(set(labels)) == len(labels)


# ---------------------------------------------------------------------------
# Aliases point where their name says
# ---------------------------------------------------------------------------


def test_every_alias_resolves_to_a_real_residual():
    for alias, code in RESIDUAL_ALIASES.items():
        assert code in ALL_RESIDUALS, f'{alias!r} -> {code!r}, which is not a residual'


@pytest.mark.parametrize(
    ('alias', 'expected'),
    [
        ('response_centered', 'R1'),
        ('mean_removed', 'R1'),
        ('within_cell', 'R2'),
        ('noise', 'R2'),
        ('interaction', 'R3'),
        ('structure_removed', 'R3'),
        ('time_main_effects', 'R4'),
        ('time_structure_removed', 'R4'),
        ('design_condition_main_effects', 'R5'),
        ('design_factor_main_effects', 'R6'),
    ],
)
def test_alias_meanings(alias, expected):
    assert RESIDUAL_ALIASES[alias] == expected


def test_noise_is_R2_not_R5():
    """The specific regression. 'Noise' is unexplained within-cell variation — R2.

    Every doc uses the word that way ("within-cell noise", "the noise floor",
    "irreducible noise (R2)"). R5 is design-condition main effects, which is the
    explained part.
    """
    assert RESIDUAL_ALIASES['noise'] == 'R2'
    assert RESIDUAL_LABELS['R2'] == 'Within-Cell'
    assert RESIDUAL_LABELS['R5'] == 'Design Condition Main Effects'


def test_main_effect_aliases_do_not_collide():
    """Design *condition* and design *factor* effects are different residuals.

    They are one word apart in English and R5 vs R6 in the methodology; an alias table
    that conflated them would send an analyst to the wrong chart with no error.
    """
    assert RESIDUAL_ALIASES['design_condition_main_effects'] == 'R5'
    assert RESIDUAL_ALIASES['design_factor_main_effects'] == 'R6'


def test_R6_is_reachable_by_alias():
    """R6 was absent from the table entirely."""
    assert 'R6' in set(RESIDUAL_ALIASES.values())


def test_aliases_cover_every_residual():
    assert set(RESIDUAL_ALIASES.values()) == set(ALL_RESIDUALS)


def test_no_alias_shadows_a_residual_code():
    """`execute(value='R5')` must never be intercepted as an alias."""
    assert not set(RESIDUAL_ALIASES) & set(ALL_RESIDUALS)


# ---------------------------------------------------------------------------
# What the analyst actually sees
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def study():
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv',
        na_values=['*'],
    )
    return pb.formulate(
        df, response='PM SDS 1', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME',
    )


def test_alias_used_as_a_chart_name_names_the_right_residual(study):
    """The one live consumer: the error raised when an alias is passed as `chart`."""
    with pytest.raises(ValidationError) as exc:
        study.execute(chart='noise')
    message = str(exc.value)
    assert 'R2' in message
    assert 'Within-Cell' in message, 'the error should say what R2 is, not just its code'
    assert 'R5' not in message, 'the old table sent this user to R5'


def test_error_names_the_residual_for_every_alias(study):
    """No alias may produce a bare code with no explanation of what it means.

    This is also what proves every alias is *reachable* — see the ordering test below.
    """
    for alias, code in RESIDUAL_ALIASES.items():
        with pytest.raises(ValidationError) as exc:
            study.execute(chart=alias)
        message = str(exc.value)
        assert code in message, f'{alias}: message omits {code}'
        assert RESIDUAL_LABELS[code] in message, f'{alias}: message omits the label'


def test_underscored_aliases_are_reachable(study):
    """Aliases are matched before the old-`residual_chart` syntax is parsed.

    Reversed, `'_' in chart` claims every multi-word alias first, hands it to the
    old-syntax parser, which splits `within_cell` into `within` + `cell`, finds no base
    chart, and raises the generic message. Silent: the alias stayed in the table looking
    supported. Every alias in the table now has an underscore, so this ordering is the
    difference between the table working and being decorative.
    """
    with pytest.raises(ValidationError) as exc:
        study.execute(chart='within_cell')
    assert 'residual alias' in str(exc.value)
    assert 'Invalid chart name' not in str(exc.value)


def test_old_syntax_guidance_still_works(study):
    """The branch that had to move must keep doing its job."""
    with pytest.raises(ValidationError, match='no longer supported'):
        study.execute(chart='R5_Xbar')
    with pytest.raises(ValidationError, match='no longer supported'):
        study.execute(chart='rc_R5_Xbar')


def test_alias_with_a_base_chart_still_resolves(study):
    """`noise_Xbar` is old syntax whose residual half is an alias — now R2, not R5."""
    with pytest.raises(ValidationError) as exc:
        study.execute(chart='noise_Xbar')
    message = str(exc.value)
    assert 'no longer supported' in message
    assert "value='R2'" in message


def test_genuinely_unknown_names_still_report_cleanly(study):
    with pytest.raises(ValidationError, match='Invalid chart name|Unknown chart'):
        study.execute(chart='not_a_chart')
    with pytest.raises(ValidationError, match='Unknown chart'):
        study.execute(chart='nonsense')
