"""Truth-table tests for `Study._validate_by_parameter`.

The function builds a focused error-message tree:
  - No-factors specs reject any by= except None/[]
  - X/mR with factors require an explicit by=
  - X/mR refuse the time variable in by= (time is the x-axis)
  - Xbar/S allow factors + time in by=
  - Invalid names trigger case-insensitive matching, then fuzzy matching
  - Duplicates trigger a UserWarning and dedupe

The audit (#82) flagged this as a 145-line C901-suppressed if/elif
tree with no dedicated test file. This file is that test file.
"""

from __future__ import annotations

import warnings

import pytest

from processbehavior.exceptions import FactorNotFoundError, ValidationError
from processbehavior.formulation_spec import FormulationSpec
from processbehavior.study import Study

# ---------------------------------------------------------------------------
# Stub harness
# ---------------------------------------------------------------------------


class _StubStudy:
    """Minimal stand-in for Study; provides only what _validate_by_parameter reads."""

    def __init__(self, spec: FormulationSpec, n_factor_combos: int = 4):
        self._spec = spec
        self._n_factor_combos = n_factor_combos

    def _get_factor_combinations(self) -> int:
        return self._n_factor_combos


def _validate(
    spec: FormulationSpec,
    by: list[str] | None,
    base_chart: str,
    n_factor_combos: int = 4,
) -> list[str] | None:
    stub = _StubStudy(spec, n_factor_combos)
    return Study._validate_by_parameter(stub, by, base_chart)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


SPEC_TWO_FACTORS_TIME = FormulationSpec(
    response_var='y', rsg_vars=('f1', 'f2'), time_var='t'
)
SPEC_NO_FACTORS = FormulationSpec(response_var='y', time_var='t')
SPEC_NO_FACTORS_NO_TIME = FormulationSpec(response_var='y')


# ---------------------------------------------------------------------------
# No-factors case: by=None or by=[] OK; anything else rejected
# ---------------------------------------------------------------------------


class TestNoFactors:
    @pytest.mark.parametrize('by', [None, []])
    def test_no_factors_by_none_or_empty(self, by):
        assert _validate(SPEC_NO_FACTORS, by, 'Xbar') == []

    def test_no_factors_by_with_value_rejected(self):
        with pytest.raises(ValidationError, match='No factors defined'):
            _validate(SPEC_NO_FACTORS, ['t'], 'Xbar')


# ---------------------------------------------------------------------------
# X/mR with factors: by=None forbidden
# ---------------------------------------------------------------------------


class TestXmrRequiresExplicitBy:
    @pytest.mark.parametrize('chart', ['X', 'mR'])
    def test_xmr_with_factors_requires_by(self, chart):
        with pytest.raises(ValidationError, match='require explicit'):
            _validate(SPEC_TWO_FACTORS_TIME, None, chart)

    @pytest.mark.parametrize('chart', ['X', 'mR'])
    def test_xmr_by_empty_returns_empty(self, chart):
        assert _validate(SPEC_TWO_FACTORS_TIME, [], chart) == []

    @pytest.mark.parametrize('chart', ['X', 'mR'])
    def test_xmr_by_factors_returns_list(self, chart):
        assert _validate(SPEC_TWO_FACTORS_TIME, ['f1', 'f2'], chart) == ['f1', 'f2']

    @pytest.mark.parametrize('chart', ['X', 'mR'])
    def test_xmr_time_in_by_rejected(self, chart):
        """Time is the x-axis for X/mR; rejecting it is a methodology rule."""
        with pytest.raises(ValidationError, match='Time is the x-axis'):
            _validate(SPEC_TWO_FACTORS_TIME, ['t'], chart)


# ---------------------------------------------------------------------------
# Xbar/S with factors: by=None means rsg_key-level (returns None)
# ---------------------------------------------------------------------------


class TestXbarSDefaults:
    @pytest.mark.parametrize('chart', ['Xbar', 'S'])
    def test_xbar_s_by_none_returns_none(self, chart):
        """by=None means "use rsg_key-level grouping" — preserved as None."""
        assert _validate(SPEC_TWO_FACTORS_TIME, None, chart) is None

    @pytest.mark.parametrize('chart', ['Xbar', 'S'])
    def test_xbar_s_time_in_by_allowed(self, chart):
        """Xbar/S accept time in by — different semantics from X/mR."""
        assert _validate(SPEC_TWO_FACTORS_TIME, ['t'], chart) == ['t']

    @pytest.mark.parametrize('chart', ['Xbar', 'S'])
    def test_xbar_s_factors_plus_time(self, chart):
        assert _validate(
            SPEC_TWO_FACTORS_TIME, ['f1', 'f2', 't'], chart
        ) == ['f1', 'f2', 't']


# ---------------------------------------------------------------------------
# Invalid names: case-insensitive then fuzzy matching in error messages
# ---------------------------------------------------------------------------


class TestInvalidNameSuggestions:
    def test_case_mismatch_suggested(self):
        """Case-insensitive match → suggestion in error message."""
        with pytest.raises(FactorNotFoundError, match='Did you mean'):
            _validate(SPEC_TWO_FACTORS_TIME, ['F1'], 'Xbar')

    def test_typo_close_match_suggested(self):
        """Fuzzy close match → suggestion in error message."""
        with pytest.raises(FactorNotFoundError, match='Did you mean'):
            _validate(SPEC_TWO_FACTORS_TIME, ['f11'], 'Xbar')

    def test_unrelated_name_no_suggestion(self):
        """No close match → no "Did you mean" hint, just the valid list."""
        with pytest.raises(FactorNotFoundError) as exc_info:
            _validate(SPEC_TWO_FACTORS_TIME, ['zzz_unrelated'], 'Xbar')
        msg = str(exc_info.value)
        assert 'is not a valid by variable' in msg
        assert 'Did you mean' not in msg

    def test_error_lists_valid_dimensions(self):
        """The error message names what *is* valid."""
        with pytest.raises(FactorNotFoundError, match='Valid:'):
            _validate(SPEC_TWO_FACTORS_TIME, ['nope'], 'Xbar')


# ---------------------------------------------------------------------------
# Deduplication: duplicate values warn + are removed
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_duplicate_values_warn_and_dedupe(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            result = _validate(SPEC_TWO_FACTORS_TIME, ['f1', 'f1', 'f2'], 'Xbar')
        assert result == ['f1', 'f2']
        assert any(
            issubclass(w.category, UserWarning) and 'Duplicate values' in str(w.message)
            for w in captured
        )

    def test_no_duplicates_no_warning(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            result = _validate(SPEC_TWO_FACTORS_TIME, ['f1', 'f2'], 'Xbar')
        assert result == ['f1', 'f2']
        assert not any(
            issubclass(w.category, UserWarning) and 'Duplicate' in str(w.message)
            for w in captured
        )

    def test_dedup_preserves_first_occurrence_order(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = _validate(SPEC_TWO_FACTORS_TIME, ['f2', 'f1', 'f2'], 'Xbar')
        assert result == ['f2', 'f1']


# ---------------------------------------------------------------------------
# String coercion: by='f1' → ['f1']
# ---------------------------------------------------------------------------


class TestStringCoercion:
    @pytest.mark.parametrize('chart', ['Xbar', 'S', 'X', 'mR'])
    def test_string_by_normalized_to_list(self, chart):
        """by= can be a bare string — the function normalizes to a 1-list."""
        assert _validate(SPEC_TWO_FACTORS_TIME, 'f1', chart) == ['f1']  # type: ignore[arg-type]
