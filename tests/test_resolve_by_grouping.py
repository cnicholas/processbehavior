"""Truth-table tests for `Analysis._resolve_by_grouping`.

The function decides three things from `spec` + `request.by`:
  - groupby_cols   : how Xbar/S aggregates rows
  - ybar_col       : which pre-cached cell-mean column to reuse, if any
  - stratify_by    : which factor (if any) to break out as separate charts

The branches are enumerable — this file enumerates them. Adding a new
branch (or removing one) without updating this file should break a test.

The audit (#82) flagged that a dead `'Ybar'` branch survived for months
because no test asserted this function's output; this file's purpose
is to be that pinning regression net.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from processbehavior.analysis import Analysis
from processbehavior.formulation_spec import ChartRequest, FormulationSpec

# ---------------------------------------------------------------------------
# Stub harness — _resolve_by_grouping only reads self.spec and self.request
# ---------------------------------------------------------------------------


@dataclass
class _StubAnalysis:
    spec: FormulationSpec
    request: ChartRequest


def _resolve(
    spec: FormulationSpec,
    by: list[str] | None,
    value_col: str,
) -> tuple[list[str], str | None, list[str]]:
    """Invoke `_resolve_by_grouping` with a stub `self`."""
    stub = _StubAnalysis(
        spec=spec,
        request=ChartRequest(chart='Xbar', by=tuple(by) if by is not None else None),
    )
    return Analysis._resolve_by_grouping(stub, value_col)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Specs covering every shape of spec _resolve_by_grouping inspects
# ---------------------------------------------------------------------------


SPEC_FACTORS_AND_TIME = FormulationSpec(
    response_var='y', rsg_vars=('f1', 'f2'), time_var='t'
)
SPEC_FACTORS_ONLY = FormulationSpec(
    response_var='y', rsg_vars=('f1', 'f2'),
)
SPEC_TIME_ONLY = FormulationSpec(
    response_var='y', time_var='t',
)
SPEC_BARE = FormulationSpec(
    response_var='y',
)


# ---------------------------------------------------------------------------
# by=None — Kt-level (cell) aggregation defaults
# ---------------------------------------------------------------------------


class TestByNone:
    """by=None means "default aggregation for the spec shape"."""

    def test_factors_and_time_uses_cell_grain_with_ybar_kt(self):
        """rsg_vars + time_var → groupby=[*rsg_vars, time_var], ybar=Ybar_kt."""
        groupby, ybar, stratify = _resolve(SPEC_FACTORS_AND_TIME, None, 'y')
        assert groupby == ['f1', 'f2', 't']
        assert ybar == 'Ybar_kt'
        assert stratify == []

    def test_factors_only_uses_rsg_col_with_ybar_k(self):
        """rsg_vars only → groupby=[rsg_var_name], ybar=Ybar_k."""
        groupby, ybar, stratify = _resolve(SPEC_FACTORS_ONLY, None, 'y')
        assert groupby == ['rsg']
        assert ybar == 'Ybar_k'
        assert stratify == []

    def test_time_only_uses_time_with_ybar_t(self):
        """time_var only → groupby=[time_var], ybar=Ybar_t."""
        groupby, ybar, stratify = _resolve(SPEC_TIME_ONLY, None, 'y')
        assert groupby == ['t']
        assert ybar == 'Ybar_t'
        assert stratify == []

    def test_bare_spec_no_grouping_no_ybar(self):
        """No factors and no time → groupby=[], ybar=None."""
        groupby, ybar, stratify = _resolve(SPEC_BARE, None, 'y')
        assert groupby == []
        assert ybar is None
        assert stratify == []


# ---------------------------------------------------------------------------
# by=[] — semantic alias for by=None on Xbar/S
# ---------------------------------------------------------------------------


class TestByEmpty:
    """by=[] is normalized to by=None inside _resolve_by_grouping."""

    @pytest.mark.parametrize(
        'spec',
        [SPEC_FACTORS_AND_TIME, SPEC_FACTORS_ONLY, SPEC_TIME_ONLY, SPEC_BARE],
    )
    def test_by_empty_matches_by_none(self, spec):
        """For every spec shape, by=[] returns the same triple as by=None."""
        assert _resolve(spec, [], 'y') == _resolve(spec, None, 'y')


# ---------------------------------------------------------------------------
# by=[all factors] — rsg-level aggregation with Ybar_k optimization
# ---------------------------------------------------------------------------


class TestByAllFactors:
    """by matching rsg_vars in order → groupby=[rsg_var_name], ybar=Ybar_k."""

    def test_all_factors_uses_ybar_k(self):
        groupby, ybar, stratify = _resolve(SPEC_FACTORS_AND_TIME, ['f1', 'f2'], 'y')
        assert groupby == ['rsg']
        assert ybar == 'Ybar_k'
        assert stratify == []

    def test_all_factors_out_of_order_falls_through(self):
        """by=[f2, f1] doesn't match rsg_vars order → no Ybar_k optimization."""
        groupby, ybar, stratify = _resolve(SPEC_FACTORS_AND_TIME, ['f2', 'f1'], 'y')
        # Not the rsg_vars order, so falls to "partial subset" path
        # (cell_key_vars match check excludes time_var too)
        # Result: groupby is the literal `by`, no ybar optimization
        assert groupby == ['f2', 'f1']
        assert ybar is None
        assert stratify == []


# ---------------------------------------------------------------------------
# by=[time_var] — time-aggregation; stratify by rsg when factors exist + response
# ---------------------------------------------------------------------------


class TestByTimeOnly:
    """by=[time_var] uses Ybar_t; stratifies by factors when charting response."""

    def test_time_only_with_factors_stratifies_by_rsg(self):
        """Factors exist + charting response → stratify by rsg, ybar=Ybar_t."""
        groupby, ybar, stratify = _resolve(SPEC_FACTORS_AND_TIME, ['t'], 'y')
        assert groupby == ['t']
        assert ybar == 'Ybar_t'
        assert stratify == ['rsg']

    def test_time_only_no_factors_no_stratify(self):
        """No factors → no stratification."""
        groupby, ybar, stratify = _resolve(SPEC_TIME_ONLY, ['t'], 'y')
        assert groupby == ['t']
        assert ybar == 'Ybar_t'
        assert stratify == []

    def test_time_only_charting_residual_no_stratify(self):
        """Charting residual (R5) → no Ybar optimization, no stratification."""
        groupby, ybar, stratify = _resolve(SPEC_FACTORS_AND_TIME, ['t'], 'R5')
        assert groupby == ['t']
        assert ybar is None
        assert stratify == []


# ---------------------------------------------------------------------------
# by=[all factors + time] — cell-level aggregation, Ybar_kt
# ---------------------------------------------------------------------------


class TestByFactorsAndTime:
    """by matching the full (factors, time) set → cell aggregation."""

    def test_factors_plus_time_uses_ybar_kt(self):
        groupby, ybar, stratify = _resolve(
            SPEC_FACTORS_AND_TIME, ['f1', 'f2', 't'], 'y'
        )
        # Order from rsg_vars + [time_var], not the user's order
        assert groupby == ['f1', 'f2', 't']
        assert ybar == 'Ybar_kt'
        assert stratify == []


# ---------------------------------------------------------------------------
# Partial subset — falls through to "use by as-is" with no Ybar optimization
# ---------------------------------------------------------------------------


class TestPartialSubset:
    """When by doesn't match a known aggregation level, groupby=by and ybar=None."""

    def test_partial_subset_one_factor_of_two(self):
        groupby, ybar, stratify = _resolve(SPEC_FACTORS_AND_TIME, ['f1'], 'y')
        assert groupby == ['f1']
        assert ybar is None
        assert stratify == []


# ---------------------------------------------------------------------------
# Residual charting — ybar_col is always None
# ---------------------------------------------------------------------------


class TestResidualValueCol:
    """When charting a residual (value_col != response_var), ybar_col is None
    across every branch — there's no pre-cached residual mean."""

    @pytest.mark.parametrize('residual', ['R1', 'R2', 'R3', 'R4', 'R5'])
    def test_no_ybar_for_residuals_with_by_none(self, residual):
        _, ybar, _ = _resolve(SPEC_FACTORS_AND_TIME, None, residual)
        assert ybar is None

    @pytest.mark.parametrize('residual', ['R1', 'R2', 'R3', 'R4', 'R5'])
    def test_no_ybar_for_residuals_with_by_all_factors(self, residual):
        _, ybar, _ = _resolve(SPEC_FACTORS_AND_TIME, ['f1', 'f2'], residual)
        assert ybar is None

    @pytest.mark.parametrize('residual', ['R1', 'R2', 'R3', 'R4', 'R5'])
    def test_no_ybar_for_residuals_with_by_time(self, residual):
        _, ybar, _ = _resolve(SPEC_FACTORS_AND_TIME, ['t'], residual)
        assert ybar is None


# ---------------------------------------------------------------------------
# Return-shape invariant
# ---------------------------------------------------------------------------


class TestReturnShape:
    """Every path returns (list[str], str|None, list[str])."""

    @pytest.mark.parametrize(
        'spec,by',
        [
            (SPEC_FACTORS_AND_TIME, None),
            (SPEC_FACTORS_AND_TIME, []),
            (SPEC_FACTORS_AND_TIME, ['f1']),
            (SPEC_FACTORS_AND_TIME, ['f1', 'f2']),
            (SPEC_FACTORS_AND_TIME, ['t']),
            (SPEC_FACTORS_AND_TIME, ['f1', 'f2', 't']),
            (SPEC_FACTORS_ONLY, None),
            (SPEC_TIME_ONLY, None),
            (SPEC_BARE, None),
        ],
    )
    def test_return_triple_types(self, spec, by):
        groupby, ybar, stratify = _resolve(spec, by, 'y')
        assert isinstance(groupby, list)
        assert all(isinstance(c, str) for c in groupby)
        assert ybar is None or isinstance(ybar, str)
        assert isinstance(stratify, list)
        assert all(isinstance(c, str) for c in stratify)

    @pytest.mark.parametrize(
        'spec,by',
        [
            (SPEC_FACTORS_AND_TIME, None),
            (SPEC_FACTORS_AND_TIME, ['f1', 'f2']),
            (SPEC_FACTORS_AND_TIME, ['t']),
            (SPEC_FACTORS_AND_TIME, ['f1', 'f2', 't']),
            (SPEC_FACTORS_ONLY, None),
            (SPEC_TIME_ONLY, None),
        ],
    )
    def test_ybar_col_is_canonical_value_or_none(self, spec, by):
        """ybar_col must be one of {Ybar_kt, Ybar_k, Ybar_t, None}."""
        _, ybar, _ = _resolve(spec, by, 'y')
        assert ybar in {None, 'Ybar_kt', 'Ybar_k', 'Ybar_t'}, (
            f'Unexpected ybar={ybar!r}; the documented canonical values '
            f'are Ybar_kt / Ybar_k / Ybar_t (or None).'
        )
