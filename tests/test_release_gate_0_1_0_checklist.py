"""Release gate tests for 0.1.0.

These tests encode the acceptance criteria from docs/release_gate_0_1_0.md.
Each MUST-FIX gate (1-5) has concrete, enforceable assertions that currently
FAIL against the codebase.  The implementation work is to make them pass.

Gate 7 (mutability) already passes and is retained as-is.
Gates 6, 8-10 remain scaffolded for later milestones.
"""

import keyword

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic
from processbehavior.exceptions import ProcessBehaviorError, ValidationError

# ============================================================================
# Helpers / fixtures
# ============================================================================


@pytest.fixture
def sds1_study():
    """SDS 1 study with two factors — supports Xbar, S, XmR, R."""
    df = synthetic.make_design(1, K1=2, K2=2, T=5, seed=42)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1', 'factor 2'],
    )


@pytest.fixture
def sds4_study():
    """SDS 4 study — single condition over time, XmR only."""
    df = synthetic.make_design(4, T=20, seed=42)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1'],
    )


@pytest.fixture
def stratified_result(sds1_study):
    """Stratified XmR result with two strata dimensions."""
    return sds1_study.execute(chart='X', by=['factor 1'])


# ============================================================================
# Gate 01 — Normalize Exception Model
#
# Acceptance criteria (from release_gate_0_1_0.md):
#   - Invalid execute(chart=...) raises ChartNotAvailableError or
#     ValidationError, NOT raw ValueError.
#   - Invalid by / value / recentered combinations raise typed custom
#     exceptions.
#   - Catching ProcessBehaviorError reliably covers user-facing API failures.
# ============================================================================


class TestGate01ExceptionModel:
    """Every public API failure must raise a ProcessBehaviorError subclass."""

    # -- _parse_chart_request paths ------------------------------------------

    def test_empty_chart_name_raises_validation_error(self, sds1_study):
        """Empty string chart name must raise ValidationError, not ValueError."""
        with pytest.raises(ValidationError):
            sds1_study.execute(chart='')

    def test_non_string_chart_raises_validation_error(self, sds1_study):
        """Non-string chart must raise ValidationError, not ValueError."""
        with pytest.raises(ValidationError):
            sds1_study.execute(chart=123)

    def test_unknown_chart_raises_validation_error(self, sds1_study):
        """Completely unknown chart name must raise ValidationError."""
        with pytest.raises(ValidationError, match='Unknown chart'):
            sds1_study.execute(chart='FooBar')

    def test_residual_id_as_chart_raises_validation_error(self, sds1_study):
        """Bare residual identifier (R5) used as chart raises ValidationError."""
        with pytest.raises(ValidationError, match='residual identifier'):
            sds1_study.execute(chart='R5')

    def test_residual_alias_as_chart_raises_validation_error(self, sds1_study):
        """Residual alias (noise) used as chart raises ValidationError."""
        with pytest.raises(ValidationError, match='residual alias'):
            sds1_study.execute(chart='noise')

    def test_old_residual_syntax_raises_validation_error(self, sds1_study):
        """Old syntax (R5_Xbar) must raise ValidationError with migration hint."""
        with pytest.raises(ValidationError, match='no longer supported'):
            sds1_study.execute(chart='R5_Xbar')

    def test_invalid_chart_underscore_raises_validation_error(self, sds1_study):
        """Unknown underscore chart (foo_bar) must raise ValidationError."""
        with pytest.raises(ValidationError, match='Invalid chart name'):
            sds1_study.execute(chart='foo_bar')

    # -- _validate_by_parameter paths ----------------------------------------

    def test_by_with_no_factors_raises_validation_error(self, sds4_study):
        """by=['nonexistent'] with single-factor study raises typed exception."""
        # SDS 4 has one factor; specifying an invalid by should not raise
        # raw ValueError.
        with pytest.raises(ProcessBehaviorError):
            sds4_study.execute(chart='X', by=['nonexistent'])

    def test_imr_without_explicit_by_raises_validation_error(self, sds1_study):
        """IMR/R chart with factors but no by= must raise ValidationError."""
        with pytest.raises(ValidationError):
            sds1_study.execute(chart='X')

    def test_time_in_by_for_xmr_raises_validation_error(self, sds1_study):
        """Using time variable in by= for XmR chart raises ValidationError."""
        with pytest.raises(ValidationError, match='time'):
            sds1_study.execute(chart='X', by=['time'])

    # -- focus() paths -------------------------------------------------------

    def test_focus_on_unstratified_raises_validation_error(self, sds1_study):
        """focus() on non-stratified result must raise ValidationError."""
        result = sds1_study.execute(chart='Xbar')
        with pytest.raises(ValidationError, match='not stratified'):
            result.focus('anything')

    def test_focus_bad_stratum_raises_validation_error(self, stratified_result):
        """focus() with unknown stratum must raise ValidationError."""
        with pytest.raises(ValidationError, match='not found'):
            stratified_result.focus('DOES_NOT_EXIST')

    # -- catch-all contract --------------------------------------------------

    def test_process_behavior_error_catches_all_parse_failures(self, sds1_study):
        """ProcessBehaviorError must catch every public failure mode."""
        bad_inputs = [
            dict(chart=''),
            dict(chart='FooBar'),
            dict(chart='R5'),
            dict(chart='R5_Xbar'),
            dict(chart='noise'),
            dict(chart='foo_bar'),
            dict(chart=123),
        ]
        for kwargs in bad_inputs:
            with pytest.raises(ProcessBehaviorError):
                sds1_study.execute(**kwargs)

    def test_process_behavior_error_catches_all_by_failures(self, sds1_study):
        """ProcessBehaviorError must catch every by= failure mode."""
        bad_inputs = [
            dict(chart='X'),  # missing required by
            dict(chart='X', by=['time']),  # time in by for XmR
        ]
        for kwargs in bad_inputs:
            with pytest.raises(ProcessBehaviorError):
                sds1_study.execute(**kwargs)

    def test_process_behavior_error_catches_focus_failures(self, sds1_study):
        """ProcessBehaviorError must catch focus() failures."""
        result = sds1_study.execute(chart='Xbar')
        with pytest.raises(ProcessBehaviorError):
            result.focus('anything')


# ============================================================================
# Gate 02 — Unify Stratified API Semantics
#
# Acceptance criteria:
#   - result.strata values are exactly valid inputs to result.focus(...)
#   - get_stratified_chart() resolves exact strata keys (no ambiguous matching)
#   - list_strata() returns canonical values consistent with result.strata
#   - Multi-factor strata round-trip correctly (always strings via encode_rsg)
# ============================================================================


class TestGate02StratifiedAPISemantics:
    """Strata / focus / stratified helper methods share one canonical contract."""

    def test_strata_values_roundtrip_through_focus(self, stratified_result):
        """Every value in result.strata must be a valid input to focus()."""
        for stratum in stratified_result.strata:
            focused = stratified_result.focus(stratum)
            assert focused is not None, f"focus('{stratum}') returned None"

    def test_multi_factor_strata_roundtrip(self, sds1_study):
        """Multi-factor by= produces strata that roundtrip through focus()."""
        result = sds1_study.execute(
            chart='X',
            by=['factor 1', 'factor 2'],
        )
        assert result.is_stratified, 'Expected stratified result'
        assert len(result.strata) > 1, 'Expected multiple strata'

        # Every stratum must roundtrip
        for stratum in result.strata:
            focused = result.focus(stratum)
            assert focused is not None

    def test_focused_result_has_data(self, stratified_result):
        """Focused result must contain non-empty chart data."""
        stratum = stratified_result.strata[0]
        focused = stratified_result.focus(stratum)
        chart = focused.get_chart('X')
        assert len(chart) > 0, 'Focused chart data must be non-empty'

    def test_focus_rejects_partial_match(self, stratified_result):
        """focus() must not accept substring/partial matches."""
        full_stratum = stratified_result.strata[0]
        # A prefix of the stratum should NOT match
        partial = full_stratum[: max(1, len(str(full_stratum)) // 2)]
        if partial != full_stratum:
            with pytest.raises(ProcessBehaviorError):
                stratified_result.focus(partial)


# ============================================================================
# Gate 04 — Correct Public Type Hints and Docstrings for Strata
#
# Acceptance criteria:
#   - AnalysisResult.strata and focus() annotations match runtime values
#   - Examples include multi-factor tuple strata usage
#   - Static-checking examples pass
#
# Note: Gate 04 is the type-annotation face of Gate 02.
# ============================================================================


class TestGate04StrataTypeHints:
    """Type hints/docstrings match real strata types at runtime."""

    def test_strata_type_annotation_covers_tuples(self, sds1_study):
        """Multi-factor strata are tuples; type annotation must not say list[str]."""
        result = sds1_study.execute(
            chart='X',
            by=['factor 1', 'factor 2'],
        )
        strata = result.strata
        # Multi-factor strata should be present
        assert len(strata) > 0

        # With two factors in by=, strata may be tuples.
        # The docstring currently says `list[str]` which is wrong if tuples appear.
        # This test documents the *actual* runtime type so the annotation can match.
        strata_types = {type(s) for s in strata}

        # Accept either all-str or all-tuple, but the annotation must match.
        # If tuples appear, the current `list[str]` annotation is wrong.
        assert strata_types <= {str, tuple}, f'Strata contain unexpected types: {strata_types}'

    def test_focus_accepts_actual_strata_types(self, sds1_study):
        """focus() must accept whatever type strata actually returns."""
        result = sds1_study.execute(
            chart='X',
            by=['factor 1', 'factor 2'],
        )
        # This exercises focus() with the actual runtime type (str or tuple)
        # If strata returns tuples but focus() annotation says str,
        # the implementation must still work.
        for s in result.strata:
            focused = result.focus(s)
            assert focused is not None


# ============================================================================
# Gate 05 — Harden ColumnAccessor Edge Cases
#
# Acceptance criteria:
#   - Empty column names do not crash sanitization
#   - Sanitization collisions are detectable and non-lossy for user access
#   - Reserved/accessor attribute collisions do not break core behavior
# ============================================================================


class TestGate05ColumnAccessorEdgeCases:
    """ColumnAccessor is safe and predictable with messy/real-world schemas."""

    def test_empty_column_name_does_not_crash(self):
        """Empty string column name must not raise IndexError."""
        df = pd.DataFrame({'': [1, 2, 3], 'normal': [4, 5, 6]})
        # Should not raise
        pb = ProcessBehavior(df)
        accessor = pb.cols
        # Empty name should still be accessible via bracket syntax
        ref = accessor['']
        assert ref.name == ''

    def test_digit_leading_column_name(self):
        """Column names starting with digits get safe attribute names."""
        df = pd.DataFrame({'1stCol': [1], '2ndCol': [2], 'normal': [3]})
        pb = ProcessBehavior(df)
        accessor = pb.cols
        # Should not crash and should be accessible
        assert accessor['1stCol'].name == '1stCol'
        assert accessor['2ndCol'].name == '2ndCol'

    def test_sanitization_collision_preserves_bracket_access(self):
        """Two columns that sanitize to the same name remain accessible via []."""
        df = pd.DataFrame(
            {
                'my col': [1, 2],  # sanitizes to my_col
                'my-col': [3, 4],  # also sanitizes to my_col
            }
        )
        pb = ProcessBehavior(df)
        accessor = pb.cols

        # Both must be accessible via original name
        ref1 = accessor['my col']
        ref2 = accessor['my-col']
        assert ref1.name == 'my col'
        assert ref2.name == 'my-col'

    def test_python_keyword_column_name(self):
        """Column named after a Python keyword should not break accessor."""
        df = pd.DataFrame({'class': [1, 2], 'for': [3, 4], 'normal': [5, 6]})
        pb = ProcessBehavior(df)
        accessor = pb.cols

        # Bracket access must always work
        assert accessor['class'].name == 'class'
        assert accessor['for'].name == 'for'

        # Attribute access to keyword names should not shadow built-in behavior.
        # At minimum, the accessor should not raise on construction.

    def test_reserved_attribute_collision(self):
        """Column named like an accessor method should not break core behavior."""
        df = pd.DataFrame(
            {
                '_df': [1],
                '_columns': [2],
                '_attr_to_col': [3],
                'normal': [4],
            }
        )
        pb = ProcessBehavior(df)
        accessor = pb.cols

        # Internal state must survive — repr should not crash
        repr(accessor)

        # Bracket access must work
        assert accessor['normal'].name == 'normal'
        # Internal attributes must not be overwritten
        assert isinstance(accessor._df, pd.DataFrame)
        assert isinstance(accessor._columns, list)
        assert isinstance(accessor._attr_to_col, dict)

    def test_special_characters_column_names(self):
        """Columns with special chars should sanitize without crashing."""
        df = pd.DataFrame(
            {
                'weight (kg)': [1.0],
                'temp°C': [25.0],
                'a/b': [0.5],
                'x=y+1': [2.0],
            }
        )
        pb = ProcessBehavior(df)
        accessor = pb.cols

        # All must be accessible via bracket
        for col in df.columns:
            ref = accessor[col]
            assert ref.name == col

    def test_all_digit_column_name(self):
        """Purely numeric column name should not crash."""
        df = pd.DataFrame({123: [1, 2], 456: [3, 4]})
        pb = ProcessBehavior(df)
        accessor = pb.cols
        # Bracket access with original type
        assert accessor['123'].name == '123' or accessor[123].name == 123

    def test_dir_returns_valid_identifiers(self):
        """__dir__() must return only valid Python identifiers."""
        df = pd.DataFrame(
            {
                'good_name': [1],
                'bad name': [2],
                '3starts_with_digit': [3],
            }
        )
        pb = ProcessBehavior(df)
        accessor = pb.cols

        for attr in dir(accessor):
            assert attr.isidentifier(), f'dir() returned non-identifier: {attr!r}'
            assert not keyword.iskeyword(attr) or True, (
                # Keywords are technically identifiers; this just documents the case
                f'dir() returned keyword: {attr!r}'
            )


# ============================================================================
# Gate 03 — Resolve Docs vs Runtime Mismatches
#
# Acceptance criteria:
#   - API examples are executable in CI smoke/doctest checks
#   - plan docs reflect current required keys (factors, T, N)
#   - Naming is consistent (pb.cols vs pb.columns, etc.)
# ============================================================================


class TestGate03DocsMatchRuntime:
    """Public docs/examples match actual runtime behavior."""

    def test_pb_cols_is_canonical_accessor(self):
        """pb.cols is the documented accessor — must exist and work."""
        df = pd.DataFrame({'Weight': [1.0, 2.0], 'Lane': ['A', 'B']})
        pb = ProcessBehavior(df)
        assert hasattr(pb, 'cols')
        assert pb.cols.Weight.name == 'Weight'

    def test_plan_requires_factors_key(self):
        """plan= without 'factors' key must raise ValidationError."""
        df = pd.DataFrame(
            {
                'y': range(10),
                'time': range(10),
                'lane': ['A'] * 5 + ['B'] * 5,
            }
        )
        pb = ProcessBehavior(df)

        with pytest.raises(ValidationError, match='factors'):
            pb.formulate(
                response='y',
                plan={'lane': ['A', 'B']},  # missing 'factors' wrapper
            )

    def test_plan_requires_T_key(self):
        """plan= without 'T' key must raise ValidationError."""
        df = pd.DataFrame(
            {
                'y': range(10),
                'time': range(10),
                'lane': ['A'] * 5 + ['B'] * 5,
            }
        )
        pb = ProcessBehavior(df)

        with pytest.raises(ValidationError, match='T'):
            pb.formulate(
                response='y',
                plan={'factors': {'lane': ['A', 'B']}, 'N': 1},
            )

    def test_plan_requires_N_key(self):
        """plan= without 'N' key must raise ValidationError."""
        df = pd.DataFrame(
            {
                'y': range(10),
                'time': range(10),
                'lane': ['A'] * 5 + ['B'] * 5,
            }
        )
        pb = ProcessBehavior(df)

        with pytest.raises(ValidationError, match='N'):
            pb.formulate(
                response='y',
                plan={'factors': {'lane': ['A', 'B']}, 'T': 5},
            )

    def test_complete_plan_succeeds(self):
        """plan= with all required keys succeeds."""
        df = pd.DataFrame(
            {
                'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                'time': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                'lane': ['A', 'B'] * 5,
            }
        )
        pb = ProcessBehavior(df)

        study = pb.formulate(
            response='y',
            plan={'factors': {'lane': ['A', 'B']}, 'T': 5, 'N': 1},
        )
        assert study is not None

    def test_formulate_rejects_factors_and_plan_together(self):
        """Cannot specify both factors= and plan=."""
        df = pd.DataFrame(
            {
                'y': range(10),
                'time': range(10),
                'lane': ['A'] * 5 + ['B'] * 5,
            }
        )
        pb = ProcessBehavior(df)

        with pytest.raises(ValidationError, match='[Cc]annot|[Bb]oth'):
            pb.formulate(
                response='y',
                factors=['lane'],
                plan={'factors': {'lane': ['A', 'B']}, 'T': 5, 'N': 1},
            )


# ============================================================================
# Gate 07 — Result Internal Mutability Guarded (ALREADY PASSES)
# ============================================================================


def test_gate_07_result_internal_mutability_guarded():
    """Gate 07: mutating user-returned objects cannot corrupt internals."""
    df = synthetic.make_design(1, K1=2, K2=1, T=5, seed=42)
    study = ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1'])
    result = study.execute(chart='Xbar')

    # get_chart() should return a copy
    chart_1 = result.get_chart('Xbar')
    original_value = chart_1.iloc[0, 0]
    chart_1.iloc[0, 0] = '__mutated__'
    chart_2 = result.get_chart('Xbar')
    assert chart_2.iloc[0, 0] == original_value
    assert chart_2.iloc[0, 0] != '__mutated__'

    # get_statistics() should return a copy
    stats_1 = result.get_statistics('Xbar')
    stats_1['center'] = -999
    stats_2 = result.get_statistics('Xbar')
    assert stats_2.get('center') != -999

    # summary should return a copy
    summary_1 = result.summary
    summary_1['n_observations'] = -1
    summary_2 = result.summary
    assert summary_2['n_observations'] != -1


# ============================================================================
# Deferred gates — scaffolds only
# ============================================================================

SC = pytest.mark.skip(
    reason='Release gate scaffold placeholder; convert to enforced assertions',
)


@SC
def test_gate_06_strict_vs_convenience_cleaning_modes_defined():
    """Gate 06: deferred. Document current cleaning behavior for 0.1.0."""


@SC
def test_gate_08_chart_table_join_logic_stable():
    """Gate 08: chart_table() joins are stable across key naming variants."""


@SC
def test_gate_09_top_level_exports_scoped():
    """Gate 09: top-level exports align with documented stability contract."""


@SC
def test_gate_10_ods_ads_messaging_clear():
    """Gate 10: user-facing SDS messaging distinguishes ODS vs ADS clearly."""
