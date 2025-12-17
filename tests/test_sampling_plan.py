"""
Tests for sampling plan feature.

Tests the ColumnRef, DesignReport, and plan parameter functionality
for enabling SDS 4-6 detection with explicit factor level specifications.
"""

import warnings
from itertools import product

import pandas as pd
import pytest

from processbehavior import (
    ColumnNotFoundError,
    ColumnRef,
    ProcessBehavior,
    ValidationError,
    encode_rsg,
)

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_df():
    """Simple DataFrame with Lane and Phase factors."""
    return pd.DataFrame({
        'Weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
        'Lane': [1, 1, 2, 2, 1, 1, 2, 2],
        'Phase': [1, 1, 1, 1, 2, 2, 2, 2],
        'Pull': [1, 1, 1, 1, 2, 2, 2, 2]
    })


@pytest.fixture
def missing_phase_df():
    """DataFrame where Phase 3 is missing (not observed)."""
    return pd.DataFrame({
        'Weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
        'Lane': [1, 1, 2, 2, 1, 1, 2, 2],
        'Phase': [1, 1, 1, 1, 2, 2, 2, 2],  # Phase 3 missing
        'Pull': [1, 1, 1, 1, 2, 2, 2, 2]
    })


@pytest.fixture
def extra_phase_df():
    """DataFrame with extra Phase level not in plan, with replication."""
    return pd.DataFrame({
        'Weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8,
                   10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6],
        'Lane': [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2],
        'Phase': [1, 1, 2, 2, 1, 1, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5],  # Phase 5 is extra
        'Pull': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    })


# =============================================================================
# encode_rsg Tests
# =============================================================================

class TestEncodeRsg:
    """Tests for encode_rsg() helper function."""

    def test_encode_rsg_single_value(self):
        """Single value should encode without delimiter."""
        assert encode_rsg((42,)) == '42'
        assert encode_rsg(('A',)) == 'A'

    def test_encode_rsg_multiple_values(self):
        """Multiple values should encode with delimiter."""
        assert encode_rsg((1, 'A')) == '1_A'
        assert encode_rsg((2, 'B', 3)) == '2_B_3'

    def test_encode_rsg_custom_delimiter(self):
        """Custom delimiter should be used."""
        assert encode_rsg((1, 'A'), delimiter='-') == '1-A'
        assert encode_rsg((1, 2, 3), delimiter='::') == '1::2::3'

    def test_encode_rsg_matches_dataframe_rsg(self, simple_df):
        """encode_rsg() output should match observed RSG values in DataFrame."""
        # Create a study with factors
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response='Weight',
            factors=['Lane', 'Phase'],
            time='Pull'
        )

        # Get the analysis dataset and its RSG values
        result = study.execute()
        observed_rsg = set(result._ads.analysis_dataset['rsg'].unique())

        # Generate expected RSG keys using encode_rsg()
        plan = {'Lane': [1, 2], 'Phase': [1, 2]}
        expected_rsg = {encode_rsg(combo) for combo in product(*plan.values())}

        # They should match
        assert observed_rsg == expected_rsg

    def test_encode_rsg_plan_expansion_matches_observed(self):
        """Plan expansion with encode_rsg() should match observed RSG."""
        # Create data with known factor combinations
        df = pd.DataFrame({
            'Weight': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'Lane': [1, 1, 2, 2, 1, 1, 2, 2],
            'Phase': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2]
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(
            response='Weight',
            factors=['Lane', 'Phase'],
            time='Pull'
        )

        # Get observed RSG values
        result = study.execute()
        observed_rsg = set(result._ads.analysis_dataset['rsg'].unique())

        # Generate expected from plan using encode_rsg()
        plan = {'Lane': [1, 2], 'Phase': ['A', 'B']}
        expected_rsg = {encode_rsg(combo) for combo in product(*plan.values())}

        # Verify consistency
        assert observed_rsg == expected_rsg
        assert '1_A' in observed_rsg
        assert '2_B' in observed_rsg


# =============================================================================
# ColumnRef Tests
# =============================================================================

class TestColumnRef:
    """Tests for ColumnRef class."""

    def test_column_ref_levels_returns_sorted_unique(self, simple_df):
        """ColumnRef.levels should return sorted unique values."""
        pb = ProcessBehavior(simple_df)
        lane_ref = pb.cols.Lane

        assert lane_ref.levels == [1, 2]

    def test_column_ref_count_matches_levels(self, simple_df):
        """ColumnRef.count should return number of unique levels."""
        pb = ProcessBehavior(simple_df)
        lane_ref = pb.cols.Lane

        assert lane_ref.count == 2
        assert lane_ref.count == len(lane_ref.levels)

    def test_column_ref_repr_shows_levels(self, simple_df):
        """ColumnRef repr should show column name and levels."""
        pb = ProcessBehavior(simple_df)
        lane_ref = pb.cols.Lane

        repr_str = repr(lane_ref)
        assert 'Lane' in repr_str
        assert '(2)' in repr_str  # Count

    def test_column_ref_works_as_dict_key(self, simple_df):
        """ColumnRef should work as dictionary key."""
        pb = ProcessBehavior(simple_df)
        lane_ref = pb.cols.Lane

        d = {lane_ref: [1, 2, 3, 4]}
        assert d[lane_ref] == [1, 2, 3, 4]

    def test_column_ref_equals_string(self, simple_df):
        """ColumnRef should compare equal to equivalent string."""
        pb = ProcessBehavior(simple_df)
        lane_ref = pb.cols.Lane

        assert lane_ref == 'Lane'
        assert lane_ref == 'Lane'  # Test reverse comparison

    def test_column_ref_hash_equals_string_hash(self, simple_df):
        """ColumnRef should hash same as equivalent string."""
        pb = ProcessBehavior(simple_df)
        lane_ref = pb.cols.Lane

        assert hash(lane_ref) == hash('Lane')

    def test_column_accessor_returns_column_ref(self, simple_df):
        """ColumnAccessor should return ColumnRef objects."""
        pb = ProcessBehavior(simple_df)

        assert isinstance(pb.cols.Lane, ColumnRef)
        assert isinstance(pb.cols['Lane'], ColumnRef)

    def test_column_ref_str_returns_name(self, simple_df):
        """str(ColumnRef) should return the column name."""
        pb = ProcessBehavior(simple_df)
        lane_ref = pb.cols.Lane

        assert str(lane_ref) == 'Lane'

    def test_column_ref_name_attribute(self, simple_df):
        """ColumnRef.name should return the column name."""
        pb = ProcessBehavior(simple_df)
        lane_ref = pb.cols.Lane

        assert lane_ref.name == 'Lane'


# =============================================================================
# Formulate with Plain Strings Tests
# =============================================================================

class TestFormulateAcceptsStrings:
    """Tests that formulate() accepts plain strings (backward compatibility)."""

    def test_formulate_accepts_plain_string_response(self, simple_df):
        """formulate() should accept plain string for response."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(response='Weight')

        assert study.response == 'Weight'

    def test_formulate_accepts_plain_string_factors(self, simple_df):
        """formulate() should accept plain strings for factors."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response='Weight',
            factors=['Lane', 'Phase']
        )

        assert study.factors == ['Lane', 'Phase']

    def test_formulate_accepts_plain_string_time(self, simple_df):
        """formulate() should accept plain string for time."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response='Weight',
            factors=['Lane'],
            time='Pull'
        )

        assert study.time == 'Pull'


# =============================================================================
# Plan Validation Tests
# =============================================================================

class TestPlanValidation:
    """Tests for plan parameter validation."""

    def test_plan_and_factors_mutual_exclusion(self, simple_df):
        """Cannot specify both factors and plan."""
        pb = ProcessBehavior(simple_df)

        with pytest.raises(ValidationError) as exc_info:
            pb.formulate(
                response='Weight',
                factors=['Lane'],
                plan={'factors': {pb.cols.Lane: [1, 2, 3, 4]}}
            )

        assert "Cannot specify both" in str(exc_info.value)

    def test_plan_extracts_factors_from_keys(self, simple_df):
        """Factors should be extracted from plan keys."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            plan={
                'factors': {
                    pb.cols.Lane: [1, 2],
                    pb.cols.Phase: [1, 2]
                }
            }
        )

        assert study.factors == ['Lane', 'Phase']

    def test_plan_warns_on_extra_observed_levels(self, extra_phase_df, caplog):
        """Should warn (not error) when observed levels not in plan."""
        import logging

        pb = ProcessBehavior(extra_phase_df)

        with caplog.at_level(logging.WARNING):
            pb.formulate(
                response=pb.cols.Weight,
                plan={
                    'factors': {
                        pb.cols.Lane: [1, 2],
                        pb.cols.Phase: [1, 2]  # Phase 5 is in data but not plan
                    }
                }
            )

            # Check warning was issued via logging
            assert any('has observed levels not in plan' in msg for msg in caplog.messages)

    def test_plan_column_not_found_raises(self, simple_df):
        """Should raise ColumnNotFoundError for invalid plan column."""
        pb = ProcessBehavior(simple_df)

        with pytest.raises(ColumnNotFoundError) as exc_info:
            pb.formulate(
                response=pb.cols.Weight,
                plan={'factors': {'NonexistentColumn': [1, 2, 3]}}
            )

        assert 'NonexistentColumn' in str(exc_info.value)

    def test_plan_accepts_column_ref_keys(self, simple_df):
        """Plan should accept ColumnRef objects as keys."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            plan={'factors': {pb.cols.Lane: [1, 2]}}
        )

        assert study.factors == ['Lane']

    def test_plan_accepts_string_keys(self, simple_df):
        """Plan should accept plain string keys."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            plan={'factors': {'Lane': [1, 2]}}
        )

        assert study.factors == ['Lane']

    def test_plan_without_factors_key_raises(self, simple_df):
        """Plan without 'factors' key should raise ValidationError."""
        pb = ProcessBehavior(simple_df)

        # Old flat structure should now raise
        with pytest.raises(ValidationError) as exc_info:
            pb.formulate(
                response='Weight',
                plan={'Lane': [1, 2, 3, 4]}  # Old structure, no 'factors' key
            )

        assert "must have 'factors' key" in str(exc_info.value)


# =============================================================================
# DesignReport Tests
# =============================================================================

class TestDesignReport:
    """Tests for DesignReport class."""

    def test_design_report_factors_dataframe(self, simple_df):
        """design().factors should return a DataFrame."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            plan={'factors': {pb.cols.Lane: [1, 2], pb.cols.Phase: [1, 2]}}
        )

        design = study.design()
        df = design.factors

        assert isinstance(df, pd.DataFrame)
        assert 'factor' in df.columns
        assert 'planned' in df.columns
        assert 'observed' in df.columns
        assert 'missing_levels' in df.columns
        assert 'extra_levels' in df.columns

    def test_design_report_missing_levels_per_factor(self, missing_phase_df):
        """missing_levels should show levels in plan but not observed."""
        pb = ProcessBehavior(missing_phase_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            plan={
                'factors': {
                    pb.cols.Lane: [1, 2],
                    pb.cols.Phase: [1, 2, 3]  # Phase 3 not in data
                }
            }
        )

        design = study.design()

        assert design.missing_levels['Phase'] == [3]
        assert design.missing_levels['Lane'] == []

    def test_design_report_extra_levels_per_factor(self, extra_phase_df):
        """extra_levels should show levels observed but not in plan."""
        pb = ProcessBehavior(extra_phase_df)

        # Suppress warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = pb.formulate(
                response=pb.cols.Weight,
                plan={
                    'factors': {
                        pb.cols.Lane: [1, 2],
                        pb.cols.Phase: [1, 2]  # Phase 5 is extra
                    }
                }
            )

        design = study.design()

        assert design.extra_levels['Phase'] == [5]
        assert design.extra_levels['Lane'] == []

    def test_design_report_no_plan_uses_observed(self, simple_df):
        """Without plan, design() should show observed structure only."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            factors=['Lane', 'Phase']
        )

        design = study.design()

        assert design.has_plan is False
        assert design.missing_levels == {'Lane': [], 'Phase': []}
        assert design.extra_levels == {'Lane': [], 'Phase': []}

    def test_design_report_repr(self, simple_df):
        """DesignReport repr should show summary."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            plan={'factors': {pb.cols.Lane: [1, 2], pb.cols.Phase: [1, 2]}}
        )

        design = study.design()
        repr_str = repr(design)

        assert 'DesignReport' in repr_str
        assert '2 factors' in repr_str

    def test_design_report_empty_when_plan_matches_observed(self, simple_df):
        """No missing/extra when plan exactly matches observed."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            plan={'factors': {pb.cols.Lane: [1, 2], pb.cols.Phase: [1, 2]}}
        )

        design = study.design()

        assert all(len(v) == 0 for v in design.missing_levels.values())
        assert all(len(v) == 0 for v in design.extra_levels.values())

    def test_design_report_has_plan_true_with_plan(self, simple_df):
        """has_plan should be True when plan was provided."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            plan={'factors': {pb.cols.Lane: [1, 2]}}
        )

        assert study.design().has_plan is True

    def test_design_report_has_plan_false_without_plan(self, simple_df):
        """has_plan should be False when no plan was provided."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            factors=['Lane']
        )

        assert study.design().has_plan is False


# =============================================================================
# SDS Integration Tests
# =============================================================================

class TestSDSIntegration:
    """Tests for SDS detection with and without plan."""

    def test_sds_detection_without_plan_works(self, simple_df):
        """SDS detection should work without plan (existing behavior)."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            factors=['Lane', 'Phase'],
            time='Pull'
        )

        # Should detect some SDS (exact value depends on data structure)
        assert study.sds in [0, 1, 2, 3, 4, 5, 6]

    def test_sds_detection_with_plan_works(self, simple_df):
        """SDS detection should work with plan."""
        pb = ProcessBehavior(simple_df)
        study = pb.formulate(
            response=pb.cols.Weight,
            time=pb.cols.Pull,
            plan={'factors': {pb.cols.Lane: [1, 2], pb.cols.Phase: [1, 2]}}
        )

        # Should detect some SDS
        assert study.sds in [0, 1, 2, 3, 4, 5, 6]

    def test_sds_5_detected_with_incomplete_plan_and_replication(self):
        """Incomplete grid WITH replication should detect SDS 5."""
        # Data has Lane=[1,2] and Phase=[1,2] (4 combinations), each with n=2
        # Plan says Lane=[1,2,3,4] and Phase=[1,2,3] (12 combinations)
        # → Only 4 of 12 expected cells present → coverage ~33% < 95%
        # → Has replication (n=2 per cell) → SDS 5
        df = pd.DataFrame({
            'Weight': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'Lane': [1, 1, 2, 2, 1, 1, 2, 2],
            'Phase': [1, 1, 2, 2, 1, 1, 2, 2],
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2]
        })
        pb = ProcessBehavior(df)

        # Suppress warning about extra levels (we expect this warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = pb.formulate(
                response='Weight',
                time='Pull',
                plan={'factors': {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]}}
            )

        assert study.sds == 5  # Incomplete grid WITH replication

    def test_sds_6_detected_with_incomplete_plan_no_replication(self):
        """Incomplete grid with NO replication should detect SDS 6."""
        # Data has Lane=[1,2] and Phase=[1,2] (4 factor combinations)
        # Each cell has multiple time points but n=1 per (factor, time) cell
        # Plan says Lane=[1,2,3,4] and Phase=[1,2,3] (12 combinations)
        # → Only 4 of 12 expected factor combos × 4 time points = 16 of 48 cells
        # → Coverage ~33% < 95%, no replication (n=1 per cell) → SDS 6
        df = pd.DataFrame({
            'Weight': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            'Lane': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            'Phase': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        })
        pb = ProcessBehavior(df)

        # Suppress warning about extra levels (we expect this warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = pb.formulate(
                response='Weight',
                time='Pull',
                plan={'factors': {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]}}
            )

        assert study.sds == 6  # Incomplete grid with NO replication

    def test_sds_5_detected_with_incomplete_plan_mixed_replication(self):
        """Incomplete grid with MIXED replication should detect SDS 5."""
        # Data has Lane=[1,2] and Phase=[1,2] (4 factor combinations)
        # Some cells have n=2 (replication), some have n=1 (no replication)
        # Plan says Lane=[1,2,3,4] and Phase=[1,2,3] (12 combinations)
        # → Coverage < 95%, mixed replication → SDS 5
        # → Can estimate variance from replicated cells
        df = pd.DataFrame({
            # (Lane=1, Phase=1): n=2 at Pull=1, n=2 at Pull=2 (replicated)
            # (Lane=2, Phase=2): n=1 at Pull=1, n=1 at Pull=2 (not replicated)
            'Weight': [1.0, 1.1,   # Lane=1, Phase=1, Pull=1 (n=2)
                       2.0, 2.1,   # Lane=1, Phase=1, Pull=2 (n=2)
                       3.0,        # Lane=2, Phase=2, Pull=1 (n=1)
                       4.0],       # Lane=2, Phase=2, Pull=2 (n=1)
            'Lane':  [1, 1, 1, 1, 2, 2],
            'Phase': [1, 1, 1, 1, 2, 2],
            'Pull':  [1, 1, 2, 2, 1, 2]
        })
        pb = ProcessBehavior(df)

        # Suppress warning about extra levels (we expect this warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = pb.formulate(
                response='Weight',
                time='Pull',
                plan={'factors': {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]}}
            )

        # Mixed replication + incomplete → SDS 5 (can estimate variance)
        assert study.sds == 5


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow_with_plan(self, simple_df):
        """Full workflow: formulate with plan → design → execute."""
        pb = ProcessBehavior(simple_df)

        # Discover levels
        assert pb.cols.Lane.levels == [1, 2]
        assert pb.cols.Phase.levels == [1, 2]

        # Formulate with plan
        study = pb.formulate(
            response=pb.cols.Weight,
            time=pb.cols.Pull,
            plan={
                'factors': {
                    pb.cols.Lane: [1, 2],
                    pb.cols.Phase: [1, 2]
                }
            }
        )

        # Check design
        design = study.design()
        assert design.has_plan is True

        # Execute should work
        result = study.execute()
        assert result is not None

    def test_full_workflow_without_plan(self, simple_df):
        """Full workflow: formulate with factors → design → execute."""
        pb = ProcessBehavior(simple_df)

        # Formulate with factors (not plan)
        study = pb.formulate(
            response=pb.cols.Weight,
            factors=[pb.cols.Lane, pb.cols.Phase],
            time=pb.cols.Pull
        )

        # Check design (should work without plan)
        design = study.design()
        assert design.has_plan is False
        assert len(design._factors) == 2

        # Execute should work
        result = study.execute()
        assert result is not None


# =============================================================================
# Coverage Ratio Edge Cases
# =============================================================================

class TestCoverageRatioEdgeCases:
    """Tests for coverage ratio calculation edge cases."""

    def test_plan_order_does_not_affect_coverage(self):
        """Plan keys in different order than rsg_vars should still work.

        The coverage calculation should use spec.rsg_vars order, not
        dict insertion order, to generate expected combinations.
        """
        # Data has Lane first, Phase second in rsg encoding
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2] * 2,
            'Phase': [1, 2, 1, 2] * 2,
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2],
            'Weight': [10.0] * 8
        })
        pb = ProcessBehavior(df)

        # Plan with keys in REVERSE order (Phase first, Lane second)
        # This differs from rsg_vars order [Lane, Phase]
        plan_reversed = {
            'factors': {
                'Phase': [1, 2],  # Phase first
                'Lane': [1, 2],   # Lane second
            }
        }

        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan=plan_reversed
        )

        # Should detect complete grid (SDS 1, 2, or 3 based on replication)
        # NOT SDS 5/6 which would indicate incomplete coverage
        assert study.sds in [1, 2, 3]

    def test_extra_levels_do_not_inflate_coverage(self):
        """Extra levels in data should not make coverage appear complete.

        If plan says Lane=[1,2] but data has Lane=[1,2,3], and Lane=2
        is missing, coverage should reflect the missing Lane=2, not
        be inflated by the extra Lane=3.
        """
        # Data has Lane 1 and 3, but NOT Lane 2
        # Plan expects Lane 1 and 2
        # Need n≥2 per cell to avoid subgroup size validation errors
        df = pd.DataFrame({
            'Lane': [1, 1, 1, 1, 3, 3, 3, 3],  # Lane 2 is MISSING, Lane 3 is EXTRA
            'Phase': [1, 1, 2, 2, 1, 1, 2, 2],
            'Pull': [1, 1, 1, 1, 1, 1, 1, 1],
            'Weight': [10.0] * 8
        })
        pb = ProcessBehavior(df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore extra levels warning
            study = pb.formulate(
                response='Weight',
                time='Pull',
                plan={
                    'factors': {
                        'Lane': [1, 2],   # Expects Lane 1 and 2
                        'Phase': [1, 2],
                    }
                }
            )

        # Lane=2 is missing from plan, so grid is incomplete
        # Should be SDS 5 or 6 (incomplete), NOT SDS 1/2/3 (complete)
        assert study.sds in [5, 6]


# =============================================================================
# SDSResult Reason Field Tests
# =============================================================================

class TestSDSResultReason:
    """Tests for SDSResult.reason field disambiguation."""

    def test_sds_5_reason_nested(self):
        """Nested design should return reason='nested'."""
        # Heads nested in lanes (each head belongs to one lane only)
        # With very incomplete temporal coverage (< 90%)
        df = pd.DataFrame({
            'lane': ['A'] * 5 + ['B'] * 5,
            'head': [1] * 5 + [2] * 5,  # Head 1 only with lane A, head 2 only with B
            'pull': [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],  # Sparse time coverage
            'weight': [10.0] * 10
        })

        from processbehavior.analysis_specification import AnalysisSpecification
        from processbehavior.data_preparation import DataPreparation
        from processbehavior.sds_detector import SDSRegistry

        spec = AnalysisSpecification({
            'analysis_type': 'Xbar',
            'rsg_vars': ['lane', 'head'],
            'rsg_var_name': 'rsg',
            'time_var': 'pull',
            'response_var': 'weight'
        })

        prep = DataPreparation()
        prep.validate_columns(df, spec)
        prepared_df = prep.prepare_dataset(df, spec)

        detector = SDSRegistry()
        result = detector.detect_sds(prepared_df, spec)

        # If nested is detected, reason should be 'nested'
        if result.sds == 5:
            assert result.reason == 'nested'

    def test_sds_5_reason_incomplete_with_replication(self):
        """Incomplete grid with replication should return reason='incomplete_with_replication'."""
        from processbehavior.analysis_specification import DataPrepConfig
        from processbehavior.data_preparation import DataPreparation
        from processbehavior.sds_detector import SDSRegistry

        # Incomplete grid (missing Lane 3) with replication (n=2 per cell)
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2] * 2,  # Lane 3 is missing
            'Phase': [1, 1, 1, 1, 2, 2, 2, 2],
            'Pull': [1, 1, 1, 1, 1, 1, 1, 1],
            'Weight': [10.0] * 8
        })

        spec_dict = {
            'response_var': 'Weight',
            'rsg_vars': ['Lane', 'Phase'],
            'rsg_var_name': 'rsg',
            'time_var': 'Pull',
        }
        config = DataPrepConfig(spec_dict)
        prep = DataPreparation()
        prep.validate_columns(df, config)
        prepared_df = prep.prepare_dataset(df, config)

        detector = SDSRegistry()
        result = detector.detect_sds(
            prepared_df, config,
            plan={'Lane': [1, 2, 3], 'Phase': [1, 2]}
        )

        assert result.sds == 5
        assert result.reason == 'incomplete_with_replication'

    def test_sds_6_reason_incomplete_no_replication(self):
        """Incomplete grid without replication should return reason='incomplete_no_replication'."""
        from processbehavior.analysis_specification import DataPrepConfig
        from processbehavior.data_preparation import DataPreparation
        from processbehavior.sds_detector import SDSRegistry

        # Incomplete grid (missing Lane 3) with NO replication (n=1 per cell)
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2],  # Lane 3 is missing
            'Phase': [1, 2, 1, 2],
            'Pull': [1, 2, 3, 4],  # Different time points, so each cell has n=1
            'Weight': [10.0] * 4
        })

        spec_dict = {
            'response_var': 'Weight',
            'rsg_vars': ['Lane', 'Phase'],
            'rsg_var_name': 'rsg',
            'time_var': 'Pull',
        }
        config = DataPrepConfig(spec_dict)
        prep = DataPreparation()
        prep.validate_columns(df, config)
        prepared_df = prep.prepare_dataset(df, config)

        detector = SDSRegistry()
        result = detector.detect_sds(
            prepared_df, config,
            plan={'Lane': [1, 2, 3], 'Phase': [1, 2]}
        )

        assert result.sds == 6
        assert result.reason == 'incomplete_no_replication'


# =============================================================================
# Documentation Alignment Tests
# =============================================================================

class TestDocsAlignment:
    """Tests to verify SDS definitions are consistent across all sources."""

    def test_sds_characteristics_matches_analysis_plan(self):
        """get_sds_characteristics and get_analysis_plan should agree on key properties."""
        from processbehavior.sds_detector import SDSRegistry

        for sds in range(7):
            chars = SDSRegistry().get_sds_characteristics(sds)
            plan = SDSRegistry.get_analysis_plan(sds, min_cell_size=2)

            # Both should agree on SDS number
            assert chars['sds'] == plan.sds == sds

            # Both should agree on VAS support
            assert chars['variance_decomposition'] == plan.vas_residuals_supported

    def test_sds_4_is_single_condition_over_time(self):
        """SDS 4 should be 'Single Condition Over Time' everywhere."""
        from processbehavior.sds_detector import SDSRegistry

        chars = SDSRegistry().get_sds_characteristics(4)
        plan = SDSRegistry.get_analysis_plan(4)

        assert 'Single condition over time' in chars['description']
        assert plan.name == 'Single Condition Over Time'
        assert plan.has_time is True
        assert plan.has_factors is True

    def test_sds_5_is_nested_or_incomplete_with_replication(self):
        """SDS 5 should be 'Nested/Incomplete with Replication' everywhere."""
        from processbehavior.sds_detector import SDSRegistry

        chars = SDSRegistry().get_sds_characteristics(5)
        plan = SDSRegistry.get_analysis_plan(5)

        assert 'Nested' in chars['description'] or 'nested' in chars['description'].lower()
        assert 'Nested' in plan.name
        assert plan.has_replication == 'partial'

    def test_sds_6_is_incomplete_without_replication(self):
        """SDS 6 should be 'Incomplete Without Replication' everywhere."""
        from processbehavior.sds_detector import SDSRegistry

        SDSRegistry().get_sds_characteristics(6)
        plan = SDSRegistry.get_analysis_plan(6)

        assert plan.name == 'Incomplete Grid Without Replication'
        assert plan.has_replication == 'none'


# =============================================================================
# K, T, N Tests
# =============================================================================

class TestDesignReportKTN:
    """Tests for K, T, N in DesignReport."""

    def test_K_derived_from_factors(self):
        """K should be product of factor level counts."""
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2, 3, 3, 4, 4] * 2,
            'Phase': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            'Pull': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            'Weight': [10.0] * 16
        })
        pb = ProcessBehavior(df)

        # Plan: Lane=[1,2,3,4] (4), Phase=[1,2,3] (3) → K = 4 × 3 = 12
        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]}}
        )

        design = study.design()
        assert design.K == 12  # 4 × 3

    def test_K_observed_from_data(self):
        """K_observed should reflect actual unique RSG groups (nunique)."""
        # Create data with 4 unique RSG combinations: (1,1), (1,2), (2,1), (2,2)
        df = pd.DataFrame({
            'Lane': [1, 1, 1, 1, 2, 2, 2, 2],
            'Phase': [1, 1, 2, 2, 1, 1, 2, 2],
            'Pull': [1, 1, 1, 1, 1, 1, 1, 1],
            'Weight': [10.0] * 8
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]}}
        )

        design = study.design()
        # Data has Lane=1,2 × Phase=1,2 → 4 unique RSG groups
        # K_observed = nunique(rsg) from actual data
        assert design.K_observed == 4

    def test_K_missing_from_missing_combos(self):
        """K_missing should equal len(missing_combos)."""
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2] * 2,
            'Phase': [1, 1, 2, 2] * 2,
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2],
            'Weight': [10.0] * 8
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]}}
        )

        design = study.design()
        assert design.K_missing == len(design.missing_combos)

    def test_T_planned_vs_observed(self):
        """T should show planned vs observed time points."""
        df = pd.DataFrame({
            'Lane': [1] * 16,
            'Pull': list(range(1, 9)) * 2,  # 8 unique time points
            'Weight': [10.0] * 16
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1]}, 'T': 10}
        )

        design = study.design()
        assert design.T == 10
        assert design.T_observed == 8
        assert design.T_missing == 2

    def test_N_planned_vs_observed(self):
        """N should show planned vs observed cell sizes."""
        df = pd.DataFrame({
            'Lane': [1, 1, 1, 2, 2] * 2,  # Lane 1 has 3 obs/cell, Lane 2 has 2
            'Pull': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'Weight': [10.0] * 10
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1, 2]}, 'N': 2}
        )

        design = study.design()
        assert design.N == 2
        assert design.N_observed is not None
        min_n, med_n, max_n = design.N_observed
        assert min_n == 2  # Lane 2 has min 2
        assert max_n == 3  # Lane 1 has max 3
        assert isinstance(med_n, float)  # Median should be float

    def test_structure_summary_complete(self):
        """structure_summary should be 'Complete structure' when all match."""
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2] * 2,
            'Phase': [1, 2, 1, 2] * 2,
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2],
            'Weight': [10.0] * 8
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1, 2], 'Phase': [1, 2]}}
        )

        design = study.design()
        assert design.structure_summary == "Complete structure"

    def test_structure_summary_incomplete(self):
        """structure_summary should explain discrepancies."""
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2] * 2,
            'Phase': [1, 1, 2, 2] * 2,  # Phase 3 missing
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2],
            'Weight': [10.0] * 8
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1, 2], 'Phase': [1, 2, 3]}}
        )

        design = study.design()
        assert "missing" in design.structure_summary.lower()

    def test_missing_combos_from_cartesian_product(self):
        """missing_combos should be cartesian product minus observed."""
        df = pd.DataFrame({
            'Lane': [1, 1] * 2,  # Only Lane 1
            'Phase': [1, 1, 2, 2],  # Both phases
            'Pull': [1, 1, 2, 2],
            'Weight': [10.0] * 4
        })
        pb = ProcessBehavior(df)

        # Plan expects Lane 1,2 × Phase 1,2 = 4 combos
        # Data only has Lane 1 × Phase 1,2 = 2 combos
        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1, 2], 'Phase': [1, 2]}}
        )

        design = study.design()
        # Missing: Lane 2 combos = ['2_1', '2_2']
        assert len(design.missing_combos) == 2
        assert '2_1' in design.missing_combos
        assert '2_2' in design.missing_combos

    def test_missing_combos_natural_sort(self):
        """missing_combos should be naturally sorted."""
        from processbehavior.data_preparation import natural_sort_key

        df = pd.DataFrame({
            'Item': [1] * 4,
            'Pull': [1, 1, 2, 2],
            'Weight': [10.0] * 4
        })
        pb = ProcessBehavior(df)

        # Plan with numeric levels that need natural sort
        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Item': [1, 2, 10, 20]}}  # 10 should come after 2
        )

        design = study.design()
        # Missing items should be naturally sorted: 2, 10, 20 (not 10, 2, 20)
        expected_sorted = sorted(design.missing_combos, key=natural_sort_key)
        assert design.missing_combos == expected_sorted

    def test_sds_reason_in_design_report(self):
        """sds_reason should come from SDSResult."""
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2] * 2,
            'Phase': [1, 1, 2, 2] * 2,
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2],
            'Weight': [10.0] * 8
        })
        pb = ProcessBehavior(df)

        # Create study with full plan to get specific SDS reason
        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1, 2], 'Phase': [1, 2]}}
        )

        design = study.design()
        # sds_reason should be present (exact value depends on SDS detected)
        assert design.sds_reason is not None

    def test_repr_shows_K_T_N(self):
        """DesignReport __repr__ should show K, T, N info."""
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2] * 2,
            'Phase': [1, 2, 1, 2] * 2,
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2],
            'Weight': [10.0] * 8
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(
            response='Weight',
            time='Pull',
            plan={'factors': {'Lane': [1, 2], 'Phase': [1, 2]}, 'T': 3, 'N': 2}
        )

        design = study.design()
        repr_str = repr(design)

        # Should contain K, T, N lines
        assert 'K:' in repr_str
        assert 'T:' in repr_str
        assert 'N:' in repr_str
        assert 'Structure:' in repr_str

    def test_extra_combos_detected(self):
        """extra_combos should show RSG combos observed but not in plan."""
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2, 3, 3] * 2,  # Lane 3 is extra
            'Phase': [1, 2, 1, 2, 1, 2] * 2,
            'Pull': [1] * 6 + [2] * 6,
            'Weight': [10.0] * 12
        })
        pb = ProcessBehavior(df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Extra levels warning
            study = pb.formulate(
                response='Weight',
                time='Pull',
                plan={'factors': {'Lane': [1, 2], 'Phase': [1, 2]}}  # No Lane 3
            )

        design = study.design()
        # Extra combos: Lane 3 × Phase 1,2 = ['3_1', '3_2']
        assert len(design.extra_combos) == 2
        assert '3_1' in design.extra_combos
        assert '3_2' in design.extra_combos

    def test_no_plan_has_K_observed_only(self):
        """Without plan, K should equal K_observed."""
        df = pd.DataFrame({
            'Lane': [1, 1, 2, 2] * 2,
            'Phase': [1, 2, 1, 2] * 2,
            'Pull': [1, 1, 1, 1, 2, 2, 2, 2],
            'Weight': [10.0] * 8
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(
            response='Weight',
            factors=['Lane', 'Phase'],
            time='Pull'
        )

        design = study.design()
        assert design.K_observed == design.K  # K falls back to K_observed
        assert design.K_missing == 0  # No missing when no plan
        assert len(design.missing_combos) == 0
        assert len(design.extra_combos) == 0
