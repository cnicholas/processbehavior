"""
Tests for garbage character and NA value handling in ProcessBehavior.

These tests verify that real-world messy data with non-standard NA indicators
is properly cleaned and handled gracefully, including financial/monetary
formatted strings (currency symbols, thousands separators, accounting negatives).
"""

from pathlib import Path

import pandas as pd
import pytest

from processbehavior import ProcessBehavior


class TestDefaultGarbageCharacterHandling:
    """Test automatic handling of common garbage characters."""

    def test_asterisk_treated_as_na(self):
        """Test that '*' is automatically converted to NA."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'Y': [235.5, '*', 237.2, 239.1]
        })

        pdf = ProcessBehavior(df)

        # Asterisk should be converted to NA
        assert pd.isna(pdf.data['Y'].iloc[1])
        # Other values should remain
        assert pdf.data['Y'].iloc[0] == 235.5
        assert pdf.data['Y'].iloc[2] == 237.2

    def test_question_mark_treated_as_na(self):
        """Test that '?' is automatically converted to NA."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3],
            'Y': [235.5, '?', 237.2]
        })

        pdf = ProcessBehavior(df)
        assert pd.isna(pdf.data['Y'].iloc[1])

    def test_double_dash_treated_as_na(self):
        """Test that '--' is automatically converted to NA."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3],
            'FACTOR': ['A', '--', 'C']
        })

        pdf = ProcessBehavior(df)
        assert pd.isna(pdf.data['FACTOR'].iloc[1])

    def test_nd_treated_as_na(self):
        """Test that 'ND' (Not Detected) is automatically converted to NA."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3],
            'Y': [235.5, 'ND', 237.2]
        })

        pdf = ProcessBehavior(df)
        assert pd.isna(pdf.data['Y'].iloc[1])

    def test_bdl_treated_as_na(self):
        """Test that 'BDL' (Below Detection Limit) is automatically converted to NA."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3],
            'Y': [235.5, 'BDL', 237.2]
        })

        pdf = ProcessBehavior(df)
        assert pd.isna(pdf.data['Y'].iloc[1])

    def test_bql_treated_as_na(self):
        """Test that 'BQL' (Below Quantification Limit) is automatically converted to NA."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3],
            'Y': [235.5, 'BQL', 237.2]
        })

        pdf = ProcessBehavior(df)
        assert pd.isna(pdf.data['Y'].iloc[1])

    def test_lod_treated_as_na(self):
        """Test that '<LOD' (Below Limit of Detection) is automatically converted to NA."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3],
            'Y': [235.5, '<LOD', 237.2]
        })

        pdf = ProcessBehavior(df)
        assert pd.isna(pdf.data['Y'].iloc[1])

    def test_multiple_garbage_characters_in_same_column(self):
        """Test handling multiple types of garbage in one column."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4, 5, 6],
            'Y': [235.5, '*', '?', 'ND', 'BDL', 237.2]
        })

        pdf = ProcessBehavior(df)

        # All garbage should be NA
        assert pd.isna(pdf.data['Y'].iloc[1])
        assert pd.isna(pdf.data['Y'].iloc[2])
        assert pd.isna(pdf.data['Y'].iloc[3])
        assert pd.isna(pdf.data['Y'].iloc[4])
        # Valid values should remain
        assert pdf.data['Y'].iloc[0] == 235.5
        assert pdf.data['Y'].iloc[5] == 237.2

    def test_garbage_characters_across_multiple_columns(self):
        """Test garbage character handling across multiple columns."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'Y': [235.5, '*', 237.2, 239.1],
            'FACTOR': ['A', 'B', '--', 'D']
        })

        pdf = ProcessBehavior(df)

        # Check Y column
        assert pd.isna(pdf.data['Y'].iloc[1])
        # Check FACTOR column
        assert pd.isna(pdf.data['FACTOR'].iloc[2])

    def test_case_variations_treated_as_na(self):
        """Test that case variations are handled (ND vs n/d)."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'Y': [235.5, 'ND', 'n/d', 'N/D']
        })

        pdf = ProcessBehavior(df)

        # All should be NA
        assert pd.isna(pdf.data['Y'].iloc[1])
        assert pd.isna(pdf.data['Y'].iloc[2])
        assert pd.isna(pdf.data['Y'].iloc[3])


class TestCustomNAValues:
    """Test user-specified custom NA values."""

    def test_custom_na_values_are_handled(self):
        """Test that custom NA values are properly converted."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'Y': [235.5, '-999', 237.2, '9999']
        })

        pdf = ProcessBehavior(df, na_values=['-999', '9999'])

        # Custom NA values should be converted
        assert pd.isna(pdf.data['Y'].iloc[1])
        assert pd.isna(pdf.data['Y'].iloc[3])
        # Valid values should remain
        assert pdf.data['Y'].iloc[0] == 235.5
        assert pdf.data['Y'].iloc[2] == 237.2

    def test_custom_na_combined_with_defaults(self):
        """Test that custom NA values are combined with default garbage characters."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4, 5],
            'Y': [235.5, '*', '-999', 'ND', 237.2]
        })

        pdf = ProcessBehavior(df, na_values=['-999'])

        # Both default and custom should be NA
        assert pd.isna(pdf.data['Y'].iloc[1])  # Default: *
        assert pd.isna(pdf.data['Y'].iloc[2])  # Custom: -999
        assert pd.isna(pdf.data['Y'].iloc[3])  # Default: ND

    def test_empty_na_values_list(self):
        """Test that passing empty list still uses defaults."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3],
            'Y': [235.5, '*', 237.2]
        })

        pdf = ProcessBehavior(df, na_values=[])

        # Default garbage characters should still work
        assert pd.isna(pdf.data['Y'].iloc[1])


class TestDataIntegrity:
    """Test that data cleaning doesn't corrupt valid data."""

    def test_numeric_data_unchanged(self):
        """Test that numeric data is not affected."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'Y': [235.5, 237.2, 239.1, 236.8]
        })

        pdf = ProcessBehavior(df)

        # All values should be unchanged
        pd.testing.assert_series_equal(pdf.data['Y'], df['Y'])

    def test_valid_string_data_unchanged(self):
        """Test that valid string data is not affected."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'FACTOR': ['A', 'B', 'C', 'D']
        })

        pdf = ProcessBehavior(df)

        # All values should be unchanged
        pd.testing.assert_series_equal(pdf.data['FACTOR'], df['FACTOR'])

    def test_negative_numbers_not_treated_as_na(self):
        """Test that negative numbers are preserved (single dash is NOT treated as NA)."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'Y': [235.5, -5.2, 237.2, -10.8]
        })

        pdf = ProcessBehavior(df)

        # Negative numbers should be preserved
        assert pdf.data['Y'].iloc[1] == -5.2
        assert pdf.data['Y'].iloc[3] == -10.8

    def test_original_dataframe_not_modified(self):
        """Test that the original DataFrame is not modified (immutability)."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3],
            'Y': [235.5, '*', 237.2]
        })

        original_values = df['Y'].copy()
        ProcessBehavior(df)

        # Original should be unchanged
        pd.testing.assert_series_equal(df['Y'], original_values)


class TestWarningMessages:
    """Test that appropriate warnings are logged."""

    def test_warning_logged_when_garbage_found(self, caplog):
        """Test that a warning is logged when garbage characters are found."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3],
            'Y': [235.5, '*', 237.2]
        })

        with caplog.at_level('WARNING'):
            ProcessBehavior(df)

        # Should have warning about garbage values
        assert 'garbage/NA values' in caplog.text
        assert 'Y' in caplog.text

    def test_warning_shows_column_counts(self, caplog):
        """Test that warning shows how many values per column."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'Y': [235.5, '*', '?', 237.2],
            'FACTOR': ['A', '--', 'C', 'D']
        })

        with caplog.at_level('WARNING'):
            ProcessBehavior(df)

        # Should show counts for both columns
        assert 'Y: 2 values' in caplog.text
        assert 'FACTOR: 1 values' in caplog.text

    def test_no_warning_when_no_garbage(self, caplog):
        """Test that no warning is logged for clean data."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'Y': [235.5, 237.2, 239.1, 236.8]
        })

        with caplog.at_level('WARNING'):
            ProcessBehavior(df)

        # Should not have warning
        assert 'garbage/NA values' not in caplog.text


class TestIntegrationWithAnalysis:
    """Test that garbage character handling works end-to-end with analysis."""

    def test_analysis_works_with_cleaned_data(self):
        """Test that analysis runs successfully with cleaned data.

        With garbage values ('*', 'ND') converted to NA, some cells become empty:
        - Cell (TIME=2, FACTOR=A): Y='*' → NA → N_kt=0
        - Cell (TIME=3, FACTOR=B): Y='ND' → NA → N_kt=0

        This creates an incomplete structure (SDS 5: incomplete, no replication).
        SDS 5 supports XmR chart, not Xbar.
        """
        df = pd.DataFrame({
            'TIME': [1, 1, 2, 2, 3, 3, 4, 4],
            'FACTOR': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'Y': [235.5, 237.2, '*', 239.1, 236.8, 'ND', 238.3, 237.5]
        })

        pdf = ProcessBehavior(df)

        # Analysis should run without crashing
        study = pdf.formulate(
            response=pdf.cols.Y,
            factors=[pdf.cols.FACTOR],
            time=pdf.cols.TIME
        )

        # SDS 5 detected due to empty cells (NA responses, no replication)
        assert study.observed_design_state.sds == 5

        # Use XmR chart (valid for SDS 5) with explicit stratification
        result = study.execute(chart='XmR', by=['FACTOR'])

        # Should complete successfully
        assert result is not None
        assert 'summary' in dir(result)

    def test_mixed_garbage_in_real_world_scenario(self):
        """Test realistic scenario with various garbage characters."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4, 5, 6, 7, 8],
            'LANE': ['L1', 'L2', 'L1', 'L2', 'L1', 'L2', 'L1', 'L2'],
            'WEIGHT': [236.5, '*', 237.8, '?', 'BDL', 238.2, '--', 239.1]
        })

        pdf = ProcessBehavior(df, na_values=['-999'])

        # Should handle gracefully
        assert pdf is not None
        assert len(pdf.data) == 8

        # Should have 4 NA values in WEIGHT column
        na_count = pdf.data['WEIGHT'].isna().sum()
        assert na_count == 4


# =========================================================================
# Numeric String Cleaning (currency, thousands sep, accounting negatives)
# =========================================================================

class TestNumericStringCleaning:
    """Test automatic cleaning of formatted numeric strings."""

    def test_dollar_sign_cleaned(self):
        df = pd.DataFrame({'Y': ['$1.50', '$2.00', '$3.50']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [1.50, 2.00, 3.50]

    def test_dollar_sign_with_space_cleaned(self):
        df = pd.DataFrame({'Y': ['$ 1.50', '$ 2.00', '$ 3.50']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [1.50, 2.00, 3.50]

    def test_negative_dollar_cleaned(self):
        df = pd.DataFrame({'Y': ['-$1.50', '-$ 2.00', '-$3.50']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [-1.50, -2.00, -3.50]

    def test_accounting_negative_cleaned(self):
        df = pd.DataFrame({'Y': ['(1.50)', '(2.00)', '(3.50)']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [-1.50, -2.00, -3.50]

    def test_accounting_negative_with_dollar(self):
        df = pd.DataFrame({'Y': ['($1.50)', '($ 2.00)', '($3.50)']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [-1.50, -2.00, -3.50]

    def test_thousands_separator_cleaned(self):
        df = pd.DataFrame({'Y': ['$1,234.56', '1,000', '$1,234,567.89']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [1234.56, 1000.0, 1234567.89]

    def test_euro_symbol_cleaned(self):
        df = pd.DataFrame({'Y': ['\u20ac100', '\u20ac200', '\u20ac300']})
        pdf = ProcessBehavior(df)
        assert pd.api.types.is_numeric_dtype(pdf.data['Y'])
        assert list(pdf.data['Y']) == [100, 200, 300]

    def test_pound_symbol_cleaned(self):
        df = pd.DataFrame({'Y': ['\u00a3100', '\u00a3200', '\u00a3300']})
        pdf = ProcessBehavior(df)
        assert pd.api.types.is_numeric_dtype(pdf.data['Y'])
        assert list(pdf.data['Y']) == [100, 200, 300]

    def test_yen_symbol_cleaned(self):
        df = pd.DataFrame({'Y': ['\u00a5100', '\u00a5200', '\u00a5300']})
        pdf = ProcessBehavior(df)
        assert pd.api.types.is_numeric_dtype(pdf.data['Y'])
        assert list(pdf.data['Y']) == [100, 200, 300]

    def test_percentage_sign_cleaned(self):
        df = pd.DataFrame({'Y': ['25.5%', '30%', '100%']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [25.5, 30.0, 100.0]

    def test_whitespace_around_values(self):
        df = pd.DataFrame({'Y': [' $1.50 ', '  $2.00  ', ' $3.50']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [1.50, 2.00, 3.50]

    def test_compound_format(self):
        """Accounting negative + currency + thousands separator."""
        df = pd.DataFrame({'Y': ['($ 1,234.56)', '$ 2,345.67', '-$ 3,456.78']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [-1234.56, 2345.67, -3456.78]

    def test_plain_numeric_strings_converted(self):
        """Strings that are just numbers (no formatting) should also convert."""
        df = pd.DataFrame({'Y': ['1.50', '2.00', '3.50']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [1.50, 2.00, 3.50]


class TestNumericStringDataIntegrity:
    """Ensure numeric string cleaning doesn't produce false positives."""

    def test_pure_text_column_unchanged(self):
        df = pd.DataFrame({'LABEL': ['Red', 'Blue', 'Green']})
        pdf = ProcessBehavior(df)
        assert pd.api.types.is_string_dtype(pdf.data['LABEL'])
        assert list(pdf.data['LABEL']) == ['Red', 'Blue', 'Green']

    def test_numeric_column_unchanged(self):
        df = pd.DataFrame({'Y': [1.0, 2.0, 3.0]})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert list(pdf.data['Y']) == [1.0, 2.0, 3.0]

    def test_integer_column_unchanged(self):
        df = pd.DataFrame({'Y': [1, 2, 3]})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'int64'
        assert list(pdf.data['Y']) == [1, 2, 3]

    def test_below_threshold_column_unchanged(self):
        """Column with < 80% convertible values stays as object."""
        df = pd.DataFrame({
            'LABEL': ['Red', '$1.50', 'Blue', 'Green', 'Orange',
                       'Yellow', 'Purple', 'Pink', 'Brown', 'Black']
        })
        pdf = ProcessBehavior(df)
        assert pd.api.types.is_string_dtype(pdf.data['LABEL'])

    def test_original_dataframe_not_modified(self):
        df = pd.DataFrame({'Y': ['$1.50', '$2.00', '$3.50']})
        original_values = list(df['Y'])
        ProcessBehavior(df)
        assert list(df['Y']) == original_values
        assert pd.api.types.is_string_dtype(df['Y'])

    def test_na_values_preserved(self):
        df = pd.DataFrame({'Y': ['$1.50', pd.NA, '$3.00']})
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].iloc[0] == 1.50
        assert pd.isna(pdf.data['Y'].iloc[1])
        assert pdf.data['Y'].iloc[2] == 3.00

    def test_negative_numbers_not_mangled(self):
        """Negative sign should not be confused with formatting."""
        df = pd.DataFrame({'Y': [-1.5, -2.0, -3.5]})
        pdf = ProcessBehavior(df)
        assert list(pdf.data['Y']) == [-1.5, -2.0, -3.5]


class TestNumericStringWithGarbageChars:
    """Test interaction between garbage character cleaning and numeric string cleaning."""

    def test_garbage_and_monetary_combined(self):
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4],
            'Y': ['$1.50', '*', '(3.00)', 'ND']
        })
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].iloc[0] == 1.50
        assert pd.isna(pdf.data['Y'].iloc[1])
        assert pdf.data['Y'].iloc[2] == -3.00
        assert pd.isna(pdf.data['Y'].iloc[3])

    def test_garbage_cleaned_before_monetary(self):
        """Garbage chars become NA first, then monetary cleaning on the rest."""
        df = pd.DataFrame({
            'TIME': [1, 2, 3, 4, 5],
            'Y': ['$10.00', '*', '$20.00', 'BDL', '$30.00']
        })
        pdf = ProcessBehavior(df)
        assert pdf.data['Y'].dtype == 'float64'
        assert pdf.data['Y'].iloc[0] == 10.0
        assert pd.isna(pdf.data['Y'].iloc[1])
        assert pdf.data['Y'].iloc[2] == 20.0
        assert pd.isna(pdf.data['Y'].iloc[3])
        assert pdf.data['Y'].iloc[4] == 30.0


class TestNumericStringWarnings:
    """Test warning messages for numeric string cleaning."""

    def test_warning_logged_when_monetary_cleaned(self, caplog):
        df = pd.DataFrame({'Y': ['$1.50', '$2.00', '$3.50']})
        with caplog.at_level('WARNING'):
            ProcessBehavior(df)
        assert 'Cleaned numeric formatting' in caplog.text
        assert 'Y' in caplog.text

    def test_no_warning_when_no_monetary(self, caplog):
        df = pd.DataFrame({'Y': [1.0, 2.0, 3.0]})
        with caplog.at_level('WARNING'):
            ProcessBehavior(df)
        assert 'Cleaned numeric formatting' not in caplog.text

    def test_warning_shows_column_names(self, caplog):
        df = pd.DataFrame({
            'REVENUE': ['$100', '$200'],
            'COST': ['$50', '$75']
        })
        with caplog.at_level('WARNING'):
            ProcessBehavior(df)
        assert 'REVENUE' in caplog.text
        assert 'COST' in caplog.text


class TestGrossRevenueDatabase:
    """Validate against Tom's real-world financial dataset."""

    @pytest.fixture
    def gross_rev_df(self):
        csv_path = Path(__file__).parent.parent / 'validation' / 'GROSSREVATTENDERRORDATABASE.csv'
        return pd.read_csv(csv_path)

    def test_gross_revenue_column_converts_to_float(self, gross_rev_df):
        pdf = ProcessBehavior(gross_rev_df)
        assert pdf.data['GROSS REVENUE ERROR'].dtype == 'float64'

    def test_all_350_values_converted(self, gross_rev_df):
        pdf = ProcessBehavior(gross_rev_df)
        assert pdf.data['GROSS REVENUE ERROR'].notna().sum() == 350

    def test_attendance_column_stays_numeric(self, gross_rev_df):
        pdf = ProcessBehavior(gross_rev_df)
        assert pdf.data['ATTENDANCE COUNT ERROR'].dtype == 'int64'

    def test_text_columns_not_mangled(self, gross_rev_df):
        pdf = ProcessBehavior(gross_rev_df)
        assert pd.api.types.is_string_dtype(pdf.data['EVENT DOW'])
        assert pd.api.types.is_string_dtype(pdf.data['PRIMARY PARTNER'])
        assert pd.api.types.is_string_dtype(pdf.data['HOME TEAM'])

    def test_integer_columns_untouched(self, gross_rev_df):
        pdf = ProcessBehavior(gross_rev_df)
        assert pdf.data['YR-WEEK'].dtype == 'int64'
        assert pdf.data['YR-MONTH'].dtype == 'int64'

    def test_spot_check_values(self, gross_rev_df):
        pdf = ProcessBehavior(gross_rev_df)
        rev = pdf.data['GROSS REVENUE ERROR']
        assert rev.iloc[0] == pytest.approx(-1815.36)
        assert rev.iloc[3] == pytest.approx(29794.37)

    def test_end_to_end_formulate_execute(self, gross_rev_df):
        pdf = ProcessBehavior(gross_rev_df)
        study = pdf.formulate(
            response='GROSS REVENUE ERROR',
            factors=['HOME TEAM'],
            time='YR-WEEK'
        )
        # XmR with factors requires explicit by parameter
        result = study.execute(chart='XmR', by=['HOME TEAM'])
        assert result is not None
