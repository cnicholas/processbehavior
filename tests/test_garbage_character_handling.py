"""
Tests for garbage character and NA value handling in ProcessBehavior.

These tests verify that real-world messy data with non-standard NA indicators
is properly cleaned and handled gracefully.
"""

import pandas as pd

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
        """Test that analysis runs successfully with cleaned data."""
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
        result = study.execute()

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
