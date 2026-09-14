"""Series-length precision statement (#114).

One line under Structure in the design report: where sigma comes from at the observed
structure, and how precise it is at the observed T. A fact beside the design state, never a
verdict on it. These tests pin the sentences, the table they draw on, and the invariant that
the design state, recommendation and chart menu do not move with T.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.series_length import SeriesLengthPrecision, assess_series_length
from processbehavior.spc_constants import D2_N2, MR_SIGMA_INTERVAL_80


def _stable(T, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'t': range(T), 'y': rng.normal(5.5, 1.2, T)})


# ---------------------------------------------------------------------------
# The pure function
# ---------------------------------------------------------------------------


class TestAssessSeriesLength:
    def test_unreplicated_in_table_range_states_the_interval(self):
        r = assess_series_length(T=4, within_cell_df=0, sigma_from_time=True)
        assert r.n_moving_ranges == 3
        assert r.mr_interval_80 == MR_SIGMA_INTERVAL_80[4]
        assert r.description == (
            'T=4. Sigma for X/mR rests on 3 moving ranges; on a stable process, '
            '80% of such estimates fall between 0.43x and 1.68x the true sigma.'
        )

    def test_unreplicated_beyond_table_is_bounded_by_the_last_row(self):
        r = assess_series_length(T=400, within_cell_df=0, sigma_from_time=True)
        assert r.n_moving_ranges == 399 and r.mr_interval_80 is None
        assert '399 moving ranges' in r.description
        assert 'within 0.79x to 1.22x the true sigma or narrower' in r.description

    def test_two_points_is_a_single_moving_range(self):
        r = assess_series_length(T=2, within_cell_df=0, sigma_from_time=True)
        assert r.n_moving_ranges == 1 and r.mr_interval_80 is None
        assert r.description == 'T=2. Sigma for X/mR rests on a single moving range.'

    def test_replicated_does_not_depend_on_T(self):
        r1 = assess_series_length(T=1, within_cell_df=25, sigma_from_time=False)
        assert r1.n_moving_ranges is None and r1.mr_interval_80 is None
        assert r1.description == (
            'T=1. Sigma rests on within-cell replication (25 degrees of freedom), not on the time sequence.'
        )
        r_none = assess_series_length(T=None, within_cell_df=25, sigma_from_time=False)
        assert r_none.description.startswith('Sigma rests on within-cell replication (25 degrees of freedom)')

    def test_partial_replication_states_both_facts(self):
        r = assess_series_length(T=8, within_cell_df=32, sigma_from_time=True)
        assert '7 moving ranges' in r.description
        assert 'Replicated cells also carry 32 within-cell degrees of freedom for Xbar/S.' in r.description

    def test_no_time_and_no_replication_says_nothing(self):
        r = assess_series_length(T=None, within_cell_df=0, sigma_from_time=True)
        assert r.description == ''

    def test_no_verdict_words(self):
        for T in (2, 3, 4, 6, 12, 20, 30, 31, 400):
            text = assess_series_length(T, 0, True).description.lower()
            for word in ('short', 'adequate', 'provisional', 'warning', 'insufficient', 'too few'):
                assert word not in text, (T, word)

    def test_result_is_frozen(self):
        r = assess_series_length(T=4, within_cell_df=0, sigma_from_time=True)
        assert isinstance(r, SeriesLengthPrecision)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.T = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class TestIntervalTable:
    def test_covers_3_through_30_and_narrows_monotonically(self):
        assert sorted(MR_SIGMA_INTERVAL_80) == list(range(3, 31))
        spreads = [hi / lo for lo, hi in (MR_SIGMA_INTERVAL_80[n] for n in range(3, 31))]
        assert all(a > b for a, b in zip(spreads, spreads[1:], strict=False))

    @pytest.mark.parametrize('n', [4, 20])
    def test_rows_regenerate_from_the_validation_script_seed(self, n):
        """validation/short_series_bands.py: default_rng([20260903, n]), 100,000 reps."""
        rng = np.random.default_rng([20260903, n])
        z = rng.standard_normal((100_000, n))
        v = np.abs(np.diff(z, axis=1)).mean(axis=1) / D2_N2
        p10, p90 = np.percentile(v, [10, 90])
        assert (round(p10, 3), round(p90, 3)) == MR_SIGMA_INTERVAL_80[n]


# ---------------------------------------------------------------------------
# Through Study and the design report
# ---------------------------------------------------------------------------


class TestStudySurface:
    def test_unreplicated_series_reports_moving_ranges(self):
        st = pb.formulate(_stable(4), response='y', time='t')
        assert st.series_length.T == 4 and st.series_length.n_moving_ranges == 3
        assert st.series_length_description == st.series_length.description
        assert 'Series length: T=4. Sigma for X/mR rests on 3 moving ranges' in repr(st.design())

    def test_design_state_and_menu_do_not_move_with_T(self):
        """The invariant #114's first table demonstrates: ADS 2 is ADS 2 at T=3 and T=300."""
        short, long = (pb.formulate(_stable(T), response='y', time='t') for T in (3, 300))
        assert short.analytical_design_state.sds == long.analytical_design_state.sds == 2
        assert short.ads_reason == long.ads_reason
        assert short.recommended_chart == long.recommended_chart == 'X'
        assert short.valid_charts == long.valid_charts
        assert short.series_length.n_moving_ranges == 2 and long.series_length.n_moving_ranges == 299

    def test_replicated_single_period_reports_within_cell_df(self):
        """Tom's T=1 file: 5 conditions x 6 replicates, one time point — a complete study."""
        df = pd.DataFrame(
            {
                'TIME': [1] * 30,
                'TEMP': np.repeat([0, 25, 50, 75, 100], 6),
                'LIFE': np.arange(30, dtype=float) + 55,
            }
        )
        st = pb.formulate(df, response='LIFE', factors=['TEMP'], time='TIME')
        assert st.analytical_design_state.sds == 1
        sl = st.series_length
        assert sl.T == 1 and not sl.sigma_from_time and sl.within_cell_df == 25
        assert 'Series length: T=1. Sigma rests on within-cell replication (25 degrees of freedom)' in repr(st.design())

    def test_no_time_variable_with_replication_still_states_the_source(self):
        df = pd.DataFrame({'TEMP': np.repeat([0, 25, 50], 4), 'LIFE': np.arange(12, dtype=float)})
        st = pb.formulate(df, response='LIFE', factors=['TEMP'])
        assert st.series_length.T is None
        assert 'within-cell replication (9 degrees of freedom); no time sequence' in repr(st.design())

    def test_no_warning_is_emitted_for_a_short_series(self, recwarn):
        pb.formulate(_stable(3), response='y', time='t').design()
        assert not [w for w in recwarn if 'Series length' in str(w.message) or 'moving range' in str(w.message)]
