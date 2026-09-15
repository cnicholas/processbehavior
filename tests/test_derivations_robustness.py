"""Derived variables honour their contract: evaluate/validate never raise on routine data.

An adversarial pass (2026-09-15) found the contract broken by infinities in the source column
(pd.cut "bins must increase monotonically"), range labels colliding at six significant digits
(pd.cut "labels must be unique"), unvalidated custom labels, and unvalidated parameter types. It
also found silent wrong answers: non-finite transform outputs passing with n_invalid=0, a z-score
of a constant column that never triggered on_invalid, more bins than distinct values with no
message, and a serializer that turned a label spelled "NaN" into a float. These tests pin the
fixes and keep a seeded fuzz in the suite so the next regression is caught here, not by a user.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior import Derivation, evaluate
from processbehavior.exceptions import ValidationError

T, B = Derivation.transform, Derivation.bin


# ---------------------------------------------------------------------------
# Construction rejects what evaluate could not honour
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_numpy_integer_n_is_accepted_and_normalised(self):
        assert B('x', n=np.int64(4)).params['n'] == 4
        assert type(B('x', n=np.int64(4)).params['n']) is int

    @pytest.mark.parametrize('n', [True, 4.0, 0, -1, '4'])
    def test_non_integral_or_nonpositive_n_is_rejected(self, n):
        with pytest.raises(ValidationError, match='integer n > 0'):
            B('x', n=n)

    @pytest.mark.parametrize('breaks', [[1, math.inf], [1, math.nan, 2], ['1', '2'], [2, 1], [1, 1]])
    def test_breaks_must_be_finite_ascending_numbers(self, breaks):
        with pytest.raises(ValidationError):
            B('x', method='breaks', breaks=breaks)

    def test_breaks_are_normalised_to_floats(self):
        assert B('x', method='breaks', breaks=[np.int64(1), 2]).params['breaks'] == [1.0, 2.0]

    @pytest.mark.parametrize('kw', [dict(shift='a'), dict(shift=math.inf), dict(shift=True)])
    def test_shift_must_be_a_finite_number(self, kw):
        with pytest.raises(ValidationError, match='shift must be a finite number'):
            T('x', 'log', **kw)

    @pytest.mark.parametrize('exponent', ['2', math.nan, None])
    def test_exponent_must_be_a_finite_number(self, exponent):
        with pytest.raises(ValidationError):
            T('x', 'power', exponent=exponent)

    def test_on_invalid_must_be_error_or_na(self):
        with pytest.raises(ValidationError, match="on_invalid must be 'error' or 'na'"):
            T('x', 'log', on_invalid='banana')

    @pytest.mark.parametrize(
        'labels, match',
        [(['a', 'a'], 'unique'), (['a', None], 'null'), (['a', math.nan], 'null'), ((), 'empty'), ([1, '1'], 'unique')],
    )
    def test_explicit_labels_are_non_empty_null_free_and_unique(self, labels, match):
        with pytest.raises(ValidationError, match=match):
            B('x', n=2, bin_labels=labels)

    def test_direct_construction_is_validated_too(self):
        """The app's from_dict path and the factories share one gate."""
        with pytest.raises(ValidationError, match='unique'):
            Derivation(
                family='bin',
                column='x',
                function='bin',
                params={'method': 'equal_freq', 'n': 2, 'bin_labels': ['a', 'a']},
            )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_is_json_safe_with_numpy_inputs(self):
        spec = T('x', 'power', exponent=np.float64(0.5)).with_fitted(
            {'edges': np.array([1.0, np.inf]), 'mu': np.float32(2)}
        )
        text = json.dumps(spec.to_dict())
        back = Derivation.from_dict(json.loads(text))
        assert back.fitted['edges'] == [1.0, math.inf] and back.fitted['mu'] == 2.0

    def test_labels_that_spell_the_float_tags_stay_strings(self):
        spec = B('x', n=3, bin_labels=['NaN', 'Infinity', '-Infinity'])
        back = Derivation.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert back.params['bin_labels'] == ['NaN', 'Infinity', '-Infinity']

    def test_non_finite_numbers_round_trip_under_numeric_keys(self):
        spec = B('x', method='breaks', breaks=[1.0]).with_fitted(
            {'edges': [-math.inf, 1.0, math.inf], 'sigma': math.nan}
        )
        back = Derivation.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert back.fitted['edges'] == [-math.inf, 1.0, math.inf] and math.isnan(back.fitted['sigma'])

    def test_from_dict_without_id_is_a_validation_error(self):
        with pytest.raises(ValidationError, match="needs an 'id'"):
            Derivation.from_dict({'family': 'transform', 'column': 'x', 'function': 'log'})


# ---------------------------------------------------------------------------
# Transforms: every non-finite result is a violation
# ---------------------------------------------------------------------------


class TestTransformNonFinite:
    @pytest.mark.parametrize(
        'fn, values',
        [
            ('log', [1.0, np.inf]),
            ('log10', [1.0, np.inf]),
            ('sqrt', [4.0, np.inf]),
            ('square', [1.0, 1e200]),
            ('inverse', [1.0, 1e-320]),
            ('power', [1.0, 10.0]),
        ],
    )
    def test_inf_in_or_out_is_flagged_never_returned(self, fn, values):
        spec = T('x', fn, exponent=1e6) if fn == 'power' else T('x', fn)
        r = evaluate(spec, pd.Series(values))
        assert r.n_invalid == 1 and list(r.invalid_index) == [1]
        assert np.isfinite(r.values.iloc[0]) and pd.isna(r.values.iloc[1])

    def test_negative_infinity_is_a_violation_even_for_square(self):
        r = evaluate(T('x', 'square'), pd.Series([-np.inf, 2.0]))
        assert r.n_invalid == 1 and r.values.iloc[1] == 4.0

    def test_zscore_of_a_constant_column_is_a_violation_so_on_invalid_applies(self):
        r = evaluate(T('x', 'zscore'), pd.Series([5.0, 5.0, np.nan, 5.0]))
        assert r.n_invalid == 3 and r.values.isna().all()
        assert 'zscore is undefined' in r.message
        df = pd.DataFrame({'t': range(1, 5), 'lane': list('ABAB'), 'y': 5.0})
        with pytest.raises(ValidationError, match="Derived variable 'y_zscore'"):
            pb.ProcessBehavior(df).transform('y', 'zscore').formulate(response='y_zscore', factors=['lane'], time='t')

    def test_arcsin_on_percentages_gets_a_hint(self):
        r = evaluate(T('x', 'arcsin'), pd.Series([0.0, 25.0, 50.0, 100.0]))
        assert r.n_invalid == 3 and 'divide by 100' in r.message

    def test_finite_data_is_unchanged(self):
        r = evaluate(T('x', 'log'), pd.Series([1.0, math.e, np.nan]))
        assert r.n_invalid == 0 and r.values.round(6).tolist()[:2] == [0.0, 1.0] and pd.isna(r.values.iloc[2])


# ---------------------------------------------------------------------------
# Bins: infinities, label collisions, bin-count caps, non-numeric sources
# ---------------------------------------------------------------------------


class TestBinRobustness:
    @pytest.mark.parametrize('method', ['equal_freq', 'equal_width', 'sd'])
    def test_infinities_leave_the_fit_and_are_counted(self, method):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.inf, -np.inf, np.nan])
        r = evaluate(B('x', method=method, n=2), s)
        assert r.n_invalid == 2 and sorted(r.invalid_index) == [5, 6]
        assert r.values.iloc[:5].notna().all() and r.values.iloc[5:].isna().all()
        assert '2 non-finite value(s) cannot be binned' in r.message
        assert all(math.isfinite(e) for e in r.fitted['edges'][1:-1])

    def test_only_infinities_is_reported_not_raised(self):
        r = evaluate(B('x', n=2), pd.Series([np.inf, -np.inf]))
        assert r.fitted['n_bins'] == 0 and r.n_invalid == 2 and 'no finite values' in r.message

    def test_tiny_range_labels_are_distinct(self):
        s = pd.Series([1.00000001, 1.00000002, 1.00000003, 1.00000004, 1.00000005])
        r = evaluate(B('x', method='equal_width', n=4), s)
        cats = list(r.values.cat.categories)
        assert len(set(cats)) == 4 and r.values.notna().all()

    def test_near_constant_sd_bins_do_not_raise(self):
        r = evaluate(B('x', method='sd'), pd.Series([-0.085, -0.085, -0.085, -0.085000001]))
        assert r.fitted['n_bins'] == 5 and r.values.notna().all()

    def test_more_bins_than_distinct_values_is_capped_with_a_message(self):
        r = evaluate(B('x', n=1000), pd.Series(np.arange(1.0, 21.0)))
        assert r.fitted['n_bins'] == 20 and 'only 20 distinct values' in r.message
        r = evaluate(B('x', method='equal_width', n=9), pd.Series([1.0, 2.0, 3.0]))
        assert r.fitted['n_bins'] == 3 and r.values.notna().all()

    def test_ordinal_beyond_five_bins_says_so(self):
        r = evaluate(B('x', n=6, bin_labels='ordinal'), pd.Series(np.arange(1.0, 61.0)))
        assert 'ordinal labels are defined for 2 to 5 bins' in r.message

    @pytest.mark.parametrize(
        'column',
        [
            pd.Series(pd.to_datetime(['2024-01-01', '2024-06-01', '2025-01-01'])),
            pd.Series(pd.Categorical([1.0, 2.0, 3.0])),
        ],
    )
    def test_non_numeric_sources_derive_nothing_without_raising(self, column):
        column.name = 'src'
        for spec in (B('src', n=2), T('src', 'log')):
            r = evaluate(spec, column)
            assert r.values.isna().all() and r.n_invalid == 0 and 'is not numeric' in r.message

    def test_finite_data_edges_are_unchanged(self):
        r = evaluate(B('x', n=4), pd.Series(np.arange(1.0, 21.0)))
        assert r.fitted['edges'] == [1.0, 5.75, 10.5, 15.25, 20.0]
        assert list(r.values.cat.categories) == ['[1, 5.75)', '[5.75, 10.5)', '[10.5, 15.25)', '[15.25, 20]']


# ---------------------------------------------------------------------------
# The contract itself: a seeded fuzz over every method
# ---------------------------------------------------------------------------


def _random_series(rng, n):
    kind = rng.integers(0, 6)
    if kind == 0:
        x = rng.normal(size=n)
    elif kind == 1:
        x = rng.integers(0, 3, size=n).astype(float)  # heavy ties
    elif kind == 2:
        x = np.full(n, rng.normal())  # constant
    elif kind == 3:
        x = rng.normal(size=n) * 10.0 ** rng.integers(-12, 12)  # extreme scales
    elif kind == 4:
        x = rng.normal(size=n)
        x[rng.random(n) < 0.3] = np.nan
    else:
        x = rng.normal(size=n)
        x[rng.random(n) < 0.1] = np.inf
        x[rng.random(n) < 0.05] = -np.inf
    return pd.Series(x)


def _all_specs():
    specs = [
        B('x', method=m, n=n, bin_labels=bl, right=r)
        for m in ('equal_freq', 'equal_width')
        for n in (1, 2, 3, 4, 7, 50)
        for bl in ('range', 'ordinal', 'number')
        for r in (False, True)
    ]
    specs += [B('x', method='sd', bin_labels=bl) for bl in ('range', 'ordinal', 'number')]
    specs += [B('x', method='breaks', breaks=[-1, 0, 1])]
    specs += [T('x', fn) for fn in ('log', 'log10', 'sqrt', 'arcsin', 'inverse', 'square', 'zscore')]
    specs += [T('x', 'power', exponent=e) for e in (-1, 0.5, 2, 3)]
    return specs


@pytest.mark.slow
def test_evaluate_never_raises_and_never_returns_infinity():
    """400 random series x every spec: no exception, no ±inf output, every finite input binned."""
    rng = np.random.default_rng(20260915)
    specs = _all_specs()
    for _ in range(400):
        s = _random_series(rng, int(rng.integers(0, 40)))
        finite = s.notna() & np.isfinite(s.astype(float))
        for spec in specs:
            r = evaluate(spec, s)  # must not raise
            if spec.family == 'transform':
                assert not np.isinf(r.values.to_numpy(dtype=float)).any(), (spec.function, s.tolist()[:6])
                assert r.n_invalid >= int((s.notna() & ~finite).sum())
            elif r.fitted.get('n_bins'):
                assert r.values[finite].notna().all(), (spec.params, s.tolist()[:6])
                assert r.values[~finite].isna().all()


def test_evaluate_never_raises_quick():
    """A 40-series slice of the fuzz that runs in the default (not slow) selection."""
    rng = np.random.default_rng(1)
    specs = _all_specs()
    for _ in range(40):
        s = _random_series(rng, int(rng.integers(0, 30)))
        for spec in specs:
            r = evaluate(spec, s)
            if spec.family == 'transform':
                assert not np.isinf(r.values.to_numpy(dtype=float)).any()


# ---------------------------------------------------------------------------
# ProcessBehavior error paths that had no test
# ---------------------------------------------------------------------------


class TestProcessBehaviorDerivationErrors:
    @pytest.fixture
    def pbd(self):
        df = pd.DataFrame({'t': range(1, 7), 'lane': list('ABABAB'), 'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        return pb.ProcessBehavior(df).transform('y', 'log').bin('y', n=2, label='yb')

    def test_remove_unknown_id_lists_the_attached_ids(self, pbd):
        ids = [d.id for d in pbd.derivations]
        with pytest.raises(ValidationError) as excinfo:
            pbd.remove_derived('nope')
        assert str(excinfo.value) == f"No derivation with id 'nope'. Attached ids: {ids}."

    def test_replace_unknown_id_raises(self, pbd):
        with pytest.raises(ValidationError, match='No derivation with id'):
            pbd.replace_derived('nope', B('y', n=3))

    def test_replace_with_a_colliding_name_raises(self, pbd):
        with pytest.raises(ValidationError, match='already exists'):
            pbd.replace_derived(pbd.derivations[1].id, B('y', n=3, label='y_log'))

    def test_free_functions_delegate(self, pbd):
        assert pb.derivations(pbd) == pbd.derivations
        fewer = pb.remove_derived(pbd, pbd.derivations[0].id)
        assert len(fewer.derivations) == 1
        swapped = pb.replace_derived(pbd, pbd.derivations[1].id, B('y', n=3, label='yb'))
        assert swapped.derivations[1].params['n'] == 3
