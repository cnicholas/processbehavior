"""Tests for Derived Variables (transforms + binning).

Covers the spec engine (evaluate/validate), the Derivation value object and its
round-trip, the fluent ProcessBehavior verbs, and the formulate() integration
(referenceability, binned-factor ordering, fit-freezing, reformulation gating,
and the one-frame identity gate).
"""

import math
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior import Derivation, ProcessBehavior, evaluate, make_design, validate

# ---------------------------------------------------------------------------
# Derivation construction / validation
# ---------------------------------------------------------------------------


def test_transform_factory_and_output_name():
    d = Derivation.transform('rate', 'arcsin')
    assert d.family == 'transform' and d.function == 'arcsin'
    assert d.output_name == 'rate_arcsin'
    assert Derivation.transform('x', 'log', label='lx').output_name == 'lx'


def test_ln_canonicalizes_to_log():
    assert Derivation.transform('x', 'ln').function == 'log'
    assert Derivation.transform('x', 'ln').output_name == 'x_log'


def test_bin_factory_function_is_literal_bin():
    d = Derivation.bin('weight', method='equal_freq', n=4)
    assert d.family == 'bin' and d.function == 'bin'
    assert d.output_name == 'weight_bin'  # not weight_equal_freq
    assert d.params['method'] == 'equal_freq'


@pytest.mark.parametrize(
    'factory',
    [
        lambda: Derivation.transform('x', 'nope'),               # unknown transform
        lambda: Derivation.transform('x', 'power'),              # power needs exponent
        lambda: Derivation(family='bin', column='x', function='log'),  # bin must be function='bin'
        lambda: Derivation(family='transform', column='x', function='bin'),  # log/sqrt only
        lambda: Derivation.bin('x', method='nope'),              # unknown method
        lambda: Derivation.bin('x', method='breaks', breaks=[3, 1, 2]),  # not ascending
        lambda: Derivation.bin('x', method='equal_freq', n=0),   # n must be > 0
        lambda: Derivation.bin('x', bin_labels='weird'),         # unknown style
        lambda: Derivation.transform('', 'log'),                 # empty column
    ],
)
def test_invalid_construction_raises(factory):
    with pytest.raises(pb.ValidationError):
        factory()


# ---------------------------------------------------------------------------
# evaluate() — transforms
# ---------------------------------------------------------------------------


def test_transform_values_and_domain_violations():
    s = pd.Series([1.0, math.e, np.nan, -2.0, 0.0])
    r = evaluate(Derivation.transform('x', 'log'), s)
    assert r.values.iloc[0] == pytest.approx(0.0)
    assert r.values.iloc[1] == pytest.approx(1.0)
    assert pd.isna(r.values.iloc[2])          # NA in -> NA out
    assert r.n_invalid == 2                    # -2 and 0 (NA not counted)
    assert list(r.invalid_index) == [3, 4]


def test_shift_resolves_domain():
    s = pd.Series([0.0, -0.5, 3.0])
    r = evaluate(Derivation.transform('x', 'log', shift=1.0), s)  # log(x+1)
    assert r.n_invalid == 0
    assert r.values.iloc[0] == pytest.approx(0.0)


def test_zscore_fits_mu_sigma_ddof1():
    r = evaluate(Derivation.transform('x', 'zscore'), pd.Series([1.0, 2, 3, 4, 5]))
    assert r.fitted['mu'] == pytest.approx(3.0)
    assert r.fitted['sigma'] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert float(r.values.mean()) == pytest.approx(0.0, abs=1e-12)


def test_power_and_inverse_domains():
    rp = evaluate(Derivation.transform('x', 'power', exponent=0.5), pd.Series([4.0, -1.0]))
    assert rp.values.iloc[0] == pytest.approx(2.0)
    assert rp.n_invalid == 1  # sqrt of negative not representable
    ri = evaluate(Derivation.transform('x', 'inverse'), pd.Series([2.0, 0.0]))
    assert ri.values.iloc[0] == pytest.approx(0.5)
    assert ri.n_invalid == 1  # 1/0


def test_arcsin_closed_boundary_clamps_within_epsilon_flags_beyond():
    s = pd.Series([0.0, 1.0, 1.0000000002, 1.05])
    r = evaluate(Derivation.transform('p', 'arcsin'), s)
    assert r.values.iloc[0] == pytest.approx(0.0)
    assert r.values.iloc[1] == pytest.approx(math.pi / 2)
    assert r.values.iloc[2] == pytest.approx(math.pi / 2)  # 1.0000000002 -> clamped
    assert pd.isna(r.values.iloc[3])                       # 1.05 -> flagged
    assert r.n_invalid == 1
    assert list(r.invalid_index) == [3]


def test_na_and_violations_are_distinct():
    # rows: NA, valid, violation(-1), NA, valid
    s = pd.Series([np.nan, 4.0, -1.0, np.nan, 9.0])
    r = evaluate(Derivation.transform('x', 'sqrt'), s)
    assert r.n_invalid == 1
    assert list(r.invalid_index) == [2]          # only the genuine violation
    assert 0 not in r.invalid_index and 3 not in r.invalid_index  # no NA-row labels
    assert pd.isna(r.values.iloc[0]) and pd.isna(r.values.iloc[3])
    assert r.values.iloc[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# evaluate() — binning
# ---------------------------------------------------------------------------


def test_equal_freq_labels_and_fitted_edges():
    r = evaluate(Derivation.bin('w', n=4, bin_labels='ordinal'), pd.Series(range(1, 17), dtype=float))
    assert list(r.values.cat.categories) == ['Low', 'Medium-Low', 'Medium-High', 'High']
    assert r.values.cat.ordered
    assert r.fitted['n_bins'] == 4
    assert len(r.fitted['edges']) == 5


def test_tie_drop_uses_fitted_count_labels_and_message():
    tie = pd.Series([1, 1, 1, 1, 1, 1, 2, 3, 4, 5], dtype=float)
    r = evaluate(Derivation.bin('t', n=5, bin_labels='ordinal'), tie)
    assert r.fitted['n_bins'] == 3                      # qcut dropped duplicate edges
    assert r.fitted['labels'] == ['Low', 'Medium', 'High']  # keyed on FITTED count, not 5
    assert 'requested 5 bins' in r.message and 'produced 3' in r.message


def test_range_labels_from_fitted_edges():
    r = evaluate(Derivation.bin('w', method='equal_width', n=2, bin_labels='range'),
                 pd.Series([0.0, 10.0]))
    # edges [0, 5, 10] -> two range labels from fitted edges
    assert r.fitted['edges'] == [0.0, 5.0, 10.0]
    assert r.values.cat.categories[0].startswith('[0')


def test_breaks_out_of_range_tails():
    r = evaluate(Derivation.bin('x', method='breaks', breaks=[2.0, 4.0]), pd.Series([0.0, 3.0, 9.0]))
    cats = list(r.values.cat.categories)
    assert cats[0].startswith('<') and cats[-1].startswith('>=')  # open tails
    assert r.fitted['edges'][0] == -math.inf and r.fitted['edges'][-1] == math.inf


def test_sd_method_zones_and_fitted_mu_sigma():
    s = pd.Series(np.arange(0, 100, dtype=float))
    r = evaluate(Derivation.bin('x', method='sd'), s)
    assert r.fitted['method'] == 'sd'
    assert 'mu' in r.fitted and 'sigma' in r.fitted
    assert r.fitted['n_bins'] == 5  # (-inf, -2s, -s, +s, +2s, inf)


def test_edge_closure_follows_right():
    s = pd.Series([0.0, 2.0, 3.0])
    left = evaluate(Derivation.bin('x', method='breaks', breaks=[2.0], right=False), s)
    right = evaluate(Derivation.bin('x', method='breaks', breaks=[2.0], right=True), s)
    # 2.0 falls in the upper bin under left-closed, lower bin under right-closed
    assert str(left.values.iloc[1]).startswith('>=')
    assert str(right.values.iloc[1]).startswith('<')


def test_explicit_labels_length_mismatch_falls_back_with_message():
    r = evaluate(Derivation.bin('w', n=4, bin_labels=['a', 'b']), pd.Series(range(1, 17), dtype=float))
    assert r.message and 'labels supplied' in r.message
    assert len(r.values.cat.categories) == 4   # fell back to range labels


def test_bin_passes_na_through():
    r = evaluate(Derivation.bin('w', n=2), pd.Series([1.0, 2.0, np.nan, 3.0, 4.0]))
    assert pd.isna(r.values.iloc[2])
    assert r.n_invalid == 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_roundtrip_value_equality_and_explicit_id():
    d = Derivation.bin('w', n=4, breaks=None).with_fitted({'edges': [1.0, 2.0, math.inf], 'n_bins': 2})
    d2 = Derivation.from_dict(d.to_dict())
    assert d2 == d                       # content equality
    assert d2.id == d.id                 # explicit: compare=False would hide a fresh id
    assert d2.fitted['edges'][-1] == math.inf  # inf survived JSON-safe encoding


def test_from_dict_takes_id_but_fresh_construction_mints():
    a = Derivation.transform('x', 'log')
    b = Derivation.transform('x', 'log')
    assert a == b and a.id != b.id       # equal content, distinct identity


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_structured_results():
    df = pd.DataFrame({'x': [1.0, 2, 3, 4], 'x_log': [0, 0, 0, 0]})
    assert validate(Derivation.transform('x', 'log'), df).ok is False  # collision with x_log
    assert validate(Derivation.transform('missing', 'log'), df).issues[0]['code'] == 'column_not_found'
    ok = validate(Derivation.transform('x', 'sqrt'), df)
    assert ok.ok is True


def test_validate_label_count_against_fitted_not_requested():
    df = pd.DataFrame({'t': [1, 1, 1, 1, 1, 1, 2, 3, 4, 5]})
    # request 5 bins with 5 labels, but ties produce 3 bins -> mismatch caught
    res = validate(Derivation.bin('t', n=5, bin_labels=['a', 'b', 'c', 'd', 'e']), df)
    assert res.ok is False
    assert any(i['code'] == 'label_count' for i in res.issues)


# ---------------------------------------------------------------------------
# Fluent verbs on ProcessBehavior
# ---------------------------------------------------------------------------


def _pb():
    df = make_design(1, seed=1).copy()
    df['y'] = df['y'].abs() + 1.0  # ensure positive for log
    return ProcessBehavior(df)


def test_verbs_return_new_pb_original_unchanged():
    base = _pb()
    derived = base.transform('y', 'log')
    assert base.derivations == ()
    assert [d.output_name for d in derived.derivations] == ['y_log']


def test_collision_and_missing_column_raise_at_attach():
    base = _pb()
    with pytest.raises(pb.ValidationError):
        base.transform('does_not_exist', 'log')
    with pytest.raises(pb.ValidationError):
        base.transform('y', 'log').transform('y', 'log')  # second y_log collides


def test_chaining_rejected_but_same_source_twice_allowed():
    base = _pb()
    # two derivations off the same ORIGINAL column both attach
    two = base.transform('y', 'square').bin('y', n=3)
    assert {d.output_name for d in two.derivations} == {'y_square', 'y_bin'}
    # deriving from a DERIVED column is rejected (not an original column)
    with pytest.raises(pb.ValidationError):
        base.transform('y', 'square').transform('y_square', 'log')


def test_remove_and_replace_by_id():
    base = _pb().transform('y', 'square')
    spec_id = base.derivations[0].id
    assert base.remove_derived(spec_id).derivations == ()
    swapped = base.replace_derived(spec_id, Derivation.transform('y', 'log'))
    assert swapped.derivations[0].function == 'log'
    assert swapped.derivations[0].id == swapped.derivations[0].id  # sanity
    with pytest.raises(pb.ValidationError):
        base.remove_derived('nope')


# ---------------------------------------------------------------------------
# formulate() integration
# ---------------------------------------------------------------------------


def test_derived_columns_referenceable_and_fits_frozen():
    study = _pb().transform('y', 'square').bin('y', n=3, bin_labels='ordinal').formulate(
        response='y_square', factors=['y_bin'], time='time'
    )
    assert 'y_square' in study.dataset.columns
    assert 'y_bin' in study.dataset.columns
    names = {d.output_name for d in study.derivations}
    assert names == {'y_square', 'y_bin'}
    bin_spec = next(d for d in study.derivations if d.family == 'bin')
    assert bin_spec.fitted['n_bins'] >= 2  # edges frozen on the study


def test_binned_ordinal_factor_charts_in_bin_order():
    study = _pb().bin('y', n=4, bin_labels='ordinal').formulate(
        response='y', factors=['y_bin'], time='time'
    )
    cats = list(study.dataset['y_bin'].cat.categories)
    assert cats == ['Low', 'Medium-Low', 'Medium-High', 'High']


def test_reformulation_gating():
    base = _pb()
    study1 = base.formulate(response='y', factors=['factor 1'], time='time')
    assert study1.derivations == ()
    assert 'y_square' not in study1.dataset.columns
    # adding a derivation produces a NEW pb; study1 is unaffected
    study2 = base.transform('y', 'square').formulate(
        response='y_square', factors=['factor 1'], time='time'
    )
    assert 'y_square' in study2.dataset.columns
    assert 'y_square' not in study1.dataset.columns  # immutable prior study


def test_no_derivation_path_is_identity():
    base = _pb()
    frame, resolved = base._materialize_derivations()
    assert frame is base.data       # no copy when nothing pending
    assert resolved == ()


def test_on_invalid_error_raises_structured_at_formulate():
    df = make_design(1, seed=2).copy()
    df.loc[df.index[0], 'y'] = -5.0   # force a log-domain violation
    p = ProcessBehavior(df).transform('y', 'log', on_invalid='error')
    with pytest.raises(pb.ValidationError, match='domain violation'):
        p.formulate(response='y_log', factors=['factor 1'], time='time')


def test_on_invalid_na_coerces_instead_of_raising():
    df = make_design(1, seed=2).copy()
    df.loc[df.index[0], 'y'] = -5.0
    # on_invalid='na' must not raise at materialization; the violation becomes NaN.
    p = ProcessBehavior(df).transform('y', 'log', on_invalid='na')
    frame, resolved = p._materialize_derivations()
    assert frame['y_log'].isna().any()


def test_one_frame_identity_gate():
    """validate / ODS-detect / ADS must all receive the SAME augmented frame."""
    import processbehavior.analysis_dataset as ads_mod
    import processbehavior.data_preparation as dp_mod
    import processbehavior.sds_detector as sds_mod

    captured = {}
    orig_validate = dp_mod.DataPreparation.validate_columns
    orig_detect = sds_mod.SDSRegistry.detect_sds_from_structure
    orig_ads = ads_mod.AnalysisDataSet.__init__

    # validate_columns is also called *inside* AnalysisDataSet.prepare_dataset
    # with a copy; capture only the FIRST (outer, in _formulate_from_frame) call.
    def cap_validate(self, df, spec, *a, **k):
        captured.setdefault('validate', df)
        return orig_validate(self, df, spec, *a, **k)

    def cap_detect(self, df, spec, *a, **k):
        captured.setdefault('detect', df)
        return orig_detect(self, df, spec, *a, **k)

    def cap_ads(self, df, spec, *a, **k):
        captured.setdefault('ads', df)
        return orig_ads(self, df, spec, *a, **k)

    with mock.patch.object(dp_mod.DataPreparation, 'validate_columns', cap_validate), \
         mock.patch.object(sds_mod.SDSRegistry, 'detect_sds_from_structure', cap_detect), \
         mock.patch.object(ads_mod.AnalysisDataSet, '__init__', cap_ads):
        _pb().transform('y', 'square').formulate(
            response='y_square', factors=['factor 1'], time='time'
        )

    # Pairwise identity across the CAPTURED references (not presence checks).
    assert captured['validate'] is captured['detect'] is captured['ads']
    assert 'y_square' in captured['validate'].columns


def test_preordered_categorical_factor_order_preserved_without_derivations():
    # A pre-ordered categorical factor in non-natsort order, no derivations.
    df = pd.DataFrame({
        'grade': pd.Categorical(['B', 'A', 'C'] * 8, categories=['C', 'B', 'A'], ordered=True),
        'time': np.repeat(np.arange(1, 9), 3),
        'y': np.linspace(10.0, 20.0, 24),
    })
    study = ProcessBehavior(df).formulate(response='y', factors=['grade'], time='time')
    # Supplied order C < B < A is preserved, not natsorted to A, B, C.
    assert list(study.dataset['rsg'].cat.categories) == ['C', 'B', 'A']
