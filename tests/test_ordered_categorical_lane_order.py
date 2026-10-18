"""An ordered pandas Categorical factor orders the lanes on every chart (issue #144, PR 4).

The Xbar/S aggregation groups on the factor column and so followed an ordered Categorical's
category order; the canonical row order in ``DataPreparation.build_keys`` sorted the raw labels,
so X/mR lanes, phases and ``strata`` disagreed (``after, before`` for a stage ``before < after``).
Plain string and numeric factors keep their previous order, which the golden masters and the
Bishop gate pin.
"""

from __future__ import annotations

import pandas as pd

import processbehavior as pb


def _staged_frame(labels=('before', 'after')) -> pd.DataFrame:
    rows = []
    for base, stage in zip((10, 20), labels, strict=True):
        for t in range(1, 4):
            for v in (0, 1, 2):
                rows.append({'stage': stage, 'time': t, 'y': float(base + t + v)})
    return pd.DataFrame(rows)


def _lane_order(study, **kwargs) -> list[str]:
    data = study.execute(chart='X', by=[], phased=True, **kwargs).get_chart('X')
    return list(dict.fromkeys(data['phase']))


class TestOrderedCategorical:
    def test_every_view_follows_the_category_order(self):
        df = _staged_frame()
        ordered = df.assign(stage=pd.Categorical(df['stage'], categories=['before', 'after'], ordered=True))
        study = pb.formulate(ordered, response='y', time='time', factors=['stage'])

        assert _lane_order(study) == ['before', 'after']
        assert study.execute(chart='X', by=['stage']).strata == ['before', 'after']
        xbar = study.execute(chart='Xbar', phased=True)
        assert list(xbar.phase_statistics('Xbar')['phase']) == ['before', 'after']
        boundaries = study.execute(chart='X', by=[]).charts['X']['metadata']['lane_boundaries']
        assert [(b['position'], b['label']) for b in boundaries] == [(9, 'after')]

        frame = study._ads.analysis_dataset.sort_values('sort_key')
        assert list(frame['stage'].astype(str).unique()) == ['before', 'after']
        assert list(frame['sort_key']) == list(range(len(frame)))

    def test_plain_strings_still_sort_lexically(self):
        study = pb.formulate(_staged_frame(), response='y', time='time', factors=['stage'])
        assert _lane_order(study) == ['after', 'before']
        assert study.execute(chart='X', by=['stage']).strata == ['after', 'before']
        assert list(study.execute(chart='Xbar', phased=True).phase_statistics('Xbar')['phase']) == ['after', 'before']

    def test_sortable_labels_need_no_categorical(self):
        df = _staged_frame(labels=('1 before', '2 after'))
        study = pb.formulate(df, response='y', time='time', factors=['stage'])
        assert _lane_order(study) == ['1 before', '2 after']

    def test_two_factors_with_one_ordered(self):
        df = _staged_frame()
        two = pd.concat([df.assign(machine=1), df.assign(machine=2, y=df['y'] + 0.5)], ignore_index=True)
        two['stage'] = pd.Categorical(two['stage'], categories=['before', 'after'], ordered=True)
        study = pb.formulate(two, response='y', time='time', factors=['stage', 'machine'])
        assert _lane_order(study) == ['before_1', 'before_2', 'after_1', 'after_2']
        assert study.execute(chart='X', by=['stage']).strata == ['before', 'after']

    def test_rows_within_a_cell_keep_source_order(self):
        df = _staged_frame()
        ordered = df.assign(stage=pd.Categorical(df['stage'], categories=['before', 'after'], ordered=True))
        study = pb.formulate(ordered, response='y', time='time', factors=['stage'])
        frame = study._ads.analysis_dataset.sort_values('sort_key')
        for _, cell in frame.groupby(['stage', 'time'], observed=True, sort=False):
            assert list(cell['obs_id']) == sorted(cell['obs_id'])
