"""`ControlChartFigure` delegation, and honest static-export errors.

Two bugs are pinned here.

The class docstring promised "full access to Plotly's API" and delivered six
methods: `write_html`, `write_image`, `update_traces`, `add_hline` — every
plotly user's muscle memory — raised AttributeError, and the `.figure` escape
hatch was undiscoverable from the error.

And `save_image` decided what went wrong by substring-matching "kaleido" in the
message, so a *present* kaleido that could not find a browser was reported as
"Image export requires kaleido", sending people to reinstall a package they
already had.
"""

import copy

import plotly.graph_objects as go
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets.synthetic import make_design
from processbehavior.plotting.control_chart import ControlChartFigure, _translate_image_error

pytestmark = pytest.mark.plotting


@pytest.fixture(scope='module')
def figure():
    df = make_design(state=1, seed=42)
    study = ProcessBehavior(df).formulate(
        response='y', factors=['factor 1', 'factor 2'], time='time'
    )
    return study.execute().plot()


class TestPlotlyDelegation:
    def test_is_the_wrapper_not_a_figure(self, figure):
        """Delegation, not inheritance — `.figure` remains the real one."""
        assert isinstance(figure, ControlChartFigure)
        assert isinstance(figure.figure, go.Figure)

    def test_delegates_attributes(self, figure):
        assert len(figure.data) == len(figure.figure.data)
        assert figure.layout is figure.figure.layout

    @pytest.mark.parametrize('method', ['update_traces', 'add_hline', 'add_vline', 'update_xaxes'])
    def test_delegates_methods(self, figure, method):
        assert callable(getattr(figure, method))

    def test_own_methods_are_not_shadowed(self, figure):
        """__getattr__ only fires on lookup failure, so these stay ours."""
        assert figure.update_layout.__qualname__.startswith('ControlChartFigure')
        assert figure.save_html.__qualname__.startswith('ControlChartFigure')

    def test_dir_includes_both_surfaces(self, figure):
        names = dir(figure)
        assert 'save_html' in names
        assert 'update_traces' in names

    def test_unknown_attribute_still_raises(self, figure):
        with pytest.raises(AttributeError):
            figure.definitely_not_a_plotly_method  # noqa: B018

    def test_dunder_lookup_refused(self, figure):
        """Letting dunders delegate makes copy/pickle/IPython behave erratically."""
        with pytest.raises(AttributeError):
            figure.__deepcopy__  # noqa: B018

    def test_copy_works(self, figure):
        assert isinstance(copy.copy(figure), ControlChartFigure)


class TestPlotlySpellings:
    def test_write_html_writes(self, figure, tmp_path):
        target = tmp_path / 'chart.html'
        figure.write_html(str(target))
        assert target.stat().st_size > 1000

    def test_save_html_still_writes(self, figure, tmp_path):
        target = tmp_path / 'chart2.html'
        figure.save_html(target)
        assert target.stat().st_size > 1000

    def test_write_image_translates_errors(self, figure, monkeypatch, tmp_path):
        """The alias must not bypass translation and leak kaleido's raw error."""
        monkeypatch.setattr(
            figure.figure, 'write_image', lambda *a, **k: (_ for _ in ()).throw(ImportError('no kaleido'))
        )
        with pytest.raises(ImportError, match='requires kaleido'):
            figure.write_image(str(tmp_path / 'x.png'))


class TestImageErrorTranslation:
    """Three failures needing three different fixes must not collapse into one."""

    def test_missing_kaleido_is_import_error(self):
        translated = _translate_image_error(ImportError('No module named kaleido'))
        assert isinstance(translated, ImportError)
        assert 'pip install' in str(translated)

    def test_missing_chrome_is_not_reported_as_missing_kaleido(self):
        class ChromeNotFoundError(Exception):
            pass

        ChromeNotFoundError.__module__ = 'kaleido.errors'
        translated = _translate_image_error(
            ChromeNotFoundError('Kaleido requires Google Chrome to be installed')
        )
        assert not isinstance(translated, ImportError)
        assert isinstance(translated, RuntimeError)
        assert 'chrome' in str(translated).lower()
        assert 'plotly_get_chrome' in str(translated)

    def test_unrelated_error_passes_through_unchanged(self):
        original = ValueError('width must be positive')
        assert _translate_image_error(original) is original

    def test_save_image_chains_the_cause(self, figure, monkeypatch, tmp_path):
        monkeypatch.setattr(
            figure.figure, 'write_image', lambda *a, **k: (_ for _ in ()).throw(ImportError('no kaleido'))
        )
        with pytest.raises(ImportError) as exc:
            figure.save_image(tmp_path / 'x.png')
        assert exc.value.__cause__ is not None
