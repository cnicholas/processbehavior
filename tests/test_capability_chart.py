"""
Tests for process capability visualization chart.

Tests assert semantic presence (shapes, annotations, trace types), not exact
numeric bins or strict ordering. Follows the existing test_capability.py pattern.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from processbehavior.capability import CapabilityResult, SpecLimits
from processbehavior.plotting import create_capability_chart
from processbehavior.plotting.themes import get_theme

# ============================================================================
# Fixtures
# ============================================================================


def _make_cap_two_sided(*, with_r2: bool = True, sigma_hat: float = 1.5) -> CapabilityResult:
    """Two-sided CapabilityResult with known values."""
    specs = SpecLimits(usl=243, lsl=233, target=238)
    y_bar = 238.0
    n = 100

    pp = (243 - 233) / (6 * sigma_hat)
    ppk_lower = (y_bar - 233) / (3 * sigma_hat)
    ppk_upper = (243 - y_bar) / (3 * sigma_hat)
    ppk = min(ppk_lower, ppk_upper)

    cp = cpk = cpk_lower = cpk_upper = sigma_hat_r2 = None
    potential_values = None
    reason = 'no R2' if not with_r2 else None
    if with_r2:
        sigma_hat_r2 = 0.8
        cp = (243 - 233) / (6 * sigma_hat_r2)
        cpk_lower = (y_bar - 233) / (3 * sigma_hat_r2)
        cpk_upper = (243 - y_bar) / (3 * sigma_hat_r2)
        cpk = min(cpk_lower, cpk_upper)
        # Synthetic R2 recentered to y_bar: tight distribution around y_bar
        rng = np.random.default_rng(123)
        potential_values = y_bar + rng.normal(0, 0.8, size=n)

    # Potential outside: computed from potential_values if available
    pot_n_below = pot_n_above = pot_pct_below = pot_pct_above = None
    if potential_values is not None:
        pot_n_below = int(np.sum(potential_values < specs.lsl))
        pot_n_above = int(np.sum(potential_values > specs.usl))
        pot_pct_below = pot_n_below / n * 100
        pot_pct_above = pot_n_above / n * 100

    return CapabilityResult(
        specs=specs,
        n=n,
        y_bar=y_bar,
        s=sigma_hat * 0.99,
        sigma_hat=sigma_hat,
        pp=pp,
        ppk_lower=ppk_lower,
        ppk_upper=ppk_upper,
        ppk=ppk,
        sigma_hat_r2=sigma_hat_r2,
        cp=cp,
        cpk_lower=cpk_lower,
        cpk_upper=cpk_upper,
        cpk=cpk,
        potential_unavailable_reason=reason,
        potential_values=potential_values,
        z_lower=(y_bar - 233) / sigma_hat,
        z_upper=(243 - y_bar) / sigma_hat,
        n_below_lsl=2,
        n_above_usl=3,
        n_outside=5,
        pct_below_lsl=2.0,
        pct_above_usl=3.0,
        pct_outside=5.0,
        potential_n_below_lsl=pot_n_below,
        potential_n_above_usl=pot_n_above,
        potential_pct_below_lsl=pot_pct_below,
        potential_pct_above_usl=pot_pct_above,
        round_to=3,
    )


def _make_cap_usl_only() -> CapabilityResult:
    """USL-only CapabilityResult."""
    specs = SpecLimits(usl=242)
    y_bar = 238.0
    sigma_hat = 1.5
    return CapabilityResult(
        specs=specs,
        n=100,
        y_bar=y_bar,
        s=1.485,
        sigma_hat=sigma_hat,
        pp=None,
        ppk_lower=None,
        ppk_upper=(242 - y_bar) / (3 * sigma_hat),
        ppk=(242 - y_bar) / (3 * sigma_hat),
        sigma_hat_r2=None,
        cp=None,
        cpk_lower=None,
        cpk_upper=None,
        cpk=None,
        potential_unavailable_reason='no R2',
        z_lower=None,
        z_upper=(242 - y_bar) / sigma_hat,
        n_below_lsl=None,
        n_above_usl=5,
        n_outside=5,
        pct_below_lsl=None,
        pct_above_usl=5.0,
        pct_outside=5.0,
        round_to=3,
    )


def _make_cap_lsl_only() -> CapabilityResult:
    """LSL-only CapabilityResult."""
    specs = SpecLimits(lsl=234)
    y_bar = 238.0
    sigma_hat = 1.5
    return CapabilityResult(
        specs=specs,
        n=100,
        y_bar=y_bar,
        s=1.485,
        sigma_hat=sigma_hat,
        pp=None,
        ppk_lower=(y_bar - 234) / (3 * sigma_hat),
        ppk_upper=None,
        ppk=(y_bar - 234) / (3 * sigma_hat),
        sigma_hat_r2=None,
        cp=None,
        cpk_lower=None,
        cpk_upper=None,
        cpk=None,
        potential_unavailable_reason='no R2',
        z_lower=(y_bar - 234) / sigma_hat,
        z_upper=None,
        n_below_lsl=3,
        n_above_usl=None,
        n_outside=3,
        pct_below_lsl=3.0,
        pct_above_usl=None,
        pct_outside=3.0,
        round_to=3,
    )


def _make_cap_sigma_zero() -> CapabilityResult:
    """CapabilityResult with sigma_hat=0 (all identical values)."""
    specs = SpecLimits(usl=243, lsl=233)
    return CapabilityResult(
        specs=specs,
        n=50,
        y_bar=238.0,
        s=0.0,
        sigma_hat=0.0,
        pp=float('inf'),
        ppk_lower=float('inf'),
        ppk_upper=float('inf'),
        ppk=float('inf'),
        sigma_hat_r2=None,
        cp=None,
        cpk_lower=None,
        cpk_upper=None,
        cpk=None,
        potential_unavailable_reason='no R2',
        z_lower=float('inf'),
        z_upper=float('inf'),
        n_below_lsl=0,
        n_above_usl=0,
        n_outside=0,
        pct_below_lsl=0.0,
        pct_above_usl=0.0,
        pct_outside=0.0,
        round_to=3,
    )


@pytest.fixture
def sample_values():
    """100 normally-distributed values around 238."""
    rng = np.random.default_rng(42)
    return rng.normal(238, 1.5, size=100)


# ============================================================================
# Histogram Trace
# ============================================================================


class TestHistogram:
    """Verify the histogram trace."""

    def test_single_histogram_trace(self, sample_values):
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values)
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histograms) == 1

    def test_nbins_passed_through(self, sample_values):
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values, nbins=20)
        hist = fig.data[0]
        assert hist.nbinsx == 20

    def test_x_label_applied(self, sample_values):
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values, x_label='Measurement (mm)')
        assert fig.layout.xaxis.title.text == 'Measurement (mm)'


# ============================================================================
# Spec Lines (Shapes)
# ============================================================================


def _get_vline_shapes(fig: go.Figure) -> list:
    """Extract vline shapes from figure (type='line', x0==x1)."""
    return [s for s in (fig.layout.shapes or []) if s.type == 'line' and s.x0 == s.x1]


def _get_vrect_shapes(fig: go.Figure) -> list:
    """Extract vrect shapes from figure (type='rect')."""
    return [s for s in (fig.layout.shapes or []) if s.type == 'rect']


class TestSpecLines:
    """Verify spec limit vertical lines."""

    @pytest.mark.parametrize(
        'make_cap, expected_present, expected_absent',
        [
            pytest.param(
                _make_cap_two_sided,
                [233, 243],
                [],
                id='two_sided-lsl_and_usl',
            ),
            pytest.param(
                _make_cap_usl_only,
                [242],
                [234],
                id='usl_only-usl_no_lsl',
            ),
            pytest.param(
                _make_cap_lsl_only,
                [234],
                [],
                id='lsl_only-lsl_present',
            ),
        ],
    )
    def test_spec_lines(self, sample_values, make_cap, expected_present, expected_absent):
        cap = make_cap()
        fig = create_capability_chart(cap, sample_values)
        vlines = _get_vline_shapes(fig)
        x_values = [s.x0 for s in vlines]
        for val in expected_present:
            assert val in x_values, f'Expected {val} in vline x-values'
        for val in expected_absent:
            assert val not in x_values, f'Expected {val} NOT in vline x-values'

    def test_target_line_present(self, sample_values):
        """Two-sided with target=238 → target vline present."""
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values)
        vlines = _get_vline_shapes(fig)
        x_values = [s.x0 for s in vlines]
        assert 238 in x_values  # target


# ============================================================================
# Out-of-Spec Shading (vrects)
# ============================================================================


def _wide_values():
    """Values that extend well beyond typical spec limits."""
    rng = np.random.default_rng(99)
    return rng.normal(238, 5, size=200)


class TestOutOfSpecShading:
    """Verify vrect shading for out-of-spec regions."""

    @pytest.mark.parametrize(
        'make_cap, expected_count, check_fn',
        [
            pytest.param(
                _make_cap_two_sided,
                2,
                None,
                id='two_sided-two_vrects',
            ),
            pytest.param(
                _make_cap_usl_only,
                1,
                lambda vrects: vrects[0].x0 >= 240,
                id='usl_only-usl_vrect',
            ),
            pytest.param(
                _make_cap_lsl_only,
                1,
                lambda vrects: vrects[0].x1 <= 236,
                id='lsl_only-lsl_vrect',
            ),
        ],
    )
    def test_vrect_shading(self, make_cap, expected_count, check_fn):
        """With wide data, correct vrects appear per spec configuration."""
        cap = make_cap()
        vals = _wide_values()
        fig = create_capability_chart(cap, vals)
        vrects = _get_vrect_shapes(fig)
        assert len(vrects) == expected_count
        if check_fn is not None:
            assert check_fn(vrects)


# ============================================================================
# Natural Process Limits
# ============================================================================


class TestNPLLines:
    """Verify NPL lines (mean, LNPL, UNPL)."""

    def test_npl_lines_present_normal_sigma(self, sample_values):
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values)
        vlines = _get_vline_shapes(fig)
        x_values = [s.x0 for s in vlines]

        # Mean line
        assert cap.y_bar in x_values

        # LNPL and UNPL
        lnpl = cap.y_bar - 3 * cap.sigma_hat
        unpl = cap.y_bar + 3 * cap.sigma_hat
        assert lnpl in x_values
        assert unpl in x_values

    def test_sigma_zero_only_mean(self, sample_values):
        """When sigma=0, only mean line — no LNPL/UNPL."""
        cap = _make_cap_sigma_zero()
        fig = create_capability_chart(cap, sample_values)
        vlines = _get_vline_shapes(fig)
        x_values = [s.x0 for s in vlines]

        # Mean line present
        assert cap.y_bar in x_values

        # LNPL and UNPL would be at y_bar (same point) — should NOT be drawn
        # Count how many lines are at y_bar: should be exactly 1 (the mean)
        mean_lines = [v for v in vlines if v.x0 == cap.y_bar]
        assert len(mean_lines) == 1


# ============================================================================
# Annotation Box
# ============================================================================


def _get_annotation_text(fig: go.Figure) -> str:
    """Get text from the capability index annotation (top-right, monospace)."""
    for ann in fig.layout.annotations:
        if hasattr(ann, 'font') and ann.font and ann.font.family == 'monospace':
            return ann.text
    return ''


class TestAnnotationBox:
    """Verify capability index annotation content."""

    @pytest.mark.parametrize(
        'make_cap_kwargs, chart_kwargs, expected_present, expected_absent',
        [
            pytest.param(
                {},
                {},
                ['PP  Index', 'PPL Index', 'PPU Index'],
                [],
                id='two_sided-shows_pp_ppl_ppu',
            ),
            pytest.param(
                {'with_r2': True},
                {},
                ['PP  Index'],
                ['Cp ', 'Cpk'],
                id='two_sided_with_r2-current_omits_cp_cpk',
            ),
            pytest.param(
                {'with_r2': False},
                {},
                ['PP  Index', 'PPL Index', 'PPU Index'],
                ['Cp ', 'Cpk'],
                id='no_r2-omits_cp_cpk',
            ),
            pytest.param(
                {'with_r2': True},
                {'show_potential': False},
                ['PP  Index'],
                ['Cp ', 'Cpk'],
                id='show_potential_false-omits_cp_cpk',
            ),
        ],
    )
    def test_annotation_indices(
        self,
        sample_values,
        make_cap_kwargs,
        chart_kwargs,
        expected_present,
        expected_absent,
    ):
        cap = _make_cap_two_sided(**make_cap_kwargs)
        fig = create_capability_chart(cap, sample_values, **chart_kwargs)
        text = _get_annotation_text(fig)
        for label in expected_present:
            assert label in text, f"Expected '{label}' in annotation"
        for label in expected_absent:
            assert label not in text, f"Expected '{label}' NOT in annotation"

    @pytest.mark.parametrize(
        'make_cap, expected_labels',
        [
            pytest.param(_make_cap_usl_only, ['PPU Index', 'Pct Above USL'], id='usl_only'),
            pytest.param(_make_cap_lsl_only, ['PPL Index', 'Pct Below LSL'], id='lsl_only'),
        ],
    )
    def test_one_sided_annotations(self, sample_values, make_cap, expected_labels):
        cap = make_cap()
        fig = create_capability_chart(cap, sample_values)
        text = _get_annotation_text(fig)
        for label in expected_labels:
            assert label in text, f"Expected '{label}' in annotation"

    def test_shows_n_and_sigma(self, sample_values):
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values)
        text = _get_annotation_text(fig)
        assert 'n = 100' in text
        assert '\u03c3\u0302' in text

    def test_two_sided_shows_pct_below_and_above(self, sample_values):
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values)
        text = _get_annotation_text(fig)
        assert 'Pct Below LSL' in text
        assert 'Pct Above USL' in text


# ============================================================================
# Title
# ============================================================================


class TestTitle:
    """Verify title generation."""

    @pytest.mark.parametrize(
        'make_cap, expected_substrings',
        [
            pytest.param(
                _make_cap_two_sided,
                ['LSL=233', 'USL=243', 'Target=238'],
                id='two_sided',
            ),
            pytest.param(
                _make_cap_usl_only,
                ['USL=242'],
                id='usl_only',
            ),
        ],
    )
    def test_auto_title(self, sample_values, make_cap, expected_substrings):
        cap = make_cap()
        fig = create_capability_chart(cap, sample_values)
        for substr in expected_substrings:
            assert substr in fig.layout.title.text

    def test_custom_title(self, sample_values):
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values, title='Custom Title')
        assert fig.layout.title.text == 'Custom Title'


# ============================================================================
# Theme
# ============================================================================


class TestTheme:
    """Verify theme colors are applied."""

    @pytest.mark.parametrize(
        'theme_name, chart_kwargs',
        [
            pytest.param('dark', {'theme': get_theme('dark')}, id='custom_dark'),
            pytest.param('processbehavior', {}, id='default'),
        ],
    )
    def test_theme_data_color(self, sample_values, theme_name, chart_kwargs):
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values, **chart_kwargs)
        hist = fig.data[0]
        assert hist.marker.color == get_theme(theme_name).data_color


# ============================================================================
# Convenience Method
# ============================================================================


class TestConvenienceMethod:
    """Verify CapabilityResult.plot() works."""

    def test_plot_returns_figure(self, sample_values):
        cap = _make_cap_two_sided()
        fig = cap.plot(sample_values)
        assert isinstance(fig, go.Figure)

    def test_plot_passes_kwargs(self, sample_values):
        cap = _make_cap_two_sided()
        fig = cap.plot(sample_values, title='Via .plot()', nbins=15)
        assert fig.layout.title.text == 'Via .plot()'
        assert fig.data[0].nbinsx == 15


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge cases for values normalization and special inputs."""

    def test_nan_values_dropped(self):
        """NaN values in the input are dropped silently."""
        cap = _make_cap_two_sided()
        values = [238.0, np.nan, 237.0, np.nan, 239.0]
        fig = create_capability_chart(cap, values)
        assert isinstance(fig, go.Figure)
        # Histogram should have 3 values
        assert len(fig.data[0].x) == 3

    @pytest.mark.parametrize(
        'make_values',
        [
            pytest.param(lambda: [237.0, 238.0, 239.0, 240.0], id='list'),
            pytest.param(
                lambda: __import__('pandas').Series([237.0, 238.0, 239.0, 240.0]),
                id='pandas_series',
            ),
        ],
    )
    def test_accepts_array_like_input(self, make_values):
        """Various array-like inputs work."""
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, make_values())
        assert isinstance(fig, go.Figure)

    def test_width_height(self, sample_values):
        """Width and height parameters are applied."""
        cap = _make_cap_two_sided()
        fig = create_capability_chart(cap, sample_values, width=1200, height=700)
        assert fig.layout.width == 1200
        assert fig.layout.height == 700


# ============================================================================
# View Parameter
# ============================================================================


class TestViewParameter:
    """Tests for the view='current'|'potential' parameter."""

    def test_view_current_is_default(self, sample_values):
        """cap.plot(values) produces same output as cap.plot(values, view='current')."""
        cap = _make_cap_two_sided(with_r2=True)
        fig_default = create_capability_chart(cap, sample_values)
        fig_explicit = create_capability_chart(cap, sample_values, view='current')
        # Same annotation text
        assert _get_annotation_text(fig_default) == _get_annotation_text(fig_explicit)
        # Same number of shapes (vlines)
        assert len(fig_default.layout.shapes) == len(fig_explicit.layout.shapes)

    def test_view_potential_histograms_recentered_r2(self, sample_values):
        """view='potential' → histogram uses y_bar + R2, not raw Y values."""
        cap = _make_cap_two_sided(with_r2=True)
        fig = create_capability_chart(cap, sample_values, view='potential')
        hist = [t for t in fig.data if isinstance(t, go.Histogram)][0]
        # Histogram x-data should be potential_values, not raw sample_values
        np.testing.assert_array_equal(hist.x, cap.potential_values)

    def test_view_potential_no_npl_lines(self, sample_values):
        """view='potential' → no NPL vlines (only spec lines + mean)."""
        cap = _make_cap_two_sided(with_r2=True)
        fig = create_capability_chart(cap, sample_values, view='potential')
        vlines = _get_vline_shapes(fig)
        x_values = [s.x0 for s in vlines]

        # NPL lines should NOT be present
        lnpl_r2 = cap.y_bar - 3 * cap.sigma_hat_r2
        unpl_r2 = cap.y_bar + 3 * cap.sigma_hat_r2
        assert lnpl_r2 not in x_values
        assert unpl_r2 not in x_values

        # Mean line should still be present
        assert cap.y_bar in x_values

    @pytest.mark.parametrize(
        'extra_kwargs, expected_yaxis',
        [
            pytest.param({}, 'Percent', id='default_percent'),
            pytest.param(
                {'histnorm': 'probability density'},
                'Probability Density',
                id='histnorm_override',
            ),
        ],
    )
    def test_view_potential_yaxis_label(self, sample_values, extra_kwargs, expected_yaxis):
        """view='potential' y-axis label reflects histnorm setting."""
        cap = _make_cap_two_sided(with_r2=True)
        fig = create_capability_chart(cap, sample_values, view='potential', **extra_kwargs)
        assert fig.layout.yaxis.title.text == expected_yaxis

    def test_view_current_still_shows_npl_lines(self, sample_values):
        """Regression: current view still has NPL lines."""
        cap = _make_cap_two_sided(with_r2=True)
        fig = create_capability_chart(cap, sample_values, view='current')
        vlines = _get_vline_shapes(fig)
        x_values = [s.x0 for s in vlines]

        lnpl = cap.y_bar - 3 * cap.sigma_hat
        unpl = cap.y_bar + 3 * cap.sigma_hat
        assert lnpl in x_values
        assert unpl in x_values

    def test_view_potential_annotation_shows_cp_cpl_cpu(self, sample_values):
        """view='potential' → annotation has CP/CPL/CPU Index, not PP/PPL/PPU."""
        cap = _make_cap_two_sided(with_r2=True)
        fig = create_capability_chart(cap, sample_values, view='potential')
        text = _get_annotation_text(fig)
        assert 'CP  Index' in text
        assert 'CPL Index' in text
        assert 'CPU Index' in text
        assert 'PP  Index' not in text
        assert 'PPL Index' not in text
        assert 'PPU Index' not in text
        assert '\u03c3\u0302(R2)' in text
        assert 'Pct Below LSL' in text
        assert 'Pct Above USL' in text

    def test_view_potential_without_r2_raises(self, sample_values):
        """SDS without R2 → ValidationError with helpful message."""
        from processbehavior.exceptions import ValidationError

        cap = _make_cap_two_sided(with_r2=False)
        with pytest.raises(ValidationError, match='Cannot plot potential capability'):
            create_capability_chart(cap, sample_values, view='potential')

    def test_invalid_view_raises(self, sample_values):
        """view='foo' → ValueError."""
        cap = _make_cap_two_sided()
        with pytest.raises(ValueError, match='view must be'):
            create_capability_chart(cap, sample_values, view='foo')

    def test_view_potential_title(self, sample_values):
        """view='potential' auto-title says 'Potential Capability'."""
        cap = _make_cap_two_sided(with_r2=True)
        fig = create_capability_chart(cap, sample_values, view='potential')
        assert 'Potential Capability' in fig.layout.title.text


# ============================================================================
# Paired Parameter
# ============================================================================


class TestPairedParameter:
    """Tests for the paired=True two-panel facet."""

    def test_paired_structure(self, sample_values):
        """paired=True → 2 subplots, 2 histograms, 2 annotation boxes."""
        cap = _make_cap_two_sided(with_r2=True)
        fig = create_capability_chart(cap, sample_values, paired=True)
        assert isinstance(fig, go.Figure)
        # Two subplots
        assert fig.layout.xaxis2 is not None
        assert fig.layout.yaxis2 is not None
        # Two histogram traces
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histograms) == 2
        # Two monospace annotation boxes
        mono_annotations = [
            a for a in fig.layout.annotations if hasattr(a, 'font') and a.font and a.font.family == 'monospace'
        ]
        assert len(mono_annotations) == 2

    def test_paired_without_r2_warns_and_falls_back(self, sample_values):
        """No R2 → warning + single current panel."""
        cap = _make_cap_two_sided(with_r2=False)
        with pytest.warns(UserWarning, match='Potential capability unavailable'):
            fig = create_capability_chart(cap, sample_values, paired=True)
        # Falls back to single chart — only one histogram
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histograms) == 1

    def test_paired_subplot_titles(self, sample_values):
        """paired=True → subplot titles include 'Current' and 'Potential'."""
        cap = _make_cap_two_sided(with_r2=True)
        fig = create_capability_chart(cap, sample_values, paired=True)
        annotation_texts = [a.text for a in fig.layout.annotations]
        assert 'Current Capability' in annotation_texts
        assert 'Potential Capability' in annotation_texts
