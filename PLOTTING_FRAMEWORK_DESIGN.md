# Plotting Framework Design for ProcessBehavior

## Executive Summary

**Recommendation: Option 3 (Plotly with plotly.graph_objects + Abstraction Layer)**

This approach provides the best balance of:
- Interactive web-based output (critical for modern analysis)
- Faceting support out-of-the-box
- Extensibility for future enhancements
- Pythonic, intuitive API
- Zero server requirements (static HTML export)

---

## Design Goals

Based on the Pythonic Hadley persona:

1. **Human-First API** - Simple, natural syntax: `result.plot()` just works
2. **Consistency** - Same API whether plotting one chart or faceted charts
3. **Composability** - Build complex visualizations from simple parts
4. **Progressive Disclosure** - Simple by default, powerful when needed
5. **Fail Helpful** - Clear guidance when something goes wrong
6. **Faceting Support** - First-class citizen for stratified/grouped charts

---

## Current State Analysis

The `AnalysisResult` object already has:
- ✅ Well-structured chart data (`charts` dict)
- ✅ Statistics for each chart
- ✅ Stratified chart support
- ✅ Signal detection (beyond_limits)
- ✅ Rich metadata (SDS, summary)

**Missing:**
- ❌ No plotting capability
- ❌ No visualization framework
- ❌ No faceting/small multiples support

---

## Option 1: Matplotlib + Seaborn

### API Design
```python
# Simple case
result.plot()  # Auto-plots all charts

# Specific chart
result.plot(chart='Xbar')

# Faceted (stratified)
result.plot(facet_by='Operator')  # Small multiples

# Customization
result.plot(
    chart='Xbar',
    style='dark',
    highlight_signals=True,
    figsize=(12, 6)
)
```

### Implementation Approach
```python
# processbehavior/plotting/matplotlib_backend.py

from typing import Optional, Literal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

class MatplotlibPlotter:
    """Matplotlib backend for control charts."""

    def plot_control_chart(
        self,
        data: pd.DataFrame,
        statistics: dict,
        chart_type: str,
        ax: Optional[plt.Axes] = None,
        highlight_signals: bool = True
    ) -> plt.Axes:
        """Plot a single control chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        x = data.index
        y = data['mean'] if 'mean' in data.columns else data.iloc[:, 0]

        ax.plot(x, y, 'o-', color='steelblue', label='Data')

        # Plot control limits
        if 'ucl' in statistics:
            ax.axhline(statistics['ucl'], color='red',
                      linestyle='--', label='UCL')
        if 'lcl' in statistics:
            ax.axhline(statistics['lcl'], color='red',
                      linestyle='--', label='LCL')
        if 'Mean' in statistics:
            ax.axhline(statistics['Mean'], color='green',
                      linestyle='-', label='Center')

        # Highlight signals
        if highlight_signals and 'beyond_limits' in data.columns:
            signals = data[data['beyond_limits'] != 0]
            if not signals.empty:
                ax.scatter(signals.index, signals[y.name],
                          color='red', s=100, zorder=5,
                          label='Signal')

        ax.set_title(f"{chart_type} Chart")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_faceted(
        self,
        charts: dict,
        ncols: int = 2,
        figsize: Optional[tuple] = None
    ) -> Figure:
        """Plot multiple charts in small multiples."""
        n_charts = len(charts)
        nrows = (n_charts + ncols - 1) // ncols

        if figsize is None:
            figsize = (ncols * 6, nrows * 4)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_charts > 1 else [axes]

        for idx, (name, chart_info) in enumerate(charts.items()):
            self.plot_control_chart(
                data=chart_info['data'],
                statistics=chart_info['statistics'],
                chart_type=name,
                ax=axes[idx]
            )

        # Hide unused subplots
        for idx in range(n_charts, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig
```

### Pros
✅ **Mature & Stable** - Battle-tested, widely used
✅ **Publication Quality** - Perfect for papers/reports
✅ **Highly Customizable** - Full control over every element
✅ **Large Community** - Extensive documentation, examples
✅ **No Dependencies** - Ships with scientific Python
✅ **Works Everywhere** - Jupyter, scripts, web (static)

### Cons
❌ **Static Output** - No interactivity (hover, zoom, pan)
❌ **Verbose Customization** - Requires many lines for styling
❌ **Not Web-Native** - Need to export to images for web
❌ **Faceting Complexity** - Manual subplot management
❌ **Less Modern Feel** - Not as polished as newer tools

### Use Cases
- Academic papers and reports
- Offline analysis workflows
- Environments without JavaScript
- Maximum control over appearance

---

## Option 2: Plotly Express (High-Level)

### API Design
```python
# Simple case - auto-detects everything
result.plot()

# Specific chart with faceting
result.plot(chart='Xbar', facet_col='Operator')

# Advanced customization
result.plot(
    chart='Xbar',
    facet_col='Operator',
    facet_col_wrap=3,
    color='Shift',
    template='plotly_white',
    height=600
)

# Export options
fig = result.plot()
fig.write_html('charts.html')  # Interactive HTML
fig.write_image('charts.png')  # Static image
```

### Implementation Approach
```python
# processbehavior/plotting/plotly_express_backend.py

import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

class PlotlyExpressPlotter:
    """Plotly Express backend for control charts."""

    def plot_control_chart(
        self,
        data: pd.DataFrame,
        statistics: dict,
        chart_type: str,
        facet_col: Optional[str] = None,
        template: str = 'plotly_white'
    ) -> go.Figure:
        """Plot control chart with Plotly Express."""

        # Prepare data for plotting
        df = data.copy()
        y_col = 'mean' if 'mean' in df.columns else df.columns[0]

        # Create base plot
        fig = px.line(
            df,
            x=df.index,
            y=y_col,
            facet_col=facet_col,
            template=template,
            title=f"{chart_type} Control Chart",
            labels={'x': 'Observation', 'y': chart_type}
        )

        # Add control limits as horizontal lines
        # (Requires iterating through facets if faceted)
        for trace in fig.data:
            # Add UCL
            if 'ucl' in statistics:
                fig.add_hline(
                    y=statistics['ucl'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="UCL"
                )
            # Add LCL
            if 'lcl' in statistics:
                fig.add_hline(
                    y=statistics['lcl'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="LCL"
                )
            # Add centerline
            if 'Mean' in statistics:
                fig.add_hline(
                    y=statistics['Mean'],
                    line_color="green",
                    annotation_text="Mean"
                )

        # Highlight signals
        if 'beyond_limits' in df.columns:
            signals = df[df['beyond_limits'] != 0]
            if not signals.empty:
                fig.add_scatter(
                    x=signals.index,
                    y=signals[y_col],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='x'),
                    name='Signal',
                    showlegend=True
                )

        fig.update_layout(hovermode='x unified')
        return fig
```

### Pros
✅ **Interactive by Default** - Hover, zoom, pan out-of-the-box
✅ **Web-Native** - Exports to standalone HTML
✅ **Concise API** - Very Pythonic, minimal code
✅ **Built-in Faceting** - `facet_col`, `facet_row` parameters
✅ **Modern Look** - Beautiful default styling
✅ **Animation Support** - Can animate over time
✅ **Free & Open Source** - No licensing issues

### Cons
❌ **Less Control** - High-level API hides complexity
❌ **Facet Limitations** - Control limits need manual handling per facet
❌ **Learning Curve** - Different paradigm from matplotlib
❌ **Larger Bundle Size** - ~3MB JavaScript for web
❌ **Template Constraints** - Custom styling can be tricky

### Use Cases
- Web dashboards and interactive reports
- Exploratory data analysis
- Quick prototyping
- Modern web-first applications

---

## Option 3: Plotly graph_objects (Low-Level) + Abstraction Layer ⭐ **RECOMMENDED**

### API Design
```python
# Dead simple - follows Hadley philosophy
result.plot()  # Auto-generates appropriate charts

# Single chart
result.plot(chart='Xbar')

# Faceted (stratified charts)
result.plot(facet=True)  # Automatically detects stratification
result.plot(chart='Imr', facet_by='Operator')

# Fine-grained control
result.plot(
    chart='Xbar',
    facet_by='Operator',
    highlight_signals=True,
    show_rules=True,  # Show run rules
    template='plotly_white',
    width=1200,
    height=600
)

# Residuals plotting
result.plot_residuals()  # R1-R5 faceted plot
result.plot_residuals(residual='R5')  # Single residual

# Effects plotting
result.plot_effects()  # Main effects
result.plot_interactions()  # Interaction plot

# Export
fig = result.plot()
fig.save_html('report.html')  # Our custom method
fig.save_image('chart.png')  # Requires kaleido
fig.show()  # Display in browser/notebook
```

### Implementation Approach

#### Core Architecture
```python
# processbehavior/plotting/__init__.py
"""
Plotting framework for ProcessBehavior.

Provides an intuitive, extensible API for creating control charts
with built-in support for faceting, interactivity, and export.

Design Philosophy:
- Simple tasks are trivial: result.plot()
- Complex tasks are possible: result.plot(facet_by='X', template='Y')
- Consistent API across all chart types
- Progressive disclosure of complexity
"""

from .plotter import Plotter
from .control_chart import ControlChartFigure

__all__ = ['Plotter', 'ControlChartFigure']
```

#### Main Plotter Class
```python
# processbehavior/plotting/plotter.py

from typing import Optional, Literal, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from .control_chart import ControlChartFigure
from .themes import THEMES, apply_theme

class Plotter:
    """
    Unified plotting interface for ProcessBehavior analysis results.

    This class orchestrates chart creation with built-in support for:
    - Single and faceted control charts
    - Automatic detection of chart types
    - Residual and effects visualization
    - Interactive and static export

    Examples
    --------
    Simple plotting:

    >>> plotter = Plotter(result)
    >>> fig = plotter.plot()
    >>> fig.show()

    Faceted charts:

    >>> fig = plotter.plot(facet_by='Operator')

    Customize appearance:

    >>> fig = plotter.plot(
    ...     chart='Xbar',
    ...     template='minimal',
    ...     highlight_signals=True
    ... )
    """

    def __init__(self, analysis_result):
        """
        Initialize plotter with an AnalysisResult.

        Parameters
        ----------
        analysis_result : AnalysisResult
            The result object from analysis
        """
        self.result = analysis_result
        self.charts = analysis_result.charts
        self.summary = analysis_result.summary

    def plot(
        self,
        chart: Optional[str] = None,
        facet: bool = False,
        facet_by: Optional[str] = None,
        ncols: int = 2,
        highlight_signals: bool = True,
        show_limits: bool = True,
        show_rules: bool = False,
        template: str = 'processbehavior',
        width: int = 1000,
        height: Optional[int] = None,
        title: Optional[str] = None
    ) -> ControlChartFigure:
        """
        Create control chart visualization.

        This is the main entry point for plotting. It automatically
        determines the best visualization based on your data structure.

        Parameters
        ----------
        chart : str, optional
            Specific chart to plot ('Xbar', 'Sbar', 'Imr', etc.)
            If None, plots all available charts
        facet : bool, default False
            Whether to create faceted plot for stratified data
        facet_by : str, optional
            Variable to facet by (overrides auto-detection)
        ncols : int, default 2
            Number of columns in faceted layout
        highlight_signals : bool, default True
            Whether to highlight points beyond control limits
        show_limits : bool, default True
            Whether to show control limit lines
        show_rules : bool, default False
            Whether to show additional run rules
        template : str, default 'processbehavior'
            Visual theme ('processbehavior', 'minimal', 'dark')
        width : int, default 1000
            Figure width in pixels
        height : int, optional
            Figure height in pixels (auto-calculated if None)
        title : str, optional
            Custom title for the figure

        Returns
        -------
        ControlChartFigure
            Interactive figure object with .show(), .save_html(), etc.

        Examples
        --------
        Auto-plot everything:

        >>> fig = plotter.plot()

        Specific chart:

        >>> fig = plotter.plot(chart='Xbar')

        Faceted by operator:

        >>> fig = plotter.plot(facet_by='Operator', ncols=3)

        Custom styling:

        >>> fig = plotter.plot(
        ...     template='dark',
        ...     highlight_signals=True,
        ...     show_rules=True
        ... )
        """
        # Validate inputs
        if chart and chart not in self.charts:
            available = list(self.charts.keys())
            raise ValueError(
                f"Chart '{chart}' not found.\n"
                f"Available charts: {available}\n"
                f"Hint: Use plotter.list_charts() to see all options"
            )

        # Determine what to plot
        if chart:
            charts_to_plot = {chart: self.charts[chart]}
        elif facet or facet_by or self.summary['is_stratified']:
            # Auto-detect stratified charts
            charts_to_plot = self.result.get_stratified_charts()
        else:
            # Plot all standard charts
            charts_to_plot = {
                k: v for k, v in self.charts.items()
                if k in ['Xbar', 'Sbar', 'Imr', 'R', 'all']
            }

        # Calculate height if not specified
        if height is None:
            n_charts = len(charts_to_plot)
            nrows = (n_charts + ncols - 1) // ncols
            height = nrows * 400

        # Create figure
        if len(charts_to_plot) == 1:
            fig = self._plot_single_chart(
                list(charts_to_plot.values())[0],
                list(charts_to_plot.keys())[0],
                highlight_signals=highlight_signals,
                show_limits=show_limits,
                width=width,
                height=height
            )
        else:
            fig = self._plot_faceted(
                charts_to_plot,
                ncols=ncols,
                highlight_signals=highlight_signals,
                show_limits=show_limits,
                width=width,
                height=height
            )

        # Apply theme
        fig = apply_theme(fig, template)

        # Set title
        if title:
            fig.update_layout(title=title)
        elif not title and len(charts_to_plot) == 1:
            chart_name = list(charts_to_plot.keys())[0]
            fig.update_layout(title=f"{chart_name} Control Chart")

        # Wrap in our custom figure class
        return ControlChartFigure(fig, self.result)

    def _plot_single_chart(
        self,
        chart_info: dict,
        chart_name: str,
        highlight_signals: bool,
        show_limits: bool,
        width: int,
        height: int
    ) -> go.Figure:
        """Create a single control chart."""
        data = chart_info['data']
        stats = chart_info['statistics']

        fig = go.Figure()

        # Determine value column
        value_col = self._get_value_column(data, chart_name)
        x_col = self._get_x_column(data)

        # Main data trace
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[value_col],
            mode='lines+markers',
            name='Data',
            marker=dict(size=8, color='steelblue'),
            line=dict(color='steelblue', width=2),
            hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'
        ))

        # Control limits
        if show_limits:
            # UCL
            if 'ucl' in stats and stats['ucl'] != 'Varies':
                fig.add_hline(
                    y=stats['ucl'],
                    line_dash='dash',
                    line_color='red',
                    annotation_text='UCL',
                    annotation_position='right'
                )

            # LCL
            if 'lcl' in stats and stats['lcl'] != 'Varies':
                fig.add_hline(
                    y=stats['lcl'],
                    line_dash='dash',
                    line_color='red',
                    annotation_text='LCL',
                    annotation_position='right'
                )

            # Centerline
            center_key = self._get_center_key(stats)
            if center_key and center_key in stats:
                fig.add_hline(
                    y=stats[center_key],
                    line_color='green',
                    annotation_text='Center',
                    annotation_position='right'
                )

        # Highlight signals
        if highlight_signals and 'beyond_limits' in data.columns:
            signals = data[data['beyond_limits'] != 0]
            if not signals.empty:
                fig.add_trace(go.Scatter(
                    x=signals[x_col],
                    y=signals[value_col],
                    mode='markers',
                    name='Signal',
                    marker=dict(
                        size=14,
                        color='red',
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='Signal<br>%{x}<br>%{y:.3f}<extra></extra>'
                ))

        # Layout
        fig.update_layout(
            width=width,
            height=height,
            xaxis_title='Observation',
            yaxis_title=value_col.capitalize(),
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def _plot_faceted(
        self,
        charts: dict,
        ncols: int,
        highlight_signals: bool,
        show_limits: bool,
        width: int,
        height: int
    ) -> go.Figure:
        """Create faceted control charts."""
        n_charts = len(charts)
        nrows = (n_charts + ncols - 1) // ncols

        # Create subplot grid
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=list(charts.keys()),
            vertical_spacing=0.1,
            horizontal_spacing=0.08
        )

        # Plot each chart
        for idx, (chart_name, chart_info) in enumerate(charts.items()):
            row = idx // ncols + 1
            col = idx % ncols + 1

            data = chart_info['data']
            stats = chart_info['statistics']

            value_col = self._get_value_column(data, chart_name)
            x_col = self._get_x_column(data)

            # Main trace
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[value_col],
                    mode='lines+markers',
                    name=chart_name,
                    marker=dict(size=6, color='steelblue'),
                    line=dict(color='steelblue', width=1.5),
                    showlegend=False,
                    hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'
                ),
                row=row,
                col=col
            )

            # Control limits as shapes (more efficient for facets)
            if show_limits:
                x_range = [data[x_col].min(), data[x_col].max()]

                # UCL
                if 'ucl' in stats and stats['ucl'] != 'Varies':
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats['ucl'], y1=stats['ucl'],
                        line=dict(color='red', dash='dash', width=1),
                        row=row, col=col
                    )

                # LCL
                if 'lcl' in stats and stats['lcl'] != 'Varies':
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats['lcl'], y1=stats['lcl'],
                        line=dict(color='red', dash='dash', width=1),
                        row=row, col=col
                    )

                # Centerline
                center_key = self._get_center_key(stats)
                if center_key and center_key in stats:
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats[center_key], y1=stats[center_key],
                        line=dict(color='green', width=1.5),
                        row=row, col=col
                    )

            # Signals
            if highlight_signals and 'beyond_limits' in data.columns:
                signals = data[data['beyond_limits'] != 0]
                if not signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=signals[x_col],
                            y=signals[value_col],
                            mode='markers',
                            marker=dict(size=10, color='red', symbol='x'),
                            showlegend=False,
                            hovertemplate='Signal<br>%{x}<br>%{y:.3f}<extra></extra>'
                        ),
                        row=row,
                        col=col
                    )

        # Update layout
        fig.update_layout(
            width=width,
            height=height,
            hovermode='closest'
        )

        return fig

    def plot_residuals(
        self,
        residual: Optional[str] = None,
        ncols: int = 2,
        template: str = 'processbehavior'
    ) -> ControlChartFigure:
        """
        Plot VAS residuals (R1-R5).

        Parameters
        ----------
        residual : str, optional
            Specific residual to plot ('R1', 'R2', etc.)
            If None, plots all residuals
        ncols : int, default 2
            Number of columns in faceted layout
        template : str, default 'processbehavior'
            Visual theme

        Returns
        -------
        ControlChartFigure
            Interactive figure
        """
        if not self.result.has_residuals:
            raise ValueError(
                "No residuals available.\n"
                "VAS residuals are only calculated for Xbar/S analysis "
                "with SDS 1, 2, or 3."
            )

        residuals = self.result.residuals

        if residual:
            # Single residual
            if residual not in residuals.columns:
                available = [c for c in residuals.columns if c.startswith('R')]
                raise ValueError(
                    f"Residual '{residual}' not found.\n"
                    f"Available: {available}"
                )

            fig = self._plot_residual_series(residuals, residual)
        else:
            # All residuals
            residual_cols = [c for c in residuals.columns if c.startswith('R')]
            fig = self._plot_residual_facets(residuals, residual_cols, ncols)

        fig = apply_theme(fig, template)
        return ControlChartFigure(fig, self.result)

    def plot_effects(
        self,
        effect_type: Literal['main', 'interaction', 'both'] = 'both',
        template: str = 'processbehavior'
    ) -> ControlChartFigure:
        """
        Plot main effects and/or interactions.

        Parameters
        ----------
        effect_type : {'main', 'interaction', 'both'}, default 'both'
            Which effects to plot
        template : str, default 'processbehavior'
            Visual theme

        Returns
        -------
        ControlChartFigure
            Interactive figure
        """
        if not self.result.has_effects and effect_type in ['main', 'both']:
            raise ValueError("No main effects available")

        if not self.result.has_interactions and effect_type in ['interaction', 'both']:
            raise ValueError("No interaction effects available")

        # Implementation would create appropriate plots
        # (main effects plots, interaction plots)
        pass

    def list_charts(self) -> list[str]:
        """Get list of available charts."""
        return list(self.charts.keys())

    # Helper methods
    def _get_value_column(self, data: pd.DataFrame, chart_name: str) -> str:
        """Determine the value column for a chart."""
        if 'mean' in data.columns:
            return 'mean'
        elif 's' in data.columns:
            return 's'
        elif 'mr' in data.columns:
            return 'mr'
        else:
            # Get first numeric column
            numeric_cols = data.select_dtypes(include='number').columns
            return numeric_cols[0] if len(numeric_cols) > 0 else data.columns[0]

    def _get_x_column(self, data: pd.DataFrame) -> str:
        """Determine the x-axis column."""
        if 'x' in data.columns:
            return 'x'
        else:
            return data.index.name or 'index'

    def _get_center_key(self, stats: dict) -> Optional[str]:
        """Get the centerline statistic key."""
        for key in ['Mean', 'mean', 'S', 'mR']:
            if key in stats:
                return key
        return None
```

#### Custom Figure Wrapper
```python
# processbehavior/plotting/control_chart.py

import plotly.graph_objects as go
from pathlib import Path
from typing import Optional

class ControlChartFigure:
    """
    Wrapper around plotly Figure with domain-specific methods.

    This class extends Plotly figures with processbehavior-specific
    functionality while maintaining full access to Plotly's API.

    Examples
    --------
    >>> fig = result.plot()
    >>> fig.show()  # Display in browser/notebook
    >>> fig.save_html('report.html')  # Export to HTML
    >>> fig.save_image('chart.png')  # Export to image
    >>> fig.update_layout(title='Custom Title')  # Full Plotly API
    """

    def __init__(self, plotly_fig: go.Figure, analysis_result):
        """
        Wrap a Plotly figure with enhanced functionality.

        Parameters
        ----------
        plotly_fig : go.Figure
            Underlying Plotly figure
        analysis_result : AnalysisResult
            Source analysis result for metadata
        """
        self._fig = plotly_fig
        self._result = analysis_result

    def show(self):
        """Display figure in browser or notebook."""
        self._fig.show()

    def save_html(
        self,
        filepath: str | Path,
        include_plotlyjs: bool = True,
        auto_open: bool = False
    ):
        """
        Save as standalone HTML file.

        Parameters
        ----------
        filepath : str or Path
            Output file path
        include_plotlyjs : bool, default True
            Whether to include plotly.js (makes file larger but standalone)
        auto_open : bool, default False
            Whether to open in browser after saving
        """
        filepath = Path(filepath)

        self._fig.write_html(
            str(filepath),
            include_plotlyjs='cdn' if not include_plotlyjs else True,
            auto_open=auto_open
        )

        print(f"✓ Saved interactive chart to: {filepath}")

    def save_image(
        self,
        filepath: str | Path,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: float = 2.0
    ):
        """
        Save as static image (requires kaleido).

        Parameters
        ----------
        filepath : str or Path
            Output file path (.png, .jpg, .svg, .pdf)
        width : int, optional
            Image width in pixels
        height : int, optional
            Image height in pixels
        scale : float, default 2.0
            Scale factor for resolution
        """
        try:
            self._fig.write_image(
                str(filepath),
                width=width,
                height=height,
                scale=scale
            )
            print(f"✓ Saved static image to: {filepath}")
        except Exception as e:
            if 'kaleido' in str(e).lower():
                raise ImportError(
                    "Image export requires kaleido.\n"
                    "Install with: pip install kaleido\n"
                    "Or use .save_html() for interactive HTML export"
                ) from e
            raise

    def add_annotation(self, text: str, x, y, **kwargs):
        """Add text annotation to the figure."""
        self._fig.add_annotation(text=text, x=x, y=y, **kwargs)
        return self

    def update_layout(self, **kwargs):
        """Update figure layout (full Plotly API)."""
        self._fig.update_layout(**kwargs)
        return self

    # Expose underlying figure for full Plotly access
    @property
    def figure(self) -> go.Figure:
        """Get underlying Plotly figure."""
        return self._fig

    def __repr__(self):
        return f"ControlChartFigure(charts={list(self._result.charts.keys())})"
```

#### Integration with AnalysisResult
```python
# Add to processbehavior/analysis_result.py

class AnalysisResult:
    # ... existing code ...

    def plot(self, **kwargs) -> 'ControlChartFigure':
        """
        Create interactive control chart visualization.

        This is the main plotting method. It automatically determines
        the best visualization for your data and chart type.

        Parameters
        ----------
        **kwargs : dict
            Plotting options passed to Plotter.plot()

            Common options:
            - chart: str - Specific chart to plot
            - facet_by: str - Variable to facet by
            - highlight_signals: bool - Highlight out-of-control points
            - template: str - Visual theme
            - width/height: int - Figure dimensions

        Returns
        -------
        ControlChartFigure
            Interactive figure with .show(), .save_html(), etc.

        Examples
        --------
        Simple plotting:

        >>> result.plot()

        Specific chart:

        >>> result.plot(chart='Xbar')

        Faceted visualization:

        >>> result.plot(facet_by='Operator', ncols=3)

        Custom styling:

        >>> fig = result.plot(
        ...     highlight_signals=True,
        ...     template='dark',
        ...     width=1200
        ... )
        >>> fig.save_html('report.html')
        """
        from .plotting import Plotter
        plotter = Plotter(self)
        return plotter.plot(**kwargs)

    def plot_residuals(self, **kwargs) -> 'ControlChartFigure':
        """Plot VAS residuals (R1-R5)."""
        from .plotting import Plotter
        plotter = Plotter(self)
        return plotter.plot_residuals(**kwargs)

    def plot_effects(self, **kwargs) -> 'ControlChartFigure':
        """Plot main effects and interactions."""
        from .plotting import Plotter
        plotter = Plotter(self)
        return plotter.plot_effects(**kwargs)
```

### Pros
✅ **Best of Both Worlds** - Low-level control + high-level convenience
✅ **Fully Interactive** - Hover, zoom, pan, export built-in
✅ **Extensible** - Easy to add new chart types and features
✅ **Web-Native** - Perfect for dashboards and reports
✅ **Faceting First-Class** - `make_subplots` handles complex layouts
✅ **Pythonic API** - Follows Hadley philosophy perfectly
✅ **Type-Safe** - Full type hints for IDE support
✅ **Progressive Disclosure** - Simple by default, powerful when needed
✅ **Consistent** - Same API across all chart types
✅ **Future-Proof** - Can add animations, 3D, etc.

### Cons
⚠️ **Larger Dependency** - Plotly is ~3MB JavaScript
⚠️ **Learning Curve** - graph_objects more complex than Express
⚠️ **Static Export Requires Extra** - kaleido for PNG/PDF
⚠️ **Performance** - Large datasets (>10k points) may be slow in browser

### Use Cases
- ✅ Modern web applications and dashboards
- ✅ Interactive analysis and exploration
- ✅ Stakeholder reports (HTML export)
- ✅ Jupyter notebook workflows
- ✅ Any scenario needing faceting + interactivity

---

## Option 4: Bokeh (Server-Based Interactive)

### API Design
```python
# Simple
result.plot()

# With Bokeh server
result.plot(server=True, port=5006)

# Faceted
result.plot(facet_by='Operator', layout='grid')
```

### Pros
✅ **Highly Interactive** - Real-time data updates
✅ **Server Capabilities** - Can connect to live data
✅ **Python Callbacks** - Business logic in Python, not JavaScript
✅ **Beautiful Default** - Modern, clean aesthetics

### Cons
❌ **Requires Server** - Can't create standalone HTML easily
❌ **Complex Deployment** - Need Bokeh server infrastructure
❌ **Steeper Learning Curve** - Different paradigm
❌ **Less Mature** - Smaller community than Plotly

---

## Option 5: Altair (Declarative Vega-Lite)

### API Design
```python
# Declarative approach
result.plot()  # Generates Vega-Lite spec

# Faceting is natural
result.plot(facet='Operator', columns=3)
```

### Pros
✅ **Declarative** - Specify WHAT not HOW
✅ **Concise** - Very few lines of code
✅ **Grammar of Graphics** - True Hadley Wickham style
✅ **JSON Export** - Can save/share specifications

### Cons
❌ **Limited Complexity** - High-level only
❌ **Data Size Limits** - Vega-Lite has 5000 row limit
❌ **Less Control** - Can't customize everything
❌ **Smaller Ecosystem** - Fewer examples

---

## Recommendation: Option 3 (Plotly graph_objects + Abstraction) ⭐

### Why This is the Best Choice

#### 1. **Aligns Perfectly with Pythonic Hadley Philosophy**

```python
# Simple tasks are trivial
result.plot()  # Just works!

# Complex tasks are possible
result.plot(
    chart='Xbar',
    facet_by='Operator',
    highlight_signals=True,
    template='dark',
    width=1200
)

# Consistent API
result.plot()
result.plot_residuals()
result.plot_effects()
```

#### 2. **Faceting is First-Class**

Plotly's `make_subplots` provides:
- Grid layouts (rows × columns)
- Shared axes options
- Individual control over each subplot
- Efficient rendering

```python
# Faceting just works
result.plot(facet_by='Operator')

# Auto-detects stratified analysis
result.plot(facet=True)
```

#### 3. **Interactive + Web-Native**

- Hover tooltips show exact values
- Zoom and pan for exploration
- Export to standalone HTML (no server needed!)
- Works in Jupyter, scripts, and web apps
- Can embed in larger dashboards

#### 4. **Extensible for Future Needs**

```python
# Today: Control charts
result.plot()

# Tomorrow: Custom annotations
fig = result.plot()
fig.add_annotation("Process change", x=50, y=10)

# Future: Animations, 3D, etc.
# Full Plotly API available via fig.figure
```

#### 5. **Progressive Disclosure**

```python
# Beginner: Zero configuration
result.plot()

# Intermediate: Common options
result.plot(highlight_signals=True, template='minimal')

# Advanced: Full control
fig = result.plot()
fig.figure.update_traces(marker=dict(size=10))
fig.figure.update_layout(font=dict(family="Arial"))
```

#### 6. **Modern UX Expectations**

Users expect:
- ✅ Interactive charts (not static images)
- ✅ Tooltips on hover
- ✅ Zoom/pan capabilities
- ✅ Easy export to reports
- ✅ Works on mobile

Plotly delivers all of this out-of-the-box.

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Create `processbehavior/plotting/` module
2. Implement `Plotter` class with `plot()` method
3. Implement `ControlChartFigure` wrapper
4. Add `result.plot()` to `AnalysisResult`
5. Create default theme

### Phase 2: Chart Types (Week 2)
1. Single chart plotting (Xbar, S, Imr, R)
2. Faceted plotting for stratified analysis
3. Control limits and signals
4. Hover templates and tooltips

### Phase 3: Advanced Features (Week 3)
1. Residual plotting (`plot_residuals()`)
2. Effects plotting (`plot_effects()`)
3. Custom themes and templates
4. Export functionality (HTML, PNG)

### Phase 4: Documentation & Examples (Week 4)
1. Comprehensive docstrings
2. Usage examples in docs
3. Tutorial notebooks
4. Gallery of chart types

---

## Example Usage (End User Experience)

```python
from processbehavior import ProcessDataFrame

# Load and analyze data
pdf = ProcessDataFrame('fillweight.csv')
result = pdf.analyze(
    response_var='weight',
    factors=['lane', 'head'],
    time='pull'
)

# Dead simple - auto-plots everything
result.plot()

# Specific chart
result.plot(chart='Xbar')

# Faceted by operator (stratified analysis was run)
result.plot(facet_by='Operator', ncols=3)

# Residuals
result.plot_residuals()  # All R1-R5
result.plot_residuals(residual='R5')  # Just R5

# Effects
result.plot_effects()

# Customization
fig = result.plot(
    chart='Xbar',
    highlight_signals=True,
    template='dark',
    width=1400,
    height=600,
    title='Fill Weight Control Chart - Production Line A'
)

# Export
fig.save_html('report.html')  # Interactive
fig.save_image('chart.png')  # Static (requires kaleido)

# Or display
fig.show()  # Opens in browser or shows in notebook
```

---

## Dependencies

```toml
# Add to pyproject.toml
[project]
dependencies = [
    "pandas>=2.0",
    # ... existing ...
    "plotly>=5.18",  # Core plotting
]

[project.optional-dependencies]
image-export = [
    "kaleido>=0.2.1",  # For static image export
]
```

---

## Summary Comparison

| Feature | Matplotlib | Plotly Express | **Plotly GO** | Bokeh | Altair |
|---------|-----------|----------------|---------------|-------|--------|
| **Interactive** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Web-Native** | ⚠️ | ✅ | ✅ | ⚠️ | ✅ |
| **Faceting** | ⚠️ | ✅ | ✅✅ | ✅ | ✅✅ |
| **Extensibility** | ✅ | ⚠️ | ✅✅ | ✅ | ❌ |
| **API Simplicity** | ⚠️ | ✅✅ | ✅ | ⚠️ | ✅✅ |
| **Standalone Export** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Fine Control** | ✅✅ | ⚠️ | ✅✅ | ✅ | ❌ |
| **Learning Curve** | Medium | Low | Medium | High | Low |
| **Bundle Size** | Small | Large | Large | Large | Medium |
| **Community** | Huge | Large | Large | Medium | Small |

**✅✅ = Excellent | ✅ = Good | ⚠️ = Adequate | ❌ = Poor**

---

## Final Recommendation

**Go with Option 3: Plotly graph_objects + Custom Abstraction Layer**

This approach delivers:
1. ✅ **Best user experience** - Interactive, modern, intuitive
2. ✅ **Perfect for faceting** - First-class small multiples support
3. ✅ **Web-ready** - Standalone HTML export, no server needed
4. ✅ **Pythonic API** - Follows Hadley design philosophy
5. ✅ **Future-proof** - Can grow with the package
6. ✅ **Progressive disclosure** - Simple default, powerful when needed

The abstraction layer ensures users get a clean, domain-specific API while we retain full access to Plotly's power under the hood.
