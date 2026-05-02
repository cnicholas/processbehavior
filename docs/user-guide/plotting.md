# Plotting & Themes

ProcessBehavior provides interactive Plotly-based visualizations with professional styling and multiple customization options.

## Basic Plotting

After analysis, call `plot()` to create a visualization:

```python
result = study.execute()

# Basic chart
fig = result.plot()
fig.show()
```

## Plot Options

The `plot()` method accepts many customization options:

```python
fig = result.plot(
    chart=None,              # Specific chart or None for default/all
    facet=False,             # Create faceted plot for stratified data
    ncols=2,                 # Columns in faceted layout
    show_limits=True,        # Show control limits (UCL, LCL)
    show_zones=False,        # Show zone shading (A, B, C)
    highlight_signals=True,  # Highlight out-of-control points
    show_rules=False,        # Show all WECO rule violations
    show_stats=False,        # Display statistics box
    theme='processbehavior',     # Theme name
    width=1000,              # Figure width in pixels
    height=None,             # Figure height (auto if None)
    title=None               # Custom title
)
```

## Common Visualization Patterns

### Simple Chart with Zones

```python
fig = result.plot(
    show_zones=True,      # Shaded 1σ, 2σ, 3σ zones
    highlight_signals=True     # Red markers for beyond-limits points
)
fig.show()
```

### Full Analysis View

```python
fig = result.plot(
    show_zones=True,
    highlight_signals=True,
    show_rules=True,      # All WECO rule violations
    show_stats=True       # Statistics box (CL, UCL, LCL)
)
fig.show()
```

### Specific Chart

```python
# Just the Xbar chart
fig = result.plot(chart='Xbar', show_zones=True)

# Just the S chart
fig = result.plot(chart='S', show_zones=True)
```

### Stratified Faceted View

```python
# For stratified X analysis
result = study.execute(study.charts.X)

# All lanes in one figure
fig = result.plot(
    facet=True,
    ncols=2,              # 2 columns of charts
    show_zones=True,
    highlight_signals=True
)
fig.show()
```

### Individual Stratum

```python
# Focus on one lane
fig = result.plot(
    chart='X_Lane_A',
    show_zones=True,
    show_rules=True,
    show_stats=True
)
fig.show()
```

## Built-in Themes

ProcessBehavior includes four professional themes:

### processbehavior (default)

Professional SPC styling with clear data visibility.

```python
fig = result.plot(theme='processbehavior')
```

### minimal

Light background with minimal annotations.

```python
fig = result.plot(theme='minimal')
```

### dark

Dark theme with high contrast colors.

```python
fig = result.plot(theme='dark')
```

### ggplot

Inspired by ggplot2's aesthetics.

```python
fig = result.plot(theme='ggplot')
```

## Listing Available Themes

```python
from processbehavior.plotting import list_themes

print(list_themes())
# ['processbehavior', 'minimal', 'dark', 'ggplot']
```

## Custom Themes

Create your own theme with `ChartTheme`:

```python
from processbehavior.plotting import ChartTheme, register_theme

custom = ChartTheme(
    name='company',

    # Data appearance
    data_color='navy',
    data_marker_size=10,
    data_line_width=1.5,

    # Signal highlighting
    signal_color='orange',
    signal_marker_size=14,

    # Control lines
    center_color='darkgreen',
    limit_color='darkred',
    center_line_width=2.0,
    limit_line_width=1.5,
    limit_line_dash='dash',

    # Zone shading
    zone_a_color='rgba(255, 255, 200, 0.3)',
    zone_b_color='rgba(200, 255, 255, 0.3)',
    zone_c_color='rgba(200, 255, 200, 0.3)',

    # Typography
    font_family='Arial',
    font_size=12,
    title_font_size=16,

    # Background
    plot_bgcolor='white',
    paper_bgcolor='white',
    gridcolor='lightgray'
)

# Register the theme
register_theme(custom)

# Use it
fig = result.plot(theme='company')
```

## ControlChartFigure Methods

The `plot()` method returns a `ControlChartFigure` with additional methods:

### Show in Browser

```python
fig = result.plot()
fig.show()  # Opens in default browser
```

### Save as HTML

```python
fig.save_html('chart.html')
# or
fig.save_html('chart.html', include_plotlyjs=True)  # Standalone file
```

### Save as Image

Requires kaleido package:

```python
# pip install kaleido

fig.save_image('chart.png')
fig.save_image('chart.pdf')
fig.save_image('chart.svg')
```

### Access Underlying Plotly Figure

```python
plotly_fig = fig.figure  # Standard plotly.graph_objects.Figure
plotly_fig.update_layout(...)  # Full Plotly customization
```

## Zone Shading

Zones represent standard deviation bands:

| Zone | Range | Color (default) |
|------|-------|-----------------|
| A | 2σ to 3σ | Light yellow |
| B | 1σ to 2σ | Light blue |
| C | 0 to 1σ | Light green |

```python
# Enable zone shading
fig = result.plot(show_zones=True)
```

## Signal Markers

Signals are highlighted differently based on the rule violated:

```python
# Just beyond-limits signals (Rule 1)
fig = result.plot(highlight_signals=True)

# All WECO rules
fig = result.plot(show_rules=True)
```

## Statistics Box

Display control limit values on the chart:

```python
fig = result.plot(show_stats=True)

# Shows:
# CL = 100.23
# UCL = 106.45
# LCL = 94.01
```

## Responsive Sizing

```python
# Fixed size
fig = result.plot(width=1200, height=600)

# Auto height based on content
fig = result.plot(width=1000, height=None)
```

## Multiple Charts

### Side-by-Side in Jupyter

```python
from IPython.display import display

fig_xbar = result.plot(chart='Xbar', show_zones=True)
fig_s = result.plot(chart='S', show_zones=True)

display(fig_xbar.figure, fig_s.figure)
```

### Combined in Subplots

For advanced layouts, access the underlying Plotly figure:

```python
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, subplot_titles=['Xbar', 'S'])
# Add traces from result charts...
```

## Best Practices

1. **Start simple** - Add features incrementally
2. **Use zones for context** - Helps interpret point positions
3. **Enable signals for monitoring** - Highlights actionable items
4. **Choose appropriate theme** - Match your organization's style
5. **Save interactive HTML** - Allows exploration without Python

## Example: Complete Report Chart

```python
# Full-featured analysis chart
fig = result.plot(
    chart='Xbar',
    show_zones=True,
    highlight_signals=True,
    show_stats=True,
    theme='processbehavior',
    title='Fill Weight Analysis - Xbar Chart',
    width=1200
)

# Save for report
fig.save_html('fillweight_xbar.html')
fig.save_image('fillweight_xbar.png')
```

## Next Steps

- [Excel Export](excel-export.md) - Include charts in Excel exports
- [Signal Detection](../tutorials/signal-detection.ipynb) - Visualizing rule violations
- [Chart Types](chart-types.md) - Available chart types
