"""HTML report generation.

Extracted from plotter.py to isolate report building from chart rendering.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .themes import ChartTheme, get_theme

if TYPE_CHECKING:
    from ..analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


def generate_report(
    result: AnalysisResult,
    filepath: str,
    include_charts: bool = True,
    include_residuals: bool = True,
    include_effects: bool = True,
    include_summary: bool = True,
    theme: str | ChartTheme = 'processbehavior',
    width: int = 1200,
    title: str | None = None,
) -> None:
    """Generate a comprehensive HTML report with all visualizations.

    Creates a single-page HTML file with analysis summary, control charts,
    residual diagnostics, and effects analysis.

    Parameters
    ----------
    result : AnalysisResult
        The analysis result to report on.
    filepath : str
        Output HTML file path.
    include_charts : bool, default True
        Include control charts.
    include_residuals : bool, default True
        Include residual diagnostic plots (if available).
    include_effects : bool, default True
        Include effects bar charts (if available).
    include_summary : bool, default True
        Include analysis summary section.
    theme : str or ChartTheme, default 'processbehavior'
        Visual theme for all charts.
    width : int, default 1200
        Width of charts in pixels.
    title : str, optional
        Report title (defaults to "Process Behavior Analysis Report").
    """
    # Import here to avoid circular import at module level
    from .plotter import Plotter

    plotter = Plotter(result)

    sections: list[str] = []
    report_title = title or "Process Behavior Analysis Report"
    esc = html.escape

    # Resolve theme for plotter calls
    resolved_theme = get_theme(theme) if isinstance(theme, str) else theme

    # Summary section
    if include_summary:
        sections.append(_build_summary_section(result, esc))

    # Control charts section
    if include_charts:
        fig = plotter.plot(theme=resolved_theme, width=width, height=500)
        chart_html = fig.figure.to_html(full_html=False, include_plotlyjs=False)
        sections.append(f"""
        <div class="section">
            <h2>Control Charts</h2>
            {chart_html}
        </div>
        """)

    # Residuals section
    if include_residuals and result.has_residuals:
        from .residuals import plot_residuals

        fig = plot_residuals(result, theme=resolved_theme, width=width, height=350)
        residual_html = fig.figure.to_html(full_html=False, include_plotlyjs=False)
        sections.append(f"""
        <div class="section">
            <h2>Residual Diagnostics</h2>
            <p>R1 residuals showing process variation over time.</p>
            {residual_html}
        </div>
        """)

    # Effects section
    if include_effects and result.has_effects:
        try:
            fig = plotter.plot_effects(
                effect_type='all', theme=resolved_theme, width=width, height=400,
            )
            effects_html = fig.figure.to_html(full_html=False, include_plotlyjs=False)
            sections.append(f"""
            <div class="section">
                <h2>Main Effects Analysis</h2>
                <p>Factor and time effects showing contribution to process variation.</p>
                {effects_html}
            </div>
            """)
        except (ValueError, Exception):
            pass  # No effects to plot

    # Build full HTML
    html_content = _build_html_page(report_title, sections, esc)
    Path(filepath).write_text(html_content)
    logger.info(f"Report generated: {filepath}")


def _build_summary_section(result, esc):
    """Build the HTML summary table."""
    summary = result.summary
    has_res = 'Yes' if summary['has_residuals'] else 'No'
    has_eff = 'Yes' if summary['has_effects'] else 'No'
    return f"""
    <div class="section">
        <h2>Analysis Summary</h2>
        <table class="summary-table">
            <tr><td><strong>ADS</strong></td><td>{esc(str(summary['analytical_sds']))} - {
                esc(str(summary['analytical_sds_description']))}</td></tr>
            <tr><td><strong>Response Variable</strong></td><td>{
                esc(str(summary['response_var']))}</td></tr>
            <tr><td><strong>Observations</strong></td><td>{
                esc(str(summary['n_observations']))}</td></tr>
            <tr><td><strong>Charts</strong></td><td>{
                esc(', '.join(str(c) for c in summary['chart_types']))}</td></tr>
            <tr><td><strong>Signals Detected</strong></td><td>{
                esc(str(summary['n_signals_total']))}</td></tr>
            <tr><td><strong>Has Residuals</strong></td><td>{has_res}</td></tr>
            <tr><td><strong>Has Effects</strong></td><td>{has_eff}</td></tr>
        </table>
    </div>
    """


def _build_html_page(report_title, sections, esc):
    """Build the full HTML page."""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{esc(report_title)}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4a90a4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4a90a4;
            margin-top: 30px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-table {{
            border-collapse: collapse;
            width: 100%;
            max-width: 600px;
        }}
        .summary-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
        }}
        .summary-table tr:last-child td {{
            border-bottom: none;
        }}
        .footer {{
            text-align: center;
            color: #888;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>{esc(report_title)}</h1>
    {''.join(sections)}
    <div class="footer">
        Generated with ProcessBehavior - Statistical Process Control for Python
    </div>
</body>
</html>
    """
