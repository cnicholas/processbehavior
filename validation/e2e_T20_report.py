"""
E2E Self-Report: processbehavior analyses on PBTESTDATABASE_T20.csv

Runs every PM SDS variable (Y-1..Y-6) through the same analysis suite as
the Bishop validator, plus capability and loss-function. Outputs HTML to
share with Tom for manual review. Expected values are NOT compared here —
the EXPECTED dict is a placeholder for a future pass once Tom returns his
reference values.

Usage:
    python validation/e2e_T20_report.py
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from processbehavior import ProcessBehavior
from processbehavior.exceptions import ValidationError, ChartNotAvailableError


# --- Configuration ---

VALIDATION_CSV = Path(__file__).parent / 'PBTESTDATABASE_T20.csv'
OUTPUT_HTML = Path(__file__).parent / 'e2e_T20_report.html'

RESPONSE_COLS = ['Y-1', 'Y-2', 'Y-3', 'Y-4', 'Y-5', 'Y-6']

SPEC_LSL = 232.0
SPEC_USL = 242.0
SPEC_TARGET = 237.0

# Placeholder: per-row reference values from Tom. Once populated, add a
# PASS/FAIL column to the analyses table.
EXPECTED: dict[tuple, dict] = {}


# --- Analysis catalogue ---

@dataclass(frozen=True)
class AnalysisSpec:
    """One row in the analyses table for a given column."""
    label: str
    chart: str
    by: tuple[str, ...] | None
    value: str | None
    recentered: bool
    companion: bool


def analysis_catalogue(ads: int, factor1: str, factor2: str, time_var: str) -> list[AnalysisSpec]:
    """Return the suite of analyses for a given Analytical Design State.

    Mirrors what `validation/e2e_bishop_report.py` runs in `run_sds_validation`,
    organized by ADS (ODS 4/5/6 collapse to ADS 1/2/3 so this catalogue
    covers all six Y-N columns).
    """
    specs: list[AnalysisSpec] = []

    # Overall + stratified primary chart
    if ads == 1:
        specs.append(AnalysisSpec('Overall (Xbar+S)', 'Xbar', (), None, False, True))
        specs.append(AnalysisSpec('Stratified by PT (Xbar+S)', 'Xbar', (time_var,), None, False, True))
    else:  # ADS 2, 3
        specs.append(AnalysisSpec('Overall (X+mR)', 'X', (), None, False, True))
        specs.append(AnalysisSpec('Stratified by F1,F2 (X+mR)', 'X', (factor1, factor2), None, False, True))

    # PDC effects (R6 recentered) -- Xbar + S
    specs.append(AnalysisSpec('PDC effects (Xbar)', 'Xbar', (factor1, factor2), 'R6', True, False))
    specs.append(AnalysisSpec('PDC effects (S)', 'S', (factor1, factor2), 'R6', False, False))

    # PT effects (R3 recentered) -- Xbar + S
    specs.append(AnalysisSpec('PT effects (Xbar)', 'Xbar', (time_var,), 'R3', True, False))
    specs.append(AnalysisSpec('PT effects (S)', 'S', (time_var,), 'R3', True, False))

    # F1 effects (R6 recentered)
    specs.append(AnalysisSpec('F1 effects (Xbar)', 'Xbar', (factor1,), 'R6', True, False))
    specs.append(AnalysisSpec('F1 effects (S)', 'S', (factor1,), 'R6', False, False))

    # F2 effects (R6 recentered)
    specs.append(AnalysisSpec('F2 effects (Xbar)', 'Xbar', (factor2,), 'R6', True, False))
    specs.append(AnalysisSpec('F2 effects (S)', 'S', (factor2,), 'R6', False, False))

    # Interaction (R3 recentered)
    if ads == 1:
        specs.append(AnalysisSpec('Interaction (Xbar)', 'Xbar', (factor1, factor2, time_var), 'R3', True, False))
        specs.append(AnalysisSpec('Interaction (S)', 'S', (factor1, factor2, time_var), 'R3', True, False))
    else:
        specs.append(AnalysisSpec('Interaction (X+mR)', 'X', (), 'R3', True, True))

    # Maximum information (R2 individual)
    specs.append(AnalysisSpec('Max info (X on R2)', 'X', (), 'R2', False, False))

    return specs


# --- Per-column analysis runner ---

def _extract_stats(result, chart_type: str) -> tuple[Any, Any, Any]:
    """Return (CL, LPL, UPL) for a chart in result. None for 'Varies'."""
    stats = result.get_statistics(chart_type)
    def _f(v):
        if v is None or v == 'Varies':
            return v if v == 'Varies' else None
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return _f(stats.get('center')), _f(stats.get('lpl', None)), _f(stats.get('upl', None))


def _safe_chart_table(result, chart_type: str):
    try:
        return result.chart_table(chart_type)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def _run_analysis(study, spec: AnalysisSpec) -> dict:
    """Run one AnalysisSpec; return a row dict with CL/LPL/UPL or 'N/A: ...'.

    For companion runs (Xbar+S or X+mR), this returns rows for both charts.
    For single-chart runs, returns just the requested chart.
    """
    base = {
        'label': spec.label,
        'chart': spec.chart,
        'by': list(spec.by) if spec.by is not None else None,
        'value': spec.value,
        'recentered': spec.recentered,
        'companion': spec.companion,
    }
    try:
        kwargs = dict(chart=spec.chart, by=list(spec.by) if spec.by is not None else None,
                      companion=spec.companion)
        if spec.value:
            kwargs['value'] = spec.value
        if spec.recentered:
            kwargs['recentered'] = True
        result = study.execute(**kwargs)
    except (ValidationError, ChartNotAvailableError) as e:
        return [{**base, 'error': str(e).split('\n')[0]}]
    except Exception as e:
        return [{**base, 'error': f'{type(e).__name__}: {str(e).splitlines()[0]}'}]

    # Determine which charts to extract
    chart_keys: list[str]
    if spec.companion:
        if spec.chart == 'Xbar':
            chart_keys = ['Xbar', 'S']
        elif spec.chart == 'X':
            chart_keys = ['X', 'mR']
        else:
            chart_keys = [spec.chart]
    else:
        chart_keys = [spec.chart]

    rows = []
    for ck in chart_keys:
        try:
            cl, lpl, upl = _extract_stats(result, ck)
        except (KeyError, ValueError) as e:
            rows.append({**base, 'chart': ck, 'error': f'no statistics for {ck}: {e}'})
            continue
        rows.append({
            **base,
            'chart': ck,
            'cl': cl,
            'lpl': lpl,
            'upl': upl,
            'chart_table': _safe_chart_table(result, ck),
        })
    return rows


def _run_capability(study) -> dict | str:
    try:
        cap = study.capability(usl=SPEC_USL, lsl=SPEC_LSL, target=SPEC_TARGET)
        return cap.as_dict()
    except Exception as e:
        return f'{type(e).__name__}: {str(e).splitlines()[0]}'


def _run_loss(study) -> dict | str:
    try:
        loss = study.loss_function(target=SPEC_TARGET)
        return loss.as_dict()
    except Exception as e:
        return f'{type(e).__name__}: {str(e).splitlines()[0]}'


def run_column(pb, col: str) -> dict:
    """Run the full analysis suite for one Y-N column."""
    print(f'\n=== {col} ===')
    try:
        study = pb.formulate(
            response=col,
            factors=['FACTOR 1', 'FACTOR 2'],
            time='PRODUCTION TIME',
        )
    except Exception as e:
        return {
            'column': col,
            'error': f'formulate() raised: {type(e).__name__}: {e}',
        }

    ods = study.observed_design_state.sds
    ads = study.analytical_design_state.sds
    ods_reason = study.observed_design_state.reason
    ads_reason = study.analytical_design_state.reason
    print(f'  ODS={ods} ({ods_reason})  ADS={ads} ({ads_reason})')

    catalogue = analysis_catalogue(
        ads, factor1='FACTOR 1', factor2='FACTOR 2', time_var='PRODUCTION TIME'
    )

    analyses = []
    for spec in catalogue:
        rows = _run_analysis(study, spec)
        analyses.extend(rows)
        for r in rows:
            if 'error' in r:
                print(f'  [N/A] {spec.label} ({r["chart"]}): {r["error"]}')
            else:
                print(f'  [OK ] {spec.label} ({r["chart"]}): CL={r["cl"]}')

    capability = _run_capability(study)
    loss = _run_loss(study)
    print(f'  capability: {"OK" if isinstance(capability, dict) else capability}')
    print(f'  loss      : {"OK" if isinstance(loss, dict) else loss}')

    return {
        'column': col,
        'ods': ods,
        'ods_reason': ods_reason,
        'ads': ads,
        'ads_reason': ads_reason,
        'analyses': analyses,
        'capability': capability,
        'loss': loss,
    }


# --- HTML rendering ---

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #fafafa; color: #222; }
h1 { color: #333; }
h2 { color: #555; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
h3 { color: #666; margin-top: 20px; }
.column-card { background: #fff; padding: 15px 20px; border-radius: 8px; margin: 20px 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.state { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px;
  background: #e3f2fd; color: #0d47a1; margin-right: 8px; }
.state.ads { background: #f3e5f5; color: #4a148c; }
.pending { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px;
  background: #fff3e0; color: #e65100; }
table.main { border-collapse: collapse; width: 100%; margin: 10px 0; }
table.main th { background: #f5f5f5; padding: 8px 12px; text-align: left; font-size: 13px;
  border-bottom: 2px solid #ddd; }
table.main td { padding: 6px 12px; border-bottom: 1px solid #eee; font-size: 13px; vertical-align: top; }
table.main tr.error td { background: #fafafa; color: #777; font-style: italic; }
table.kv { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
table.kv td { padding: 3px 14px 3px 0; }
table.kv td.k { color: #555; font-weight: 500; }
table.chart-table { border-collapse: collapse; margin: 5px 0; font-size: 11px; }
table.chart-table th, table.chart-table td { padding: 3px 8px; border: 1px solid #ddd; }
table.chart-table th { background: #f0f0f0; }
details { margin: 4px 0; }
summary { cursor: pointer; color: #1565c0; font-size: 12px; }
.timestamp { font-size: 12px; color: #999; margin-top: 30px; }
.error-note { color: #c62828; font-style: italic; font-size: 12px; }
"""


def _fmt(v, decimals=3):
    if v is None:
        return '-'
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    try:
        return f'{v:.{decimals}f}'
    except (TypeError, ValueError):
        return str(v)


def _render_chart_table(table) -> str:
    if table is None:
        return ''
    html = '<details><summary>chart_table()</summary><table class="chart-table"><tr>'
    for col in table.columns:
        html += f'<th>{col}</th>'
    html += '</tr>'
    for _, row in table.iterrows():
        html += '<tr>'
        for col in table.columns:
            v = row[col]
            html += f'<td>{_fmt(v, 4) if isinstance(v, float) else v}</td>'
        html += '</tr>'
    html += '</table></details>'
    return html


def _render_analyses_table(analyses: list[dict]) -> str:
    html = '<table class="main"><tr>'
    html += '<th>Analysis</th><th>Chart</th><th>by</th><th>value</th>'
    html += '<th>recentered</th><th>CL</th><th>LPL</th><th>UPL</th><th>Details</th></tr>'
    for r in analyses:
        if 'error' in r:
            html += '<tr class="error">'
            html += f'<td>{r["label"]}</td><td>{r["chart"]}</td>'
            html += f'<td>{r.get("by") or "[]"}</td><td>{r.get("value") or "-"}</td>'
            html += f'<td>{r.get("recentered", False)}</td>'
            html += f'<td colspan="3" class="error-note">N/A — {r["error"]}</td><td>-</td>'
            html += '</tr>'
            continue
        html += '<tr>'
        html += f'<td>{r["label"]}</td><td>{r["chart"]}</td>'
        html += f'<td>{r.get("by") or "[]"}</td><td>{r.get("value") or "-"}</td>'
        html += f'<td>{r.get("recentered", False)}</td>'
        html += f'<td>{_fmt(r.get("cl"))}</td><td>{_fmt(r.get("lpl"))}</td>'
        html += f'<td>{_fmt(r.get("upl"))}</td>'
        html += f'<td>{_render_chart_table(r.get("chart_table"))}</td>'
        html += '</tr>'
    html += '</table>'
    return html


def _render_capability(cap) -> str:
    if isinstance(cap, str):
        return f'<p class="error-note">N/A — {cap}</p>'
    rows = [
        ('LSL', cap['lsl']), ('USL', cap['usl']), ('Target', cap['target']),
        ('n', cap['n']),
        ('y_bar', cap['y_bar']), ('s', cap['s']),
        ('sigma_hat', cap['sigma_hat']), ('sigma_hat_r2', cap['sigma_hat_r2']),
        ('Pp', cap['pp']), ('Ppk_lower', cap['ppk_lower']),
        ('Ppk_upper', cap['ppk_upper']), ('Ppk', cap['ppk']),
        ('Cp', cap['cp']), ('Cpk_lower', cap['cpk_lower']),
        ('Cpk_upper', cap['cpk_upper']), ('Cpk', cap['cpk']),
        ('potential_unavailable_reason', cap['potential_unavailable_reason']),
        ('Z_lower', cap['z_lower']), ('Z_upper', cap['z_upper']),
        ('n_below_lsl', cap['n_below_lsl']), ('n_above_usl', cap['n_above_usl']),
        ('n_outside', cap['n_outside']),
        ('pct_below_lsl', cap['pct_below_lsl']),
        ('pct_above_usl', cap['pct_above_usl']),
        ('pct_outside', cap['pct_outside']),
        ('potential_n_below_lsl', cap['potential_n_below_lsl']),
        ('potential_n_above_usl', cap['potential_n_above_usl']),
        ('potential_pct_below_lsl', cap['potential_pct_below_lsl']),
        ('potential_pct_above_usl', cap['potential_pct_above_usl']),
    ]
    html = '<table class="kv">'
    for k, v in rows:
        html += f'<tr><td class="k">{k}</td><td>{_fmt(v)}</td></tr>'
    html += '</table>'
    return html


def _render_loss(loss) -> str:
    if isinstance(loss, str):
        return f'<p class="error-note">N/A — {loss}</p>'
    rows = [
        ('target', loss['target']), ('target_is_default', loss['target_is_default']),
        ('y_bar', loss['y_bar']),
        ('n', loss['n']), ('K (PDC levels)', loss['K']),
        ('T_periods', loss['T_periods']), ('SDS', loss['sds']),
        ('total', loss['total']),
        ('centering', f'{_fmt(loss["centering"])}  ({_fmt(loss["pct_centering"], 1)}%)'),
        ('unexplained', f'{_fmt(loss["unexplained"])}  ({_fmt(loss["pct_unexplained"], 1)}%)'),
        ('pdc', f'{_fmt(loss["pdc"])}  ({_fmt(loss["pct_pdc"], 1)}%)'),
        ('time', f'{_fmt(loss["time"])}  ({_fmt(loss["pct_time"], 1)}%)'),
        ('interaction', f'{_fmt(loss["interaction"])}  ({_fmt(loss["pct_interaction"], 1)}%)'),
    ]
    if 'pdc_by_factor' in loss:
        for fname, val in loss['pdc_by_factor'].items():
            rows.append((f'  pdc[{fname}]', val))
        rows.append(('  pdc_factor_interaction', loss['pdc_factor_interaction']))
    html = '<table class="kv">'
    for k, v in rows:
        html += f'<tr><td class="k">{k}</td><td>{_fmt(v) if not isinstance(v, str) else v}</td></tr>'
    html += '</table>'
    return html


def render_html(per_column_results: list[dict]) -> str:
    html = '<!DOCTYPE html><html><head><meta charset="utf-8">'
    html += '<title>T20 E2E Self-Report — processbehavior</title>'
    html += f'<style>{CSS}</style></head><body>'
    html += '<h1>T20 E2E Self-Report</h1>'
    html += (
        '<p>processbehavior analyses on <code>PBTESTDATABASE_T20.csv</code> '
        '(Y-1 through Y-6) for manual comparison against Tom Bishop\'s reference. '
        '<span class="pending">Pending Tom\'s expected values</span></p>'
    )
    html += (
        '<p style="font-size:12px; color:#777;">Specs: '
        f'LSL={SPEC_LSL}, USL={SPEC_USL}, Target={SPEC_TARGET}</p>'
    )

    for r in per_column_results:
        col = r['column']
        if 'error' in r:
            html += f'<div class="column-card"><h2>{col}</h2>'
            html += f'<p class="error-note">{r["error"]}</p></div>'
            continue
        html += '<div class="column-card">'
        html += f'<h2>{col}'
        html += f' <span class="state">ODS {r["ods"]} — {r["ods_reason"]}</span>'
        html += f' <span class="state ads">ADS {r["ads"]} — {r["ads_reason"]}</span>'
        html += '</h2>'

        html += '<h3>Process behavior chart analyses</h3>'
        html += _render_analyses_table(r['analyses'])

        html += '<h3>Capability</h3>'
        html += _render_capability(r['capability'])

        html += '<h3>Loss function</h3>'
        html += _render_loss(r['loss'])

        html += '</div>'

    html += f'<p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
    html += '</body></html>'
    return html


# --- Main ---

def main() -> int:
    if not VALIDATION_CSV.exists():
        print(f'ERROR: {VALIDATION_CSV} not found')
        return 2

    print(f'Loading {VALIDATION_CSV.name} ...')
    df = pd.read_csv(VALIDATION_CSV)
    pb = ProcessBehavior(df)

    results = [run_column(pb, col) for col in RESPONSE_COLS]

    print(f'\nWriting {OUTPUT_HTML} ...')
    OUTPUT_HTML.write_text(render_html(results))
    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
