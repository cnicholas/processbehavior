"""
E2E Validation Report: processbehavior vs Tom Bishop's VAS Analyses

Compares the library's output with Tom Bishop's VAS (Minitab) analyses, value by value, for every
run in RUNS: PM SDS 1-6 and PM INERT SDS 1-6 (PBTESTDATABASE_T100.csv), the known-effects data
(PBTESTKNOWNEFFECTS_T100.csv) and the Medicare ACO data (aco_per_capita_expenditure.csv).
Each run's reference values, spec limits and target live in
tests/fixtures/bishop_analyses/<run>.json; a reference that is not yet available is "pending" and
never fails the gate. Writes an HTML report and docs/reference/validation.md; exits 1 on any
disagreement.

Usage:
    python validation/e2e_bishop_report.py
"""
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from processbehavior import ProcessBehavior

# --- Configuration ---

DATA = {
    'T100': Path(__file__).parent / 'PBTESTDATABASE_T100.csv',
    'KNOWN': Path(__file__).parent / 'PBTESTKNOWNEFFECTS_T100.csv',
    'ACO': Path(__file__).parent / 'aco_per_capita_expenditure.csv',
}
FIXTURES_DIR = Path(__file__).parent.parent / 'tests' / 'fixtures' / 'bishop_analyses'
OUTPUT_HTML = Path(__file__).parent / 'e2e_bishop_report.html'
TOLERANCE = 0.01  # half-unit of last decimal place

# Sampling design state -> analytic design state: incomplete grids (SDS 4-6) are analysed as the
# complete design that survives tidying.
ADS_OF_SDS = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}


@dataclass(frozen=True)
class Run:
    """One VAS analysis to reproduce: a response column of a data source, at its ADS."""

    id: str        # also the reference file: tests/fixtures/bishop_analyses/<id>.json
    label: str
    data: str      # key of DATA
    response: str
    ads: int
    factors: tuple[str, ...] = ('FACTOR 1', 'FACTOR 2')
    time: str = 'PRODUCTION TIME'


RUNS = [
    *[Run(f'pm_sds_{k}', f'PM SDS {k}', 'T100', f'PM SDS {k}', ADS_OF_SDS[k]) for k in range(1, 7)],
    Run('pm_inert_sds_1', 'PM INERT', 'T100', 'PM INERT', 1),
    # PM INERT SDS k: the PM INERT column with PM SDS k's missing pattern (see load_frames)
    *[Run(f'pm_inert_sds_{k}', f'PM INERT SDS {k}', 'T100', f'PM INERT SDS {k}', ADS_OF_SDS[k]) for k in range(2, 7)],
    Run('known_effects_sds_1', 'PM SDS 1 KNOWN', 'KNOWN', 'PM SDS 1 KNOWN', 1),
    # Medicare per-capita expenditure: 24 organisations x 4 years, one reading each (issue #114)
    Run('aco_medicare', 'Medicare ACO (PCE)', 'ACO', 'PER CAPITA EXPENDITURE', 2, factors=('ACO',), time='YEAR'),
]

# Tom reports capability to 2 decimals and loss shares to 1; tolerance is half a unit of the last place.
CAPABILITY_TOLERANCE = 0.01
# (Tom's label) -> (CapabilityResult.as_dict key). Order shown is the order
# they appear in the report table; "Current" block then "Potential" block.
CAPABILITY_METRICS = [
    ('Current PP',          'pp'),
    ('Current PPU Index',   'ppk_upper'),
    ('Current PPL',         'ppk_lower'),
    ('Current % below LSL', 'pct_below_lsl'),
    ('Current % above USL', 'pct_above_usl'),
    ('Potential PP',          'cp'),
    ('Potential PPU Index',   'cpk_upper'),
    ('Potential PPL',         'cpk_lower'),
    ('Potential % below LSL', 'potential_pct_below_lsl'),
    ('Potential % above USL', 'potential_pct_above_usl'),
]

# Loss shares are compared in percent, to 1 decimal.
LOSS_TOLERANCE = 0.05
# (Tom's label) -> (LossResult.as_dict key, or synthetic key). Synthetic
# keys (pct_pdc_f1, pct_pdc_f2, pct_pdc_factor_interaction) are computed
# inside _build_loss_results since LossResult only exposes the absolute
# pdc_by_factor / pdc_factor_interaction values, not percentages.
LOSS_METRICS = [
    ('mean',        'pct_centering'),
    ('unexplained', 'pct_unexplained'),
    ('pdc',         'pct_pdc'),
    ('pt',          'pct_time'),
    ('pdcxpt',      'pct_interaction'),
    ('F1',          'pct_pdc_f1'),
    ('F2',          'pct_pdc_f2'),
    ('PDC Int',     'pct_pdc_factor_interaction'),
]


def context_to_stratum(context_subtitle: str, n_factors: int = 2) -> str:
    """Convert a slide's context 'PDC RSG - 1-1' (two factors) or 'PDC RSG - ACO-001' (one) to our stratum key."""
    level = context_subtitle.split(' - ', 1)[1]  # '1-1' or 'ACO-001'
    if n_factors == 1:
        return level
    f1, f2 = level.split('-')
    return f'{f1}_{f2}'  # Strata normalized to strings per #73


def close(actual, expected, tol=TOLERANCE):
    """Check if two values are close within tolerance."""
    if actual is None or expected is None:
        return actual is None and expected is None
    if math.isnan(actual) or math.isnan(expected):
        return False
    return abs(actual - expected) <= tol


def classify_page(item):
    """Classify a JSON page item into a test category."""
    title = item.get('analysis_title', '')
    context = item.get('context_subtitle')

    if title == 'Analysis of Original Performance Measurement Behavior':
        return 'overall' if context is None else 'stratified'
    elif 'Main Effects' in title:
        if 'Production Time' in title:
            return 'pt_effects'
        elif context is not None:
            return 'factor_effects'
        else:
            return 'pdc_effects'
    elif 'Interaction' in title or 'Analysis of R3 Residuals' in title:
        return 'interaction'
    elif title == 'Test of Maximum Information':
        return 'max_info'
    elif 'Taguchi Loss' in title:
        return 'loss_function'
    elif 'Capability' in title:
        return 'capability'
    elif 'Maximum Information Analysis' in title:
        return 'max_info_histogram'
    else:
        return 'skip'


def is_location_chart(item):
    """True if this page shows a location chart (Xbar or X), False for dispersion (S or mR)."""
    variable = item.get('variable') or ''
    chart_title = item.get('chart_title') or ''
    if variable and 'STDEV' in variable.upper():
        return False
    if variable and 'MOVING RANGE' in variable.upper():
        return False
    if 'Standard Deviation' in chart_title:
        return False
    return 'Moving Range' not in chart_title


def _coerce_float(v):
    """Cast numpy scalars to Python float so downstream `is True` checks work."""
    if v is None:
        return None
    return float(v)


def _build_capability_results(expected, cap_dict):
    """Validate process-capability indices against the run's reference values."""
    rows = []
    for label, field in CAPABILITY_METRICS:
        actual = _coerce_float(cap_dict.get(field))
        exp = expected.get(field)
        match = bool(close(actual, exp, tol=CAPABILITY_TOLERANCE)) if exp is not None else None
        rows.append({
            'label': label, 'field': field,
            'expected': exp, 'actual': actual, 'match': match,
        })
    return rows


def _augment_loss_with_factor_percentages(loss_dict):
    """Derive per-factor pdc percentages from absolutes (LossResult doesn't expose them).

    Returns a shallow copy with synthetic keys ``pct_pdc_f1``, ``pct_pdc_f2``,
    ``pct_pdc_factor_interaction`` added when the loss decomposition includes
    a multi-factor breakdown. No-op otherwise.
    """
    derived = dict(loss_dict)
    total = loss_dict.get('total')
    pdc_by_factor = loss_dict.get('pdc_by_factor') or {}
    pdc_fi = loss_dict.get('pdc_factor_interaction')
    if not total or total <= 0:
        return derived
    # The validation dataset uses "FACTOR 1" / "FACTOR 2" naming; resolve them
    # positionally so this stays robust if factor names ever change.
    factor_keys = list(pdc_by_factor.keys())
    if len(factor_keys) >= 1:
        derived['pct_pdc_f1'] = 100 * pdc_by_factor[factor_keys[0]] / total
    if len(factor_keys) >= 2:
        derived['pct_pdc_f2'] = 100 * pdc_by_factor[factor_keys[1]] / total
    if pdc_fi is not None:
        derived['pct_pdc_factor_interaction'] = 100 * pdc_fi / total
    return derived


def _build_loss_results(expected, loss_dict):
    """Validate loss-function decomposition (percentages) against the run's reference values."""
    derived = _augment_loss_with_factor_percentages(loss_dict)
    rows = []
    for label, field in LOSS_METRICS:
        actual = _coerce_float(derived.get(field))
        exp = expected.get(field)
        match = bool(close(actual, exp, tol=LOSS_TOLERANCE)) if exp is not None else None
        rows.append({
            'label': label, 'field': field,
            'expected': exp, 'actual': actual, 'match': match,
        })
    return rows


def run_validation(run, pb, study, ref):  # noqa: C901
    """Run all validations for one run (a response column at its ADS) against its reference file.

    Returns
    -------
    dict
        ``{'charts': [...], 'capability': [...], 'loss': [...]}`` where each
        list contains per-row dicts ready for HTML rendering.
    """
    results = []
    items = ref['items']
    specs = ref['specs']
    tol = ref.get('chart_tolerance', TOLERANCE)
    factors, time = list(run.factors), run.time
    f_str, t_str = '[' + ', '.join(factors) + ']', f'[{time}]'

    # Pre-compute results we'll need
    computed = {}

    # Xbar/S wherever some subgroups are replicated: ADS 1, and ADS 3, where the 10-1 manual makes
    # Xbar/S VAS's default (Tom's 10/3/2026 run draws PM SDS 3 that way). ADS 2 uses X/mR.
    xbar_s = run.ads in (1, 3)

    # Overall charts
    if xbar_s:
        computed['overall'] = study.execute(chart='Xbar', by=[], companion=True)
        computed['stratified'] = study.execute(chart='Xbar', by=[time], companion=True)
    else:
        computed['overall'] = study.execute(chart='X', by=[], companion=True)
        computed['stratified'] = study.execute(chart='X', by=factors, companion=True)

    # Effects charts — all SDS types
    # PDC effects (pages 20-21)
    computed['pdc_effects_xbar'] = study.execute(
        chart='Xbar', by=factors,
        value='R6', recentered=True
    )
    computed['pdc_effects_s'] = study.execute(
        chart='S', by=factors, value='R6'
    )
    # PT effects (pages 22-23) — Xbar/S of R4 by time for all SDS (R4 carries the period effect,
    # 10-1 manual Eq 14-16). When charted by=[time], each time subgroup has multiple factor
    # levels, giving n>1 subgroups even in SDS 2, so Xbar/S is correct.
    computed['pt_effects_xbar'] = study.execute(
        chart='Xbar', by=[time], value='R4', recentered=True
    )
    computed['pt_effects_s'] = study.execute(
        chart='S', by=[time], value='R4', recentered=True
    )
    # Interaction (pages 28-29)
    if xbar_s:
        computed['interaction_xbar'] = study.execute(
            chart='Xbar', by=[*factors, time],
            value='R3', recentered=True
        )
        computed['interaction_s'] = study.execute(
            chart='S', by=[*factors, time],
            value='R3', recentered=True
        )
    else:
        computed['r3_xmr'] = study.execute(
            chart='X', by=[], value='R3', recentered=True, companion=True
        )
    # Individual factor effects (pages 24-27), when the conditions are built from two factors
    if len(factors) >= 2:
        for tag, factor in (('f1', factors[0]), ('f2', factors[1])):
            computed[f'{tag}_effects_xbar'] = study.execute(chart='Xbar', by=[factor], value='R6', recentered=True)
            computed[f'{tag}_effects_s'] = study.execute(chart='S', by=[factor], value='R6')

    computed['max_info_xmr'] = study.execute(chart='X', by=[], value='R2')
    computed['loss'] = study.loss_function(target=specs['target'])
    computed['capability'] = study.capability(lsl=specs['lsl'], usl=specs['usl'], target=specs['target'])

    for item in items:
        page = item['page_number']
        category = classify_page(item)
        chart_title = item.get('chart_title', '')
        chart_subtitle = item.get('chart_subtitle', '')
        context = item.get('context_subtitle')
        expected_cl = item.get('CL')
        expected_lbl = item.get('LBL')
        expected_ubl = item.get('UBL')
        is_location = is_location_chart(item)

        base = {
            'run': run.id,
            'page': page,
            'chart_title': chart_title,
            'chart_subtitle': chart_subtitle,
            'context_subtitle': context or '',
            'category': category,
        }

        def append_result(
            chart_type, by_str, value, recentered, actual_cl, actual_lpl, actual_upl, table,
            _base=base, _ecl=expected_cl, _elbl=expected_lbl, _eubl=expected_ubl, **extra
        ):
            results.append({
                **_base,
                'chart_type': chart_type,
                'by': by_str,
                'value': value,
                'recentered': recentered,
                'expected_cl': _ecl,
                'expected_lbl': _elbl,
                'expected_ubl': _eubl,
                'actual_cl': actual_cl,
                'actual_lpl': actual_lpl,
                'actual_upl': actual_upl,
                'match_cl': close(actual_cl, _ecl, tol) if _ecl is not None else None,
                'match_lpl': close(actual_lpl, _elbl, tol) if _elbl is not None else None,
                'match_upl': close(actual_upl, _eubl, tol) if _eubl is not None else None,
                'chart_table': table,
                **extra,
            })

        def stats_from(result_obj, chart_type):
            stats = result_obj.get_statistics(chart_type)
            center = stats['center']
            lpl_val = stats.get('lpl')
            upl_val = stats.get('upl')
            if stats.get('limits_vary') and lpl_val is None and upl_val is None:
                # Limits step with subgroup size (10-1 manual Eqs 11-16/17). VAS/Minitab labels the
                # limits of the last subgroup, so that is the value its slide carries.
                table = safe_chart_table(result_obj, chart_type)
                if table is not None and len(table):
                    lpl_val, upl_val = table['lpl'].iloc[-1], table['upl'].iloc[-1]
            cl = float(center) if center is not None else None
            lpl = float(lpl_val) if lpl_val is not None else None
            upl = float(upl_val) if upl_val is not None else None
            return cl, lpl, upl

        def primary_chart_type(_loc=is_location):
            if xbar_s:
                return 'Xbar' if _loc else 'S'
            return 'X' if _loc else 'mR'

        if category == 'overall':
            overall = computed['overall']
            chart_type = primary_chart_type()
            cl, lpl, upl = stats_from(overall, chart_type)
            append_result(chart_type, '[]', 'response', False, cl, lpl, upl, safe_chart_table(overall, chart_type))

        elif category == 'stratified':
            stratum = context_to_stratum(context, len(run.factors))
            focused = computed['stratified'].focus(stratum)
            chart_type = primary_chart_type()
            cl, lpl, upl = stats_from(focused, chart_type)
            by_str = t_str if xbar_s else f_str
            append_result(chart_type, by_str, 'response', False, cl, lpl, upl,
                          safe_chart_table(focused, chart_type), stratum=str(stratum))

        elif category == 'pdc_effects':
            result_obj = computed['pdc_effects_xbar'] if is_location else computed['pdc_effects_s']
            chart_type = 'Xbar' if is_location else 'S'
            cl, lpl, upl = stats_from(result_obj, chart_type)
            recentered = is_location
            append_result(chart_type, f_str, 'R6', recentered, cl, lpl, upl,
                          safe_chart_table(result_obj, chart_type))

        elif category == 'pt_effects':
            result_obj = computed['pt_effects_xbar'] if is_location else computed['pt_effects_s']
            chart_type = 'Xbar' if is_location else 'S'
            cl, lpl, upl = stats_from(result_obj, chart_type)
            append_result(chart_type, t_str, 'R4', True, cl, lpl, upl,
                          safe_chart_table(result_obj, chart_type))

        elif category == 'factor_effects':
            if context and 'F1' in context:
                xbar_r = computed['f1_effects_xbar']
                s_r = computed['f1_effects_s']
                by_str = f'[{factors[0]}]'
            elif context and 'F2' in context:
                xbar_r = computed['f2_effects_xbar']
                s_r = computed['f2_effects_s']
                by_str = f'[{factors[1]}]'
            else:
                append_result('?', '?', 'R6', True, None, None, None, None)
                continue

            result_obj = xbar_r if is_location else s_r
            chart_type = 'Xbar' if is_location else 'S'
            cl, lpl, upl = stats_from(result_obj, chart_type)
            recentered = is_location  # S is NOT recentered per notebook
            append_result(chart_type, by_str, 'R6', recentered, cl, lpl, upl,
                          safe_chart_table(result_obj, chart_type))

        elif category == 'interaction':
            if xbar_s:
                result_obj = computed['interaction_xbar'] if is_location else computed['interaction_s']
                chart_type = 'Xbar' if is_location else 'S'
                cl, lpl, upl = stats_from(result_obj, chart_type)
                append_result(chart_type, '[' + ', '.join([*factors, time]) + ']', 'R3', True,
                              cl, lpl, upl, safe_chart_table(result_obj, chart_type))
            else:
                r3 = computed['r3_xmr']
                if is_location:
                    cl, lpl, upl = stats_from(r3, 'X')
                    append_result('X', '[]', 'R3', True, cl, lpl, upl, safe_chart_table(r3, 'X'))
                else:
                    cl, lpl, upl = stats_from(r3, 'mR')
                    append_result('mR', '[]', 'R3', True, cl, lpl, upl, safe_chart_table(r3, 'mR'))

        elif category == 'max_info':
            mi_xmr = computed['max_info_xmr']
            cl, lpl, upl = stats_from(mi_xmr, 'X')
            append_result('X', '[]', 'R2', False, cl, lpl, upl, safe_chart_table(mi_xmr, 'X'))

        elif category in ('loss_function', 'capability', 'max_info_histogram'):
            # Covered by the dedicated Capability and Loss-function sub-tables
            # rendered per SDS by generate_html(); skip the redundant chart-table
            # row that would otherwise show as grey "Deferred".
            continue

        else:
            # No other category produces a meaningful chart-row; drop it.
            continue

    # Compare at full precision: Tom's figures are displayed to 1-2 decimals and the tolerance
    # absorbs that; rounding our side first would double-count the rounding.
    capability_results = _build_capability_results(
        ref.get('capability', {}), computed['capability'].as_dict(round_to=12)
    )
    loss_results = _build_loss_results(ref.get('loss', {}), computed['loss'].as_dict(round_to=12))

    return {
        'charts': results,
        'capability': capability_results,
        'loss': loss_results,
    }


def fmt(val, decimals=2):
    """Format a value for display."""
    if val is None:
        return '-'
    return f'{val:.{decimals}f}'


def match_class(match_val):
    """CSS class for a match result."""
    if match_val is None:
        return 'skip'
    return 'pass' if match_val else 'fail'


def safe_chart_table(result, chart_type):
    """Try to get chart_table, return None on error."""
    try:
        return result.chart_table(chart_type)
    except (ValueError, KeyError, TypeError):
        return None


def render_chart_table_html(table):
    """Render a chart_table DataFrame as a collapsible HTML detail."""
    if table is None:
        return ''
    html = '<details><summary>chart_table()</summary><table class="chart-table"><tr>'
    for col in table.columns:
        html += f'<th>{col}</th>'
    html += '</tr>'
    for _, row in table.iterrows():
        html += '<tr>'
        for col in table.columns:
            val = row[col]
            if isinstance(val, float):
                html += f'<td>{val:.4f}</td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</table></details>'
    return html


def _render_metric_table(title: str, rows, value_decimals: int, tolerance: float) -> str:
    """Render a small capability- or loss-style metric table (4 columns)."""
    sub_pass = sum(1 for r in rows if r['match'] is True)
    sub_fail = sum(1 for r in rows if r['match'] is False)
    sub_skip = sum(1 for r in rows if r['match'] is None)

    html = (
        f'<h3 style="color:#666; margin-top:18px; margin-bottom:6px; font-size:15px;">{title}'
        f' <small style="color:#888; font-weight:normal;">(tol &plusmn;{tolerance})</small></h3>'
    )
    html += (
        f'<div class="summary" style="margin:4px 0; padding:6px 12px; font-size:12px;">'
        f'<span class="pass-count">{sub_pass} passed</span> / '
        f'<span class="fail-count">{sub_fail} failed</span> / '
        f'<span class="skip-count">{sub_skip} pending</span>'
        f'</div>'
    )
    html += '<table class="main"><tr>'
    html += '<th>Metric</th><th>Expected</th><th>Actual</th><th>Match</th></tr>'
    for r in rows:
        cls = match_class(r['match'])
        mark = 'PASS' if r['match'] is True else ('FAIL' if r['match'] is False else 'pending')
        exp = fmt(r['expected'], decimals=value_decimals)
        act = fmt(r['actual'], decimals=value_decimals)
        html += (
            f'<tr class="{cls}"><td>{r["label"]}</td>'
            f'<td>{exp}</td><td>{act}</td><td>{mark}</td></tr>'
        )
    html += '</table>'
    return html


def generate_html(all_results, aux_by_run, runs):  # noqa: C901
    """Generate the full HTML report.

    Parameters
    ----------
    all_results : list[dict]
        Per-chart-row result dicts, each tagged with its run id.
    aux_by_run : dict[str, dict]
        Per run id: ``{'capability': [...], 'loss': [...], 'specs': {...}, 'loss_note': str | None}``.
    runs : list[Run]
        The runs, in report order.
    """
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>E2E Bishop Validation Report — processbehavior</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #fafafa; }
h1 { color: #333; }
h2 { color: #555; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
.summary { background: #fff; padding: 15px; border-radius: 8px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.summary .pass-count { color: #2e7d32; font-weight: bold; }
.summary .fail-count { color: #c62828; font-weight: bold; }
.summary .skip-count { color: #757575; }
table.main { border-collapse: collapse; width: 100%; margin: 10px 0;
  background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
table.main th { background: #f5f5f5; padding: 8px 12px; text-align: left;
  font-size: 13px; border-bottom: 2px solid #ddd; }
table.main td { padding: 6px 12px; border-bottom: 1px solid #eee; font-size: 13px; }
table.main tr.pass td { background: #e8f5e9; }
table.main tr.fail td { background: #ffebee; }
table.main tr.skip td { background: #f5f5f5; color: #999; }
table.chart-table { border-collapse: collapse; margin: 5px 0; font-size: 11px; }
table.chart-table th, table.chart-table td { padding: 3px 8px; border: 1px solid #ddd; }
table.chart-table th { background: #f0f0f0; }
details { margin: 4px 0; }
summary { cursor: pointer; color: #1565c0; font-size: 12px; }
.tolerance { font-size: 12px; color: #888; }
.timestamp { font-size: 12px; color: #999; margin-top: 20px; }
</style>
</head>
<body>
<h1>E2E Bishop Validation Report</h1>
<p>Comparison of processbehavior library output against Tom Bishop's VAS Minitab analyses, run by run.</p>
<p class="tolerance">Tolerance: &plusmn;0.01 (half-unit of last decimal place in reference data)</p>
"""
    # Global summary
    total_checks = 0
    total_pass = 0
    total_fail = 0
    total_skip = 0

    for r in all_results:
        for key in ('match_cl', 'match_lpl', 'match_upl'):
            val = r.get(key)
            if val is True:
                total_checks += 1
                total_pass += 1
            elif val is False:
                total_checks += 1
                total_fail += 1
            else:
                total_skip += 1

    # Also fold in capability + loss matches from auxiliary results
    for aux in aux_by_run.values():
        for row in aux.get('capability', []) + aux.get('loss', []):
            if row['match'] is True:
                total_checks += 1
                total_pass += 1
            elif row['match'] is False:
                total_checks += 1
                total_fail += 1
            else:
                total_skip += 1

    html += f"""
<div class="summary">
<strong>Overall:</strong>
<span class="pass-count">{total_pass} passed</span> /
<span class="fail-count">{total_fail} failed</span> /
<span class="skip-count">{total_skip} skipped</span>
out of {total_checks + total_skip} total checks
</div>
"""

    # One section per run
    for run in runs:
        run_results = [r for r in all_results if r['run'] == run.id]
        if not run_results:
            continue

        run_pass = sum(1 for r in run_results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is True)
        run_fail = sum(1 for r in run_results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is False)
        run_skip = sum(1 for r in run_results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is None)

        html += f"""
<h2>{run.label} — Analytic Design State {run.ads}</h2>
<div class="summary">
<span class="pass-count">{run_pass} passed</span> /
<span class="fail-count">{run_fail} failed</span> /
<span class="skip-count">{run_skip} skipped</span>
</div>
<table class="main">
<tr>
    <th>Page</th>
    <th>Chart Title</th>
    <th>Context</th>
    <th>Chart Type</th>
    <th>By</th>
    <th>Value</th>
    <th>Recentered</th>
    <th>Stat</th>
    <th>Expected</th>
    <th>Actual</th>
    <th>Match</th>
    <th>Details</th>
</tr>
"""
        for r in run_results:
            # Determine overall row status
            matches = [r.get(k) for k in ('match_cl', 'match_lpl', 'match_upl')]
            if any(m is False for m in matches):
                row_class = 'fail'
            elif any(m is True for m in matches):
                row_class = 'pass'
            else:
                row_class = 'skip'

            # Build stat rows (CL, LPL, UPL)
            stats_html = ''
            for stat_name, exp_key, act_key, match_key in [
                ('CL', 'expected_cl', 'actual_cl', 'match_cl'),
                ('LPL', 'expected_lbl', 'actual_lpl', 'match_lpl'),
                ('UPL', 'expected_ubl', 'actual_upl', 'match_upl'),
            ]:
                exp = r.get(exp_key)
                act = r.get(act_key)
                m = r.get(match_key)
                if exp is not None or act is not None:
                    mark = 'PASS' if m is True else ('FAIL' if m is False else '-')
                    if stats_html:
                        stats_html += '<br>'
                    stats_html += f'{stat_name}'

            # For the first stat line, build the full row
            stat_entries = []
            for stat_name, exp_key, act_key, match_key in [
                ('CL', 'expected_cl', 'actual_cl', 'match_cl'),
                ('LPL', 'expected_lbl', 'actual_lpl', 'match_lpl'),
                ('UPL', 'expected_ubl', 'actual_upl', 'match_upl'),
            ]:
                exp = r.get(exp_key)
                act = r.get(act_key)
                m = r.get(match_key)
                if exp is not None or act is not None:
                    mark = 'PASS' if m is True else ('FAIL' if m is False else '-')
                    stat_entries.append((stat_name, fmt(exp), fmt(act), mark))

            if not stat_entries:
                stat_entries = [('-', '-', '-', '-')]

            chart_table_html = render_chart_table_html(r.get('chart_table'))
            note = r.get('note', '')

            # Render one row per stat
            for i, (sn, ev, av, mk) in enumerate(stat_entries):
                if i == 0:
                    rowspan = len(stat_entries)
                    html += f'<tr class="{row_class}">'
                    html += f'<td rowspan="{rowspan}">{r["page"]}</td>'
                    html += f'<td rowspan="{rowspan}">{r["chart_title"]}<br><small>{r["chart_subtitle"]}</small></td>'
                    html += f'<td rowspan="{rowspan}">{r["context_subtitle"]}</td>'
                    html += f'<td rowspan="{rowspan}">{r["chart_type"]}</td>'
                    html += f'<td rowspan="{rowspan}">{r["by"]}</td>'
                    html += f'<td rowspan="{rowspan}">{r["value"]}</td>'
                    html += f'<td rowspan="{rowspan}">{r["recentered"]}</td>'
                else:
                    html += f'<tr class="{row_class}">'

                html += f'<td>{sn}</td><td>{ev}</td><td>{av}</td><td>{mk}</td>'

                if i == 0:
                    html += f'<td rowspan="{rowspan}">{chart_table_html}{note}</td>'
                html += '</tr>'

        html += '</table>'

        # Capability + loss-function sections under each SDS
        aux = aux_by_run.get(run.id, {})
        specs = aux.get('specs', {})
        cap_rows = aux.get('capability', [])
        loss_rows = aux.get('loss', [])
        if cap_rows:
            html += _render_metric_table(
                f"Process Capability (LSL={specs.get('lsl')}, USL={specs.get('usl')}, Target={specs.get('target')})",
                cap_rows, value_decimals=2, tolerance=CAPABILITY_TOLERANCE,
            )
        if loss_rows:
            html += _render_metric_table(
                'Taguchi Loss Function (% of total)',
                loss_rows, value_decimals=1, tolerance=LOSS_TOLERANCE,
            )
            if aux.get('loss_note'):
                html += f'<p class="tolerance">Pending: {aux["loss_note"]}</p>'

    from datetime import datetime
    html += f'<p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
    html += '</body></html>'
    return html


# ---------------------------------------------------------------------------
# MyST summary page (docs/reference/validation.md)
# ---------------------------------------------------------------------------

MYST_OUTPUT = Path(__file__).parent.parent / 'docs' / 'reference' / 'validation.md'


def _tally(all_results, aux_by_run):
    """(passed, failed, pending) per run id, over charts + capability + loss."""
    per_run: dict[str, list[int]] = {}
    for row in all_results:
        counts = per_run.setdefault(row['run'], [0, 0, 0])
        for key in ('match_cl', 'match_lpl', 'match_upl'):
            match = row.get(key)
            counts[0 if match is True else 1 if match is False else 2] += 1
    for run_id, aux in aux_by_run.items():
        counts = per_run.setdefault(run_id, [0, 0, 0])
        for row in aux['capability'] + aux['loss']:
            match = row['match']
            counts[0 if match is True else 1 if match is False else 2] += 1
    return per_run


def generate_myst_summary(all_results, aux_by_run, runs):
    """A short, committable summary of the validation run.

    Deliberately not the full HTML report: this is the page a reader lands on,
    so it carries the totals and what they cover, and links to the detailed
    artifact. It also carries no timestamp — a generated page that changes on
    every run is a permanent diff, so provenance is the library version instead.
    """
    import processbehavior

    per_run = _tally(all_results, aux_by_run)
    total_pass = sum(v[0] for v in per_run.values())
    total_fail = sum(v[1] for v in per_run.values())
    total_skip = sum(v[2] for v in per_run.values())
    data_files = {'T100': '`PBTESTDATABASE_T100.csv`', 'KNOWN': '`PBTESTKNOWNEFFECTS_T100.csv`',
                  'ACO': '`aco_per_capita_expenditure.csv`'}

    lines = [
        '# Validation against Bishop\'s reference results',
        '',
        'Every analytical output this library produces is checked, number by number,',
        "against Dr. Thomas A. Bishop's VAS results (Minitab) for the same data.",
        'Not "inspired by" and not spot-checked — the chart centers, control limits,',
        'capability indices and loss-function components are compared to the reference',
        'and the run fails if any of them disagree.',
        '',
        f'**{total_pass} assertions pass**'
        + (f', {total_fail} fail' if total_fail else ', 0 fail')
        + (f', {total_skip} have no reference value yet.' if total_skip else '.'),
        '',
        f'Generated by `validation/e2e_bishop_report.py` from processbehavior '
        f'{processbehavior.__version__}.',
        '',
        '## Coverage',
        '',
        '| Run | Data | Analytic design state | Assertions | No reference yet | Result |',
        '|---|---|---|---|---|---|',
    ]
    for run in runs:
        passed, failed, skipped = per_run.get(run.id, [0, 0, 0])
        verdict = '✅ all pass' if failed == 0 else f'❌ {failed} failing'
        lines.append(
            f'| `{run.label}` | {data_files[run.data]} | ADS {run.ads} '
            f'| {passed + failed} | {skipped} | {verdict} |'
        )

    lines += [
        '',
        'PM SDS 1-6 are one fill-weight dataset with deletions; SDS 4-6 are incomplete grids, analysed',
        'as the complete design that survives tidying (ADS 1-3). PM INERT is pure noise with the same',
        'six missing-data patterns. The known-effects data are built from known condition, time and',
        'interaction effects plus noise. The Medicare data are per-capita expenditure for 24',
        'organisations over 4 years, one reading each (issue #114).',
        '',
        '## What is covered',
        '',
        '- **Charts** — center line and both natural process limits, for the primary',
        '  chart and for each valid chart x residual pair at that design state.',
        '- **Capability** — Pp, Ppk (upper and lower), Cp, Cpk, and the percentage of',
        '  the distribution beyond each specification limit.',
        '- **Loss function** — the Taguchi loss decomposition and its components.',
        '',
        '## Values with no reference yet',
        '',
        '- Charts whose limits vary with subgroup size (unequal subgroups, as in SDS 3, 4 and 6):',
        '  VAS prints "UNEQUAL" instead of a single limit, so the slides carry no value to compare.',
        '- The factor S charts of SDS 1 and 4 (pages 25 and 27), which VAS draws on condition',
        '  subgroups where the manual uses one subgroup per factor level; awaiting Dr. Bishop.',
        '- PM SDS 5 loss shares: its VAS run used the process mean as the target; the gate uses',
        '  237 for all six PM runs, pending a VAS rerun at 237.',
        '',
        '## Reproducing this',
        '',
        '```bash',
        'python validation/e2e_bishop_report.py',
        '```',
        '',
        'That writes this page and a detailed HTML report',
        '(`validation/e2e_bishop_report.html`) with every compared value, expected',
        'beside actual, grouped by run.',
        '',
    ]
    return '\n'.join(lines)


def load_frames():
    """Read every data source. PM INERT SDS k (k = 2..6) is the PM INERT column with PM SDS k's
    missing pattern, which is how the PM INERT SDS 2-6 columns of Dr. Bishop's VAS runs are built."""
    frames = {name: pd.read_csv(path) for name, path in DATA.items()}
    t100 = frames['T100']
    inert = pd.to_numeric(t100['PM INERT'], errors='coerce')
    for k in range(2, 7):
        present = pd.to_numeric(t100[f'PM SDS {k}'], errors='coerce').notna()
        t100[f'PM INERT SDS {k}'] = inert.where(present)
    return frames


def main():
    print('Loading validation data...')
    frames = load_frames()
    pbs = {name: ProcessBehavior(df) for name, df in frames.items()}

    all_results = []
    aux_by_run = {}   # {run id: {'capability': [...], 'loss': [...], 'specs': {...}, 'loss_note': ...}}
    total_fails = 0

    for run in RUNS:
        print(f'\nProcessing {run.label}...')
        pb = pbs[run.data]
        study = pb.formulate(response=run.response, factors=list(run.factors), time=run.time)
        ads = study.analytical_design_state
        print(f'  Analytical SDS: {ads}')
        if ads.sds != run.ads:
            print(f'  ADS MISMATCH: expected {run.ads}')
            total_fails += 1

        ref = json.loads((FIXTURES_DIR / f'{run.id}.json').read_text())
        payload = run_validation(run, pb, study, ref)
        chart_results = payload['charts']
        all_results.extend(chart_results)
        aux_by_run[run.id] = {
            'capability': payload['capability'],
            'loss': payload['loss'],
            'specs': ref['specs'],
            'loss_note': ref.get('loss_note'),
        }

        passes = sum(
            1 for r in chart_results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is True
        )
        fails = sum(
            1 for r in chart_results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is False
        )
        for row in payload['capability'] + payload['loss']:
            if row['match'] is True:
                passes += 1
            elif row['match'] is False:
                fails += 1
        print(f'  Results: {passes} passed, {fails} failed')
        total_fails += fails

    print('\nGenerating HTML report...')
    OUTPUT_HTML.write_text(generate_html(all_results, aux_by_run, RUNS))
    print(f'Report written to: {OUTPUT_HTML}')

    MYST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MYST_OUTPUT.write_text(generate_myst_summary(all_results, aux_by_run, RUNS))
    print(f'Docs summary written to: {MYST_OUTPUT}')

    # The reports above are evidence; this exit code is the gate. Reference
    # disagreement must fail the build itself, not just drift the docs page —
    # otherwise committing the drifted page makes CI green with wrong numbers.
    # 'pending' rows (match is None: no reference value yet) are never fatal.
    if total_fails:
        print(f'\nVALIDATION FAILED: {total_fails} assertion(s) diverge from Bishop reference values.')
        sys.exit(1)


if __name__ == '__main__':
    main()
