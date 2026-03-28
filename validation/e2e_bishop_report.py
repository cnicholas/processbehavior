"""
E2E Validation Report: processbehavior vs Tom Bishop's VAS Analyses

Generates an HTML report comparing our library output against Tom Bishop's
Minitab reference analyses for SDS 1-3. Uses TABVASTESTDATABASE.csv as input.

Usage:
    python validation/e2e_bishop_report.py
"""
import json
import math
from pathlib import Path

import pandas as pd

from processbehavior import ProcessBehavior

# --- Configuration ---

VALIDATION_CSV = Path(__file__).parent / 'TABVASTESTDATABASE.csv'
FIXTURES_DIR = Path(__file__).parent.parent / 'tests' / 'fixtures' / 'bishop_analyses'
OUTPUT_HTML = Path(__file__).parent / 'e2e_bishop_report.html'
TOLERANCE = 0.01  # half-unit of last decimal place

SPEC_LSL = 232
SPEC_USL = 242
SPEC_TARGET = 237

SDS_CONFIGS = {
    1: {'response_attr': 'PM_SDS_1', 'json': 'vassds1analysis.json', 'chart': 'Xbar'},
    2: {'response_attr': 'PM_SDS_2', 'json': 'vassds2analysis.json', 'chart': 'XmR'},
    3: {'response_attr': 'PM_SDS_3', 'json': 'vassds3analysis.json', 'chart': 'XmR'},
}


def context_to_stratum(context_subtitle: str, sds: int) -> str | tuple:
    """Convert JSON context_subtitle 'PDC RSG - 1-1' to our stratum key."""
    # Extract '1-1' from 'PDC RSG - 1-1'
    level = context_subtitle.split(' - ')[1]  # '1-1'
    f1, f2 = level.split('-')
    if sds == 1:
        return f'{f1}_{f2}'  # SDS 1 strata are strings like '1_1'
    else:
        return (int(f1), int(f2))  # SDS 2/3 strata are tuples like (1, 1)


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
    variable = item.get('variable')
    context = item.get('context_subtitle')

    if title == 'Analysis of Original Performance Measurement Behavior':
        if context is None:
            return 'overall'
        else:
            return 'stratified'
    elif 'Process Design Condition Main Effects' in title and context is None:
        return 'pdc_effects'
    elif 'Process Design Factor Main Effects' in title and context is None:
        # PDC-level S chart (same as pdc_effects) when no factor context
        return 'pdc_effects'
    elif 'Process Design Factor Main Effects' in title and context is not None:
        return 'factor_effects'
    elif 'Process Design Condition Main Effects' in title and context is not None:
        return 'factor_effects'
    elif 'Production Time Main Effects' in title:
        return 'pt_effects'
    elif 'Interaction' in title or 'Analysis of R3 Residuals' in title:
        return 'interaction'
    elif title == 'Test of Maximum Information':
        return 'max_info'
    elif 'Taguchi Loss' in title:
        return 'loss_function'
    elif 'Process Capability' in title or 'Current Process Capability' in title or 'Potential Process Capability' in title:
        return 'capability'
    elif 'Maximum Information Analysis' in title:
        return 'max_info_histogram'
    else:
        return 'skip'


def is_location_chart(item):
    """True if this page shows a location chart (Xbar or XmR), False for dispersion (S or R)."""
    variable = item.get('variable') or ''
    chart_title = item.get('chart_title') or ''
    if variable and 'STDEV' in variable.upper():
        return False
    if variable and 'MOVING RANGE' in variable.upper():
        return False
    if 'Standard Deviation' in chart_title:
        return False
    if 'Moving Range' in chart_title:
        return False
    return True


def run_sds_validation(sds_num, pb, study, json_data):
    """Run all validations for one SDS type. Returns list of result dicts."""
    results = []
    items = json_data['items']

    # Pre-compute results we'll need
    computed = {}

    # Overall charts
    if sds_num == 1:
        computed['overall'] = study.execute(chart='Xbar', by=[], companion=True)
        computed['stratified'] = study.execute(chart='Xbar', by=[pb.cols.PRODUCTION_TIME], companion=True)
    else:
        computed['overall'] = study.execute(chart='XmR', by=[], companion=True)
        computed['stratified'] = study.execute(chart='XmR', by=[pb.cols.FACTOR_1, pb.cols.FACTOR_2], companion=True)

    # Effects charts — all SDS types
    # PDC effects (pages 20-21)
    computed['pdc_effects_xbar'] = study.execute(
        chart='Xbar', by=[pb.cols.FACTOR_1, pb.cols.FACTOR_2],
        value='R6', recentered=True
    )
    computed['pdc_effects_s'] = study.execute(
        chart='S', by=[pb.cols.FACTOR_1, pb.cols.FACTOR_2], value='R6'
    )
    # PT effects (pages 22-23) — by=[PRODUCTION_TIME] for all SDS types
    computed['pt_effects_xbar'] = study.execute(
        chart='Xbar', by=[pb.cols.PRODUCTION_TIME], value='R3', recentered=True
    )
    computed['pt_effects_s'] = study.execute(
        chart='S', by=[pb.cols.PRODUCTION_TIME], value='R3', recentered=True
    )
    # Interaction (pages 28-29)
    if sds_num == 1:
        computed['interaction_xbar'] = study.execute(
            chart='Xbar', by=[pb.cols.FACTOR_1, pb.cols.FACTOR_2, pb.cols.PRODUCTION_TIME],
            value='R3', recentered=True
        )
        computed['interaction_s'] = study.execute(
            chart='S', by=[pb.cols.FACTOR_1, pb.cols.FACTOR_2, pb.cols.PRODUCTION_TIME],
            value='R3', recentered=True
        )
    else:
        computed['r3_xmr'] = study.execute(
            chart='XmR', by=[], value='R3', recentered=True, companion=True
        )
    # Individual factor effects (pages 24-27)
    computed['f1_effects_xbar'] = study.execute(
        chart='Xbar', by=[pb.cols.FACTOR_1], value='R6', recentered=True
    )
    computed['f1_effects_s'] = study.execute(
        chart='S', by=[pb.cols.FACTOR_1], value='R6'
    )
    computed['f2_effects_xbar'] = study.execute(
        chart='Xbar', by=[pb.cols.FACTOR_2], value='R6', recentered=True
    )
    computed['f2_effects_s'] = study.execute(
        chart='S', by=[pb.cols.FACTOR_2], value='R6'
    )

    computed['max_info_xmr'] = study.execute(chart='XmR', by=[], value='R2')
    computed['loss'] = study.loss_function(target=SPEC_TARGET)
    computed['capability'] = study.capability(lsl=SPEC_LSL, usl=SPEC_USL, target=SPEC_TARGET)

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
            'sds': sds_num,
            'page': page,
            'chart_title': chart_title,
            'chart_subtitle': chart_subtitle,
            'context_subtitle': context or '',
            'category': category,
        }

        def append_result(chart_type, by_str, value, recentered, actual_cl, actual_lpl, actual_upl, table, **extra):
            results.append({
                **base,
                'chart_type': chart_type,
                'by': by_str,
                'value': value,
                'recentered': recentered,
                'expected_cl': expected_cl,
                'expected_lbl': expected_lbl,
                'expected_ubl': expected_ubl,
                'actual_cl': actual_cl,
                'actual_lpl': actual_lpl,
                'actual_upl': actual_upl,
                'match_cl': close(actual_cl, expected_cl) if expected_cl is not None else None,
                'match_lpl': close(actual_lpl, expected_lbl) if expected_lbl is not None else None,
                'match_upl': close(actual_upl, expected_ubl) if expected_ubl is not None else None,
                'chart_table': table,
                **extra,
            })

        def stats_from(result_obj, chart_type):
            stats = result_obj.get_statistics(chart_type)
            cl = float(stats['center']) if stats['center'] != 'Varies' else None
            lpl_val = stats.get('lpl', 0)
            upl_val = stats.get('upl', 0)
            lpl = float(lpl_val) if lpl_val != 'Varies' else None
            upl = float(upl_val) if upl_val != 'Varies' else None
            return cl, lpl, upl

        if category == 'overall':
            overall = computed['overall']
            if sds_num == 1:
                chart_type = 'Xbar' if is_location else 'S'
            else:
                chart_type = 'XmR' if is_location else 'R'
            cl, lpl, upl = stats_from(overall, chart_type)
            append_result(chart_type, '[]', 'response', False, cl, lpl, upl, safe_chart_table(overall, chart_type))

        elif category == 'stratified':
            stratum = context_to_stratum(context, sds_num)
            focused = computed['stratified'].focus(stratum)
            if sds_num == 1:
                chart_type = 'Xbar' if is_location else 'S'
            else:
                chart_type = 'XmR' if is_location else 'R'
            cl, lpl, upl = stats_from(focused, chart_type)
            by_str = '[PRODUCTION_TIME]' if sds_num == 1 else '[FACTOR_1, FACTOR_2]'
            append_result(chart_type, by_str, 'response', False, cl, lpl, upl,
                          safe_chart_table(focused, chart_type), stratum=str(stratum))

        elif category == 'pdc_effects':
            result_obj = computed['pdc_effects_xbar'] if is_location else computed['pdc_effects_s']
            chart_type = 'Xbar' if is_location else 'S'
            cl, lpl, upl = stats_from(result_obj, chart_type)
            recentered = is_location
            append_result(chart_type, '[FACTOR_1, FACTOR_2]', 'R6', recentered, cl, lpl, upl,
                          safe_chart_table(result_obj, chart_type))

        elif category == 'pt_effects':
            result_obj = computed['pt_effects_xbar'] if is_location else computed['pt_effects_s']
            chart_type = 'Xbar' if is_location else 'S'
            cl, lpl, upl = stats_from(result_obj, chart_type)
            append_result(chart_type, '[PRODUCTION_TIME]', 'R3', True, cl, lpl, upl,
                          safe_chart_table(result_obj, chart_type))

        elif category == 'factor_effects':
            if context and 'F1' in context:
                xbar_r = computed['f1_effects_xbar']
                s_r = computed['f1_effects_s']
                by_str = '[FACTOR_1]'
            elif context and 'F2' in context:
                xbar_r = computed['f2_effects_xbar']
                s_r = computed['f2_effects_s']
                by_str = '[FACTOR_2]'
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
            if sds_num == 1:
                result_obj = computed['interaction_xbar'] if is_location else computed['interaction_s']
                chart_type = 'Xbar' if is_location else 'S'
                cl, lpl, upl = stats_from(result_obj, chart_type)
                append_result(chart_type, '[FACTOR_1, FACTOR_2, PRODUCTION_TIME]', 'R3', True,
                              cl, lpl, upl, safe_chart_table(result_obj, chart_type))
            else:
                r3 = computed['r3_xmr']
                if is_location:
                    cl, lpl, upl = stats_from(r3, 'XmR')
                    append_result('XmR', '[]', 'R3', True, cl, lpl, upl, safe_chart_table(r3, 'XmR'))
                else:
                    cl, lpl, upl = stats_from(r3, 'R')
                    append_result('R', '[]', 'R3', True, cl, lpl, upl, safe_chart_table(r3, 'R'))

        elif category == 'max_info':
            mi_xmr = computed['max_info_xmr']
            cl, lpl, upl = stats_from(mi_xmr, 'XmR')
            append_result('XmR', '[]', 'R2', False, cl, lpl, upl, safe_chart_table(mi_xmr, 'XmR'))

        elif category in ('loss_function', 'capability', 'max_info_histogram'):
            append_result(category, '-', '-', False, None, None, None, None,
                          note='Deferred — tested separately')

        else:
            append_result('skip', '-', '-', False, None, None, None, None)

    return results


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


def generate_html(all_results):
    """Generate the full HTML report."""
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
table.main { border-collapse: collapse; width: 100%; margin: 10px 0; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
table.main th { background: #f5f5f5; padding: 8px 12px; text-align: left; font-size: 13px; border-bottom: 2px solid #ddd; }
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
<p>Comparison of processbehavior library output against Tom Bishop's VAS Minitab analyses (SDS 1-3).</p>
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

    html += f"""
<div class="summary">
<strong>Overall:</strong>
<span class="pass-count">{total_pass} passed</span> /
<span class="fail-count">{total_fail} failed</span> /
<span class="skip-count">{total_skip} skipped</span>
out of {total_checks + total_skip} total checks
</div>
"""

    # Group by SDS
    for sds_num in [1, 2, 3]:
        sds_results = [r for r in all_results if r['sds'] == sds_num]
        if not sds_results:
            continue

        sds_pass = sum(1 for r in sds_results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is True)
        sds_fail = sum(1 for r in sds_results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is False)
        sds_skip = sum(1 for r in sds_results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is None)

        html += f"""
<h2>SDS {sds_num} — Analytic Design State {sds_num}</h2>
<div class="summary">
<span class="pass-count">{sds_pass} passed</span> /
<span class="fail-count">{sds_fail} failed</span> /
<span class="skip-count">{sds_skip} skipped</span>
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
        for r in sds_results:
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
                    mc = match_class(m)
                    mark = 'PASS' if m is True else ('FAIL' if m is False else '-')
                    if stats_html:
                        stats_html += f'<br>'
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

    from datetime import datetime
    html += f'<p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
    html += '</body></html>'
    return html


def main():
    print('Loading validation data...')
    df = pd.read_csv(VALIDATION_CSV)
    pb = ProcessBehavior(df)

    all_results = []

    for sds_num, config in SDS_CONFIGS.items():
        print(f'\nProcessing SDS {sds_num}...')
        response = getattr(pb.cols, config['response_attr'])
        study = pb.formulate(
            response=response,
            factors=[pb.cols.FACTOR_1, pb.cols.FACTOR_2],
            time=pb.cols.PRODUCTION_TIME
        )
        print(f'  Analytical SDS: {study.analytical_design_state}')

        json_path = FIXTURES_DIR / config['json']
        with open(json_path) as f:
            json_data = json.load(f)

        results = run_sds_validation(sds_num, pb, study, json_data)
        all_results.extend(results)

        # Print summary
        passes = sum(1 for r in results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is True)
        fails = sum(1 for r in results for k in ('match_cl', 'match_lpl', 'match_upl') if r.get(k) is False)
        print(f'  Results: {passes} passed, {fails} failed')

    print(f'\nGenerating HTML report...')
    html = generate_html(all_results)
    OUTPUT_HTML.write_text(html)
    print(f'Report written to: {OUTPUT_HTML}')


if __name__ == '__main__':
    main()
