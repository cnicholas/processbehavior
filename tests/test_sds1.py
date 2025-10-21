# test_analysis_dataset_sds1.py
import numpy as np
import pandas as pd
import pytest
import logging

import analysis_dataset as ad  # your module

logger = logging.getLogger(__name__)

# ========= Helpers =========

# Map of expected (source) → actual (dest) column names (RCRs excluded from direct compare)
NAME_MAP = {
    'YBAR(k,t)': 'Ybar_kt',
    'YBAR(.t)':  'Ybar_t',
    'YBAR(k.)':  'Ybar_k',
    'YBAR':      'Ybar',
    'R1':        'R1',
    'R2':        'R2',
    'R3':        'R3',
    'R4':        'R4',
    'R5':        'R5',
}

def _harmonize_dtypes(df_left: pd.DataFrame, df_right: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Cast right-hand columns to left-hand dtypes where possible so indexes/keys align."""
    left = df_left.copy()
    right = df_right.copy()
    for c in cols:
        if c in left.columns and c in right.columns:
            try:
                right[c] = right[c].astype(left[c].dtype)
            except Exception:
                # last resort: cast both to string for key alignment
                left[c]  = left[c].astype(str)
                right[c] = right[c].astype(str)
    return left, right

def add_replicate_index(df: pd.DataFrame, k_col: str, t_col: str, order_cols=None) -> pd.DataFrame:
    """Create a stable per-(k,t) replicate index so rows align reliably across frames."""
    out = df.copy()
    if order_cols:
        out = out.sort_values([k_col, t_col] + list(order_cols)).copy()
    out['__rep'] = out.groupby([k_col, t_col]).cumcount()
    return out

def compare_frames_by_map(
    df_expected: pd.DataFrame,
    df_actual: pd.DataFrame,
    name_map: dict,
    k_col: str,
    t_col: str,
    order_cols=None,
    rtol=1e-10,
    atol=1e-10,
) -> pd.DataFrame:
    """Key-align by (k,t,rep) and return a summary of mismatches for the mapped columns."""
    left  = add_replicate_index(df_expected, k_col, t_col, order_cols)
    right = add_replicate_index(df_actual,   k_col, t_col, order_cols)

    # Keep just needed columns
    left_sel  = left[[k_col, t_col, '__rep'] + list(name_map.keys())].copy()
    right_sel = right[[k_col, t_col, '__rep'] + list(name_map.values())].copy()

    # Rename actual to expected names to line them up
    right_sel = right_sel.rename(columns={v: k for k, v in name_map.items()})

    # Harmonize dtypes on keys
    (left_sel, right_sel) = _harmonize_dtypes(left_sel, right_sel, [k_col, t_col, '__rep'])

    merged = pd.merge(
        left_sel, right_sel,
        on=[k_col, t_col, '__rep'],
        how='outer', suffixes=('_exp','_act'), indicator=True, validate='one_to_one'
    )

    only_exp = merged[merged['_merge'] == 'left_only']
    only_act = merged[merged['_merge'] == 'right_only']
    assert only_exp.empty and only_act.empty, (
        f"Key mismatch after alignment: only_expected={len(only_exp)}, only_actual={len(only_act)}"
    )

    both = merged[merged['_merge'] == 'both'].copy()
    rows = []
    for exp_col in name_map.keys():
        a = pd.to_numeric(both[f"{exp_col}_exp"], errors='coerce')
        b = pd.to_numeric(both[f"{exp_col}_act"], errors='coerce')
        abs_diff = (a - b).abs()
        tol = atol + rtol * a.abs()
        mism = abs_diff > tol
        rows.append({
            "column_expected": exp_col,
            "column_actual": name_map[exp_col],
            "matched_rows": int(len(both)),
            "num_mismatch": int(mism.sum()),
            "max_abs_diff": float(abs_diff.max() if len(abs_diff) else np.nan),
        })
    return pd.DataFrame(rows).sort_values(['num_mismatch','max_abs_diff'], ascending=[False, False])

# ========= Fixture =========

@pytest.fixture
def df_SDS1():
    # File-based dataset for testing SDS1
    path = "processbehavior/datasets/data/SDS_1_ANALYSIS_RESULTS.csv"
    return pd.read_csv(path)

# ========= The Test =========

def test_analysis_dataset_sds1(df_SDS1: pd.DataFrame):
    """
    SDS1 end-to-end (order-agnostic):
      - SDS detection == 1
      - All mapped columns (means + R1–R5) match via key alignment
      - RCR identities reconstruct Y (dest) (row-wise within df_calc)
      - RCRs recomputed from source (canonical) match dest RCRs via key alignment
      - PDC×PT interaction matches on unique (k,t,time), order-agnostic
    """
    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['FACTOR 1', 'FACTOR 2'],  # k, t
        'time_var': 'TIME',
        'response_var': 'Y',
        'rsg_var_name': 'rsg',
        'round_to': 2,
    }

    # Run analysis
    analysis = ad.Analysis(df_SDS1, spec)
    ds = analysis.ads
    df_calc = ds.analysis_dataset

    # SDS check
    assert ds.sampling_design_state == 1, f"Expected SDS1, got {ds.sampling_design_state}"

    # --- Keyed comparison for mapped columns (excludes RCRs)
    k_col = spec['rsg_vars'][0]
    t_col = spec['rsg_vars'][1]
    order_cols = [spec['time_var']] if spec['time_var'] in df_SDS1.columns else None

    summary = compare_frames_by_map(
        df_expected=df_SDS1,
        df_actual=df_calc,
        name_map=NAME_MAP,
        k_col=k_col,
        t_col=t_col,
        order_cols=order_cols,
        rtol=1e-10,
        atol=1e-10,
    )
    bad = summary[summary['num_mismatch'] > 0]
    assert bad.empty, f"Mismatches found in mapped columns:\n{bad.to_string(index=False)}"

    # --- RCR identities must reconstruct Y (dest) ---
    y = spec['response_var']
    eps = 1e-10
    need_cols = {'Ybar','Ybar_k','Ybar_t','Ybar_kt','R1','R2','R3','R4','R5','RCR1','RCR2','RCR3','RCR4','RCR5'}
    missing = need_cols - set(df_calc.columns)
    assert not missing, f"Calculated dataset missing columns: {sorted(missing)}"

    # These are internal identities within df_calc and can be checked positionally
    assert np.allclose(df_calc['Ybar'] + df_calc['R1'], df_calc[y], atol=eps)
    assert np.allclose(df_calc['Ybar_kt'] + df_calc['R2'], df_calc[y], atol=eps)
    assert np.allclose((df_calc['Ybar_k'] + df_calc['Ybar_t'] - df_calc['Ybar']) + df_calc['R3'], df_calc[y], atol=eps)
    assert np.allclose((df_calc['Ybar'] + df_calc['Ybar_kt'] - df_calc['Ybar_t']) + df_calc['R4'], df_calc[y], atol=eps)
    assert np.allclose((df_calc['Ybar'] + df_calc['Ybar_kt'] - df_calc['Ybar_k']) + df_calc['R5'], df_calc[y], atol=eps)

    # --- RCRs recomputed from source (canonical) vs dest (key-aligned) ---
    required_src = {'YBAR','YBAR(k,t)','YBAR(k.)','YBAR(.t)','R1','R2','R3','R4','R5'}
    if required_src.issubset(df_SDS1.columns):
        k, t, time = spec['rsg_vars'][0], spec['rsg_vars'][1], spec['time_var']

        # harmonize dtypes before indexing/merge
        (df_SDS1_h, df_calc_h) = _harmonize_dtypes(df_SDS1, df_calc, [k, t, time])

        src = add_replicate_index(df_SDS1_h, k, t, order_cols=[time])
        dst = add_replicate_index(df_calc_h,  k, t, order_cols=[time])

        src_RCR = pd.DataFrame({
            'RCR1': src['YBAR'] + src['R1'],
            'RCR2': src['YBAR(k,t)'] + src['R2'],
            'RCR3': (src['YBAR(k.)'] + src['YBAR(.t)'] - src['YBAR']) + src['R3'],
            'RCR4': (src['YBAR'] + src['YBAR(k,t)'] - src['YBAR(.t)']) + src['R4'],
            'RCR5': (src['YBAR'] + src['YBAR(k,t)'] - src['YBAR(k.)']) + src['R5'],
        })
        src_keyed = pd.concat([src[[k, t, '__rep']].reset_index(drop=True),
                               src_RCR.reset_index(drop=True)], axis=1)

        dst_keyed = dst[[k, t, '__rep', 'RCR1','RCR2','RCR3','RCR4','RCR5']].copy()

        merged = pd.merge(
            src_keyed, dst_keyed,
            on=[k, t, '__rep'], how='outer', suffixes=('_exp','_act'),
            validate='one_to_one', indicator=True
        )

        only_exp = merged[merged['_merge'] == 'left_only']
        only_act = merged[merged['_merge'] == 'right_only']
        assert only_exp.empty and only_act.empty, (
            f"RCR key mismatch: only_expected={len(only_exp)}, only_actual={len(only_act)}"
        )

        for c in ['RCR1','RCR2','RCR3','RCR4','RCR5']:
            pd.testing.assert_series_equal(
                pd.to_numeric(merged[f'{c}_exp'], errors='coerce'),
                pd.to_numeric(merged[f'{c}_act'], errors='coerce'),
                rtol=1e-10, atol=1e-10, check_names=False
            )

    # --- PDC×PT interaction effects (unique per (k,t,time)) ---
    # Order-agnostic compare with explicit key alignment; tolerant to 800 vs 400 rows.
    if {'PDCxPT INTERACTION EFFECTS'}.issubset(df_SDS1.columns) and 'pdc_by_pt' in df_calc.columns:
        k, t, time = spec['rsg_vars'][0], spec['rsg_vars'][1], spec['time_var']

        (df_SDS1_h, df_calc_h) = _harmonize_dtypes(df_SDS1, df_calc, [k, t, time])

        calc = (
            df_calc_h
            .groupby([k, t, time], sort=False)['pdc_by_pt']
            .mean()  # constant per cell by construction; mean is idempotent
        )

        src_int = (
            df_SDS1_h
            .groupby([k, t, time], sort=False)['PDCxPT INTERACTION EFFECTS']
            .first()
            .astype(float)
        )

        common = src_int.index.intersection(calc.index)
        assert len(common) > 0, "No common (k,t,time) keys found for PDC×PT comparison."

        src_al  = src_int.loc[common].sort_index()
        calc_al = calc.loc[common].sort_index()
        # test_analysis_dataset_sds1.py
import numpy as np
import pandas as pd
import pytest
import logging

import analysis_dataset as ad  # your module

logger = logging.getLogger(__name__)

# ========= Helpers =========

# Map of expected (source) → actual (dest) column names (RCRs excluded from direct compare)
NAME_MAP = {
    'YBAR(k,t)': 'Ybar_kt',
    'YBAR(.t)':  'Ybar_t',
    'YBAR(k.)':  'Ybar_k',
    'YBAR':      'Ybar',
    'R1':        'R1',
    'R2':        'R2',
    'R3':        'R3',
    'R4':        'R4',
    'R5':        'R5',
}

def _harmonize_dtypes(df_left: pd.DataFrame, df_right: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Cast right-hand columns to left-hand dtypes where possible so indexes/keys align."""
    left = df_left.copy()
    right = df_right.copy()
    for c in cols:
        if c in left.columns and c in right.columns:
            try:
                right[c] = right[c].astype(left[c].dtype)
            except Exception:
                # last resort: cast both to string for key alignment
                left[c]  = left[c].astype(str)
                right[c] = right[c].astype(str)
    return left, right

def add_replicate_index(df: pd.DataFrame, k_col: str, t_col: str, order_cols=None) -> pd.DataFrame:
    """Create a stable per-(k,t) replicate index so rows align reliably across frames."""
    out = df.copy()
    if order_cols:
        out = out.sort_values([k_col, t_col] + list(order_cols)).copy()
    out['__rep'] = out.groupby([k_col, t_col]).cumcount()
    return out

def compare_frames_by_map(
    df_expected: pd.DataFrame,
    df_actual: pd.DataFrame,
    name_map: dict,
    k_col: str,
    t_col: str,
    order_cols=None,
    rtol=1e-10,
    atol=1e-10,
) -> pd.DataFrame:
    """Key-align by (k,t,rep) and return a summary of mismatches for the mapped columns."""
    left  = add_replicate_index(df_expected, k_col, t_col, order_cols)
    right = add_replicate_index(df_actual,   k_col, t_col, order_cols)

    # Keep just needed columns
    left_sel  = left[[k_col, t_col, '__rep'] + list(name_map.keys())].copy()
    right_sel = right[[k_col, t_col, '__rep'] + list(name_map.values())].copy()

    # Rename actual to expected names to line them up
    right_sel = right_sel.rename(columns={v: k for k, v in name_map.items()})

    # Harmonize dtypes on keys
    (left_sel, right_sel) = _harmonize_dtypes(left_sel, right_sel, [k_col, t_col, '__rep'])

    merged = pd.merge(
        left_sel, right_sel,
        on=[k_col, t_col, '__rep'],
        how='outer', suffixes=('_exp','_act'), indicator=True, validate='one_to_one'
    )

    only_exp = merged[merged['_merge'] == 'left_only']
    only_act = merged[merged['_merge'] == 'right_only']
    assert only_exp.empty and only_act.empty, (
        f"Key mismatch after alignment: only_expected={len(only_exp)}, only_actual={len(only_act)}"
    )

    both = merged[merged['_merge'] == 'both'].copy()
    rows = []
    for exp_col in name_map.keys():
        a = pd.to_numeric(both[f"{exp_col}_exp"], errors='coerce')
        b = pd.to_numeric(both[f"{exp_col}_act"], errors='coerce')
        abs_diff = (a - b).abs()
        tol = atol + rtol * a.abs()
        mism = abs_diff > tol
        rows.append({
            "column_expected": exp_col,
            "column_actual": name_map[exp_col],
            "matched_rows": int(len(both)),
            "num_mismatch": int(mism.sum()),
            "max_abs_diff": float(abs_diff.max() if len(abs_diff) else np.nan),
        })
    return pd.DataFrame(rows).sort_values(['num_mismatch','max_abs_diff'], ascending=[False, False])

# ========= Fixture =========

@pytest.fixture
def df_SDS1():
    # File-based dataset for testing SDS1
    path = "processbehavior/datasets/data/SDS_1_ANALYSIS_RESULTS.csv"
    return pd.read_csv(path)

# ========= The Test =========

def test_analysis_dataset_sds1(df_SDS1: pd.DataFrame):
    """
    SDS1 end-to-end (order-agnostic):
      - SDS detection == 1
      - All mapped columns (means + R1–R5) match via key alignment
      - RCR identities reconstruct Y (dest) (row-wise within df_calc)
      - RCRs recomputed from source (canonical) match dest RCRs via key alignment
      - PDC×PT interaction matches on unique (k,t,time), order-agnostic
    """
    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['FACTOR 1', 'FACTOR 2'],  # k, t
        'time_var': 'TIME',
        'response_var': 'Y',
        'rsg_var_name': 'rsg',
        'round_to': 2,
    }

    # Run analysis
    analysis = ad.Analysis(df_SDS1, spec)
    ds = analysis.ads
    df_calc = ds.analysis_dataset

    # SDS check
    assert ds.sampling_design_state == 1, f"Expected SDS1, got {ds.sampling_design_state}"

    # --- Keyed comparison for mapped columns (excludes RCRs)
    k_col = spec['rsg_vars'][0]
    t_col = spec['rsg_vars'][1]
    order_cols = [spec['time_var']] if spec['time_var'] in df_SDS1.columns else None

    summary = compare_frames_by_map(
        df_expected=df_SDS1,
        df_actual=df_calc,
        name_map=NAME_MAP,
        k_col=k_col,
        t_col=t_col,
        order_cols=order_cols,
        rtol=1e-10,
        atol=1e-10,
    )
    bad = summary[summary['num_mismatch'] > 0]
    assert bad.empty, f"Mismatches found in mapped columns:\n{bad.to_string(index=False)}"

    # --- RCR identities must reconstruct Y (dest) ---
    y = spec['response_var']
    eps = 1e-10
    need_cols = {'Ybar','Ybar_k','Ybar_t','Ybar_kt','R1','R2','R3','R4','R5','RCR1','RCR2','RCR3','RCR4','RCR5'}
    missing = need_cols - set(df_calc.columns)
    assert not missing, f"Calculated dataset missing columns: {sorted(missing)}"

    # These are internal identities within df_calc and can be checked positionally
    assert np.allclose(df_calc['Ybar'] + df_calc['R1'], df_calc[y], atol=eps)
    assert np.allclose(df_calc['Ybar_kt'] + df_calc['R2'], df_calc[y], atol=eps)
    assert np.allclose((df_calc['Ybar_k'] + df_calc['Ybar_t'] - df_calc['Ybar']) + df_calc['R3'], df_calc[y], atol=eps)
    assert np.allclose((df_calc['Ybar'] + df_calc['Ybar_kt'] - df_calc['Ybar_t']) + df_calc['R4'], df_calc[y], atol=eps)
    assert np.allclose((df_calc['Ybar'] + df_calc['Ybar_kt'] - df_calc['Ybar_k']) + df_calc['R5'], df_calc[y], atol=eps)

    # --- RCRs recomputed from source (canonical) vs dest (key-aligned) ---
    required_src = {'YBAR','YBAR(k,t)','YBAR(k.)','YBAR(.t)','R1','R2','R3','R4','R5'}
    if required_src.issubset(df_SDS1.columns):
        k, t, time = spec['rsg_vars'][0], spec['rsg_vars'][1], spec['time_var']

        # harmonize dtypes before indexing/merge
        (df_SDS1_h, df_calc_h) = _harmonize_dtypes(df_SDS1, df_calc, [k, t, time])

        src = add_replicate_index(df_SDS1_h, k, t, order_cols=[time])
        dst = add_replicate_index(df_calc_h,  k, t, order_cols=[time])

        src_RCR = pd.DataFrame({
            'RCR1': src['YBAR'] + src['R1'],
            'RCR2': src['YBAR(k,t)'] + src['R2'],
            'RCR3': (src['YBAR(k.)'] + src['YBAR(.t)'] - src['YBAR']) + src['R3'],
            'RCR4': (src['YBAR'] + src['YBAR(k,t)'] - src['YBAR(.t)']) + src['R4'],
            'RCR5': (src['YBAR'] + src['YBAR(k,t)'] - src['YBAR(k.)']) + src['R5'],
        })
        src_keyed = pd.concat([src[[k, t, '__rep']].reset_index(drop=True),
                               src_RCR.reset_index(drop=True)], axis=1)

        dst_keyed = dst[[k, t, '__rep', 'RCR1','RCR2','RCR3','RCR4','RCR5']].copy()

        merged = pd.merge(
            src_keyed, dst_keyed,
            on=[k, t, '__rep'], how='outer', suffixes=('_exp','_act'),
            validate='one_to_one', indicator=True
        )

        only_exp = merged[merged['_merge'] == 'left_only']
        only_act = merged[merged['_merge'] == 'right_only']
        assert only_exp.empty and only_act.empty, (
            f"RCR key mismatch: only_expected={len(only_exp)}, only_actual={len(only_act)}"
        )

        for c in ['RCR1','RCR2','RCR3','RCR4','RCR5']:
            pd.testing.assert_series_equal(
                pd.to_numeric(merged[f'{c}_exp'], errors='coerce'),
                pd.to_numeric(merged[f'{c}_act'], errors='coerce'),
                rtol=1e-10, atol=1e-10, check_names=False
            )

    # --- PDC×PT interaction effects (unique per (k,t,time)) ---
    # Order-agnostic compare with explicit key alignment; tolerant to 800 vs 400 rows.
    if {'PDCxPT INTERACTION EFFECTS'}.issubset(df_SDS1.columns) and 'pdc_by_pt' in df_calc.columns:
        k, t, time = spec['rsg_vars'][0], spec['rsg_vars'][1], spec['time_var']

        (df_SDS1_h, df_calc_h) = _harmonize_dtypes(df_SDS1, df_calc, [k, t, time])

        calc = (
            df_calc_h
            .groupby([k, t, time], sort=False)['pdc_by_pt']
            .mean()  # constant per cell by construction; mean is idempotent
        )

        src_int = (
            df_SDS1_h
            .groupby([k, t, time], sort=False)['PDCxPT INTERACTION EFFECTS']
            .first()
            .astype(float)
        )

        common = src_int.index.intersection(calc.index)
        assert len(common) > 0, "No common (k,t,time) keys found for PDC×PT comparison."

        src_al  = src_int.loc[common].sort_index()
        calc_al = calc.loc[common].sort_index()

        pd.testing.assert_series_equal(
            src_al.round(6),
            calc_al.round(6),
            rtol=1e-6, atol=1e-6, check_names=False
        )
        # test_analysis_dataset_sds1.py
import numpy as np
import pandas as pd
import pytest
import logging

import analysis_dataset as ad  # your module

logger = logging.getLogger(__name__)

# ========= Helpers =========

# Map of expected (source) → actual (dest) column names (RCRs excluded from direct compare)
NAME_MAP = {
    'YBAR(k,t)': 'Ybar_kt',
    'YBAR(.t)':  'Ybar_t',
    'YBAR(k.)':  'Ybar_k',
    'YBAR':      'Ybar',
    'R1':        'R1',
    'R2':        'R2',
    'R3':        'R3',
    'R4':        'R4',
    'R5':        'R5',
}

def _harmonize_dtypes(df_left: pd.DataFrame, df_right: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Cast right-hand columns to left-hand dtypes where possible so indexes/keys align."""
    left = df_left.copy()
    right = df_right.copy()
    for c in cols:
        if c in left.columns and c in right.columns:
            try:
                right[c] = right[c].astype(left[c].dtype)
            except Exception:
                # last resort: cast both to string for key alignment
                left[c]  = left[c].astype(str)
                right[c] = right[c].astype(str)
    return left, right

def add_replicate_index(df: pd.DataFrame, k_col: str, t_col: str, order_cols=None) -> pd.DataFrame:
    """Create a stable per-(k,t) replicate index so rows align reliably across frames."""
    out = df.copy()
    if order_cols:
        out = out.sort_values([k_col, t_col] + list(order_cols)).copy()
    out['__rep'] = out.groupby([k_col, t_col]).cumcount()
    return out

def compare_frames_by_map(
    df_expected: pd.DataFrame,
    df_actual: pd.DataFrame,
    name_map: dict,
    k_col: str,
    t_col: str,
    order_cols=None,
    rtol=1e-10,
    atol=1e-10,
) -> pd.DataFrame:
    """Key-align by (k,t,rep) and return a summary of mismatches for the mapped columns."""
    left  = add_replicate_index(df_expected, k_col, t_col, order_cols)
    right = add_replicate_index(df_actual,   k_col, t_col, order_cols)

    # Keep just needed columns
    left_sel  = left[[k_col, t_col, '__rep'] + list(name_map.keys())].copy()
    right_sel = right[[k_col, t_col, '__rep'] + list(name_map.values())].copy()

    # Rename actual to expected names to line them up
    right_sel = right_sel.rename(columns={v: k for k, v in name_map.items()})

    # Harmonize dtypes on keys
    (left_sel, right_sel) = _harmonize_dtypes(left_sel, right_sel, [k_col, t_col, '__rep'])

    merged = pd.merge(
        left_sel, right_sel,
        on=[k_col, t_col, '__rep'],
        how='outer', suffixes=('_exp','_act'), indicator=True, validate='one_to_one'
    )

    only_exp = merged[merged['_merge'] == 'left_only']
    only_act = merged[merged['_merge'] == 'right_only']
    assert only_exp.empty and only_act.empty, (
        f"Key mismatch after alignment: only_expected={len(only_exp)}, only_actual={len(only_act)}"
    )

    both = merged[merged['_merge'] == 'both'].copy()
    rows = []
    for exp_col in name_map.keys():
        a = pd.to_numeric(both[f"{exp_col}_exp"], errors='coerce')
        b = pd.to_numeric(both[f"{exp_col}_act"], errors='coerce')
        abs_diff = (a - b).abs()
        tol = atol + rtol * a.abs()
        mism = abs_diff > tol
        rows.append({
            "column_expected": exp_col,
            "column_actual": name_map[exp_col],
            "matched_rows": int(len(both)),
            "num_mismatch": int(mism.sum()),
            "max_abs_diff": float(abs_diff.max() if len(abs_diff) else np.nan),
        })
    return pd.DataFrame(rows).sort_values(['num_mismatch','max_abs_diff'], ascending=[False, False])

# ========= Fixture =========

@pytest.fixture
def df_SDS1():
    # File-based dataset for testing SDS1
    path = "processbehavior/datasets/data/SDS_1_ANALYSIS_RESULTS.csv"
    return pd.read_csv(path)

# ========= The Test =========

def test_analysis_dataset_sds1(df_SDS1: pd.DataFrame):
    """
    SDS1 end-to-end (order-agnostic):
      - SDS detection == 1
      - All mapped columns (means + R1–R5) match via key alignment
      - RCR identities reconstruct Y (dest) (row-wise within df_calc)
      - RCRs recomputed from source (canonical) match dest RCRs via key alignment
      - PDC×PT interaction matches on unique (k,t,time), order-agnostic
    """
    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['FACTOR 1', 'FACTOR 2'],  # k, t
        'time_var': 'TIME',
        'response_var': 'Y',
        'rsg_var_name': 'rsg',
        'round_to': 2,
    }

    # Run analysis
    analysis = ad.Analysis(df_SDS1, spec)
    ds = analysis.ads
    df_calc = ds.analysis_dataset

    # SDS check
    assert ds.sampling_design_state == 1, f"Expected SDS1, got {ds.sampling_design_state}"

    # --- Keyed comparison for mapped columns (excludes RCRs)
    k_col = spec['rsg_vars'][0]
    t_col = spec['rsg_vars'][1]
    order_cols = [spec['time_var']] if spec['time_var'] in df_SDS1.columns else None

    summary = compare_frames_by_map(
        df_expected=df_SDS1,
        df_actual=df_calc,
        name_map=NAME_MAP,
        k_col=k_col,
        t_col=t_col,
        order_cols=order_cols,
        rtol=1e-10,
        atol=1e-10,
    )
    bad = summary[summary['num_mismatch'] > 0]
    assert bad.empty, f"Mismatches found in mapped columns:\n{bad.to_string(index=False)}"

    # --- RCR identities must reconstruct Y (dest) ---
    y = spec['response_var']
    eps = 1e-10
    need_cols = {'Ybar','Ybar_k','Ybar_t','Ybar_kt','R1','R2','R3','R4','R5','RCR1','RCR2','RCR3','RCR4','RCR5'}
    missing = need_cols - set(df_calc.columns)
    assert not missing, f"Calculated dataset missing columns: {sorted(missing)}"

    # These are internal identities within df_calc and can be checked positionally
    assert np.allclose(df_calc['Ybar'] + df_calc['R1'], df_calc[y], atol=eps)
    assert np.allclose(df_calc['Ybar_kt'] + df_calc['R2'], df_calc[y], atol=eps)
    assert np.allclose((df_calc['Ybar_k'] + df_calc['Ybar_t'] - df_calc['Ybar']) + df_calc['R3'], df_calc[y], atol=eps)
    assert np.allclose((df_calc['Ybar'] + df_calc['Ybar_kt'] - df_calc['Ybar_t']) + df_calc['R4'], df_calc[y], atol=eps)
    assert np.allclose((df_calc['Ybar'] + df_calc['Ybar_kt'] - df_calc['Ybar_k']) + df_calc['R5'], df_calc[y], atol=eps)

    # --- RCRs recomputed from source (canonical) vs dest (key-aligned) ---
    required_src = {'YBAR','YBAR(k,t)','YBAR(k.)','YBAR(.t)','R1','R2','R3','R4','R5'}
    if required_src.issubset(df_SDS1.columns):
        k, t, time = spec['rsg_vars'][0], spec['rsg_vars'][1], spec['time_var']

        # harmonize dtypes before indexing/merge
        (df_SDS1_h, df_calc_h) = _harmonize_dtypes(df_SDS1, df_calc, [k, t, time])

        src = add_replicate_index(df_SDS1_h, k, t, order_cols=[time])
        dst = add_replicate_index(df_calc_h,  k, t, order_cols=[time])

        src_RCR = pd.DataFrame({
            'RCR1': src['YBAR'] + src['R1'],
            'RCR2': src['YBAR(k,t)'] + src['R2'],
            'RCR3': (src['YBAR(k.)'] + src['YBAR(.t)'] - src['YBAR']) + src['R3'],
            'RCR4': (src['YBAR'] + src['YBAR(k,t)'] - src['YBAR(.t)']) + src['R4'],
            'RCR5': (src['YBAR'] + src['YBAR(k,t)'] - src['YBAR(k.)']) + src['R5'],
        })
        src_keyed = pd.concat([src[[k, t, '__rep']].reset_index(drop=True),
                               src_RCR.reset_index(drop=True)], axis=1)

        dst_keyed = dst[[k, t, '__rep', 'RCR1','RCR2','RCR3','RCR4','RCR5']].copy()

        merged = pd.merge(
            src_keyed, dst_keyed,
            on=[k, t, '__rep'], how='outer', suffixes=('_exp','_act'),
            validate='one_to_one', indicator=True
        )

        only_exp = merged[merged['_merge'] == 'left_only']
        only_act = merged[merged['_merge'] == 'right_only']
        assert only_exp.empty and only_act.empty, (
            f"RCR key mismatch: only_expected={len(only_exp)}, only_actual={len(only_act)}"
        )

        for c in ['RCR1','RCR2','RCR3','RCR4','RCR5']:
            pd.testing.assert_series_equal(
                pd.to_numeric(merged[f'{c}_exp'], errors='coerce'),
                pd.to_numeric(merged[f'{c}_act'], errors='coerce'),
                rtol=1e-10, atol=1e-10, check_names=False
            )

    # --- PDC×PT interaction effects (unique per (k,t,time)) ---
    # Order-agnostic compare with explicit key alignment; tolerant to 800 vs 400 rows.
    if {'PDCxPT INTERACTION EFFECTS'}.issubset(df_SDS1.columns) and 'pdc_by_pt' in df_calc.columns:
        k, t, time = spec['rsg_vars'][0], spec['rsg_vars'][1], spec['time_var']

        (df_SDS1_h, df_calc_h) = _harmonize_dtypes(df_SDS1, df_calc, [k, t, time])

        calc = (
            df_calc_h
            .groupby([k, t, time], sort=False)['pdc_by_pt']
            .mean()  # constant per cell by construction; mean is idempotent
        )

        src_int = (
            df_SDS1_h
            .groupby([k, t, time], sort=False)['PDCxPT INTERACTION EFFECTS']
            .first()
            .astype(float)
        )

        common = src_int.index.intersection(calc.index)
        assert len(common) > 0, "No common (k,t,time) keys found for PDC×PT comparison."

        #src_al  = src_int.loc[common].sort_index()
        calc_al = calc.loc[common].sort_index()

        sanity = (
            df_calc_h
            .groupby([k, t, time], sort=False)
            .apply(lambda g: g['Ybar_kt'].iloc[0] - g['Ybar_k'].iloc[0]
                            - g['Ybar_t'].iloc[0] + g['Ybar'].iloc[0])
        )
        pd.testing.assert_series_equal(
            calc.round(6), sanity.round(6),
            rtol=1e-6, atol=1e-6, check_names=False
        )

def test_pdc_by_pt_consistency(df_SDS1):
    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['FACTOR 1', 'FACTOR 2'],  # k, t
        'time_var': 'TIME',
        'response_var': 'Y',
        'rsg_var_name': 'rsg',
        'round_to': 2,
    }
    analysis = ad.Analysis(df_SDS1, specification=spec)
    df = analysis.ads.analysis_dataset

    # Calculate expected PDC using formula
    sanity = (df.groupby(['FACTOR 1','FACTOR 2','TIME'])
                .apply(lambda g: g['Ybar_kt'].iloc[0] - g['Ybar_k'].iloc[0]
                                  - g['Ybar_t'].iloc[0] + g['Ybar'].iloc[0])
                .rename('pdc_by_pt'))

    # Get actual PDC from interactions (new architecture stores it there)
    # PDC is a Series with same index as df, so we can group it
    pdc_series = analysis.ads.interactions.get('pdc_by_pt')
    assert pdc_series is not None, "PDC should be calculated for SDS 1"

    # Add PDC to dataframe temporarily for grouping
    df_with_pdc = df.copy()
    df_with_pdc['pdc_by_pt'] = pdc_series
    calc = df_with_pdc.groupby(['FACTOR 1','FACTOR 2','TIME'])['pdc_by_pt'].mean()

    pd.testing.assert_series_equal(calc.round(6), sanity.round(6), check_names=False)