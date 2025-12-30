Validation Data for processbehavior Package
============================================

Generated: 2025-12-28
Package Version: 0.1.0 (pre-release)

Purpose
-------
These files provide validation datasets and analysis outputs for each
Sampling Design State (SDS 1-6) to verify correct behavior of the
processbehavior package.

Generation Method
-----------------
All data generated using processbehavior.datasets.synthetic.make_sds():

    from processbehavior import ProcessBehavior
    from processbehavior.datasets.synthetic import make_sds

    # Define expected factor levels based on make_sds defaults
    PLANS = {
        1: {'factors': {'factor 1': ['F1_1','F1_2','F1_3'], 'factor 2': ['F2_1','F2_2']}, 'T': 8, 'N': 5},
        2: {'factors': {'factor 1': ['F1_1','F1_2','F1_3'], 'factor 2': ['F2_1','F2_2']}, 'T': 10, 'N': 1},
        3: {'factors': {'factor 1': ['F1_1','F1_2','F1_3'], 'factor 2': ['F2_1','F2_2']}, 'T': 8, 'N': 3},
        5: {'factors': {
            'factor 1': ['Line1', 'Line2'],
            'factor 2': ['Line1_Head1', 'Line1_Head2', 'Line1_Head3',
                         'Line2_Head1', 'Line2_Head2', 'Line2_Head3']
        }, 'T': 8},
        6: {'factors': {'factor 1': ['Machine1','Machine2','Machine3'], 'factor 2': ['F2_1','F2_2']}, 'T': 80},
    }

    for sds_num in [1, 2, 3, 4, 5, 6]:
        # Generate data with specific T for SDS 4
        if sds_num == 4:
            df = make_sds(sds_num, T=40, seed=42)
        else:
            df = make_sds(sds_num, seed=42)

        # Drop cell_type column if present (SDS 3)
        if 'cell_type' in df.columns:
            df = df.drop(columns=['cell_type'])

        df.to_csv(f'validation/sds{sds_num}_data.csv', index=False)

        # Formulate study with plan (SDS 4 is single stream, no plan)
        pb = ProcessBehavior(df)
        if sds_num == 4:
            study = pb.formulate(response='y', time='time')
        else:
            study = pb.formulate(response='y', time='time', plan=PLANS[sds_num])

        result = study.execute()
        result.to_excel(f'validation/sds{sds_num}_analysis.xlsx',
            include_summary=True,
            include_charts=True,
            include_residuals=True,
            include_effects=True,
            include_interactions=True,
            include_full_dataset=True,
            include_chart_images=True,
            export_html=False
        )

Files
-----
SDS 1: Full Replication (All factor x time cells have n >= 2)
  - sds1_data.csv: K1=3, K2=2, T=8, n_min=2, n_max=5
  - sds1_analysis.xlsx: Xbar-S chart analysis with full VAS residuals
  - Plan N=5 (n_max): max expected observations per cell

SDS 2: No Replication (All factor x time cells have n = 1)
  - sds2_data.csv: K1=3, K2=2, T=10
  - sds2_analysis.xlsx: Xbar chart analysis (uses moving average variance)
  - Plan N=1: unreplicated design by definition

SDS 3: Partial Replication (Mix: some cells n=1, some n >= 2)
  - sds3_data.csv: K1=3, K2=2, T=8, p_replicated=0.5, n_when_replicated=3
  - sds3_analysis.xlsx: Xbar-S chart analysis with hybrid R2
  - Plan N=3 (n_when_replicated): expected when replicated

SDS 4: Single Condition Over Time (No factors)
  - sds4_data.csv: T=40 (single stream time series)
  - sds4_analysis.xlsx: IMR chart analysis
  - No plan: single stream, no factorial structure

SDS 5: Nested Design (Factor 2 nested within Factor 1)
  - sds5_data.csv: L=2 lines, H_per_L=3 heads, T=8, p_active=0.8
  - sds5_analysis.xlsx: Xbar-S chart analysis with nested structure
  - Plan: nested factor levels (Line1_Head1, etc.)

SDS 6: Incomplete Grid (Unstructured/irregular sampling)
  - sds6_data.csv: K1=3, K2=2, T=80, p_sampled=0.7
  - sds6_analysis.xlsx: Xbar chart analysis with incomplete coverage
  - Plan: full factorial expected, actual coverage incomplete

SDS Detection Logic (Wheeler/Bishop Methodology)
------------------------------------------------
SDS classification is based on:
- N_kt: observations per factor x time cell
- Grid coverage: proportion of expected cells that are observed
- Nesting: whether factor 2 is nested within factor 1

  SDS 1: min(N_kt) >= 2, complete grid (Full replication)
  SDS 2: N_kt = 1 for all cells, complete grid (No replication)
  SDS 3: 1 <= min(N_kt) < max(N_kt), complete grid (Partial replication)
  SDS 4: Single factor level (K=1), time series (Single condition)
  SDS 5: Factor 2 nested in Factor 1, incomplete grid (Nested design)
  SDS 6: < 75% grid coverage (Incomplete/unstructured)

Note: SDS 0 was consolidated into SDS 4. Response-only data (no factors,
no time) is now treated as SDS 4 with implicit time ordering via obs_id.

Plan Parameter Structure
------------------------
The plan parameter enables accurate SDS detection for SDS 5/6 by specifying
EXPECTED factor levels (vs just observed):

    plan = {
        'factors': {
            'factor 1': [...expected levels...],
            'factor 2': [...expected levels...]
        },
        'T': expected_time_periods,
        'N': expected_observations_per_cell  # optional
    }

This allows detection of:
- Missing factor combinations (incomplete grid -> SDS 5/6)
- Nested structures (factor 2 levels unique to factor 1)
- Replication patterns (N > 1 vs N = 1)

Verification
------------
To regenerate these files, run the generation script shown above with seed=42.
