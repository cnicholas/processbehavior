Validation Data for processbehavior Package
============================================

Generated: 2024-12-14
Package Version: 0.1.0 (pre-release)

Purpose
-------
These files provide validation datasets and analysis outputs for each
Sampling Design State (SDS 0-3) to verify correct behavior of the
processbehavior package.

Generation Method
-----------------
All data generated using processbehavior.datasets.synthetic.make_sds():

    from processbehavior import ProcessBehavior
    from processbehavior.datasets.synthetic import make_sds

    # Generate data
    df = make_sds(sds=N, seed=42, ...)

    # Formulate and execute analysis
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=[...], time='time')
    result = study.execute()

    # Export with all options enabled
    result.to_excel('sdsN_analysis.xlsx',
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
SDS 0: Individual Observations (No factors, no time)
  - sds0_data.csv: 20 observations, response only
  - sds0_analysis.xlsx: Imr chart analysis

SDS 1: Complete Design (All factor x time cells have n >= 2)
  - sds1_data.csv: K=2 factors, T=4 time periods, all cells replicated
  - sds1_analysis.xlsx: Xbar chart analysis with effects/interactions

SDS 2: Semi-Complete Design (All factor x time cells have n = 1)
  - sds2_data.csv: K=3 factors, T=5 time periods, no replication
  - sds2_analysis.xlsx: Xbar chart analysis (uses moving average variance)

SDS 3: Semi-Complete Design (Mixed: some cells n=1, some n >= 2)
  - sds3_data.csv: K=2 factors, T=5 time periods, partial replication
  - sds3_analysis.xlsx: Xbar chart analysis

SDS Detection Logic (Wheeler/Bishop Methodology)
------------------------------------------------
SDS classification is based on N_kt (observations per factor x time cell):

  SDS 0: No factors specified (individual observations)
  SDS 1: min(N_kt) >= 2 (Complete - all cells replicated)
  SDS 2: min(N_kt) = 1 AND max(N_kt) = 1 (Semi-Complete - no replication)
  SDS 3: min(N_kt) = 1 AND max(N_kt) >= 2 (Semi-Complete - partial replication)

Note: SDS 4-6 require a sampling plan to detect missing cells and are
not included in this validation set. See GitHub Issue #60 for details.

Verification
------------
To regenerate these files:

    python -c "
    from processbehavior import ProcessBehavior
    from processbehavior.datasets.synthetic import make_sds

    for sds_num in [0, 1, 2, 3]:
        df = make_sds(sds_num, seed=42)
        df.to_csv(f'validation/sds{sds_num}_data.csv', index=False)

        pb = ProcessBehavior(df)
        if sds_num == 0:
            study = pb.formulate(response='y')
        else:
            study = pb.formulate(response='y', factors=['factor 1'], time='time')

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
    "
