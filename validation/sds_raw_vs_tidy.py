"""
Compare SDS detection on raw data vs tidy data.

The "two-state problem": SDS detected on raw data (correct per Wheeler/Bishop)
can differ from SDS detected after tidying (dropping NA response rows). This
script makes that drift visible for each PM SDS variable in TABVASTESTDATABASE.
"""

import pandas as pd

from processbehavior import DataPreparation, SDSRegistry
from processbehavior.formulation_spec import FormulationSpec

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
raw_df = pd.read_csv("validation/TABVASTESTDATABASE.csv", na_values=["*"])

pm_sds_vars = [f"PM SDS {i}" for i in range(1, 7)]
registry = SDSRegistry()
prep = DataPreparation()

# ---------------------------------------------------------------------------
# Compare raw vs tidy SDS for each PM variable
# ---------------------------------------------------------------------------
rows = []
for var in pm_sds_vars:
    spec = FormulationSpec(
        response_var=var,
        rsg_vars=("FACTOR 1", "FACTOR 2"),
        time_var="PRODUCTION TIME",
    )

    # Raw detection (before dropping NA response rows)
    raw_result = registry.detect_sds_from_structure(raw_df, spec, var)

    # Tidy (drops NA response rows, builds RSG, etc.)
    tidy_df = prep.prepare_dataset(raw_df, spec)

    # Tidy detection (after NA response rows removed)
    tidy_result = registry.detect_sds_from_structure(tidy_df, spec, var)

    drift = "No" if raw_result.sds == tidy_result.sds else "YES"

    rows.append({
        "Variable": var,
        "Raw SDS": raw_result.sds,
        "Raw Reason": raw_result.reason,
        "Tidy SDS": tidy_result.sds,
        "Tidy Reason": tidy_result.reason,
        "Drift?": drift,
    })

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))
