"""
Validate VAS residuals (R1–R5) against Tom Bishop's Minitab reference output.

Each test loads a cn-tab comparison Excel file containing both input data and
Tom's known-good residuals, runs our pipeline fresh, and asserts our residuals
match within floating-point tolerance.

This is the only way to catch formula-level bugs — external ground truth,
not self-referential identity checks.
"""

import os

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessBehavior

pytestmark = pytest.mark.integration

VALIDATION_DIR = "validation"
SHEET_NAME = "VAS-ANALYSIS-DATABASE.mwx"

RESIDUAL_COLS = ["R1", "R2", "R3", "R4", "R5"]
REF_RESIDUAL_COLS = [
    "R1 RESIDUALS",
    "R2 RESIDUALS",
    "R3 RESIDUALS",
    "R4 RESIDUALS",
    "R5 RESIDUALS",
]


def _ref_path(sds_num: int) -> str:
    return f"{VALIDATION_DIR}/cn-tab comparison sds {sds_num} (3-5-26).xlsx"


def _load_reference(sds_num: int) -> pd.DataFrame:
    """Load the cn-tab comparison file for a given SDS type."""
    return pd.read_excel(_ref_path(sds_num), sheet_name=SHEET_NAME)


@pytest.mark.parametrize("sds_num", [1, 2, 3, 6])
def test_residuals_match_bishop_reference(sds_num: int) -> None:
    if not os.path.exists(_ref_path(sds_num)):
        pytest.skip(f"Reference file not found: {_ref_path(sds_num)}")
    ref = _load_reference(sds_num)

    # --- Run our pipeline fresh ---
    input_df = ref[["time", "factor 1", "factor 2", "y"]].copy()
    study = ProcessBehavior(input_df).formulate(
        response="y", factors=["factor 1", "factor 2"], time="time"
    )
    ds = study.dataset

    # --- Merge on y to align rows regardless of sort order ---
    merged = pd.merge(
        ds[["y"] + RESIDUAL_COLS],
        ref[["y"] + REF_RESIDUAL_COLS],
        on="y",
        how="inner",
    )
    assert len(merged) == len(ref), (
        f"SDS {sds_num}: row count mismatch after merge "
        f"(got {len(merged)}, expected {len(ref)})"
    )

    # --- Assert each residual matches ---
    for our_col, ref_col in zip(RESIDUAL_COLS, REF_RESIDUAL_COLS):
        ours = merged[our_col]
        toms = merged[ref_col]

        # Tom's MA2 files have NaN for the first time period (no prior period
        # for moving average). Our code outputs 0.0 there. Drop NaN rows from
        # comparison — the NaN semantics are a display choice, not a formula
        # disagreement.
        mask = toms.notna()

        np.testing.assert_allclose(
            ours[mask].to_numpy(),
            toms[mask].to_numpy(),
            atol=1e-8,
            err_msg=f"SDS {sds_num} {our_col} mismatch vs Bishop reference",
        )
