from __future__ import annotations

import pandas as pd


def imr(
    df: pd.DataFrame, response: str, by: list[str] | None = None, time_col: str | None = None
) -> dict:
    raise NotImplementedError  # implement I & MR with rational subgroup ordering
