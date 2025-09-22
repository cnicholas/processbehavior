from __future__ import annotations

import pandas as pd


def load_demo() -> pd.DataFrame:
    """Tiny synthetic dataset for smoke tests & examples."""
    n = 50
    df = pd.DataFrame(
        {'t': range(n), 'line': ['A'] * n, 'y': [10 + (i % 5) * 0.1 for i in range(n)]}
    )
    return df
