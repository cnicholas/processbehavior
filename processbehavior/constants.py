from __future__ import annotations

# Single source of truth for constants; expand as you implement real math.
# Placeholders provided so the package imports cleanly today.


def d2(n: int) -> float:
    # d2 for MR span 2 would be 1.128; here just guard
    if n < 2:
        raise ValueError('d2 undefined for n < 2')
    return 1.128 if n == 2 else 1.0  # TODO: replace with proper table/formula


def c4(n: int) -> float:
    # Placeholder c4; replace with real approximation or table
    if n < 2:
        raise ValueError('c4 undefined for n < 2')
    return 0.94
