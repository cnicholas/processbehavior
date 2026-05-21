"""
Unit tests for individual WECO signal detection functions.

Each detector is a pure function: (data, stats, value_col, [zones]) -> bool Series.
Tests use small deterministic datasets that trigger or avoid each rule.
"""

import pandas as pd

from processbehavior.signals.detectors import (
    detect_avoiding_center,
    detect_beyond_limits,
    detect_oscillation,
    detect_reduced_variation,
    detect_run,
    detect_zone_a_2_of_3,
    detect_zone_b_4_of_5,
)

# ============================================================================
# Helpers
# ============================================================================

CENTER = 50.0
SIGMA = 5.0
UPL = CENTER + 3 * SIGMA  # 65
LPL = CENTER - 3 * SIGMA  # 35


def _stats():
    return {'center': CENTER, 'upl': UPL, 'lpl': LPL}


def _zones():
    """Constant zone boundaries for Rules 2, 3, 7, 8."""
    return {
        'A_upper': (CENTER + 2 * SIGMA, CENTER + 3 * SIGMA),  # (60, 65)
        'A_lower': (CENTER - 3 * SIGMA, CENTER - 2 * SIGMA),  # (35, 40)
        'B_upper': (CENTER + SIGMA, CENTER + 2 * SIGMA),  # (55, 60)
        'B_lower': (CENTER - 2 * SIGMA, CENTER - SIGMA),  # (40, 45)
        'C_upper': (CENTER, CENTER + SIGMA),  # (50, 55)
        'C_lower': (CENTER - SIGMA, CENTER),  # (45, 50)
    }


def _df(values):
    return pd.DataFrame({'value': values})


# ============================================================================
# Rule 1: Beyond Limits
# ============================================================================


class TestRule1BeyondLimits:
    def test_violation_above(self):
        """Point above UPL should be flagged."""
        data = _df([50, 50, 50, 66, 50])  # 66 > 65
        result = detect_beyond_limits(data, _stats(), 'value')
        assert result[3]

    def test_violation_below(self):
        """Point below LPL should be flagged."""
        data = _df([50, 50, 34, 50, 50])  # 34 < 35
        result = detect_beyond_limits(data, _stats(), 'value')
        assert result[2]

    def test_clean_data(self):
        """Points within limits should not be flagged."""
        data = _df([50, 52, 48, 55, 45])
        result = detect_beyond_limits(data, _stats(), 'value')
        assert not result.any()

    def test_varying_limits(self):
        """Per-row limits should be respected."""
        data = _df([50, 60, 50])
        data['upl'] = [55, 65, 55]  # row 0: UPL=55, row 1: UPL=65
        data['lpl'] = [45, 35, 45]
        result = detect_beyond_limits(data, _stats(), 'value', limits_vary=True)
        assert not result[1]  # 60 < 65, OK


# ============================================================================
# Rule 2: 2 of 3 in Zone A
# ============================================================================


class TestRule2ZoneA:
    def test_trigger_upper(self):
        """2 of 3 consecutive points in upper Zone A should trigger."""
        # Zone A upper: (60, 65). Put 2 of 3 points there.
        data = _df([50, 61, 50, 62, 50])
        #              ^zone_a       ^zone_a
        # Window of 3 at index 3: [50, 62, 50] — only 1 in zone A
        # Actually: rolling(3) at idx 1: [50,61,50] — 1 in zone A
        # Need 2 of 3 consecutive in zone A
        data = _df([50, 61, 62, 50, 50])
        result = detect_zone_a_2_of_3(data, _stats(), 'value', _zones())
        # Window ending at idx 2: [50, 61, 62] — 2 in zone A upper
        assert result[2]

    def test_clean_data(self):
        """Points near center should not trigger Rule 2."""
        data = _df([50, 51, 49, 52, 48, 50, 51])
        result = detect_zone_a_2_of_3(data, _stats(), 'value', _zones())
        assert not result.any()


# ============================================================================
# Rule 3: 4 of 5 in Zone B or beyond
# ============================================================================


class TestRule3ZoneB:
    def test_trigger_upper(self):
        """4 of 5 consecutive points beyond 1-sigma (upper) should trigger."""
        # Zone B upper starts at 55. Put 4 of 5 points above 55.
        data = _df([50, 56, 57, 50, 58, 56])
        result = detect_zone_b_4_of_5(data, _stats(), 'value', _zones())
        # Window ending at idx 5: [56, 57, 50, 58, 56] — 4 of 5 above 55
        assert result[5]

    def test_clean_data(self):
        """Points near center should not trigger Rule 3."""
        data = _df([50, 51, 49, 52, 48, 50, 51])
        result = detect_zone_b_4_of_5(data, _stats(), 'value', _zones())
        assert not result.any()


# ============================================================================
# Rule 4: Run of 8 on same side
# ============================================================================


class TestRule4Run:
    def test_trigger_above(self):
        """8 consecutive points above center should trigger."""
        data = _df([50, 51, 52, 53, 54, 55, 56, 57, 58])
        result = detect_run(data, _stats(), 'value')
        # All points 51-58 are above 50 — 8 consecutive above
        assert result.iloc[-1]

    def test_clean_data(self):
        """Alternating above/below center should not trigger."""
        data = _df([51, 49, 51, 49, 51, 49, 51, 49, 51])
        result = detect_run(data, _stats(), 'value')
        assert not result.any()


# ============================================================================
# Rule 6: Oscillation (14 consecutive alternating)
# ============================================================================


class TestRule6Oscillation:
    def test_trigger(self):
        """14 consecutive alternating up/down should trigger."""
        # Create alternating pattern
        values = [50 + ((-1) ** i) * 3 for i in range(16)]
        data = _df(values)
        result = detect_oscillation(data, _stats(), 'value')
        assert result.any()

    def test_clean_data(self):
        """Monotonically increasing should not trigger."""
        data = _df(list(range(40, 60)))
        result = detect_oscillation(data, _stats(), 'value')
        assert not result.any()


# ============================================================================
# Rule 7: Reduced variation (15 in Zone C)
# ============================================================================


class TestRule7ReducedVariation:
    def test_trigger(self):
        """15 consecutive points within Zone C should trigger."""
        # Zone C: (45, 55). Put 15+ points there.
        values = [50.5, 49.5, 51, 49, 52, 48, 50, 51, 49, 52, 48, 50, 51, 49, 50.5, 51]
        data = _df(values)
        result = detect_reduced_variation(data, _stats(), 'value', _zones())
        assert result.any()

    def test_clean_data(self):
        """Points spread across all zones should not trigger."""
        values = [40, 60, 42, 58, 44, 56, 46, 54, 48, 52, 50, 59, 41, 55, 45, 50]
        data = _df(values)
        result = detect_reduced_variation(data, _stats(), 'value', _zones())
        assert not result.any()


# ============================================================================
# Rule 8: Avoiding center (8 outside Zone C)
# ============================================================================


class TestRule8AvoidingCenter:
    def test_trigger(self):
        """8 consecutive points outside Zone C should trigger."""
        # Zone C: (45, 55). Put 8+ points outside.
        values = [56, 44, 57, 43, 58, 42, 59, 41, 56]
        data = _df(values)
        result = detect_avoiding_center(data, _stats(), 'value', _zones())
        assert result.any()

    def test_clean_data(self):
        """Points near center should not trigger."""
        data = _df([50, 51, 49, 52, 48, 50, 51, 49, 52])
        result = detect_avoiding_center(data, _stats(), 'value', _zones())
        assert not result.any()


# ============================================================================
# Cross-cutting
# ============================================================================


class TestCrossCutting:
    def test_insufficient_data_rule4(self):
        """Fewer than 8 points should produce no violations for Rule 4."""
        data = _df([51, 52, 53, 54, 55])  # Only 5 points, all above center
        result = detect_run(data, _stats(), 'value')
        assert not result.any()

    def test_insufficient_data_rule6(self):
        """Fewer than 14 points should produce no violations for Rule 6."""
        values = [50 + ((-1) ** i) * 3 for i in range(10)]
        data = _df(values)
        result = detect_oscillation(data, _stats(), 'value')
        assert not result.any()
