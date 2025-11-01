"""
Tests for Western Electric signal detection.

Tests the signal detection framework including configuration,
detection, and result handling.
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior.signals import RuleSet, SignalConfig, SignalDetector, ZoneDefinition


class TestZoneDefinition:
    """Test zone boundary calculations."""

    def test_default_zones(self):
        """Test default zone definitions."""
        zones = ZoneDefinition()
        assert zones.A == (2.0, 3.0)
        assert zones.B == (1.0, 2.0)
        assert zones.C == (0.0, 1.0)

    def test_get_boundaries(self):
        """Test boundary calculation."""
        zones = ZoneDefinition()
        boundaries = zones.get_boundaries(center=100.0, sigma=5.0)

        # Check Zone A
        assert boundaries['A_upper'] == (110.0, 115.0)
        assert boundaries['A_lower'] == (85.0, 90.0)

        # Check Zone B
        assert boundaries['B_upper'] == (105.0, 110.0)
        assert boundaries['B_lower'] == (90.0, 95.0)

        # Check Zone C
        assert boundaries['C_upper'] == (100.0, 105.0)
        assert boundaries['C_lower'] == (95.0, 100.0)


class TestSignalConfig:
    """Test configuration class."""

    def test_default_config(self):
        """Test default configuration."""
        config = SignalConfig()
        assert config.enabled_rules == 'default'
        assert config.min_observations == 20
        assert config.use_vectorized is True
        # Test chart-type-based defaults
        assert config.get_rules_for_chart('Xbar') == ['rule_1']
        assert config.get_rules_for_chart('S') == ['rule_1']
        assert len(config.get_rules_for_chart('Imr')) == 8
        assert len(config.get_rules_for_chart('R')) == 8

    def test_preset_standard(self):
        """Test standard preset."""
        config = SignalConfig(enabled_rules='standard')
        assert config.enabled_rules == ['rule_1', 'rule_2', 'rule_3', 'rule_4']

    def test_preset_extended(self):
        """Test extended preset."""
        config = SignalConfig(enabled_rules='extended')
        assert len(config.enabled_rules) == 8
        assert 'rule_8' in config.enabled_rules

    def test_custom_rules_list(self):
        """Test custom rules list."""
        config = SignalConfig(enabled_rules=['rule_1', 'rule_5'])
        assert config.enabled_rules == ['rule_1', 'rule_5']


class TestRuleSet:
    """Test fluent API rule set builder."""

    def test_single_rule(self):
        """Test adding a single rule."""
        rules = RuleSet().beyond_limits()
        assert rules.get_rules() == ['rule_1']

    def test_multiple_rules(self):
        """Test chaining multiple rules."""
        rules = (
            RuleSet()
            .beyond_limits()
            .zone_a()
            .trend()
        )
        assert rules.get_rules() == ['rule_1', 'rule_2', 'rule_5']

    def test_all_rules(self):
        """Test adding all standard rules."""
        rules = (
            RuleSet()
            .beyond_limits()
            .zone_a()
            .zone_b()
            .run()
            .trend()
            .oscillation()
            .reduced_variation()
            .avoiding_center()
        )
        assert len(rules.get_rules()) == 8


class TestSignalDetector:
    """Test the main signal detector."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        np.random.seed(42)
        n = 30
        values = np.random.normal(100, 5, n)
        return pd.DataFrame({
            'mean': values,
            'obs_id': range(n)
        })

    @pytest.fixture
    def simple_stats(self):
        """Create simple test statistics."""
        return {
            'center': 100.0,
            'ucl': 115.0,  # center + 3*sigma
            'lcl': 85.0    # center - 3*sigma
        }

    def test_no_violations(self, simple_data, simple_stats):
        """Test detection with no violations."""
        detector = SignalDetector()
        config = SignalConfig(enabled_rules=['rule_1'], min_observations=10)

        result = detector.detect(simple_data, simple_stats, config)

        assert result.count == 0
        assert not result.has_signals
        assert len(result.flagged_observations) == 0

    def test_beyond_limits_detection(self):
        """Test Rule 1: beyond limits detection."""
        # Create data with violations
        data = pd.DataFrame({
            'mean': [100, 100, 120, 100, 100],  # 120 is beyond UCL
            'obs_id': range(5)
        })

        stats = {
            'center': 100.0,
            'ucl': 115.0,
            'lcl': 85.0
        }

        detector = SignalDetector()
        config = SignalConfig(enabled_rules=['rule_1'], min_observations=1)

        result = detector.detect(data, stats, config)

        assert result.count > 0
        assert result.has_signals
        assert 2 in result.flagged_observations  # Index 2 has value 120

    def test_trend_detection(self):
        """Test Rule 5: trend detection."""
        # Create steadily increasing data
        data = pd.DataFrame({
            'mean': [90, 92, 94, 96, 98, 100, 102],  # Increasing trend
            'obs_id': range(7)
        })

        stats = {
            'center': 96.0,
            'ucl': 110.0,
            'lcl': 82.0
        }

        detector = SignalDetector()
        config = SignalConfig(enabled_rules=['rule_5'], min_observations=1)

        result = detector.detect(data, stats, config)

        # Should detect trend (6+ consecutive increasing points)
        assert result.count > 0
        assert result.has_signals

    def test_insufficient_data(self, simple_stats):
        """Test error handling for insufficient data."""
        # Only 5 observations, but config requires 20
        data = pd.DataFrame({
            'mean': [100, 100, 100, 100, 100],
            'obs_id': range(5)
        })

        detector = SignalDetector()
        config = SignalConfig(min_observations=20)

        with pytest.raises(ValueError, match="Insufficient observations"):
            detector.detect(data, simple_stats, config)

    def test_missing_stats(self, simple_data):
        """Test error handling for missing statistics."""
        incomplete_stats = {'Mean': 100.0}  # Missing ucl and lcl

        detector = SignalDetector()
        config = SignalConfig(min_observations=10)

        with pytest.raises(ValueError, match="Missing required statistics"):
            detector.detect(simple_data, incomplete_stats, config)


class TestSignalResult:
    """Test signal result container."""

    @pytest.fixture
    def sample_result(self):
        """Create sample result for testing."""
        from processbehavior.signals import SignalResult

        violations = pd.DataFrame({
            'obs_id': [1, 2, 5, 5],
            'rule_name': ['rule_1', 'rule_1', 'rule_2', 'rule_5'],
            'rule_number': [1, 1, 2, 5],
            'description': ['Beyond limits', 'Beyond limits', 'Zone A', 'Trend'],
            'value': [120.0, 85.0, 112.0, 108.0],
            'center': [100.0, 100.0, 100.0, 100.0],
            'ucl': [115.0, 115.0, 115.0, 115.0],
            'lcl': [85.0, 85.0, 85.0, 85.0]
        })

        data = pd.DataFrame({'mean': [100] * 10})
        stats = {'center': 100, 'ucl': 115, 'lcl': 85}

        return SignalResult(violations, 'Test Chart', data, stats)

    def test_violation_count(self, sample_result):
        """Test violation counting."""
        assert sample_result.count == 4
        assert sample_result.has_signals

    def test_flagged_observations(self, sample_result):
        """Test flagged observation tracking."""
        flagged = sample_result.flagged_observations
        assert len(flagged) == 3  # obs_ids 1, 2, 5
        assert 1 in flagged
        assert 2 in flagged
        assert 5 in flagged

    def test_by_rule(self, sample_result):
        """Test grouping by rule."""
        by_rule = sample_result.by_rule
        assert 'rule_1' in by_rule
        assert 'rule_2' in by_rule
        assert 'rule_5' in by_rule
        assert len(by_rule['rule_1']) == 2  # Two Rule 1 violations

    def test_by_observation(self, sample_result):
        """Test grouping by observation."""
        by_obs = sample_result.by_observation
        assert 5 in by_obs
        # obs_id 5 has 2 violations (rule_2 and rule_5)
        assert len(by_obs[5]) == 2
