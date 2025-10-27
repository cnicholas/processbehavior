"""
Main signal detection engine.

Orchestrates rule detection and builds comprehensive results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from .config import SignalConfig
from .detectors import (
    detect_avoiding_center,
    detect_beyond_limits,
    detect_oscillation,
    detect_reduced_variation,
    detect_run,
    detect_trend,
    detect_zone_a_2_of_3,
    detect_zone_b_4_of_5,
)
from .result import SignalResult

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)


class SignalDetector:
    """
    Main signal detection engine.

    Applies Western Electric rules to control chart data and returns
    comprehensive violation information.

    Examples
    --------
    >>> detector = SignalDetector()
    >>> signals = detector.detect(data, stats, config)
    """

    # Rule mapping
    RULE_DETECTORS = {
        'rule_1': detect_beyond_limits,
        'rule_2': detect_zone_a_2_of_3,
        'rule_3': detect_zone_b_4_of_5,
        'rule_4': detect_run,
        'rule_5': detect_trend,
        'rule_6': detect_oscillation,
        'rule_7': detect_reduced_variation,
        'rule_8': detect_avoiding_center,
    }

    RULE_DESCRIPTIONS = {
        'rule_1': 'Point beyond control limits',
        'rule_2': '2 of 3 consecutive points in Zone A',
        'rule_3': '4 of 5 consecutive points in Zone B or beyond',
        'rule_4': '8+ consecutive points on same side of center',
        'rule_5': '6+ consecutive points trending',
        'rule_6': '14+ consecutive points alternating',
        'rule_7': '15+ consecutive points in Zone C',
        'rule_8': '8+ consecutive points avoiding Zone C',
    }

    def detect(
        self,
        data: pd.DataFrame,
        stats: dict,
        config: Optional[SignalConfig] = None,
        value_col: str = 'mean',
        chart_name: str = 'Chart'
    ) -> SignalResult:
        """
        Detect signals in control chart data.

        Parameters
        ----------
        data : DataFrame
            Chart data with observations
        stats : dict
            Chart statistics (ucl, lcl, center, etc.)
        config : SignalConfig, optional
            Detection configuration (uses defaults if None)
        value_col : str, default 'mean'
            Name of value column
        chart_name : str, default 'Chart'
            Name of chart for reporting

        Returns
        -------
        SignalResult
            Comprehensive signal detection results

        Examples
        --------
        >>> detector = SignalDetector()
        >>> signals = detector.detect(chart_data, chart_stats)
        >>> print(signals.summary)
        """
        config = config or SignalConfig()

        # Validate data
        self._validate_inputs(data, stats, config)

        # Calculate zones
        center = stats['center']
        sigma = (stats['ucl'] - center) / 3
        zones = config.zone_definition.get_boundaries(center, sigma)

        # Apply filtering
        filtered_data = self._filter_data(data, config)

        # Detect violations for each enabled rule
        all_violations = pd.DataFrame(index=filtered_data.index)

        for rule_name in config.enabled_rules:
            if rule_name not in self.RULE_DETECTORS:
                logger.warning(f"Unknown rule: {rule_name}, skipping")
                continue

            # Check minimum observations
            min_obs = self._get_min_observations(rule_name)
            if len(filtered_data) < min_obs:
                logger.warning(
                    f"Skipping {rule_name}: insufficient observations "
                    f"(need {min_obs}, have {len(filtered_data)})"
                )
                continue

            # Apply detector
            detector = self.RULE_DETECTORS[rule_name]

            try:
                if rule_name in ['rule_2', 'rule_3', 'rule_7', 'rule_8']:
                    # Needs zones
                    violations = detector(
                        filtered_data, stats, value_col, zones
                    )
                else:
                    violations = detector(
                        filtered_data, stats, value_col
                    )

                all_violations[rule_name] = violations

            except Exception as e:
                logger.error(f"Error detecting {rule_name}: {e}")
                all_violations[rule_name] = False

        # Build result
        return self._build_result(
            data=filtered_data,
            violations=all_violations,
            stats=stats,
            value_col=value_col,
            chart_name=chart_name
        )

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        stats: dict,
        config: SignalConfig
    ):
        """Validate inputs and provide helpful errors."""
        if data.empty:
            raise ValueError("Cannot detect signals on empty DataFrame")

        required_stats = ['ucl', 'lcl', 'center']
        missing = [s for s in required_stats if s not in stats]
        if missing:
            raise ValueError(
                f"Missing required statistics: {missing}\n"
                f"Available: {list(stats.keys())}"
            )

        if len(data) < config.min_observations:
            raise ValueError(
                f"Insufficient observations for signal detection.\n"
                f"Required: {config.min_observations}, provided: {len(data)}\n"
                f"Hint: Reduce config.min_observations or provide more data"
            )

    def _filter_data(
        self,
        data: pd.DataFrame,
        config: SignalConfig
    ) -> pd.DataFrame:
        """Apply filtering options."""
        filtered = data.copy()

        if config.ignore_first_n > 0:
            filtered = filtered.iloc[config.ignore_first_n:]

        if config.ignore_last_n > 0:
            filtered = filtered.iloc[:-config.ignore_last_n]

        return filtered

    def _get_min_observations(self, rule_name: str) -> int:
        """Get minimum observations for a rule."""
        minimums = {
            'rule_1': 1,
            'rule_2': 3,
            'rule_3': 5,
            'rule_4': 8,
            'rule_5': 6,
            'rule_6': 14,
            'rule_7': 15,
            'rule_8': 8,
        }
        return minimums.get(rule_name, 1)

    def _build_result(
        self,
        data: pd.DataFrame,
        violations: pd.DataFrame,
        stats: dict,
        value_col: str,
        chart_name: str
    ) -> SignalResult:
        """Build SignalResult from violation matrix."""
        # Create violation records
        records = []

        for idx in data.index:
            for rule_name in violations.columns:
                if violations.loc[idx, rule_name]:
                    records.append({
                        'obs_id': idx,
                        'rule_name': rule_name,
                        'rule_number': int(rule_name.split('_')[1]),
                        'description': self.RULE_DESCRIPTIONS[rule_name],
                        'value': data.loc[idx, value_col],
                        'center': stats['center'],
                        'ucl': stats['ucl'],
                        'lcl': stats['lcl']
                    })

        violation_df = pd.DataFrame(records) if records else pd.DataFrame()

        return SignalResult(
            violations=violation_df,
            chart_name=chart_name,
            data=data,
            stats=stats
        )
