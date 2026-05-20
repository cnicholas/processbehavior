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
    pass

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
        config: SignalConfig | None = None,
        value_col: str = 'mean',
        chart_name: str = 'Chart',
        chart_type: str = 'Xbar'
    ) -> SignalResult:
        """
        Detect signals in control chart data.

        Parameters
        ----------
        data : DataFrame
            Chart data with observations
        stats : dict
            Chart statistics (upl, lpl, center, etc.)
        config : SignalConfig, optional
            Detection configuration (uses defaults if None)
        value_col : str, default 'mean'
            Name of value column
        chart_name : str, default 'Chart'
            Name of chart for reporting
        chart_type : str, default 'Xbar'
            Chart type ('Xbar', 'S', 'X', 'mR') - determines applicable rules

        Returns
        -------
        SignalResult
            Comprehensive signal detection results

        Examples
        --------
        >>> detector = SignalDetector()
        >>> signals = detector.detect(chart_data, chart_stats, chart_type='X')
        >>> print(signals.summary)
        """
        config = config or SignalConfig()

        # Get applicable rules for this chart type
        applicable_rules = config.get_rules_for_chart(chart_type)

        # Log rule selection
        if logger.isEnabledFor(logging.INFO):
            all_rules = [f'rule_{i}' for i in range(1, 9)]
            skipped_rules = [r for r in all_rules if r not in applicable_rules]
            if skipped_rules:
                logger.info(
                    f"Chart '{chart_name}' (type: {chart_type}): "
                    f"Applying rules {applicable_rules}. "
                    f"Skipped {skipped_rules} (not applicable for {chart_type} charts)"
                )
            else:
                logger.info(
                    f"Chart '{chart_name}' (type: {chart_type}): "
                    f"Applying all rules {applicable_rules}"
                )

        # Calculate minimum observations needed for applicable rules
        min_obs_needed = max(
            self._get_min_observations(rule) for rule in applicable_rules
        ) if applicable_rules else 1

        # Validate data (using rule-based minimum, not global config)
        self._validate_inputs(data, stats, config, min_obs_needed)

        # Apply filtering
        filtered_data = self._filter_data(data, config)

        # Detect if limits vary
        limits_vary = self._limits_vary(stats)

        # Calculate zones (constant or per-row)
        if limits_vary:
            # Use per-row limits from data
            zones = self._calculate_per_row_zones(filtered_data, stats, config)
        else:
            # Use constant limits from stats
            center = stats['center']
            sigma = (stats['upl'] - center) / 3
            zones = config.zone_definition.get_boundaries(center, sigma)

        # Detect violations for each applicable rule
        all_violations = pd.DataFrame(index=filtered_data.index)

        for rule_name in applicable_rules:
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
                    # Zone-based rules
                    violations = detector(
                        filtered_data, stats, value_col, zones, limits_vary
                    )
                elif rule_name == 'rule_1':
                    # Rule 1 can use per-row limits
                    violations = detector(
                        filtered_data, stats, value_col, limits_vary
                    )
                else:
                    # Other rules (4, 5, 6) don't use zones
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
        config: SignalConfig,
        min_obs_for_rules: int = 1
    ):
        """Validate inputs and provide helpful errors."""
        if data.empty:
            raise ValueError("Cannot detect signals on empty DataFrame")

        required_stats = ['upl', 'lpl', 'center']
        missing = [s for s in required_stats if s not in stats]
        if missing:
            raise ValueError(
                f"Missing required statistics: {missing}\n"
                f"Available: {list(stats.keys())}"
            )

        # Use the minimum observations needed for applicable rules
        # This respects chart-type-specific rule filtering (e.g., S charts only need Rule 1)
        if len(data) < min_obs_for_rules:
            raise ValueError(
                f"Insufficient observations for signal detection.\n"
                f"Required: {min_obs_for_rules}, provided: {len(data)}\n"
                f"Hint: Provide more data"
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

        # Check if limits vary
        limits_vary = self._limits_vary(stats)

        for idx in data.index:
            for rule_name in violations.columns:
                if violations.loc[idx, rule_name]:
                    # Use per-row limits if they vary, else use stats
                    if limits_vary and 'upl' in data.columns and 'lpl' in data.columns:
                        ucl = data.loc[idx, 'upl']
                        lcl = data.loc[idx, 'lpl']
                    else:
                        ucl = stats['upl']
                        lcl = stats['lpl']

                    records.append({
                        'obs_id': idx,
                        'rule_name': rule_name,
                        'rule_number': int(rule_name.split('_')[1]),
                        'description': self.RULE_DESCRIPTIONS[rule_name],
                        'value': data.loc[idx, value_col],
                        'center': stats['center'],
                        'upl': ucl,
                        'lpl': lcl
                    })

        violation_df = pd.DataFrame(records) if records else pd.DataFrame()

        return SignalResult(
            violations=violation_df,
            chart_name=chart_name,
            data=data,
            stats=stats
        )

    def _limits_vary(self, stats: dict) -> bool:
        """Check if control limits vary (per-row limits)."""
        return bool(stats.get('limits_vary'))

    def _calculate_per_row_zones(
        self,
        data: pd.DataFrame,
        stats: dict,
        config: SignalConfig
    ) -> pd.DataFrame:
        """
        Calculate per-row zone boundaries for varying process limits.

        When process limits vary per row (e.g., Xbar with varying n),
        this calculates zone boundaries for each row based on that row's
        specific upl/lpl values.

        Parameters
        ----------
        data : DataFrame
            Chart data with 'upl' and 'lpl' columns
        stats : dict
            Chart statistics with 'center' key
        config : SignalConfig
            Configuration with zone_definition

        Returns
        -------
        DataFrame
            Per-row zone boundaries with columns like 'A_upper_lower', 'A_upper_upper', etc.
        """
        if 'upl' not in data.columns or 'lpl' not in data.columns:
            raise ValueError(
                "Cannot calculate per-row zones: data missing 'upl' or 'lpl' columns"
            )

        center = stats['center']
        zone_def = config.zone_definition

        # Calculate sigma per row
        sigma = (data['upl'] - center) / 3

        # Calculate all zone boundaries per row
        zones_df = pd.DataFrame(index=data.index)

        # Upper zones
        zones_df['A_upper_lower'] = center + zone_def.A[0] * sigma
        zones_df['A_upper_upper'] = center + zone_def.A[1] * sigma
        zones_df['B_upper_lower'] = center + zone_def.B[0] * sigma
        zones_df['B_upper_upper'] = center + zone_def.B[1] * sigma
        zones_df['C_upper_lower'] = center + zone_def.C[0] * sigma
        zones_df['C_upper_upper'] = center + zone_def.C[1] * sigma

        # Lower zones
        zones_df['A_lower_lower'] = center - zone_def.A[1] * sigma
        zones_df['A_lower_upper'] = center - zone_def.A[0] * sigma
        zones_df['B_lower_lower'] = center - zone_def.B[1] * sigma
        zones_df['B_lower_upper'] = center - zone_def.B[0] * sigma
        zones_df['C_lower_lower'] = center - zone_def.C[1] * sigma
        zones_df['C_lower_upper'] = center - zone_def.C[0] * sigma

        return zones_df
