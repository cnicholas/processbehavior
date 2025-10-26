"""
Signal detection result container.

Provides multiple ways to access and analyze detected violations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class SignalResult:
    """
    Comprehensive signal detection results.

    Provides multiple ways to access and analyze violations:
    - By rule
    - By observation
    - Summary statistics
    - Export options

    Examples
    --------
    >>> signals = result.detect_signals()
    >>> print(signals.summary)
    >>> signals.by_rule['rule_2']  # Rule 2 violations
    >>> signals.flagged_observations  # Set of obs_ids
    >>> signals.to_excel('violations.xlsx')
    """

    def __init__(
        self,
        violations: pd.DataFrame,
        chart_name: str,
        data: pd.DataFrame,
        stats: dict
    ):
        self.violations = violations
        self.chart_name = chart_name
        self.data = data
        self.stats = stats

    @property
    def count(self) -> int:
        """Total number of violations detected."""
        return len(self.violations)

    @property
    def has_signals(self) -> bool:
        """Whether any signals were detected."""
        return self.count > 0

    @property
    def flagged_observations(self) -> set:
        """Set of observation IDs that violated any rule."""
        if self.violations.empty:
            return set()
        return set(self.violations['obs_id'].unique())

    @property
    def by_rule(self) -> dict[str, pd.DataFrame]:
        """Group violations by rule."""
        if self.violations.empty:
            return {}

        return {
            rule: group
            for rule, group in self.violations.groupby('rule_name')
        }

    @property
    def by_observation(self) -> dict[Any, pd.DataFrame]:
        """Group violations by observation ID."""
        if self.violations.empty:
            return {}

        return {
            obs_id: group
            for obs_id, group in self.violations.groupby('obs_id')
        }

    def get_rule_violations(self, rule_name: str) -> pd.DataFrame:
        """
        Get all violations for a specific rule.

        Parameters
        ----------
        rule_name : str
            Name of the rule (e.g., 'rule_1')

        Returns
        -------
        DataFrame
            Violations for the specified rule
        """
        if self.violations.empty:
            return pd.DataFrame()

        return self.violations[
            self.violations['rule_name'] == rule_name
        ].copy()

    def get_observation_violations(self, obs_id: Any) -> pd.DataFrame:
        """
        Get all rule violations for a specific observation.

        Parameters
        ----------
        obs_id : any
            Observation identifier

        Returns
        -------
        DataFrame
            All rules violated by this observation
        """
        if self.violations.empty:
            return pd.DataFrame()

        return self.violations[
            self.violations['obs_id'] == obs_id
        ].copy()

    @property
    def summary(self) -> str:
        """Human-readable summary of detected signals."""
        if not self.has_signals:
            return f"✓ No signals detected in {self.chart_name}"

        lines = [
            f"\n{'=' * 70}",
            f"Signal Detection Summary: {self.chart_name}",
            f"{'=' * 70}",
            f"Total violations: {self.count}",
            f"Flagged observations: {len(self.flagged_observations)}",
            ""
        ]

        # Breakdown by rule
        rule_counts = self.violations['rule_name'].value_counts()
        lines.append("Violations by rule:")
        for rule, count in rule_counts.items():
            lines.append(f"  {rule}: {count}")

        # Show first few violations
        lines.append("\nFirst violations:")
        for _, row in self.violations.head(5).iterrows():
            lines.append(
                f"  • Obs {row['obs_id']}: {row['description']} "
                f"(value={row['value']:.3f})"
            )

        if self.count > 5:
            lines.append(f"  ... and {self.count - 5} more")

        lines.append(f"\n{'=' * 70}\n")
        return '\n'.join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get violations as DataFrame.

        Returns
        -------
        DataFrame
            Copy of violations DataFrame
        """
        return self.violations.copy()

    def to_excel(self, filepath: str):
        """
        Export violations to Excel.

        Creates two sheets:
        - Violations: Detailed violation records
        - Summary: High-level statistics

        Parameters
        ----------
        filepath : str
            Path to output Excel file

        Examples
        --------
        >>> signals.to_excel('violations.xlsx')
        """
        if self.violations.empty:
            logger.warning("No violations to export")
            return

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Violations sheet
            self.violations.to_excel(
                writer,
                sheet_name='Violations',
                index=False
            )

            # Summary sheet
            summary_data = {
                'Metric': ['Total Violations', 'Flagged Observations', 'Chart Name'],
                'Value': [
                    self.count,
                    len(self.flagged_observations),
                    self.chart_name
                ]
            }
            pd.DataFrame(summary_data).to_excel(
                writer,
                sheet_name='Summary',
                index=False
            )

        logger.info(f"✓ Exported violations to: {filepath}")

    def to_json(self, filepath: str):
        """
        Export violations to JSON.

        Parameters
        ----------
        filepath : str
            Path to output JSON file

        Examples
        --------
        >>> signals.to_json('violations.json')
        """
        self.violations.to_json(filepath, orient='records', indent=2)
        logger.info(f"✓ Exported violations to: {filepath}")

    def __repr__(self):
        return (
            f"SignalResult(violations={self.count}, "
            f"flagged_obs={len(self.flagged_observations)}, "
            f"chart='{self.chart_name}')"
        )

    def __str__(self):
        return self.summary
