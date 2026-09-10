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

    def __init__(self, violations: pd.DataFrame, chart_name: str, data: pd.DataFrame, stats: dict,
                 rules_skipped: dict[str, str] | None = None,
                 rules_evaluated: list[str] | None = None,
                 n_observations: int | None = None,
                 min_observations: int | None = None):
        self.violations = violations
        self.chart_name = chart_name
        self.data = data
        self.stats = stats
        # Applicable rules that could NOT be evaluated because the group had too few
        # points (rule_name -> reason). Lets callers distinguish "no signals found"
        # from "not fully evaluated" (e.g. a small stratified subgroup).
        self.rules_skipped = rules_skipped or {}
        # Applicable rules that did run. Together with ``rules_skipped`` this gives
        # the "k of n rules applicable" denominator the summary reports.
        self.rules_evaluated = list(rules_evaluated or [])
        # Post-filter observation count the rules were evaluated on, and the
        # advisory threshold from ``SignalConfig.min_observations``. Either may be
        # None when a result is constructed directly rather than by the detector.
        self.n_observations = n_observations
        self.min_observations = min_observations

    @property
    def count(self) -> int:
        """Total number of violations detected."""
        return len(self.violations)

    @property
    def has_signals(self) -> bool:
        """Whether any signals were detected."""
        return self.count > 0

    @property
    def rules_applicable(self) -> int:
        """Number of rules that applied to this chart (evaluated + skipped)."""
        return len(self.rules_evaluated) + len(self.rules_skipped)

    @property
    def below_min_observations(self) -> bool:
        """True when the series is shorter than the configured advisory minimum.

        Rules that could run still ran; this flags that the evaluation as a whole
        is not one the analyst should treat as complete.
        """
        return (
            self.n_observations is not None
            and self.min_observations is not None
            and self.n_observations < self.min_observations
        )

    @property
    def is_partial(self) -> bool:
        """True when some applicable rules were skipped, or the series is below
        ``min_observations``. Either way, "no signals" is not an all-clear."""
        return bool(self.rules_skipped) or self.below_min_observations

    @property
    def evaluation_status(self) -> str:
        """'complete' when every applicable rule ran on an adequate series, else 'partial'."""
        return 'partial' if self.is_partial else 'complete'

    @property
    def evaluation_note(self) -> str:
        """One-line account of what was and was not evaluated.

        Empty when the evaluation is complete. Otherwise, e.g.::

            2 of 8 rules applicable at n=4 (below min_observations=20); skipped:
            rule_3 (needs 5), rule_4 (needs 8), ...
        """
        if not self.is_partial:
            return ''
        n_txt = f'n={self.n_observations}' if self.n_observations is not None else 'this series length'
        parts = [f'{len(self.rules_evaluated)} of {self.rules_applicable} rules applicable at {n_txt}']
        if self.below_min_observations:
            parts[0] += f' (below min_observations={self.min_observations})'
        if self.rules_skipped:
            skipped = ', '.join(
                f'{rule} ({reason.split(" observations")[0]})' for rule, reason in self.rules_skipped.items()
            )
            parts.append(f'skipped: {skipped}')
        return '; '.join(parts)

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

        return {rule: group for rule, group in self.violations.groupby('rule_name', observed=True)}

    @property
    def by_observation(self) -> dict[Any, pd.DataFrame]:
        """Group violations by observation ID."""
        if self.violations.empty:
            return {}

        return {obs_id: group for obs_id, group in self.violations.groupby('obs_id', observed=True)}

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

        return self.violations[self.violations['rule_name'] == rule_name].copy()

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

        return self.violations[self.violations['obs_id'] == obs_id].copy()

    @property
    def summary(self) -> str:
        """Human-readable summary of detected signals.

        A partial evaluation never reads as an all-clear: the clean-series
        checkmark line is reserved for a complete evaluation.
        """
        if not self.has_signals:
            if self.is_partial:
                return (
                    f'⚠ Partial evaluation in {self.chart_name}: {self.evaluation_note}. '
                    f'No signals from the rules evaluated.'
                )
            return f'✓ No signals detected in {self.chart_name}'

        lines = [
            f'\n{"=" * 70}',
            f'Signal Detection Summary: {self.chart_name}',
            f'{"=" * 70}',
            f'Total violations: {self.count}',
            f'Flagged observations: {len(self.flagged_observations)}',
        ]
        if self.is_partial:
            lines.append(f'Evaluation: partial ({self.evaluation_note})')
        lines.append('')

        # Breakdown by rule
        rule_counts = self.violations['rule_name'].value_counts()
        lines.append('Violations by rule:')
        for rule, count in rule_counts.items():
            lines.append(f'  {rule}: {count}')

        # Show first few violations
        lines.append('\nFirst violations:')
        for _, row in self.violations.head(5).iterrows():
            lines.append(f'  • Obs {row["obs_id"]}: {row["description"]} (value={row["value"]:.3f})')

        if self.count > 5:
            lines.append(f'  ... and {self.count - 5} more')

        lines.append(f'\n{"=" * 70}\n')
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
            logger.warning('No violations to export')
            return

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Violations sheet
            self.violations.to_excel(writer, sheet_name='Violations', index=False)

            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Violations', 'Flagged Observations', 'Chart Name',
                    'Evaluation Status', 'Rules Evaluated / Applicable',
                ],
                'Value': [
                    self.count, len(self.flagged_observations), self.chart_name,
                    self.evaluation_status, f'{len(self.rules_evaluated)} / {self.rules_applicable}',
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        logger.info(f'✓ Exported violations to: {filepath}')

    def to_json(self, filepath: str):
        """
        Export violations to JSON (a list of violation records).

        Evaluation status is not part of this file; read ``evaluation_status``,
        ``rules_evaluated`` and ``rules_skipped`` on the result instead.

        Parameters
        ----------
        filepath : str
            Path to output JSON file

        Examples
        --------
        >>> signals.to_json('violations.json')
        """
        self.violations.to_json(filepath, orient='records', indent=2)
        logger.info(f'✓ Exported violations to: {filepath}')

    def __repr__(self):
        partial = ", evaluation='partial'" if self.is_partial else ''
        return (
            f'SignalResult(violations={self.count}, '
            f'flagged_obs={len(self.flagged_observations)}, '
            f"chart='{self.chart_name}'{partial})"
        )

    def __str__(self):
        return self.summary
