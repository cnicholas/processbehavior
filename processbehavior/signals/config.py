"""
Configuration classes for signal detection.

Provides flexible configuration through dataclasses and fluent API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

# Chart-type-based rule defaults
# ================================
# WECO rules 2-8 require sequential/temporal ordering of observations.
# - Xbar/S: Categorical comparison of subgroups → Only Rule 1 applies
# - X/mR: Sequential observations over time → All rules apply
# Easy to extend for new chart types (CUSUM, EWMA, etc.)
CHART_TYPE_RULES = {
    'Xbar': ['rule_1'],  # Rational subgroup means (categorical)
    'S': ['rule_1'],  # Rational subgroup variation (categorical)
    'X': ['rule_1', 'rule_2', 'rule_3', 'rule_4', 'rule_5', 'rule_6', 'rule_7', 'rule_8'],
    'mR': ['rule_1', 'rule_2', 'rule_3', 'rule_4', 'rule_5', 'rule_6', 'rule_7', 'rule_8'],
}


@dataclass
class ZoneDefinition:
    """
    Zone boundary definition in standard deviations from centerline.

    Zones are defined as multiples of sigma (standard deviation):
    - Zone A: 2σ to 3σ from centerline
    - Zone B: 1σ to 2σ from centerline
    - Zone C: 0σ to 1σ from centerline

    Examples
    --------
    Default zones:

    >>> zones = ZoneDefinition()
    >>> zones.A  # (2.0, 3.0)

    Custom zones:

    >>> zones = ZoneDefinition(A=(2.5, 3.5), B=(1.5, 2.5))
    """

    A: tuple[float, float] = (2.0, 3.0)
    B: tuple[float, float] = (1.0, 2.0)
    C: tuple[float, float] = (0.0, 1.0)

    def get_boundaries(self, center: float, sigma: float) -> dict:
        """
        Calculate actual zone boundaries for given center and sigma.

        Parameters
        ----------
        center : float
            Centerline value
        sigma : float
            Standard deviation

        Returns
        -------
        dict
            Zone boundaries with keys like 'A_upper', 'A_lower', etc.
        """
        return {
            'A_upper': (center + self.A[0] * sigma, center + self.A[1] * sigma),
            'A_lower': (center - self.A[1] * sigma, center - self.A[0] * sigma),
            'B_upper': (center + self.B[0] * sigma, center + self.B[1] * sigma),
            'B_lower': (center - self.B[1] * sigma, center - self.B[0] * sigma),
            'C_upper': (center + self.C[0] * sigma, center + self.C[1] * sigma),
            'C_lower': (center - self.C[1] * sigma, center - self.C[0] * sigma),
        }


@dataclass
class SignalConfig:
    """
    Configuration for signal detection.

    Provides flexible control over which rules to apply and how to apply them.

    Parameters
    ----------
    enabled_rules : list or str, default 'default'
        Which rules to apply:
        - 'default': Use chart-type-based defaults (Xbar/S: Rule 1, X/mR: Rules 1-8)
        - 'standard': Rules 1-4 (filtered by chart type)
        - 'extended': Rules 1-8 (filtered by chart type)
        - 'all': Same as extended
        - List of rule names: ['rule_1', 'rule_2', ...] (explicit override)
    zone_definition : ZoneDefinition, optional
        Custom zone definitions
    min_observations : int, default 20
        Minimum observations required
    ignore_first_n : int, default 0
        Ignore first N observations (startup period)
    ignore_last_n : int, default 0
        Ignore last N observations
    use_vectorized : bool, default True
        Use vectorized operations (faster)

    Examples
    --------
    Standard configuration:

    >>> config = SignalConfig(enabled_rules='standard')

    Custom configuration:

    >>> config = SignalConfig(
    ...     enabled_rules=['rule_1', 'rule_2', 'rule_5'],
    ...     zone_definition=ZoneDefinition(A=(2.5, 3.5)),
    ...     min_observations=30
    ... )

    Extended rules:

    >>> config = SignalConfig(enabled_rules='extended')
    """

    enabled_rules: list[str] | Literal['default', 'standard', 'extended', 'all'] = 'default'
    zone_definition: ZoneDefinition = field(default_factory=ZoneDefinition)
    min_observations: int = 20
    ignore_first_n: int = 0
    ignore_last_n: int = 0
    use_vectorized: bool = True

    def __post_init__(self):
        """Resolve rule presets to actual rule lists."""
        if isinstance(self.enabled_rules, str) and self.enabled_rules != 'default':
            # Resolve presets except 'default' (resolved per chart type)
            self.enabled_rules = self._resolve_preset(self.enabled_rules)

    def _resolve_preset(self, preset: str) -> list[str]:
        """Convert preset name to rule list."""
        presets = {
            'standard': ['rule_1', 'rule_2', 'rule_3', 'rule_4'],
            'extended': [f'rule_{i}' for i in range(1, 9)],
            'all': [f'rule_{i}' for i in range(1, 9)],
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'.\\nValid presets: {list(presets.keys())}")

        return presets[preset]

    def get_rules_for_chart(self, chart_type: str) -> list[str]:
        """
        Get applicable rules for a chart type.

        Uses chart-type-based defaults or applies filtering to user-specified rules.

        Parameters
        ----------
        chart_type : str
            Chart type ('Xbar', 'S', 'X', 'mR')

        Returns
        -------
        list of str
            Rule names to apply

        Examples
        --------
        Use defaults:

        >>> config = SignalConfig(enabled_rules='default')
        >>> config.get_rules_for_chart('Xbar')
        ['rule_1']
        >>> config.get_rules_for_chart('X')
        ['rule_1', 'rule_2', ..., 'rule_8']

        Explicit rules (no filtering):

        >>> config = SignalConfig(enabled_rules=['rule_1', 'rule_2'])
        >>> config.get_rules_for_chart('Xbar')
        ['rule_1', 'rule_2']
        """
        if isinstance(self.enabled_rules, list):
            # User provided explicit list - use as-is
            return self.enabled_rules

        if self.enabled_rules == 'default':
            # Use chart-type defaults
            return CHART_TYPE_RULES.get(chart_type, ['rule_1'])

        # This shouldn't happen (already resolved in __post_init__)
        # but handle just in case
        raise ValueError(f"Invalid enabled_rules state: {self.enabled_rules}. Expected list or 'default'.")


class RuleSet:
    """
    Fluent API for building rule configurations.

    Provides a chainable interface for configuring detection rules.

    Examples
    --------
    Build a custom rule set:

    >>> rules = (
    ...     RuleSet()
    ...         .beyond_limits()
    ...         .zone_a(consecutive=2, window=3)
    ...         .trend(length=6, direction='both')
    ... )

    Convert to list:

    >>> rule_list = rules.get_rules()
    >>> print(rule_list)
    ['rule_1', 'rule_2', 'rule_5']
    """

    def __init__(self):
        self._rules: list[str] = []
        self._parameters: dict[str, dict] = {}

    def beyond_limits(self) -> RuleSet:
        """
        Add Rule 1: Points beyond control limits.

        Detects any point beyond ±3σ from centerline.

        Returns
        -------
        RuleSet
            Self for chaining
        """
        self._rules.append('rule_1')
        return self

    def zone_a(self, consecutive: int = 2, window: int = 3) -> RuleSet:
        """
        Add Rule 2: Consecutive points in Zone A.

        Detects when 2 out of 3 consecutive points fall in Zone A
        (between 2σ and 3σ from centerline).

        Parameters
        ----------
        consecutive : int, default 2
            Number of points that must be in zone
        window : int, default 3
            Window size to check

        Returns
        -------
        RuleSet
            Self for chaining
        """
        self._rules.append('rule_2')
        self._parameters['rule_2'] = {'consecutive': consecutive, 'window': window}
        return self

    def zone_b(self, consecutive: int = 4, window: int = 5) -> RuleSet:
        """
        Add Rule 3: Consecutive points in Zone B or beyond.

        Detects when 4 out of 5 consecutive points fall in Zone B or beyond
        (beyond 1σ from centerline).

        Parameters
        ----------
        consecutive : int, default 4
            Number of points that must be in zone
        window : int, default 5
            Window size to check

        Returns
        -------
        RuleSet
            Self for chaining
        """
        self._rules.append('rule_3')
        self._parameters['rule_3'] = {'consecutive': consecutive, 'window': window}
        return self

    def run(self, length: int = 8) -> RuleSet:
        """
        Add Rule 4: Run above or below centerline.

        Detects when 8 or more consecutive points fall on the same side
        of the centerline.

        Parameters
        ----------
        length : int, default 8
            Number of consecutive points required

        Returns
        -------
        RuleSet
            Self for chaining
        """
        self._rules.append('rule_4')
        self._parameters['rule_4'] = {'length': length}
        return self

    def trend(self, length: int = 6, direction: Literal['up', 'down', 'both'] = 'both') -> RuleSet:
        """
        Add Rule 5: Trending points.

        Detects when 6 or more consecutive points are steadily
        increasing or decreasing.

        Parameters
        ----------
        length : int, default 6
            Number of consecutive points required
        direction : {'up', 'down', 'both'}, default 'both'
            Which direction trends to detect

        Returns
        -------
        RuleSet
            Self for chaining
        """
        self._rules.append('rule_5')
        self._parameters['rule_5'] = {'length': length, 'direction': direction}
        return self

    def oscillation(self, length: int = 14) -> RuleSet:
        """
        Add Rule 6: Alternating pattern.

        Detects when 14 or more consecutive points alternate up and down.

        Parameters
        ----------
        length : int, default 14
            Number of consecutive points required

        Returns
        -------
        RuleSet
            Self for chaining
        """
        self._rules.append('rule_6')
        self._parameters['rule_6'] = {'length': length}
        return self

    def reduced_variation(self, length: int = 15) -> RuleSet:
        """
        Add Rule 7: Points in Zone C (low variation).

        Detects when 15 or more consecutive points fall within Zone C
        (within ±1σ of centerline), indicating reduced variation.

        Parameters
        ----------
        length : int, default 15
            Number of consecutive points required

        Returns
        -------
        RuleSet
            Self for chaining
        """
        self._rules.append('rule_7')
        self._parameters['rule_7'] = {'length': length}
        return self

    def avoiding_center(self, length: int = 8) -> RuleSet:
        """
        Add Rule 8: Points avoiding Zone C.

        Detects when 8 or more consecutive points fall outside Zone C
        (beyond ±1σ from centerline).

        Parameters
        ----------
        length : int, default 8
            Number of consecutive points required

        Returns
        -------
        RuleSet
            Self for chaining
        """
        self._rules.append('rule_8')
        self._parameters['rule_8'] = {'length': length}
        return self

    def custom(self, name: str, detector: Callable, min_observations: int = 1) -> RuleSet:
        """
        Add a custom detection rule.

        Parameters
        ----------
        name : str
            Rule name
        detector : callable
            Detection function with signature:
            detector(data: DataFrame, stats: dict, value_col: str) -> Series[bool]
        min_observations : int, default 1
            Minimum observations required

        Returns
        -------
        RuleSet
            Self for chaining
        """
        self._rules.append(name)
        self._parameters[name] = {'detector': detector, 'min_observations': min_observations}
        return self

    def to_config(self) -> SignalConfig:
        """
        Convert to SignalConfig.

        Returns
        -------
        SignalConfig
            Configuration object
        """
        return SignalConfig(enabled_rules=self._rules)

    def get_rules(self) -> list[str]:
        """
        Get list of enabled rules.

        Returns
        -------
        list of str
            Rule names
        """
        return self._rules.copy()
