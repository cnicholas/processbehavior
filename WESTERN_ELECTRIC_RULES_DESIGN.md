# Western Electric Rules Implementation Design

## Executive Summary

**Recommendation: Option 3 (Declarative Rule Engine with Fluent API)**

This approach provides:
- **Configurable rules** - Enable/disable individual rules easily
- **Observable results** - Track which obs_ids violate which rules
- **Extensible** - Add custom rules without changing core code
- **Pythonic** - Intuitive, readable API
- **Performance** - Vectorized operations via pandas/numpy

---

## Background: Western Electric Rules

The Western Electric rules (from the 1956 Statistical Quality Control Handbook) detect non-random patterns in control charts:

### Standard Rules

1. **Rule 1: Beyond Control Limits**
   - Any point beyond ±3σ from centerline
   - Already implemented as `beyond_limits`

2. **Rule 2: Zone A (2/3 points)**
   - 2 out of 3 consecutive points in Zone A (between 2σ and 3σ)
   - Same side of centerline

3. **Rule 3: Zone B (4/5 points)**
   - 4 out of 5 consecutive points in Zone B (between 1σ and 2σ)
   - Same side of centerline

4. **Rule 4: Zone C (8+ points)**
   - 8 or more consecutive points in Zone C (between 0σ and 1σ)
   - Same side of centerline (run above/below center)

5. **Rule 5: Trend (6+ points)**
   - 6 or more consecutive points steadily increasing or decreasing

6. **Rule 6: Oscillation (14+ points)**
   - 14 or more consecutive points alternating up and down

7. **Rule 7: Zone C (15+ points)**
   - 15 or more consecutive points within Zone C (within ±1σ)
   - Indicates reduced variation

8. **Rule 8: Zone B/C (8+ points)**
   - 8 or more consecutive points avoiding Zone C
   - All points in Zone B or beyond

### Zones Definition

```
UCL (3σ)  ─────────────────────────
          Zone A (2σ to 3σ)
2σ        ─────────────────────────
          Zone B (1σ to 2σ)
1σ        ─────────────────────────
          Zone C (0σ to 1σ)
Center    ═════════════════════════
          Zone C (0σ to 1σ)
-1σ       ─────────────────────────
          Zone B (1σ to 2σ)
-2σ       ─────────────────────────
          Zone A (2σ to 3σ)
LCL (-3σ) ─────────────────────────
```

---

## Design Goals (Pythonic Hadley Philosophy)

1. **Human-First API**
   ```python
   # Should read like English
   result.detect_signals(rules=['all'])
   result.detect_signals(rules=['rule_1', 'rule_2'])
   ```

2. **Progressive Disclosure**
   ```python
   # Simple: Use defaults
   signals = result.detect_signals()

   # Advanced: Configure everything
   signals = result.detect_signals(
       rules=custom_rules,
       zone_widths={'A': (2, 3), 'B': (1, 2), 'C': (0, 1)},
       min_observations=20
   )
   ```

3. **Observable Results**
   ```python
   # Get violations with metadata
   signals.by_rule['rule_2']  # DataFrame with obs_ids
   signals.summary  # Human-readable summary
   signals.to_dataframe()  # All violations
   ```

4. **Composable & Extensible**
   ```python
   # Add custom rules
   result.add_custom_rule(
       name='custom_stability',
       detector=my_detector_function
   )
   ```

5. **Fail Helpful**
   ```python
   # Clear error messages
   "Cannot apply Rule 2 to chart with < 3 observations.
    Current chart has 2 observations.
    Hint: Ensure your data has sufficient points."
   ```

---

## Option 1: Simple Function-Based Approach

### API Design
```python
# In analysis_result.py
class AnalysisResult:

    def detect_signals(
        self,
        chart: Optional[str] = None,
        rules: List[str] = ['rule_1']
    ) -> SignalResult:
        """Detect signals using Western Electric rules."""
        # Simple function calls
        violations = {}

        if 'rule_1' in rules:
            violations['rule_1'] = detect_rule_1(data, stats)
        if 'rule_2' in rules:
            violations['rule_2'] = detect_rule_2(data, stats)
        # etc...

        return SignalResult(violations)


# Separate module: western_electric.py
def detect_rule_1(data: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Detect points beyond control limits."""
    violations = data[
        (data['value'] > stats['ucl']) |
        (data['value'] < stats['lcl'])
    ]
    return violations[['obs_id', 'value']].copy()


def detect_rule_2(data: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """2 of 3 points in Zone A."""
    # Calculate zones
    sigma = (stats['ucl'] - stats['Mean']) / 3
    zone_a_upper = stats['Mean'] + 2 * sigma
    zone_a_lower = stats['Mean'] - 2 * sigma

    # Detect upper violations
    in_upper_a = data['value'] > zone_a_upper
    rolling_sum = in_upper_a.rolling(window=3).sum()
    upper_violations = rolling_sum >= 2

    # Similar for lower...
    # Return obs_ids
```

### Implementation
```python
# processbehavior/western_electric.py

"""
Western Electric rules detection.

Simple function-based implementation of standard control chart rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_zones(center: float, sigma: float) -> Dict[str, tuple]:
    """
    Calculate zone boundaries.

    Returns
    -------
    dict
        Zone boundaries: {'A_upper': (2σ, 3σ), 'B_upper': (1σ, 2σ), etc.}
    """
    return {
        'A_upper': (center + 2*sigma, center + 3*sigma),
        'A_lower': (center - 3*sigma, center - 2*sigma),
        'B_upper': (center + sigma, center + 2*sigma),
        'B_lower': (center - 2*sigma, center - sigma),
        'C_upper': (center, center + sigma),
        'C_lower': (center - sigma, center),
    }


def detect_rule_1(
    data: pd.DataFrame,
    stats: dict,
    value_col: str = 'mean'
) -> pd.DataFrame:
    """Rule 1: Point beyond control limits."""
    violations = data[
        (data[value_col] > stats['ucl']) |
        (data[value_col] < stats['lcl'])
    ].copy()

    violations['rule'] = 'rule_1'
    violations['description'] = 'Beyond control limits'
    return violations[['rule', 'description']]


def detect_rule_2(
    data: pd.DataFrame,
    stats: dict,
    value_col: str = 'mean'
) -> pd.DataFrame:
    """Rule 2: 2 of 3 consecutive points in Zone A."""
    center = stats['Mean']
    sigma = (stats['ucl'] - center) / 3

    zones = calculate_zones(center, sigma)

    # Upper Zone A
    in_upper_a = (
        (data[value_col] > zones['A_upper'][0]) &
        (data[value_col] <= zones['A_upper'][1])
    )

    # Lower Zone A
    in_lower_a = (
        (data[value_col] >= zones['A_lower'][0]) &
        (data[value_col] < zones['A_lower'][1])
    )

    # Rolling window check
    upper_count = in_upper_a.rolling(window=3, min_periods=3).sum()
    lower_count = in_lower_a.rolling(window=3, min_periods=3).sum()

    violations = data[
        (upper_count >= 2) | (lower_count >= 2)
    ].copy()

    violations['rule'] = 'rule_2'
    violations['description'] = '2 of 3 in Zone A'
    return violations[['rule', 'description']]


# Similar functions for rules 3-8...
```

### Pros
✅ **Simple to understand** - Each rule is a standalone function
✅ **Easy to test** - Unit test each function independently
✅ **Minimal abstraction** - Direct, clear code
✅ **Quick to implement** - Straightforward approach

### Cons
❌ **Not extensible** - Hard to add custom rules
❌ **Repetitive code** - Similar patterns across rules
❌ **Configuration limited** - Fixed zone definitions
❌ **No rule composition** - Can't combine rules easily

### Use Cases
- Simple, fixed rule detection
- Minimal customization needed
- Quick prototyping

---

## Option 2: Class-Based Rule Registry

### API Design
```python
# User-facing API
result.detect_signals(
    rules=['rule_1', 'rule_2', 'rule_5'],
    enabled=True
)

# Configuration
rules_config = RulesConfiguration(
    rule_1=True,
    rule_2=True,
    rule_3=False,  # Disabled
    zone_widths={'A': (2, 3), 'B': (1, 2), 'C': (0, 1)}
)

result.detect_signals(config=rules_config)

# Custom rules
my_rule = CustomRule(
    name='consecutive_high',
    detector=lambda data, stats: data[data['value'] > stats['Mean']]
)

rules_config.add_rule(my_rule)
```

### Implementation
```python
# processbehavior/rules/rule_base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class RuleViolation:
    """Single rule violation record."""
    obs_id: int | str
    rule_name: str
    rule_number: int
    description: str
    value: float
    center: float
    zone: Optional[str] = None

    def __str__(self):
        return (
            f"Rule {self.rule_number}: {self.description}\n"
            f"  Observation: {self.obs_id}\n"
            f"  Value: {self.value:.3f} (Center: {self.center:.3f})"
        )


class Rule(ABC):
    """Base class for all Western Electric rules."""

    def __init__(self, name: str, number: int, description: str):
        self.name = name
        self.number = number
        self.description = description
        self.enabled = True

    @abstractmethod
    def detect(
        self,
        data: pd.DataFrame,
        stats: dict,
        value_col: str = 'mean'
    ) -> List[RuleViolation]:
        """
        Detect violations of this rule.

        Parameters
        ----------
        data : DataFrame
            Chart data with observations
        stats : dict
            Chart statistics (ucl, lcl, Mean, etc.)
        value_col : str
            Name of value column

        Returns
        -------
        list of RuleViolation
            Detected violations
        """
        pass

    def is_applicable(self, data: pd.DataFrame, stats: dict) -> bool:
        """Check if rule can be applied to this data."""
        return len(data) >= self.min_observations

    @property
    def min_observations(self) -> int:
        """Minimum observations required for this rule."""
        return 1


# processbehavior/rules/standard_rules.py

class Rule1(Rule):
    """Rule 1: Point beyond control limits."""

    def __init__(self):
        super().__init__(
            name='beyond_limits',
            number=1,
            description='Point beyond control limits'
        )

    def detect(self, data, stats, value_col='mean'):
        violations = []

        beyond = data[
            (data[value_col] > stats['ucl']) |
            (data[value_col] < stats['lcl'])
        ]

        for idx, row in beyond.iterrows():
            violations.append(RuleViolation(
                obs_id=idx,
                rule_name=self.name,
                rule_number=self.number,
                description=self.description,
                value=row[value_col],
                center=stats['Mean']
            ))

        return violations


class Rule2(Rule):
    """Rule 2: 2 of 3 consecutive points in Zone A."""

    def __init__(self, zone_widths: dict = None):
        super().__init__(
            name='zone_a_2_of_3',
            number=2,
            description='2 of 3 consecutive points in Zone A'
        )
        self.zone_widths = zone_widths or {'A': (2, 3)}

    @property
    def min_observations(self) -> int:
        return 3

    def detect(self, data, stats, value_col='mean'):
        violations = []
        center = stats['Mean']
        sigma = (stats['ucl'] - center) / 3

        # Calculate Zone A boundaries
        a_lower = self.zone_widths['A'][0]
        a_upper = self.zone_widths['A'][1]

        upper_zone = (center + a_lower*sigma, center + a_upper*sigma)
        lower_zone = (center - a_upper*sigma, center - a_lower*sigma)

        # Detect points in zones
        in_upper = (
            (data[value_col] > upper_zone[0]) &
            (data[value_col] <= upper_zone[1])
        )
        in_lower = (
            (data[value_col] >= lower_zone[0]) &
            (data[value_col] < lower_zone[1])
        )

        # Check consecutive sequences
        for i in range(2, len(data)):
            window_upper = in_upper.iloc[i-2:i+1].sum()
            window_lower = in_lower.iloc[i-2:i+1].sum()

            if window_upper >= 2 or window_lower >= 2:
                violations.append(RuleViolation(
                    obs_id=data.index[i],
                    rule_name=self.name,
                    rule_number=self.number,
                    description=self.description,
                    value=data.iloc[i][value_col],
                    center=center,
                    zone='A'
                ))

        return violations


class Rule5(Rule):
    """Rule 5: 6+ consecutive points trending."""

    def __init__(self):
        super().__init__(
            name='trend',
            number=5,
            description='6+ consecutive points trending'
        )

    @property
    def min_observations(self) -> int:
        return 6

    def detect(self, data, stats, value_col='mean'):
        violations = []

        # Calculate differences
        diffs = data[value_col].diff()

        # Increasing trend
        increasing = (diffs > 0).rolling(window=6, min_periods=6).sum()
        # Decreasing trend
        decreasing = (diffs < 0).rolling(window=6, min_periods=6).sum()

        trend_violations = data[
            (increasing == 6) | (decreasing == 6)
        ]

        for idx, row in trend_violations.iterrows():
            violations.append(RuleViolation(
                obs_id=idx,
                rule_name=self.name,
                rule_number=self.number,
                description=self.description,
                value=row[value_col],
                center=stats['Mean']
            ))

        return violations


# processbehavior/rules/rule_registry.py

class RuleRegistry:
    """
    Registry for managing and applying Western Electric rules.

    Examples
    --------
    >>> registry = RuleRegistry()
    >>> registry.register(Rule1())
    >>> registry.register(Rule2())
    >>> violations = registry.apply_all(data, stats)
    """

    def __init__(self):
        self._rules: Dict[str, Rule] = {}
        self._load_standard_rules()

    def _load_standard_rules(self):
        """Register all standard Western Electric rules."""
        self.register(Rule1())
        self.register(Rule2())
        self.register(Rule3())
        self.register(Rule4())
        self.register(Rule5())
        self.register(Rule6())
        self.register(Rule7())
        self.register(Rule8())

    def register(self, rule: Rule):
        """Add a rule to the registry."""
        self._rules[rule.name] = rule

    def unregister(self, rule_name: str):
        """Remove a rule from the registry."""
        if rule_name in self._rules:
            del self._rules[rule_name]

    def get_rule(self, rule_name: str) -> Rule:
        """Get a specific rule."""
        if rule_name not in self._rules:
            available = list(self._rules.keys())
            raise ValueError(
                f"Rule '{rule_name}' not found.\n"
                f"Available rules: {available}"
            )
        return self._rules[rule_name]

    def apply_all(
        self,
        data: pd.DataFrame,
        stats: dict,
        enabled_only: bool = True
    ) -> List[RuleViolation]:
        """Apply all registered rules."""
        all_violations = []

        for rule in self._rules.values():
            if enabled_only and not rule.enabled:
                continue

            if not rule.is_applicable(data, stats):
                logger.warning(
                    f"Skipping {rule.name}: insufficient observations "
                    f"(need {rule.min_observations}, have {len(data)})"
                )
                continue

            violations = rule.detect(data, stats)
            all_violations.extend(violations)

        return all_violations

    def apply_rules(
        self,
        rule_names: List[str],
        data: pd.DataFrame,
        stats: dict
    ) -> List[RuleViolation]:
        """Apply specific rules."""
        violations = []

        for name in rule_names:
            rule = self.get_rule(name)
            if rule.is_applicable(data, stats):
                violations.extend(rule.detect(data, stats))

        return violations


# processbehavior/rules/signal_result.py

class SignalResult:
    """
    Container for signal detection results.

    Provides easy access to violations by rule, observation, and zone.

    Examples
    --------
    >>> signals = result.detect_signals()
    >>> signals.summary  # Human-readable summary
    >>> signals.by_rule['rule_2']  # All Rule 2 violations
    >>> signals.to_dataframe()  # All violations as DataFrame
    """

    def __init__(self, violations: List[RuleViolation], chart_name: str):
        self.violations = violations
        self.chart_name = chart_name
        self._by_rule = None
        self._by_obs = None

    @property
    def count(self) -> int:
        """Total number of violations."""
        return len(self.violations)

    @property
    def has_signals(self) -> bool:
        """Whether any signals were detected."""
        return self.count > 0

    @property
    def by_rule(self) -> Dict[str, List[RuleViolation]]:
        """Group violations by rule."""
        if self._by_rule is None:
            self._by_rule = {}
            for v in self.violations:
                if v.rule_name not in self._by_rule:
                    self._by_rule[v.rule_name] = []
                self._by_rule[v.rule_name].append(v)
        return self._by_rule

    @property
    def by_observation(self) -> Dict[int | str, List[RuleViolation]]:
        """Group violations by observation ID."""
        if self._by_obs is None:
            self._by_obs = {}
            for v in self.violations:
                if v.obs_id not in self._by_obs:
                    self._by_obs[v.obs_id] = []
                self._by_obs[v.obs_id].append(v)
        return self._by_obs

    def to_dataframe(self) -> pd.DataFrame:
        """Convert violations to DataFrame."""
        if not self.violations:
            return pd.DataFrame()

        records = []
        for v in self.violations:
            records.append({
                'obs_id': v.obs_id,
                'rule_name': v.rule_name,
                'rule_number': v.rule_number,
                'description': v.description,
                'value': v.value,
                'center': v.center,
                'zone': v.zone
            })

        return pd.DataFrame(records)

    @property
    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.has_signals:
            return f"✓ No signals detected in {self.chart_name}"

        lines = [
            f"\n{'='*70}",
            f"Signal Detection Summary: {self.chart_name}",
            f"{'='*70}",
            f"Total violations: {self.count}",
            ""
        ]

        for rule_name, rule_violations in self.by_rule.items():
            lines.append(f"\n{rule_name}: {len(rule_violations)} violations")
            for v in rule_violations[:5]:  # Show first 5
                lines.append(f"  • Obs {v.obs_id}: {v.value:.3f}")

            if len(rule_violations) > 5:
                lines.append(f"  ... and {len(rule_violations) - 5} more")

        lines.append(f"\n{'='*70}\n")
        return '\n'.join(lines)

    def __repr__(self):
        return f"SignalResult(violations={self.count}, chart='{self.chart_name}')"
```

### Pros
✅ **Highly extensible** - Easy to add custom rules
✅ **Configurable** - Enable/disable rules, adjust parameters
✅ **Well-organized** - Clear separation of concerns
✅ **Type-safe** - Dataclasses and type hints
✅ **Testable** - Each rule is independently testable
✅ **Rich results** - SignalResult provides multiple views

### Cons
⚠️ **More complex** - Higher abstraction level
⚠️ **Boilerplate** - Each rule needs a class
⚠️ **Performance** - Object creation overhead

### Use Cases
- Production systems needing configurability
- Custom rule requirements
- Quality control applications

---

## Option 3: Declarative Rule Engine with Fluent API ⭐ **RECOMMENDED**

### API Design
```python
# Dead simple - use defaults
signals = result.detect_signals()

# Choose specific rules
signals = result.detect_signals(rules=['rule_1', 'rule_2', 'rule_5'])

# Configure rules with fluent API
signals = result.detect_signals(
    rules=RuleSet()
        .beyond_limits()
        .zone_a(consecutive=2, window=3)
        .trend(length=6)
        .custom(
            name='my_rule',
            detector=lambda data, stats: custom_logic(data, stats)
        )
)

# Or use presets
signals = result.detect_signals(rules='standard')  # Rules 1-4
signals = result.detect_signals(rules='extended')  # Rules 1-8
signals = result.detect_signals(rules='all')

# Configuration object
config = SignalConfig(
    enabled_rules=['rule_1', 'rule_2'],
    zone_widths={'A': (2, 3), 'B': (1, 2), 'C': (0, 1)},
    min_observations=20,
    ignore_first_n=5  # Don't flag first 5 points
)

signals = result.detect_signals(config=config)

# Access results - multiple ways
print(signals.summary)  # Human-readable
signals.by_rule['rule_2']  # Violations for Rule 2
signals.flagged_observations  # Set of obs_ids
signals.to_dataframe()  # DataFrame of all violations
signals.plot()  # Visual representation

# Integration with plotting
fig = result.plot(highlight_signals=signals)

# Export
signals.to_excel('violations.xlsx')
signals.to_json('violations.json')
```

### Implementation Architecture

```python
# processbehavior/signals/__init__.py
"""
Signal detection framework for control charts.

Provides flexible, declarative API for detecting Western Electric rules
and custom patterns in process behavior data.
"""

from .config import SignalConfig, RuleSet
from .detector import SignalDetector
from .result import SignalResult

__all__ = ['SignalConfig', 'RuleSet', 'SignalDetector', 'SignalResult']
```

#### Core Configuration
```python
# processbehavior/signals/config.py

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

@dataclass
class ZoneDefinition:
    """Zone boundary definition in standard deviations."""
    A: tuple[float, float] = (2.0, 3.0)
    B: tuple[float, float] = (1.0, 2.0)
    C: tuple[float, float] = (0.0, 1.0)

    def get_boundaries(self, center: float, sigma: float) -> dict:
        """Calculate actual zone boundaries."""
        return {
            'A_upper': (center + self.A[0]*sigma, center + self.A[1]*sigma),
            'A_lower': (center - self.A[1]*sigma, center - self.A[0]*sigma),
            'B_upper': (center + self.B[0]*sigma, center + self.B[1]*sigma),
            'B_lower': (center - self.B[1]*sigma, center - self.B[0]*sigma),
            'C_upper': (center + self.C[0]*sigma, center + self.C[1]*sigma),
            'C_lower': (center - self.C[1]*sigma, center - self.C[0]*sigma),
        }


@dataclass
class RuleDefinition:
    """Definition of a single detection rule."""
    name: str
    number: int
    description: str
    detector: Callable
    min_observations: int = 1
    enabled: bool = True
    parameters: dict = field(default_factory=dict)


@dataclass
class SignalConfig:
    """
    Configuration for signal detection.

    Examples
    --------
    Standard configuration:

    >>> config = SignalConfig(rules='standard')

    Custom configuration:

    >>> config = SignalConfig(
    ...     enabled_rules=['rule_1', 'rule_2', 'rule_5'],
    ...     zone_definition=ZoneDefinition(A=(2.5, 3.5)),
    ...     min_observations=30
    ... )
    """

    # Which rules to apply
    enabled_rules: List[str] | Literal['standard', 'extended', 'all'] = 'standard'

    # Zone definitions
    zone_definition: ZoneDefinition = field(default_factory=ZoneDefinition)

    # Filtering options
    min_observations: int = 20
    ignore_first_n: int = 0
    ignore_last_n: int = 0

    # Performance options
    use_vectorized: bool = True

    # Custom rules
    custom_rules: List[RuleDefinition] = field(default_factory=list)

    def __post_init__(self):
        """Resolve rule presets."""
        if isinstance(self.enabled_rules, str):
            self.enabled_rules = self._resolve_preset(self.enabled_rules)

    def _resolve_preset(self, preset: str) -> List[str]:
        """Convert preset name to rule list."""
        presets = {
            'standard': ['rule_1', 'rule_2', 'rule_3', 'rule_4'],
            'extended': [f'rule_{i}' for i in range(1, 9)],
            'all': [f'rule_{i}' for i in range(1, 9)],
        }

        if preset not in presets:
            raise ValueError(
                f"Unknown preset '{preset}'.\n"
                f"Valid presets: {list(presets.keys())}"
            )

        return presets[preset]


class RuleSet:
    """
    Fluent API for building rule configurations.

    Examples
    --------
    >>> rules = (
    ...     RuleSet()
    ...         .beyond_limits()
    ...         .zone_a(consecutive=2, window=3)
    ...         .trend(length=6, direction='both')
    ... )
    """

    def __init__(self):
        self._rules: List[str] = []
        self._parameters: Dict[str, dict] = {}

    def beyond_limits(self) -> 'RuleSet':
        """Add Rule 1: Points beyond control limits."""
        self._rules.append('rule_1')
        return self

    def zone_a(self, consecutive: int = 2, window: int = 3) -> 'RuleSet':
        """Add Rule 2: Consecutive points in Zone A."""
        self._rules.append('rule_2')
        self._parameters['rule_2'] = {
            'consecutive': consecutive,
            'window': window
        }
        return self

    def zone_b(self, consecutive: int = 4, window: int = 5) -> 'RuleSet':
        """Add Rule 3: Consecutive points in Zone B."""
        self._rules.append('rule_3')
        self._parameters['rule_3'] = {
            'consecutive': consecutive,
            'window': window
        }
        return self

    def run(self, length: int = 8) -> 'RuleSet':
        """Add Rule 4: Run above/below centerline."""
        self._rules.append('rule_4')
        self._parameters['rule_4'] = {'length': length}
        return self

    def trend(
        self,
        length: int = 6,
        direction: Literal['up', 'down', 'both'] = 'both'
    ) -> 'RuleSet':
        """Add Rule 5: Trending points."""
        self._rules.append('rule_5')
        self._parameters['rule_5'] = {
            'length': length,
            'direction': direction
        }
        return self

    def oscillation(self, length: int = 14) -> 'RuleSet':
        """Add Rule 6: Alternating pattern."""
        self._rules.append('rule_6')
        self._parameters['rule_6'] = {'length': length}
        return self

    def reduced_variation(self, length: int = 15) -> 'RuleSet':
        """Add Rule 7: Points in Zone C (low variation)."""
        self._rules.append('rule_7')
        self._parameters['rule_7'] = {'length': length}
        return self

    def avoiding_center(self, length: int = 8) -> 'RuleSet':
        """Add Rule 8: Points avoiding Zone C."""
        self._rules.append('rule_8')
        self._parameters['rule_8'] = {'length': length}
        return self

    def custom(
        self,
        name: str,
        detector: Callable,
        min_observations: int = 1
    ) -> 'RuleSet':
        """Add a custom detection rule."""
        self._rules.append(name)
        self._parameters[name] = {
            'detector': detector,
            'min_observations': min_observations
        }
        return self

    def to_config(self) -> SignalConfig:
        """Convert to SignalConfig."""
        return SignalConfig(enabled_rules=self._rules)

    def get_rules(self) -> List[str]:
        """Get list of enabled rules."""
        return self._rules.copy()
```

#### Rule Detectors (Vectorized)
```python
# processbehavior/signals/detectors.py

"""
Vectorized rule detection functions.

All detectors use pandas/numpy for performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def detect_beyond_limits(
    data: pd.DataFrame,
    stats: dict,
    value_col: str
) -> pd.Series:
    """
    Rule 1: Points beyond control limits.

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]
    return (values > stats['ucl']) | (values < stats['lcl'])


def detect_zone_a_2_of_3(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    zones: dict
) -> pd.Series:
    """
    Rule 2: 2 of 3 consecutive points in Zone A.

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]

    # Determine which zone each point is in
    in_upper_a = (values > zones['A_upper'][0]) & (values <= zones['A_upper'][1])
    in_lower_a = (values >= zones['A_lower'][0]) & (values < zones['A_lower'][1])

    # Rolling window count
    upper_count = in_upper_a.astype(int).rolling(window=3, min_periods=3).sum()
    lower_count = in_lower_a.astype(int).rolling(window=3, min_periods=3).sum()

    # Flag if 2 or more in window
    violations = (upper_count >= 2) | (lower_count >= 2)

    return violations.fillna(False)


def detect_zone_b_4_of_5(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    zones: dict
) -> pd.Series:
    """Rule 3: 4 of 5 consecutive points in Zone B or beyond."""
    values = data[value_col]

    # In Zone B or beyond (same side)
    upper = values > zones['B_upper'][0]
    lower = values < zones['B_lower'][1]

    upper_count = upper.astype(int).rolling(window=5, min_periods=5).sum()
    lower_count = lower.astype(int).rolling(window=5, min_periods=5).sum()

    violations = (upper_count >= 4) | (lower_count >= 4)
    return violations.fillna(False)


def detect_run(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    length: int = 8
) -> pd.Series:
    """Rule 4: Run of points on same side of centerline."""
    values = data[value_col]
    center = stats['Mean']

    # Above or below center
    above = (values > center).astype(int)
    below = (values < center).astype(int)

    # Count consecutive
    above_streak = above.rolling(window=length, min_periods=length).sum()
    below_streak = below.rolling(window=length, min_periods=length).sum()

    violations = (above_streak == length) | (below_streak == length)
    return violations.fillna(False)


def detect_trend(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    length: int = 6,
    direction: str = 'both'
) -> pd.Series:
    """Rule 5: Trending sequence."""
    values = data[value_col]

    # Calculate differences
    diffs = values.diff()

    if direction in ['up', 'both']:
        increasing = (diffs > 0).astype(int).rolling(
            window=length, min_periods=length
        ).sum()
        up_trend = increasing == length
    else:
        up_trend = pd.Series(False, index=values.index)

    if direction in ['down', 'both']:
        decreasing = (diffs < 0).astype(int).rolling(
            window=length, min_periods=length
        ).sum()
        down_trend = decreasing == length
    else:
        down_trend = pd.Series(False, index=values.index)

    violations = up_trend | down_trend
    return violations.fillna(False)


def detect_oscillation(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    length: int = 14
) -> pd.Series:
    """Rule 6: Alternating up-down pattern."""
    values = data[value_col]

    # Detect direction changes
    diffs = values.diff()
    changes = (diffs * diffs.shift(1) < 0).astype(int)

    # Count consecutive changes
    change_count = changes.rolling(window=length-1, min_periods=length-1).sum()

    violations = change_count == (length - 1)
    return violations.fillna(False)


def detect_reduced_variation(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    zones: dict,
    length: int = 15
) -> pd.Series:
    """Rule 7: Points within Zone C (reduced variation)."""
    values = data[value_col]

    # Within Zone C
    in_zone_c = (
        (values > zones['C_lower'][0]) &
        (values < zones['C_upper'][1])
    )

    in_c_count = in_zone_c.astype(int).rolling(
        window=length, min_periods=length
    ).sum()

    violations = in_c_count == length
    return violations.fillna(False)


def detect_avoiding_center(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    zones: dict,
    length: int = 8
) -> pd.Series:
    """Rule 8: Points avoiding Zone C."""
    values = data[value_col]

    # Outside Zone C
    outside_c = (
        (values <= zones['C_lower'][0]) |
        (values >= zones['C_upper'][1])
    )

    outside_count = outside_c.astype(int).rolling(
        window=length, min_periods=length
    ).sum()

    violations = outside_count == length
    return violations.fillna(False)
```

#### Signal Detector
```python
# processbehavior/signals/detector.py

from typing import Dict, Optional
import pandas as pd
import logging

from .config import SignalConfig, ZoneDefinition
from .detectors import *
from .result import SignalResult

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
            Chart statistics (ucl, lcl, Mean, etc.)
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
        """
        config = config or SignalConfig()

        # Validate data
        self._validate_inputs(data, stats, config)

        # Calculate zones
        center = stats['Mean']
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
            min_obs = self._get_min_observations(rule_name, config)
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

        required_stats = ['ucl', 'lcl', 'Mean']
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

    def _get_min_observations(
        self,
        rule_name: str,
        config: SignalConfig
    ) -> int:
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
                        'center': stats['Mean'],
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
```

#### Signal Result
```python
# processbehavior/signals/result.py

import pandas as pd
from typing import Dict, List, Set, Optional


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
    def flagged_observations(self) -> Set:
        """Set of observation IDs that violated any rule."""
        if self.violations.empty:
            return set()
        return set(self.violations['obs_id'].unique())

    @property
    def by_rule(self) -> Dict[str, pd.DataFrame]:
        """Group violations by rule."""
        if self.violations.empty:
            return {}

        return {
            rule: group
            for rule, group in self.violations.groupby('rule_name')
        }

    @property
    def by_observation(self) -> Dict:
        """Group violations by observation ID."""
        if self.violations.empty:
            return {}

        return {
            obs_id: group
            for obs_id, group in self.violations.groupby('obs_id')
        }

    def get_rule_violations(self, rule_name: str) -> pd.DataFrame:
        """Get all violations for a specific rule."""
        if self.violations.empty:
            return pd.DataFrame()

        return self.violations[
            self.violations['rule_name'] == rule_name
        ].copy()

    def get_observation_violations(self, obs_id) -> pd.DataFrame:
        """Get all rule violations for a specific observation."""
        if self.violations.empty:
            return pd.DataFrame()

        return self.violations[
            self.violations['obs_id'] == obs_id
        ].copy()

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        if not self.has_signals:
            return f"✓ No signals detected in {self.chart_name}"

        lines = [
            f"\n{'='*70}",
            f"Signal Detection Summary: {self.chart_name}",
            f"{'='*70}",
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

        lines.append(f"\n{'='*70}\n")
        return '\n'.join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Get violations as DataFrame."""
        return self.violations.copy()

    def to_excel(self, filepath: str):
        """Export violations to Excel."""
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
                'Value': [self.count, len(self.flagged_observations), self.chart_name]
            }
            pd.DataFrame(summary_data).to_excel(
                writer,
                sheet_name='Summary',
                index=False
            )

        logger.info(f"✓ Exported violations to: {filepath}")

    def to_json(self, filepath: str):
        """Export violations to JSON."""
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
```

#### Integration with AnalysisResult
```python
# Add to processbehavior/analysis_result.py

class AnalysisResult:
    # ... existing code ...

    def detect_signals(
        self,
        chart: Optional[str] = None,
        rules: Optional[str | List[str] | RuleSet] = None,
        config: Optional[SignalConfig] = None,
        **kwargs
    ) -> SignalResult | Dict[str, SignalResult]:
        """
        Detect Western Electric rule violations.

        This method applies configurable pattern detection rules to identify
        non-random patterns in control chart data.

        Parameters
        ----------
        chart : str, optional
            Specific chart to analyze. If None, analyzes all charts.
        rules : str, list, or RuleSet, optional
            Rules to apply:
            - 'standard': Rules 1-4 (default)
            - 'extended': Rules 1-8
            - 'all': All available rules
            - List of rule names: ['rule_1', 'rule_2', ...]
            - RuleSet: Fluent API configuration
        config : SignalConfig, optional
            Advanced configuration object
        **kwargs
            Additional parameters passed to SignalConfig

        Returns
        -------
        SignalResult or dict of SignalResult
            If chart specified: single SignalResult
            If no chart: dict mapping chart names to SignalResults

        Examples
        --------
        Simple usage (standard rules):

        >>> signals = result.detect_signals()
        >>> print(signals.summary)

        Specific chart and rules:

        >>> signals = result.detect_signals(
        ...     chart='Xbar',
        ...     rules=['rule_1', 'rule_2', 'rule_5']
        ... )

        Using fluent API:

        >>> signals = result.detect_signals(
        ...     rules=RuleSet()
        ...         .beyond_limits()
        ...         .zone_a(consecutive=2, window=3)
        ...         .trend(length=6)
        ... )

        Full configuration:

        >>> config = SignalConfig(
        ...     enabled_rules=['rule_1', 'rule_2'],
        ...     min_observations=30,
        ...     ignore_first_n=5
        ... )
        >>> signals = result.detect_signals(config=config)

        Access violations:

        >>> signals.by_rule['rule_2']  # Rule 2 violations
        >>> signals.flagged_observations  # Set of obs_ids
        >>> signals.to_excel('violations.xlsx')
        """
        from .signals import SignalDetector, SignalConfig, RuleSet

        # Build configuration
        if config is None:
            config = SignalConfig()

            # Handle rules parameter
            if rules is not None:
                if isinstance(rules, RuleSet):
                    config.enabled_rules = rules.get_rules()
                elif isinstance(rules, str):
                    config.enabled_rules = rules
                elif isinstance(rules, list):
                    config.enabled_rules = rules

            # Apply kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Initialize detector
        detector = SignalDetector()

        # Detect on specific chart or all charts
        if chart:
            if chart not in self.charts:
                raise ValueError(
                    f"Chart '{chart}' not found.\n"
                    f"Available: {self.all_charts}"
                )

            chart_info = self.charts[chart]
            return detector.detect(
                data=chart_info['data'],
                stats=chart_info['statistics'],
                config=config,
                chart_name=chart
            )

        else:
            # Detect on all charts
            results = {}
            for chart_name, chart_info in self.charts.items():
                results[chart_name] = detector.detect(
                    data=chart_info['data'],
                    stats=chart_info['statistics'],
                    config=config,
                    chart_name=chart_name
                )

            return results
```

### Pros
✅ **Best User Experience** - Multiple APIs (simple → advanced)
✅ **Highly Configurable** - Presets + custom rules + fluent API
✅ **Performance** - Vectorized operations
✅ **Extensible** - Easy to add custom rules
✅ **Rich Results** - Multiple access patterns, export options
✅ **Pythonic** - Follows Hadley philosophy perfectly
✅ **Type-Safe** - Full type hints
✅ **Observable** - Clear tracking of violations
✅ **Well-Tested** - Vectorized functions easy to unit test

### Cons
⚠️ **Most Complex** - Most code to implement
⚠️ **Learning Curve** - Multiple APIs to understand

### Use Cases
✅ Production quality control systems
✅ Flexible configuration needs
✅ Custom rule requirements
✅ Long-term maintainability

---

## Comparison Matrix

| Feature | Option 1 | Option 2 | **Option 3** |
|---------|----------|----------|--------------|
| **Ease of Use** | ✅✅ | ✅ | ✅✅✅ |
| **Configurability** | ❌ | ✅✅ | ✅✅✅ |
| **Extensibility** | ❌ | ✅✅ | ✅✅ |
| **Performance** | ✅ | ⚠️ | ✅✅ |
| **Code Complexity** | Low | Medium | High |
| **Result Richness** | ⚠️ | ✅✅ | ✅✅✅ |
| **Type Safety** | ✅ | ✅✅ | ✅✅ |
| **Custom Rules** | ❌ | ✅✅ | ✅✅✅ |
| **API Flexibility** | ⚠️ | ✅ | ✅✅✅ |
| **Pythonic** | ✅ | ✅✅ | ✅✅✅ |

---

## Recommendation: Option 3 ⭐

### Why This is Best

1. **Progressive Disclosure** - Perfect Pythonic Hadley philosophy
   ```python
   # Beginner: Just works
   signals = result.detect_signals()

   # Intermediate: Choose rules
   signals = result.detect_signals(rules=['rule_1', 'rule_2'])

   # Advanced: Full control
   signals = result.detect_signals(
       rules=RuleSet().beyond_limits().zone_a().trend(),
       config=SignalConfig(min_observations=30)
   )
   ```

2. **Observable Results** - Complete violation tracking
   ```python
   signals.flagged_observations  # Set of obs_ids
   signals.by_rule['rule_2']  # DataFrame of Rule 2 violations
   signals.to_excel('report.xlsx')  # Export
   ```

3. **Configurable** - Multiple levels
   - Presets: `'standard'`, `'extended'`, `'all'`
   - Lists: `['rule_1', 'rule_2']`
   - Fluent API: `RuleSet().beyond_limits().trend()`
   - Full config: `SignalConfig(...)`

4. **Extensible** - Custom rules easy
   ```python
   result.detect_signals(
       rules=RuleSet()
           .beyond_limits()
           .custom(
               name='my_pattern',
               detector=my_detector_function
           )
   )
   ```

5. **Performance** - Vectorized operations via pandas/numpy

6. **Integration** - Works with plotting
   ```python
   signals = result.detect_signals()
   fig = result.plot(highlight_signals=signals)
   ```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Create `processbehavior/signals/` module
2. Implement `SignalConfig` and `ZoneDefinition`
3. Implement `RuleSet` fluent API
4. Write vectorized detectors for Rules 1-4

### Phase 2: Additional Rules (Week 2)
1. Implement Rules 5-8
2. Create `SignalDetector` orchestration
3. Implement `SignalResult` class
4. Add integration to `AnalysisResult`

### Phase 3: Testing & Validation (Week 3)
1. Unit tests for each detector
2. Integration tests
3. Validate against known datasets
4. Performance benchmarking

### Phase 4: Documentation & Examples (Week 4)
1. Comprehensive docstrings
2. Usage examples
3. Tutorial notebook
4. Update main documentation

---

## Usage Examples

### Basic Usage
```python
from processbehavior import ProcessDataFrame

# Analyze data
pdf = ProcessDataFrame('quality_data.csv')
result = pdf.analyze(
    response_var='measurement',
    factors=['operator', 'machine'],
    time='hour'
)

# Detect signals (standard rules)
signals = result.detect_signals()
print(signals.summary)

# Export violations
signals.to_excel('violations.xlsx')
```

### Intermediate Usage
```python
# Specific rules
signals = result.detect_signals(
    chart='Xbar',
    rules=['rule_1', 'rule_2', 'rule_5']
)

# Check specific rule
rule_2_violations = signals.by_rule['rule_2']
print(f"Rule 2 flagged {len(rule_2_violations)} observations")

# Get flagged obs_ids
flagged = signals.flagged_observations
print(f"Flagged observations: {flagged}")
```

### Advanced Usage
```python
# Fluent API
signals = result.detect_signals(
    rules=RuleSet()
        .beyond_limits()
        .zone_a(consecutive=2, window=3)
        .zone_b(consecutive=4, window=5)
        .trend(length=6, direction='both')
        .custom(
            name='stability_check',
            detector=my_stability_detector
        )
)

# Full configuration
config = SignalConfig(
    enabled_rules='extended',
    zone_definition=ZoneDefinition(A=(2.5, 3.5)),
    min_observations=30,
    ignore_first_n=5
)

signals = result.detect_signals(config=config)

# Access results multiple ways
signals.violations  # DataFrame
signals.by_rule  # Dict by rule
signals.by_observation  # Dict by obs_id
```

### Integration with Plotting
```python
# Detect signals
signals = result.detect_signals()

# Plot with highlighted violations
fig = result.plot(highlight_signals=signals)
fig.save_html('chart_with_signals.html')

# Or get flagged obs_ids for custom plotting
flagged_ids = signals.flagged_observations
# Use in custom visualization...
```

---

## Summary

**Recommendation: Option 3 (Declarative Rule Engine with Fluent API)**

This design provides:
- ✅ **Best UX** - Simple by default, powerful when needed
- ✅ **Observable** - Clear tracking of violations and obs_ids
- ✅ **Configurable** - Multiple configuration levels
- ✅ **Extensible** - Custom rules support
- ✅ **Pythonic** - Follows Hadley philosophy
- ✅ **Performant** - Vectorized operations
- ✅ **Future-proof** - Easy to enhance

The three-tier API (simple → fluent → config) ensures users can start simple and grow into advanced features naturally.
