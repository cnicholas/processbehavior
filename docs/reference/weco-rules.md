# Western Electric Rules

The Western Electric (WECO) rules are a set of decision rules for detecting out-of-control conditions in process behavior charts. ProcessBehavior implements all 8 classic rules.

## Zone Definitions

Control charts are divided into zones based on standard deviations from the centerline:

```
UCL ─────────────────────── +3σ
      Zone A (upper)
      ─────────────────────── +2σ
      Zone B (upper)
      ─────────────────────── +1σ
      Zone C (upper)
CL  ═════════════════════════ 0
      Zone C (lower)
      ─────────────────────── -1σ
      Zone B (lower)
      ─────────────────────── -2σ
      Zone A (lower)
LCL ─────────────────────── -3σ
```

## The Eight Rules

Each false alarm rate below is the chance that the pattern appears by chance in one window of
the rule's length, for independent, normally distributed points from a stable process. Over a
whole chart these chances add up; see [Signals on pure noise, per chart](#signals-on-pure-noise-per-chart).

### Rule 1: Beyond Limits

**Pattern**: Single point beyond the 3-sigma control limits.

**Detection**: |X - CL| > 3σ

**False Alarm Rate**: 0.27% per point (1 in 370)

**Interpretation**: The strongest single signal and a reason to look for a special cause. On a
stable process about one point in 370 falls outside the limits by chance, so a long chart will
show some.

**Example Causes**:
- Equipment malfunction
- Measurement error
- Material defect
- Operator error

```python
# Always included in any rule set
signals = result.detect_signals(rules='standard')
```

### Rule 2: Zone A (2 of 3)

**Pattern**: 2 of 3 consecutive points in Zone A or beyond (same side).

**Detection**: In a window of 3 consecutive points, 2 or more are beyond 2σ on the same side.

**False Alarm Rate**: 0.31% per 3-point window

**Interpretation**: Suggests a shift. Two of three points beyond 2σ on one side is unusual.

**Example Causes**:
- Gradual equipment drift
- Environmental change
- Material batch variation

```python
from processbehavior.signals import RuleSet

rules = RuleSet().zone_a(consecutive=2)```

### Rule 3: Zone B (4 of 5)

**Pattern**: 4 of 5 consecutive points in Zone B or beyond (same side).

**Detection**: In a window of 5 consecutive points, 4 or more are beyond 1σ on the same side.

**False Alarm Rate**: 0.55% per 5-point window

**Interpretation**: Suggests a shift, smaller than one Rule 2 picks up.

**Example Causes**:
- Small sustained shift
- Calibration drift
- Slow process change

```python
rules = RuleSet().zone_b(consecutive=4)```

### Rule 4: Run

**Pattern**: 8 or more consecutive points on the same side of the centerline.

**Detection**: 8+ points all above CL or all below CL.

**False Alarm Rate**: 0.78% per 8-point window

**Interpretation**: Suggests a sustained shift. Even small shifts (< 1σ) will eventually produce runs.

**Example Causes**:
- Process adjustment
- Tool change
- New material lot
- Seasonal effect

```python
rules = RuleSet().run(length=8)
# More sensitive (shorter run)
rules = RuleSet().run(length=7)```

### Rule 5: Trend

**Pattern**: 6 or more consecutive points continuously increasing or decreasing.

**Detection**: 6+ points where each is higher (or lower) than the previous.

**False Alarm Rate**: 0.28% per 6-point window

**Interpretation**: Suggests drift. Look for a cause that builds over time.

**Example Causes**:
- Tool wear
- Chemical degradation
- Temperature change
- Fatigue effects

```python
rules = RuleSet().trend(length=6)
# More sensitive
rules = RuleSet().trend(length=5)```

### Rule 6: Oscillation

**Pattern**: 14 or more consecutive points alternating up and down.

**Detection**: 14+ points where each alternates direction from the previous.

**False Alarm Rate**: 0.46% per 14-point window

**Interpretation**: Suggests over-control or two alternating streams: each adjustment can cause the next deviation.

**Example Causes**:
- Over-adjustment (tampering)
- Two alternating streams
- Measurement round-off
- Systematic sampling issue

```python
rules = RuleSet().oscillation(length=14)```

### Rule 7: Hugging Center

**Pattern**: 15 or more consecutive points in Zone C (within 1σ of centerline).

**Detection**: 15+ points all within ±1σ of the centerline.

**False Alarm Rate**: 0.33% per 15-point window

**Interpretation**: Variation has been reduced. Could be good (process improvement) or suspicious (data manipulation, stratification).

**Example Causes**:
- Process improvement
- Incorrect subgrouping
- Mixed product/operators
- Data averaging or smoothing

```python
rules = RuleSet().reduced_variation(length=15)```

### Rule 8: Avoiding Center

**Pattern**: 8 or more consecutive points avoiding Zone C (all beyond 1σ).

**Detection**: 8+ points all outside ±1σ from centerline.

**False Alarm Rate**: 0.010% per 8-point window

**Interpretation**: Suggests a mixture of two distributions, or excessive variation.

**Example Causes**:
- Mixture of two processes
- Systematic over-adjustment
- Two distinct populations
- Subgroup selection issue

```python
rules = RuleSet().avoiding_center(length=8)```

## Rule Sets

### Standard Rules (1-4)

The most commonly used rules with lower false alarm rates.

```python
signals = result.detect_signals(rules='standard')
```

Best for:
- Routine monitoring
- Limited investigation resources
- Production environments

### Extended Rules (1-8)

All 8 rules for maximum sensitivity.

```python
signals = result.detect_signals(rules='extended')
```

Best for:
- Detailed analysis
- Research environments
- Critical processes
- When you can investigate false alarms

### Custom Rules

Select specific rules for your needs.

```python
from processbehavior.signals import RuleSet

# Just limits and trends
rules = RuleSet().beyond_limits().trend()
# Limits and runs with custom length
rules = RuleSet().beyond_limits().run(length=7)
# Full configuration
rules = (
    RuleSet()
    .beyond_limits()
    .zone_a(consecutive=2)
    .run(length=8)
    .trend(length=5)
    )

signals = result.detect_signals(rules=rules)
```

## Rule Applicability

Not all rules apply to all chart types:

| Rule | Xbar | S | X | mR |
|------|------|---|-----|---|
| 1: Beyond Limits | ✅ | ✅ | ✅ | ✅ |
| 2: Zone A | ❌ | ❌ | ✅ | ❌ |
| 3: Zone B | ❌ | ❌ | ✅ | ❌ |
| 4: Run | ❌ | ❌ | ✅ | ❌ |
| 5: Trend | ❌ | ❌ | ✅ | ❌ |
| 6: Oscillation | ❌ | ❌ | ✅ | ❌ |
| 7: Hugging Center | ❌ | ❌ | ✅ | ❌ |
| 8: Avoiding Center | ❌ | ❌ | ✅ | ❌ |

**Reason**: Xbar and S charts compare subgroups, which may not be time-ordered. Rules 2-8
assume sequential, independent points from a symmetric distribution. X charts meet that;
mR charts do not, because successive moving ranges share a reading and their distribution
is skewed, so the zone and run rules fire on pure noise. Wheeler reads the mR chart for
points above its upper limit.

These are the defaults. `rules='standard'`, `'extended'`, `'all'` or an explicit list apply
as given, on every chart.

## Sensitivity vs. False Alarms

More rules catch smaller and slower changes sooner, and give more false signals on a stable
process. The X-chart default is all eight. `rules='standard'` (Rules 1-4) or Rule 1 alone gives
fewer false signals and is slower to catch small sustained shifts.

(signals-on-pure-noise-per-chart)=
### Signals on pure noise, per chart

Over a whole chart, the chance of at least one false signal grows with its length. Share of
500 X/mR charts of pure noise (independent normal readings, no signal in them) with at least
one signal:

| Readings | X (Rule 1) | X (Rules 1-4) | X (Rules 1-8, the default) | mR (Rule 1, the default) | mR (all eight rules) |
|---|---|---|---|---|---|
| 25 | 6% | 15% | 20% | 13% | 38% |
| 50 | 12% | 35% | 50% | 33% | 78% |
| 100 | 19% | 61% | 79% | 58% | 96% |

A signal is a reason to look, not proof of a cause; on a long chart some will be noise.
`validation/chance_signals.py` reproduces this table (it is a report, not a test).

## Minimum Observations

There are two minimums, and they answer different questions.

### Structural minimum, per rule

Each rule needs a certain number of points before its pattern can occur at all.
A rule below its minimum is skipped, not failed:

| Rule | Minimum Observations |
|------|---------------------|
| 1 | 1 |
| 2 | 3 |
| 3 | 5 |
| 4 | 8 |
| 5 | 6 |
| 6 | 14 |
| 7 | 15 |
| 8 | 8 |

These live in `SignalDetector.RULE_MIN_OBSERVATIONS` and are not configurable:
a run of eight cannot exist in six points.

### Advisory minimum, per evaluation

`SignalConfig.min_observations` (default 20) is the series length below which the
evaluation as a whole is reported as **partial**, even if every rule that could run
did run. At four points, six of the eight X-chart rules are structurally inert and the
remaining two have very little to work with. Below the threshold the detector emits
a `ProcessBehaviorWarning` naming both numbers, and the result carries the detail:

```python
signals = result.detect_signals(chart='X')
signals.is_partial          # True
signals.evaluation_status   # 'partial'
signals.rules_evaluated     # ['rule_1', 'rule_2']
signals.rules_skipped       # {'rule_3': 'needs 5 observations, have 4', ...}
signals.evaluation_note     # '2 of 8 rules applicable at n=4 (below min_observations=20); skipped: ...'
```

### Reading the summary

A partial evaluation never prints the clean-series checkmark. Compare:

```
✓ No signals detected in X
```

which means every applicable rule ran on an adequate series and none fired, with:

```
⚠ Partial evaluation in X: 2 of 8 rules applicable at n=4 (below min_observations=20);
skipped: rule_3 (needs 5), rule_4 (needs 8), rule_5 (needs 6), rule_6 (needs 14),
rule_7 (needs 15), rule_8 (needs 8). No signals from the rules evaluated.
```

which means most of the examination could not be performed. The second is not an
all-clear. On a short series that is also trending, the one rule that can still fire
(Rule 1) is weakened by the trend itself, because the trend inflates the moving range
that sets the limits. Treat a partial result as "not yet examined", and see the
series-length guidance in the user guide before quoting limits from a short series.

To silence the warning in a pipeline that has already accounted for it:

```python
import warnings
from processbehavior import ProcessBehaviorWarning
warnings.simplefilter('ignore', ProcessBehaviorWarning)
```

## Interpreting Results

```python
signals = result.detect_signals(rules='extended')

# Check if any signals
if signals.has_signals:
    print(f"Found {signals.count} signals")

    # View all violations
    print(signals.violations)

    # Summary by rule
    print(signals.summary)

    # Violations for specific rule
    rule_1 = signals.by_rule.get('rule_1', [])
    print(f"Beyond limits: {len(rule_1)} violations")
```

## Visualizing Violations

```python
# Show all rule violations on chart
fig = result.plot(
    show_zones=True,   # Zone shading helps see violations
    show_rules=True    # All WECO rule violations
)
fig.show()
```

## Historical Note

The Western Electric rules were developed at Bell Telephone Laboratories and published in the *Statistical Quality Control Handbook* (1956). They remain the foundation of modern SPC practice.

Wheeler's contributions include:
- Clarifying the theoretical basis
- Recommending Rule 1 as primary
- Cautioning against over-reliance on extended rules
- Emphasizing understanding over automation

## Best Practices

1. **Start with Rule 1** - The most reliable signal
2. **Add rules incrementally** - Understand each before adding more
3. **Investigate all signals** - Don't ignore or explain away
4. **Document false alarms** - Learn from what wasn't real
5. **Adjust sensitivity to resources** - More rules = more investigation
6. **Consider the process** - Critical processes may need more sensitivity

## References

- Western Electric (1956). *Statistical Quality Control Handbook*
- Wheeler, D.J. (1995). *Advanced Topics in Statistical Process Control*
- Wheeler, D.J. & Chambers, D.S. (1992). *Understanding Statistical Process Control*
