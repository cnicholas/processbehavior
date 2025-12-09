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

### Rule 1: Beyond Limits

**Pattern**: Single point beyond the 3-sigma control limits.

**Detection**: |X - CL| > 3σ

**False Alarm Rate**: 0.27% per point (1 in 370)

**Interpretation**: Almost certainly a special cause. This is the most reliable signal.

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

**False Alarm Rate**: ~0.15% per point

**Interpretation**: Process is likely shifting. Two points near the limits is unusual.

**Example Causes**:
- Gradual equipment drift
- Environmental change
- Material batch variation

```python
from processbehavior.signals import RuleSet

rules = RuleSet().zone_a(consecutive=2).build()
```

### Rule 3: Zone B (4 of 5)

**Pattern**: 4 of 5 consecutive points in Zone B or beyond (same side).

**Detection**: In a window of 5 consecutive points, 4 or more are beyond 1σ on the same side.

**False Alarm Rate**: ~0.28% per point

**Interpretation**: Process is likely shifting, but shift is smaller than Rule 2.

**Example Causes**:
- Small sustained shift
- Calibration drift
- Slow process change

```python
rules = RuleSet().zone_b(consecutive=4).build()
```

### Rule 4: Run

**Pattern**: 8 or more consecutive points on the same side of the centerline.

**Detection**: 8+ points all above CL or all below CL.

**False Alarm Rate**: 0.39% per 8-point sequence

**Interpretation**: Process has shifted. Even small shifts (< 1σ) will eventually produce runs.

**Example Causes**:
- Process adjustment
- Tool change
- New material lot
- Seasonal effect

```python
rules = RuleSet().run(length=8).build()

# More sensitive (shorter run)
rules = RuleSet().run(length=7).build()
```

### Rule 5: Trend

**Pattern**: 6 or more consecutive points continuously increasing or decreasing.

**Detection**: 6+ points where each is higher (or lower) than the previous.

**False Alarm Rate**: 0.28% per 6-point sequence

**Interpretation**: Process is drifting. Identify and address the cause.

**Example Causes**:
- Tool wear
- Chemical degradation
- Temperature change
- Fatigue effects

```python
rules = RuleSet().trend(length=6).build()

# More sensitive
rules = RuleSet().trend(length=5).build()
```

### Rule 6: Oscillation

**Pattern**: 14 or more consecutive points alternating up and down.

**Detection**: 14+ points where each alternates direction from the previous.

**False Alarm Rate**: 0.006% per 14-point sequence

**Interpretation**: Process is being over-controlled. Each adjustment causes the next deviation.

**Example Causes**:
- Over-adjustment (tampering)
- Two alternating streams
- Measurement round-off
- Systematic sampling issue

```python
rules = RuleSet().oscillation(length=14).build()
```

### Rule 7: Hugging Center

**Pattern**: 15 or more consecutive points in Zone C (within 1σ of centerline).

**Detection**: 15+ points all within ±1σ of the centerline.

**False Alarm Rate**: 0.003% per 15-point sequence

**Interpretation**: Variation has been reduced. Could be good (process improvement) or suspicious (data manipulation, stratification).

**Example Causes**:
- Process improvement
- Incorrect subgrouping
- Mixed product/operators
- Data averaging or smoothing

```python
rules = RuleSet().hugging_center(length=15).build()
```

### Rule 8: Avoiding Center

**Pattern**: 8 or more consecutive points avoiding Zone C (all beyond 1σ).

**Detection**: 8+ points all outside ±1σ from centerline.

**False Alarm Rate**: 0.41% per 8-point sequence

**Interpretation**: Distribution is bimodal or has excessive variation.

**Example Causes**:
- Mixture of two processes
- Systematic over-adjustment
- Two distinct populations
- Subgroup selection issue

```python
rules = RuleSet().avoiding_center(length=8).build()
```

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
rules = RuleSet().beyond_limits().trend().build()

# Limits and runs with custom length
rules = RuleSet().beyond_limits().run(length=7).build()

# Full configuration
rules = (
    RuleSet()
    .beyond_limits()
    .zone_a(consecutive=2)
    .run(length=8)
    .trend(length=5)
    .build()
)

signals = result.detect_signals(rules=rules)
```

## Rule Applicability

Not all rules apply to all chart types:

| Rule | Xbar | S | IMR | R |
|------|------|---|-----|---|
| 1: Beyond Limits | ✅ | ✅ | ✅ | ✅ |
| 2: Zone A | ❌ | ❌ | ✅ | ✅ |
| 3: Zone B | ❌ | ❌ | ✅ | ✅ |
| 4: Run | ❌ | ❌ | ✅ | ✅ |
| 5: Trend | ❌ | ❌ | ✅ | ✅ |
| 6: Oscillation | ❌ | ❌ | ✅ | ✅ |
| 7: Hugging Center | ❌ | ❌ | ✅ | ✅ |
| 8: Avoiding Center | ❌ | ❌ | ✅ | ✅ |

**Reason**: Xbar and S charts compare subgroups, which may not be time-ordered. Rules 2-8 assume sequential ordering, which is guaranteed only for IMR charts.

## Sensitivity vs. False Alarms

| Rule Set | Sensitivity | False Alarm Rate |
|----------|------------|------------------|
| Rule 1 only | Low | Very low (~0.3%) |
| Standard (1-4) | Medium | Low (~1-2%) |
| Extended (1-8) | High | Moderate (~3-5%) |

**Recommendation**: Start with standard rules. Add extended rules when investigating known issues or when resources permit investigation of false alarms.

## Minimum Observations

Each rule requires a minimum number of observations:

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

ProcessBehavior automatically skips rules that can't be evaluated due to insufficient data.

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
