# Wheeler Terminology

ProcessBehavior follows Donald Wheeler's terminology and methodology as presented in his books, particularly *Understanding Statistical Process Control* and *Advanced Topics in Statistical Process Control*. This glossary maps Wheeler's terms to common alternatives.

## Core Terminology

### Process Behavior Chart

**Wheeler's Term**: Process Behavior Chart

**Common Alternatives**: Control Chart, SPC Chart, Shewhart Chart

**Definition**: A graphical tool that plots data over time against statistically-derived control limits. Used to distinguish between common cause and special cause variation.

**Wheeler's Insight**: "Process behavior charts are about understanding variation, not about control."

---

### Common Cause Variation

**Wheeler's Term**: Common Cause Variation (also: Routine Variation)

**Common Alternatives**: Random Variation, Inherent Variation, Natural Variation, Noise

**Definition**: Variation that is inherent to the process and always present. Characterized by statistical stability.

**Wheeler's Insight**: "Common causes are like the voice of the process itself."

---

### Special Cause Variation

**Wheeler's Term**: Special Cause Variation (also: Exceptional Variation)

**Common Alternatives**: Assignable Cause, Non-random Variation, Signal

**Definition**: Variation from sources outside the usual process. Indicates something different happened.

**Wheeler's Insight**: "Special causes speak loudly enough to be heard above the noise of common causes."

---

### Rational Subgroup

**Wheeler's Term**: Rational Subgroup

**Common Alternatives**: Sample Group, Factor Level, Category

**Definition**: A subgroup where items are selected to maximize similarity within and highlight differences between subgroups.

**In ProcessBehavior**: Specified via the `factors` parameter in `formulate()`.

---

### Natural Process Limits

**Wheeler's Term**: Natural Process Limits

**Common Alternatives**: Control Limits, 3-Sigma Limits

**Definition**: Limits calculated from the data that describe the natural range of common cause variation. NOT specification limits.

**Wheeler's Insight**: "Natural Process Limits are the Voice of the Process. Specification Limits are the Voice of the Customer. These are different voices."

---

## Design States (DS)

Wheeler identifies six design states that determine valid analysis approaches.

### DS 1: Full Replication

**Definition**: Every (factor × time) cell contains 2+ observations.

**ProcessBehavior Detection**: All cell counts >= 2

**Capabilities**: Full VAS analysis, exact variance estimation

---

### DS 2: No Replication

**Definition**: Every cell contains exactly 1 observation.

**ProcessBehavior Detection**: All cell counts == 1

**Capabilities**: MR-based variance estimation, approximate VAS

---

### DS 3: Partial Replication

**Definition**: Mix of replicated and unreplicated cells.

**ProcessBehavior Detection**: Some cells with n=1, others with n>=2

**Capabilities**: Hybrid variance estimation

---

### DS 4: Incomplete, No Singletons

**Definition**: Incomplete grid — empty cells present, all observed cells have N_kt >= 2.

**After cleansing**: Collapses to ADS 1 (Full Replication)

---

### DS 5: Incomplete, No Replication

**Definition**: Incomplete grid — empty cells present, all observed cells have N_kt = 1.

**After cleansing**: Collapses to ADS 2 (No Replication)

---

### DS 6: Incomplete, With Singletons

**Definition**: Incomplete grid — empty cells present, observed cells have mixed N_kt.

**After cleansing**: Collapses to ADS 3 (Partial Replication)

---

## Variance Analysis System (VAS)

Dr. Thomas A. Bishop's framework for decomposing variation into meaningful components. VAS extends Wheeler's process behavior chart methodology with a hierarchical residual decomposition (R1-R5) that isolates within-cell, interaction, time, and factor effects.

### R1: Total Deviation

**Formula**: R1 = Y - Y̅

**Meaning**: How far each observation is from the grand mean.

---

### R2: Within-Cell Residual

**Formula (DS 1)**: R2 = Y - Y̅<sub>kt</sub>

**Meaning**: Variation within subgroups, the "unexplained" portion.

**Wheeler's Insight**: This represents measurement error and short-term variation.

---

### R3: Interaction Residual

**Formula**: R3 = Y - Y̅<sub>k</sub> - Y̅<sub>t</sub> + Y̅

**Meaning**: How factor effects change over time.

**Wheeler's Insight**: Significant R3 signals mean factor behavior is inconsistent.

---

### R4: Time Effect + Unexplained

**Formula**: R4 = Y̅<sub>t</sub> - Y̅ + R2

**Meaning**: Time-related patterns combined with within-cell variation.

**Wheeler's Insight**: Chart R4 to detect trends, shifts, and cycles.

---

### R5: Factor Effect + Unexplained

**Formula**: R5 = Y̅<sub>k</sub> - Y̅ + R2

**Meaning**: Factor differences combined with within-cell variation.

**Wheeler's Insight**: Chart R5 to identify true factor differences.

---

## Chart Types

### I Chart (Individual)

**Wheeler's Term**: X Chart, Individual Chart

**Common Alternatives**: I Chart, Individuals Chart

**Definition**: Plots individual observations.

---

### mR Chart (Moving Range)

**Wheeler's Term**: mR Chart

**Common Alternatives**: MR Chart, Moving Range Chart

**Definition**: Plots |X<sub>i</sub> - X<sub>i-1</sub>|, the absolute difference between consecutive points.

---

### XmR Chart (Individual and Moving Range)

**Wheeler's Term**: XmR Chart

**Common Alternatives**: IMR Chart, I-MR Chart

**Definition**: Combined Individual and Moving Range chart.

**In ProcessBehavior**: `study.charts.XmR`

---

### Average Chart

**Wheeler's Term**: Average Chart, X̄ Chart

**Common Alternatives**: Xbar Chart, X-bar Chart

**Definition**: Plots subgroup averages.

**In ProcessBehavior**: `study.charts.Xbar`

---

### s Chart

**Wheeler's Term**: s Chart

**Common Alternatives**: S Chart, Sigma Chart

**Definition**: Plots subgroup standard deviations.

**In ProcessBehavior**: `study.charts.S`

---

## Control Limit Constants

Wheeler uses the standard SPC constants from Shewhart's work:

| Constant | Purpose | Formula Source |
|----------|---------|----------------|
| c₄ | Unbiasing s | Related to gamma function |
| A₃ | Xbar limits from s | 3 / (c₄√n) |
| B₃, B₄ | S chart limits | Functions of c₄ and n |
| d₂ | Unbiasing range | Tabulated |
| D₃, D₄ | R chart limits | Functions of d₂ |

---

## Key Principles

### The 3-Sigma Rule

**Wheeler's Principle**: Use 3-sigma limits, not 2-sigma or other values.

**Rationale**: Balances sensitivity with false alarm rate. Provides approximately 99.73% coverage for normally distributed data, but works for most distributions.

---

### Shewhart's Empirical Rule

**Wheeler's Statement**: "The power of a process behavior chart does not come from statistical theory, but from the ability to detect economically important shifts in the process."

**Implication**: Focus on practical significance, not just statistical significance.

---

### Limits ≠ Specifications

**Wheeler's Principle**: Natural Process Limits (calculated from data) are fundamentally different from Specification Limits (set by requirements).

**Common Mistake**: Comparing control limits to specifications or treating them as equivalent.

---

### First Analyze, Then Improve

**Wheeler's Principle**: Understand your process variation before attempting improvement.

**Implication**: Establish a baseline, then make changes and measure effects.

---

## Wheeler's Books

Essential references:

1. **Understanding Statistical Process Control** (with David Chambers)
   - Foundation text for SPC
   - Introduces process behavior charts

2. **Advanced Topics in Statistical Process Control**
   - VAS residual analysis
   - Design States
   - Complex designs

3. **Making Sense of Data**
   - Data analysis philosophy
   - Interpretation guidelines

4. **The Six Sigma Practitioner's Guide to Data Analysis**
   - Practical applications
   - Case studies

---

## ProcessBehavior Mapping

| Wheeler Term | ProcessBehavior API |
|--------------|---------------------|
| Process Behavior Chart | `result.plot()` |
| Rational Subgroup | `factors` parameter |
| Time Sequence | `time` parameter |
| XmR Chart | `study.charts.XmR` |
| Average Chart | `study.charts.Xbar` |
| s Chart | `study.charts.S` |
| R1-R5 Residuals | `result.residuals` |
| DS Detection | `study.observed_design_state` / `study.analytical_design_state` |
| Natural Process Limits | `result.get_statistics()` |

---

## Further Reading

- Wheeler, D.J. & Chambers, D.S. (1992). *Understanding Statistical Process Control*, 2nd ed. SPC Press.
- Wheeler, D.J. (1995). *Advanced Topics in Statistical Process Control*. SPC Press.
- Wheeler, D.J. (2000). *Understanding Variation: The Key to Managing Chaos*, 2nd ed. SPC Press.
- Shewhart, W.A. (1931). *Economic Control of Quality of Manufactured Product*. Van Nostrand.
