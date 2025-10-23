# Stratified Summary Tab: Analyst Workflow

## The Problem This Solves

When you have **stratified IMR charts** with multiple factor combinations, you need to answer:
1. **Which strata are problematic?** (Quick triage)
2. **When are the problems occurring?** (Detailed investigation)

Without a summary tab, you'd need to scroll through hundreds of rows to compare strata.

## The Solution: Two-Tab Workflow

### Tab 1: **Stratified_Summary** (Quick Triage) ⚡

**8 rows** - One per factor combination

| Stratum | Observations | Mean    | LCL     | UCL     | Signals | Signal_Rate_% |
|---------|--------------|---------|---------|---------|---------|---------------|
| 4.0_2.0 | 100          | 238.090 | 235.975 | 240.205 | **14**  | **14.0%** 🔴  |
| 2.0_2.0 | 100          | 238.921 | 237.283 | 240.559 | **12**  | **12.0%** 🔴  |
| 2.0_1.0 | 100          | 237.395 | 235.663 | 239.126 | **10**  | **10.0%** ⚠️   |
| 4.0_1.0 | 100          | 237.589 | 235.344 | 239.834 | 7       | 7.0%          |
| 1.0_1.0 | 100          | 238.108 | 235.620 | 240.596 | 6       | 6.0%          |
| 3.0_1.0 | 100          | 236.511 | 233.615 | 239.407 | 6       | 6.0%          |
| 3.0_2.0 | 100          | 236.975 | 234.100 | 239.850 | 4       | 4.0%          |
| 1.0_2.0 | 100          | 238.670 | 236.766 | 240.573 | **3**   | **3.0%** 🟢   |

**Instant Insights (in 5 seconds):**
- Stratum 4.0_2.0 has highest signal rate (14%) → **Investigate this first!**
- Stratum 1.0_2.0 is most stable (3%) → Best-performing settings
- Pattern: FACTOR 2=2 tends to have more signals than FACTOR 2=1
- Average signal rate: 7.8%

### Tab 2: **Chart_Imr_Stratified** (Detailed Drill-Down) 🔬

**800 rows** - Time-series data for all strata

Filter to Stratum 4.0_2.0 (the worst performer):

| TIME | rsg     | Y       | beyond_limits |
|------|---------|---------|---------------|
| 1    | 4.0_2.0 | 241.890 | **1** 🔴      |
| 2    | 4.0_2.0 | 238.990 | 0             |
| 3    | 4.0_2.0 | 239.100 | 0             |
| ...  | ...     | ...     | ...           |
| 13   | 4.0_2.0 | 238.300 | 0             |
| 14   | 4.0_2.0 | 236.880 | **-1** 🔴     |
| ...  | ...     | ...     | ...           |

**Detailed Findings:**
- 14 signals across 100 time points
- Problems occur at times: 1, 14, 22, 26, 28, ... (specific timestamps)
- Now you can correlate with production events, operator shifts, etc.

---

## Real-World Example: Troubleshooting Session

### Step 1: Open Stratified_Summary (30 seconds)
```
Analyst: "Stratum 4.0_2.0 has 14 signals. That's our problem area."
Action: Note down FACTOR 1=4, FACTOR 2=2 as problematic setting
```

### Step 2: Check Full Details (2 minutes)
```
Analyst: "Let me filter Chart_Imr_Stratified to rsg='4.0_2.0'"
Finding: "Signals cluster around times 13-15 and 26-28"
Action: Cross-reference with production logs
```

### Step 3: Root Cause (10 minutes)
```
Production Log: "Times 13-15: New operator shift change"
Production Log: "Times 26-28: Temperature spike in Zone 4"
Conclusion: FACTOR combination 4.0_2.0 is sensitive to temperature
```

### Step 4: Action Plan
```
1. Tighter temperature control for FACTOR 1=4, FACTOR 2=2
2. Additional training for operators on this setting
3. Monitor strata 2.0_2.0 and 2.0_1.0 (also elevated signal rates)
4. Study why 1.0_2.0 is so stable (only 3% signals)
```

**Total time: 12 minutes instead of 2 hours of Excel scrolling!**

---

## Design Philosophy: Progressive Disclosure

This follows the **Pythonic Hadley** principle:

> "Common tasks should be trivial; complex tasks should be possible"

**Common task (trivial):** "Which factor settings are problematic?"
→ **Stratified_Summary tab**: 8 rows, scan in 10 seconds ✅

**Complex task (possible):** "When exactly do problems occur for setting X?"
→ **Chart_Imr_Stratified tab**: 800 rows, full time-series detail ✅

---

## Key Features

### 1. Sorted by Severity
- Worst strata appear first
- Immediately see where to focus efforts
- No mental math required

### 2. Signal Rate %
- Normalizes across strata
- Easy comparison: "14% vs 3%"
- Answers: "How much worse is this setting?"

### 3. Compact Summary
- 8 rows fit on one screen
- No scrolling needed for comparison
- Print-friendly for meetings

### 4. Actionable
- Stratum ID directly maps to factor settings
- Can immediately adjust FACTOR 1 and FACTOR 2
- Guides experimental follow-up

---

## Comparison: Before vs After

### Before (No Summary Tab):
1. Open Chart_Imr_Stratified (800 rows)
2. Manually filter to each stratum
3. Count signals for each (8 strata × 2 min = 16 minutes)
4. Compare in your head or external spreadsheet
5. Easy to miss patterns

### After (With Summary Tab): ⚡
1. Open Stratified_Summary (8 rows)
2. **Done in 10 seconds!**
3. Top 3 problems immediately visible
4. Sorted by severity automatically

**Time saved: 15+ minutes per analysis**

---

## Technical Implementation

```python
# Automatically created for stratified IMR/I charts
result.to_excel(
    'analysis.xlsx',
    include_summary=True,
    include_charts=True
)

# Generates:
# - Stratified_Summary: High-level comparison
# - Chart_Imr_Stratified: Detailed time-series
```

**Location in Excel:** `Stratified_Summary` tab (right after Summary tab)

**Columns:**
- `Stratum`: Factor combination (e.g., "4.0_2.0")
- `Observations`: Data points per stratum
- `Mean`: Average value for this stratum
- `LCL`, `UCL`: Control limits for this stratum
- `Signals`: Count of beyond_limits points
- `Signal_Rate_%`: Signals / Observations × 100

**Sorting:** Descending by `Signals` (worst first)

---

## When to Use This

✅ **Use Stratified IMR + Summary when:**
- Multiple factor combinations (2+)
- Want to compare stability across settings
- Need quick triage for troubleshooting
- Preparing for team meetings (summary is presentation-ready)

❌ **Don't need it for:**
- Single time-series (use regular IMR)
- Combined analysis (use Xbar chart)
- When you only care about one specific stratum

---

## Bottom Line

**The Stratified_Summary tab is like a dashboard for your control charts.**

- 🚦 **Red light**: Stratum 4.0_2.0 has 14% signals
- 🟡 **Yellow light**: Strata 2.0_2.0, 2.0_1.0 need attention
- 🟢 **Green light**: Stratum 1.0_2.0 is stable

**From data → insight → action in under 1 minute.**

That's the power of good UX design! 🎯
