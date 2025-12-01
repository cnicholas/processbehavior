# 📊 processbehavior

**processbehavior** is a Python package for **Statistical Process Control (SPC)** with a focus on **analyst usability** and **UI integration**. It prepares datasets, computes control-chart statistics, and returns **tidy, well-documented tables** suitable for dashboards and deeper analysis.

## Features
- **Charts:** X̄–S, Individuals–Moving Range (I–MR), Range (R) *(R refining)*  
- **Rational subgrouping:** flexible multi-key grouping, robust validation  
- **Time support:** Year/Quarter/Month/Week/DOY extraction for analyses & plots  
- **Classic limits:** A3/B3/B4/D3/D4/d2/c4 with variable subgroup sizes  
- **Point classification:** Rule 1 (3σ) + designed to extend to Western/Nelson  
- **UI-ready outputs:** tidy `points`, `subgroups`, `limits`, `overall`, `rules`, `warnings`

## Quickstart
```python
import pandas as pd
import numpy as np
from processbehavior import Analysis

df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=30, freq="D"),
    "machine": ["A"]*15 + ["B"]*15,
    "x": np.random.normal(10, 1, 30),
})

spec = {
    'analysis_type': 'Xbar',
    'response_var': 'x',
    'rsg_vars': ['machine'],
    'time_var': 'timestamp',
    'time_unit': 'Month',
    'round_to': 3
}

result = Analysis(df, spec).calculate()

points    = result.points
subgroups = result.subgroups
limits    = result.limits
overall   = result.overall

Perfect — here’s a polished **Markdown section** you can drop into your README, docs, or PyPI page. It uses LaTeX math (via Markdown delimiters) so it will render equations nicely in most doc sites (e.g., GitHub, MkDocs, Jupyter, Quarto).

---

## 🔎 Residual System (R1–R5)

`processbehavior` extends Shewhart’s framework by partitioning variation into distinct **residuals**.
Each residual highlights a different *source of variation* so analysts can diagnose not only **if** a process is out of control, but also **why**.

---

### **R1 — Overall deviation**

* **Definition**

  $$
  R_1 = Y - \bar{Y}
  $$
* **Interpretation**
  Deviation of each observation from the global mean.
* **Use**
  Helps identify *time-related shifts* when tracked sequentially; serves as a baseline for separating common vs. special causes.

---

### **R2 — Within-cell residual**

* **Definition**

  $$
  R_2 = Y - \bar{Y}_{kt}
  $$

  if both subgroup $k$ and time $t$ exist,
  or, for one-way designs, the within-subgroup sequential residual (½-difference of adjacent observations).
* **Interpretation**
  Local “noise” after accounting for subgroup + time structure.
* **Use**
  Captures *within-subgroup variation* (measurement error, short-term fluctuation).

---

### **R3 — Interaction residual**

* **Definition**

  $$
  R_3 = \bar{Y}_{kt} - \bar{Y}_{k} - \bar{Y}_{t} + \bar{Y} - R_2
  $$
* **Interpretation**
  Residual interaction between subgroup and time effects.
* **Use**
  Highlights *factor interactions* (e.g., a machine that only drifts under certain shifts).

---

### **R4 — Time-marginal residual**

* **Definition**

  $$
  R_4 = \bar{Y}_{t} - \bar{Y} - R_2
  $$
* **Interpretation**
  Deviation of the time-period mean from the global mean.
* **Use**
  Highlights *systematic time effects* (seasonality, batch effects, trends).

---

### **R5 — Subgroup-marginal residual**

* **Definition**

  $$
  R_5 = \bar{Y}_{k} - \bar{Y} - R_2
  $$
* **Interpretation**
  Deviation of subgroup mean from the global mean.
* **Use**
  Highlights *subgroup effects* (machine-to-machine variation, operator bias).

---

## 🗺️ Why these matter

* **R1 / R4** → diagnose *time-driven shifts*.
* **R2** → isolate *pure noise* vs. signal.
* **R3** → surface *interactions*.
* **R5** → flag *consistent subgroup bias*.

Together, these residuals create a **diagnostic map of variation sources**.
In practice, plotting them side by side helps analysts understand whether signals are due to drift over time, subgroup effects, or interaction patterns — a richer picture than classic control charts alone.

---

👉 When you publish, you can also add a **note of attribution** here:

> *This residual system was developed in collaboration with Dr. Thomas Bishop (The Ohio State University), whose mentorship grounded this package in Shewhart’s philosophy of understanding sources of variation.*

---

## Automatic Stratified Control Charts ⭐

The killer feature that doesn't exist in other SPC software.

### The Problem

Traditional SPC tools force you to choose:
- ❌ **Combine all groups** → Inflated limits mask true patterns
- ❌ **Manual stratification** → Filter each group separately (tedious, error-prone)

### The Solution

**This system stratifies automatically:**
```python
# One specification
from processbehavior import Analysis

spec = {
    'analysis_type': 'Imr',
    'rsg_vars': ['lane', 'phase'],  # ← Stratify by these
    'time_var': 'pull',
    'response_var': 'fill_weight'
}

# One function call
results = Analysis(df, spec).calculate()

# Multiple charts automatically created:
# results['lane1_phase1'] - Individual chart with lane1_phase1-specific limits
# results['lane1_phase2'] - Individual chart with lane1_phase2-specific limits
# results['lane2_phase1'] - Individual chart with lane2_phase1-specific limits
# ...
```

### Key Distinction

**Grouping serves DIFFERENT purposes for different analyses:**

| Analysis | Grouping Purpose | Output | VAS Residuals |
|----------|-----------------|---------|---------------|
| **IMR/R** | **Stratification** | Separate chart per group | No (not needed) |
| **Xbar-S** | **Variance decomposition** | Variance components | Yes (that's the point) |

Both use `rsg_vars`, but for completely different reasons!
