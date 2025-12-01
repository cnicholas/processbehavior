# SDS Detection & Auto-Completion Demo

## Overview

This demo (`sds_detection_demo.py`) showcases the **automatic Sampling Design State (SDS) detection** and **auto-completion** features of the ProcessBehavior package - capabilities not available in Minitab, JMP, or other commercial SPC software.

## What It Demonstrates

### 1. **Automatic SDS Detection**
The package automatically detects the data structure (SDS 0-6) without requiring user configuration:

| SDS | Description | Detection Method |
|-----|-------------|------------------|
| **SDS 0** | No grouping or time structure | Single stream, no organization |
| **SDS 1** | Full replication (n≥2 per cell) | Multiple observations per (factor×time) cell |
| **SDS 2** | No replication (n=1 per cell) | Exactly one observation per (factor×time) cell |
| **SDS 3** | Partial replication (mixed) | Some cells n=1, others n≥2 |
| **SDS 4** | Single stream over time | One process monitored continuously (K=1) |
| **SDS 5** | Nested/hierarchical design | Factors sampled asynchronously, incomplete coverage |
| **SDS 6** | Regime changes/unstructured | Irregular patterns, sparse grids, process shifts |

### 2. **Auto-Completion**
Once SDS is detected, the package automatically completes:
- Column specifications for data preparation
- Sort requirements and sort columns
- Analysis output columns
- VAS residual calculations (when appropriate)
- Interaction analysis capabilities

### 3. **Killer Feature: Automatic Stratification**
The demo compares two analyses of the same SDS 1 data:

**Stratified IMR Analysis:**
- Creates separate I-MR charts for EACH factor level
- Each chart gets its own appropriate control limits
- **NOT available in Minitab/JMP** (requires manual filtering)

**Xbar-S Analysis:**
- Treats each (factor×time) cell as a subgroup
- Calculates VAS residual decomposition (R1-R5)
- Decomposes variance into components

## Running the Demo

```bash
# From project root
source venv/bin/activate
python examples/sds_detection_demo.py
```

## Demo Output

The demo provides detailed output for each SDS type:

```
================================================================================
TEST: SDS 1: Full Replication (Expected SDS: 1)
================================================================================

📊 Data Shape: 76 rows × 4 columns
   Columns: ['time', 'factor 1', 'factor 2', 'y']

🔍 SDS Detection Result: ✅ PASS
   Expected: SDS 1
   Detected: SDS 1

⚙️  Auto-Completed Analysis Information:
   • Has grouping: True
   • Has time: True
   • Requires sort: True
   • Sort columns: ['rsg', 'time']
   ...

📈 SDS 1 Characteristics:
   • Description: Full replication (all cells n≥2)
   • R2 Calculation Method: within_cell
   • Variance Decomposition: True
   • Interaction Analysis: True

🔬 VAS Residual Decomposition (SDS 1 Specialty):
   • R1: σ = 1.959  (Total residual)
   • R2: σ = 0.220  (Within-cell variance)
   • R3: σ = 0.457  (Interaction effects)
   • R4: σ = 1.212  (Factor main effects)
   • R5: σ = 1.630  (Time effects)
```

## Key Insights

1. ✅ **Automatic SDS detection** works for all data structures
2. ✅ **Auto-completion** provides correct column specifications
3. ✅ **VAS residuals** calculated appropriately based on SDS
4. ✅ **Stratified analyses** work automatically (killer feature!)
5. ✅ **Detection is robust** - handles edge cases intelligently

## Synthetic Data Generators

The demo uses the **unified `make_sds()` API** from `processbehavior.datasets.synthetic`:

```python
from processbehavior.datasets import synthetic

# Unified API - specify SDS type with sds parameter
# SDS 1: Full replication
df = synthetic.make_sds(sds=1, K=3, T=8, n_min=2, n_max=4, seed=42)

# SDS 2: No replication
df = synthetic.make_sds(sds=2, K=3, T=8, seed=42)

# SDS 3: Partial replication
df = synthetic.make_sds(sds=3, K=3, T=8, p_replicated=0.6, n_when_replicated=3, seed=42)

# SDS 4: Single stream
df = synthetic.make_sds(sds=4, K=3, T=50, seed=42)

# SDS 5: Nested design
df = synthetic.make_sds(sds=5, K=3, T=12, L=3, H_per_L=4, seed=42)

# SDS 6: Regime changes
df = synthetic.make_sds(sds=6, K=3, T=40, seed=42)

# Individual functions still available for backward compatibility:
# make_sds1(), make_sds2(), make_sds3(), make_sds4(), make_sds5(), make_sds6()
```

## Comparison to Commercial Software

| Feature | ProcessBehavior | Minitab/JMP |
|---------|----------------|-------------|
| **Automatic SDS detection** | ✅ Yes | ❌ No |
| **Auto-completion** | ✅ Yes | ❌ No |
| **Stratified IMR charts** | ✅ Automatic | ❌ Manual filtering required |
| **VAS residual decomposition** | ✅ Automatic | ❌ Not available |
| **Rational subgrouping intelligence** | ✅ Yes | ⚠️ Limited |

## Educational Value

This demo serves as:
- **Tutorial** on SDS classification
- **Validation** of automatic detection logic
- **Documentation** of package capabilities
- **Comparison** to manual SPC workflows

## Related Files

- `processbehavior/datasets/synthetic.py` - Data generators
- `processbehavior/sds_detector.py` - SDS detection logic
- `processbehavior/analysis_specification.py` - Auto-completion logic
- `tests/test_sds_detector.py` - Unit tests for SDS detection

## Notes

- The demo uses `seed=42` for reproducibility across all SDS types
- **No warnings**: All pandas FutureWarnings have been eliminated (as of v0.1.0)
- **Cell-level detection**: SDS classification is based on (factor × time) cell replication, not factor-level aggregation
- **100% accuracy**: Demo validates that `make_sds(sds=N)` is correctly detected as SDS N for all N ∈ {0,1,2,3,4,5,6}
- See `VERIFICATION_REPORT.md` for comprehensive validation details

---

**Created:** 2024-11-30
**Updated:** 2025-11-30 (v0.1.0 - unified API, corrected detection logic)
**Author:** ProcessBehavior Team
**Purpose:** Validate SDS detection and showcase auto-completion capabilities
