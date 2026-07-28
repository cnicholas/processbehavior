#!/usr/bin/env python
"""Performance benchmark + drift check for processbehavior.

Runs a fixed set of scenarios — design {SDS-1 exact-R2, SDS-2 MA2 worst-case} × size
{10K, 100K, 1M} × op {init, formulate, execute, execute-stratified} — prints a report table, writes
``benchmarks/last_run.json``, and compares against the committed ``benchmarks/baseline.json``.

The baseline is *same-machine* (absolute times are hardware-specific), so a regression flag
is meaningful when you re-run on the same box. Cross-machine, rely on the shape checks in
``tests/test_performance.py`` (per-row time stays flat as N grows ⇒ linear).

Usage:
    python scripts/benchmark.py                 # run + report + drift vs baseline
    python scripts/benchmark.py --quick         # skip the 1M cases
    python scripts/benchmark.py --update-baseline   # overwrite baseline.json with this run
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "benchmarks" / "baseline.json"
LAST_RUN = ROOT / "benchmarks" / "last_run.json"

REGRESS_TOL = 0.40  # current > baseline * (1 + tol) → flagged as a regression
EXTRA_COLS = 46     # noise columns, matching the historical 50-column datasets

# Each design carries the spec that actually exercises its residual path. SDS-1 uses a
# single factor (replicated (factor 1 × time) cells → exact R2); SDS-2 uses BOTH factors so
# the (factor 1 × factor 2 × time) cells are singletons (no replication → MA2 sort path).
DESIGNS = {
    1: {"factors": ["factor 1"], "label": "SDS-1 exact-R2"},
    2: {"factors": ["factor 1", "factor 2"], "label": "SDS-2 MA2"},
}


def make_dataset(sds: int, n_rows: int, seed: int = 42):
    """Build a ~n_rows dataset for the given design, plus EXTRA_COLS noise columns."""
    if sds == 1:
        n, k2 = 10, 2
        cells = max(1, n_rows // (n * k2))
        k1 = max(2, int(cells**0.5))
        t = max(2, cells // k1)
        df = synthetic.make_design(1, K1=k1, K2=k2, T=t, n_min=n, n_max=n, seed=seed)
    else:  # SDS-2: one observation per (K1 × K2 × T) cell (no replication)
        k2 = 2
        cells = max(1, n_rows // k2)
        k1 = max(2, int(cells**0.5))
        t = max(2, cells // k1)
        df = synthetic.make_design(2, K1=k1, K2=k2, T=t, seed=seed)
    rng = np.random.default_rng(seed)
    for i in range(EXTRA_COLS):
        df[f"extra_col_{i}"] = rng.normal(0, 1, len(df))
    return df


def _time(fn, repeat: int = 1):
    """Best-of-`repeat` wall time (perf_counter). Returns (seconds, last_result)."""
    best, out = float("inf"), None
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _peak_mb(fn) -> float:
    """Peak Python heap (MB) during fn(), via tracemalloc (separate pass — no time skew)."""
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def run_scenario(sds: int, target_rows: int) -> list[dict]:
    """Measure init / formulate / execute / peak-memory for one (design, size)."""
    spec = {"response": "y", "factors": DESIGNS[sds]["factors"], "time": "time"}
    exec_chart = "Xbar" if sds == 1 else "X"  # SDS-2 cells are n=1 → Individuals, not Xbar
    repeat = 1 if target_rows >= 1_000_000 else 2

    df = make_dataset(sds, target_rows)
    rows = len(df)
    input_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    t_init, pdf = _time(lambda: ProcessBehavior(df), repeat)
    t_form, study = _time(lambda: pdf.formulate(**spec), repeat)
    t_exec, _ = _time(lambda: study.execute(chart=exec_chart, by=[]), repeat)
    # Stratified X: the by=[] path above never enters it, so a regression there would be
    # invisible to the drift check. This scenario exists because that is exactly what
    # happened — a per-stratum table scan (O(strata x rows)) sat here at 26.5s/1M rows and
    # went unnoticed because nothing measured it.
    t_strat, _ = _time(lambda: study.execute(chart="X", by=list(DESIGNS[sds]["factors"])), repeat)
    peak = _peak_mb(lambda: ProcessBehavior(df).formulate(**spec))

    common = {"sds": sds, "size": target_rows, "rows": rows}
    return [
        {**common, "op": "init", "seconds": t_init},
        {**common, "op": "formulate", "seconds": t_form},
        {**common, "op": f"execute_{exec_chart.lower()}", "seconds": t_exec},
        {**common, "op": "execute_x_stratified", "seconds": t_strat},
        {**common, "op": "formulate_peak", "seconds": None, "peak_mb": peak, "input_mb": input_mb},
    ]


def run_benchmarks(quick: bool = False) -> list[dict]:
    sizes = [10_000, 100_000] + ([] if quick else [1_000_000])
    results: list[dict] = []
    for sds in (1, 2):
        for n in sizes:
            results.extend(run_scenario(sds, n))
    return results


# --------------------------------------------------------------------------- reporting
def _key(r: dict) -> str:
    return f"sds{r['sds']}|{r['size']}|{r['op']}"


def format_report(results: list[dict]) -> str:
    lines = [
        "",
        f"{'Design':<14}{'Rows':>10}  {'Op':<22}{'Time (s)':>10}{'Rows/sec':>14}{'Peak MB':>10}",
        "-" * 82,
    ]
    for r in results:
        design = DESIGNS[r["sds"]]["label"]
        if r["op"] == "formulate_peak":
            lines.append(
                f"{design:<14}{r['rows']:>10,}  {'formulate mem':<22}{'':>10}{'':>14}"
                f"{r['peak_mb']:>9.0f} ({r['peak_mb'] / r['input_mb']:.1f}x)"
            )
        else:
            tput = r["rows"] / r["seconds"] if r["seconds"] else 0
            lines.append(
                f"{design:<14}{r['rows']:>10,}  {r['op']:<22}{r['seconds']:>10.3f}{tput:>14,.0f}{'':>10}"
            )
    return "\n".join(lines)


def compare_to_baseline(results: list[dict], baseline: dict | None) -> list[str]:
    """Return human-readable drift lines (regressions and notable improvements)."""
    if not baseline:
        return ["(no baseline.json — run with --update-baseline to record one)"]
    base = baseline.get("results", {})
    drift = []
    for r in results:
        if r["op"] == "formulate_peak" or r["seconds"] is None:
            continue
        b = base.get(_key(r))
        if not b or not b.get("seconds"):
            continue
        ratio = r["seconds"] / b["seconds"]
        if ratio > 1 + REGRESS_TOL:
            drift.append(f"  REGRESSION  {_key(r):<28} {b['seconds']:.3f}s → {r['seconds']:.3f}s ({ratio:.2f}x)")
        elif ratio < 1 - REGRESS_TOL:
            drift.append(f"  improved    {_key(r):<28} {b['seconds']:.3f}s → {r['seconds']:.3f}s ({ratio:.2f}x)")
    return drift or [f"  no regressions beyond ±{REGRESS_TOL:.0%} vs baseline"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def to_document(results: list[dict]) -> dict:
    return {
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "machine": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "git_sha": _git_sha(),
            "note": "Same-machine baseline; absolute times are hardware-specific.",
        },
        "results": {_key(r): {k: v for k, v in r.items() if k not in ("sds", "size", "op")} for r in results},
    }


def load_baseline() -> dict | None:
    if BASELINE.exists():
        return json.loads(BASELINE.read_text())
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="processbehavior performance benchmark")
    ap.add_argument("--quick", action="store_true", help="skip the 1M-row cases")
    ap.add_argument("--update-baseline", action="store_true", help="overwrite baseline.json with this run")
    args = ap.parse_args()

    results = run_benchmarks(quick=args.quick)
    print(format_report(results))

    doc = to_document(results)
    LAST_RUN.parent.mkdir(exist_ok=True)
    LAST_RUN.write_text(json.dumps(doc, indent=2))

    if args.update_baseline:
        BASELINE.write_text(json.dumps(doc, indent=2))
        print(f"\nBaseline updated → {BASELINE.relative_to(ROOT)}")
    else:
        print("\nDrift vs baseline:")
        print("\n".join(compare_to_baseline(results, load_baseline())))


if __name__ == "__main__":
    main()
