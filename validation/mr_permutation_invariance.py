#!/usr/bin/env python3
"""
Permutation/invariance check for the combined ProcessBehavior ACO chart.

The combined chart (by=[]) orders points ACO-then-year, so moving ranges cross
ACO boundaries. With 24 ACOs x 4 years, 23 of 95 ranges are boundary ranges.

This script holds every observation fixed and varies only lane order. It reports:
  * within-ACO and boundary moving-range summaries;
  * X-chart and mR-chart signal counts for selected and random lane orders;
  * the shipped order's percentile among random permutations;
  * whether the high-level X-chart conclusion survives random lane order; and
  * a descriptive comparison of boundary ranges with pure between-ACO level gaps.

The boundary comparison is descriptive, not an additive/causal decomposition.

Scope. This script covers only the combined chart. It does not test the lane
pattern, which is order-invariant, and it does not evaluate organisation-level
claims. Per issue #114 those are addressed by:

    levels = study.execute(chart="Xbar", by=["ACO"], value="R5", recentered=True)
    phased = study.execute(chart="X", by=[], phased=True, companion=True)

A boundary signal is evidence about the traversal -- that those two lanes, in
that order, do not join into one stable stream.

Note that R5 is not wholly traversal-free: its points are order-free organisation
means, but in ADS 2 its limits are set from R2, which is built from the same
lane-major sequence. See #114 for the author's measurement of that dependence.

Usage:
    python3 mr_permutation_invariance.py aco_per_capita_expenditure.csv
"""

import csv
import sys

import numpy as np

A2 = 2.660   # Individuals-chart limit constant when sigma is estimated by mRbar/d2
D4 = 3.267   # Upper mR-chart constant for moving ranges of 2


def load(path, aco="ACO", year="YEAR", resp="PER CAPITA EXPENDITURE"):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    acos = sorted({r[aco] for r in rows})
    by = {a: [] for a in acos}
    for r in rows:
        by[r[aco]].append((float(r[year]), float(r[resp])))

    for a in acos:
        by[a].sort()

    lengths = {len(v) for v in by.values()}
    if len(lengths) != 1:
        sys.exit("unequal series lengths: %s" % sorted(lengths))

    T = lengths.pop()
    years = np.array([[yr for yr, _ in by[a]] for a in acos], dtype=float)
    Y = np.array([[v for _, v in by[a]] for a in acos], dtype=float)
    return acos, years, Y, T


def pieces(Y, order, T):
    """Return within-lane mRs, boundary mRs, full sequence, and full mRs."""
    seq = Y[list(order)].ravel()
    mr = np.abs(np.diff(seq))
    is_boundary = np.zeros(len(mr), dtype=bool)
    is_boundary[T - 1::T] = True
    return mr[~is_boundary], mr[is_boundary], seq, mr


def stats(Y, order, T):
    """Classical X/mR summaries for one lane ordering."""
    within, boundary, seq, mr = pieces(Y, order, T)
    mrbar = mr.mean()
    center = seq.mean()

    x_lo = center - A2 * mrbar
    x_hi = center + A2 * mrbar
    mr_ucl = D4 * mrbar

    x_signal_mask = (seq < x_lo) | (seq > x_hi)
    mr_signal_mask = mr > mr_ucl

    return {
        "within": within,
        "boundary": boundary,
        "seq": seq,
        "mr": mr,
        "mrbar": mrbar,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "x_width": x_hi - x_lo,
        "mr_ucl": mr_ucl,
        "x_signals": int(x_signal_mask.sum()),
        "mr_signals": int(mr_signal_mask.sum()),
        "x_signal_mask": x_signal_mask,
        "mr_signal_mask": mr_signal_mask,
    }


def boundary_signal_labels(acos, order, T, mr_signal_mask):
    """Labels for signalled mRs that occur specifically at lane boundaries."""
    labels = []
    order = list(order)
    for lane_pos in range(len(order) - 1):
        mr_index = (lane_pos + 1) * T - 1
        if mr_signal_mask[mr_index]:
            labels.append(f"{acos[order[lane_pos]]}|{acos[order[lane_pos + 1]]}")
    return labels


def fmt_labels(labels):
    return ", ".join(labels) if labels else "(none)"


def main(path):
    acos, years, Y, T = load(path)
    n = len(acos)
    if n < 2:
        sys.exit("need at least two organisations")

    rng = np.random.default_rng(0)
    loaded = np.arange(n)

    s0 = stats(Y, loaded, T)
    within = s0["within"]
    boundary = s0["boundary"]

    print(f"{n} organisations x {T} obs = {n*T} obs, {n*T-1} moving ranges")
    print(f"  within-ACO    {len(within):3d} ranges   mean {within.mean():10.2f}"
          "   <- invariant to lane order")
    print(f"  boundaries    {len(boundary):3d} ranges   mean {boundary.mean():10.2f}"
          "   <- varies with lane adjacency")
    print(f"  boundary/within mean ratio                 {boundary.mean()/within.mean():10.2f}x")
    print()
    print(f"  mRbar as loaded                            {s0['mrbar']:10.2f}")
    print(f"  mRbar within-only                          {within.mean():10.2f}")
    print(f"  increase from including boundaries         "
          f"{100*(s0['mrbar']/within.mean()-1):10.1f}%")
    print()

    # Two deterministic contrast orders.
    level = Y.mean(axis=1)
    sorted_order = np.argsort(level)

    low = sorted_order[: n // 2]
    high = sorted_order[n // 2:][::-1]
    interleaved = np.empty(n, dtype=int)
    interleaved[0::2] = low[: len(interleaved[0::2])]
    interleaved[1::2] = high[: len(interleaved[1::2])]

    ss = stats(Y, sorted_order, T)
    si = stats(Y, interleaved, T)

    # One Monte Carlo sample supplies all random-order summaries so the table,
    # extrema, zero-signal check, and percentile are internally consistent.
    N_RANDOM = 20_000
    mrbars = np.empty(N_RANDOM)
    widths = np.empty(N_RANDOM)
    xsignals = np.empty(N_RANDOM, dtype=int)
    mrsignals = np.empty(N_RANDOM, dtype=int)

    for k in range(N_RANDOM):
        s = stats(Y, rng.permutation(n), T)
        mrbars[k] = s["mrbar"]
        widths[k] = s["x_width"]
        xsignals[k] = s["x_signals"]
        mrsignals[k] = s["mr_signals"]

    shipped_pct = 100.0 * np.mean(mrbars <= s0["mrbar"])

    print("PERMUTATION CHECK -- observations fixed; only lane order changes")
    print()
    print("ordering              mRbar    X-limit width   X signals   mR signals")
    print(f"  as loaded         {s0['mrbar']:8.1f}     {s0['x_width']:10.1f}"
          f"       {s0['x_signals']:4d}        {s0['mr_signals']:4d}")
    print(f"  sorted by level   {ss['mrbar']:8.1f}     {ss['x_width']:10.1f}"
          f"       {ss['x_signals']:4d}        {ss['mr_signals']:4d}")
    print(f"  interleaved       {si['mrbar']:8.1f}     {si['x_width']:10.1f}"
          f"       {si['x_signals']:4d}        {si['mr_signals']:4d}")
    print(f"  {N_RANDOM//1000:2d}k random mean   {mrbars.mean():8.1f}     {widths.mean():10.1f}"
          f"       {xsignals.mean():4.2f}        {mrsignals.mean():4.2f}")
    print(f"             range  {mrbars.min():.0f}-{mrbars.max():.0f}"
          f"     {widths.min():.0f}-{widths.max():.0f}"
          f"       {xsignals.min():d}-{xsignals.max():d}"
          f"         {mrsignals.min():d}-{mrsignals.max():d}")
    print()
    print(f"  random orderings with zero X signals       {(xsignals == 0).sum():6d}"
          f" / {N_RANDOM} ({100*np.mean(xsignals == 0):.2f}%)")
    print(f"  shipped-order mRbar percentile             {shipped_pct:6.1f}")
    print()

    print("BOUNDARY mR SIGNALS")
    print(f"  as loaded       {fmt_labels(boundary_signal_labels(acos, loaded, T, s0['mr_signal_mask']))}")
    print(f"  sorted by level {fmt_labels(boundary_signal_labels(acos, sorted_order, T, ss['mr_signal_mask']))}")
    print(f"  interleaved     {fmt_labels(boundary_signal_labels(acos, interleaved, T, si['mr_signal_mask']))}")
    print()

    # Descriptive boundary/time comparison for the shipped order.
    # Every boundary changes both organisation and time: last year of i -> first year of j.
    rises = Y[:, -1] - Y[:, 0]
    lane_means = Y.mean(axis=1)
    pure_level_gaps_loaded = np.abs(np.diff(lane_means[loaded]))
    mean_boundary = boundary.mean()
    mean_level_gap = pure_level_gaps_loaded.mean()
    difference_share = (mean_boundary - mean_level_gap) / mean_boundary

    print("BOUNDARY CONSTRUCTION -- descriptive, not an additive decomposition")
    print(f"  mean within-lane first-to-last change      {rises.mean():10.1f}")
    print(f"  mean boundary |last_i - first_j|           {mean_boundary:10.1f}")
    print(f"  mean adjacent pure-level gap |mean_i-mean_j|"
          f" {mean_level_gap:10.1f}")
    print(f"  boundary / pure-level ratio                {mean_boundary/mean_level_gap:10.2f}x")
    print(f"  difference from pure-level comparison      {100*difference_share:10.1f}%"
          " of mean boundary magnitude")
    print()
    print("  NOTE: a boundary mR changes organisation AND resets time from the")
    print("  last observation of one lane to the first observation of the next.")
    print("  The comparison above describes that difference; it does not attribute")
    print("  a causal percentage to the time reset.")
    print()
    print("INTERPRETATION BOUNDARY")
    print("  This experiment does not test whether the lane pattern itself is")
    print("  order-invariant. It asks which numerical chart claims survive a change")
    print("  in lane order while every observation remains fixed.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "aco_per_capita_expenditure.csv")
