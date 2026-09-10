"""Short-series behaviour of MRbar/d2, and why a drift test cannot rescue it.

Reproduces every simulated number in issue #114 on cnicholas/processbehavior.
Self-contained: numpy, pandas, processbehavior. Nothing else, no data files.

    python short_series_sampling.py

Sections
  1  limit width vs series length, on one stable simulated process   (seed 7)
  2  sampling distribution of MRbar/d2 by n                (rng 20260903)
  3  the drift test's null under T iid normal draws       (rng 20260903)
"""
import numpy as np, pandas as pd
import processbehavior as pb
from processbehavior.spc_constants import D2_N2

MEAN, SD = 5.50, 1.20
TRUE_WIDTH = 6.0 * SD                      # natural process limits span 6 sigma
LENGTHS = (3, 4, 6, 8, 12, 20, 30, 60, 150)
NS = (3, 4, 5, 8, 12, 25)   # n=5 was computed but omitted from the table in #114
REPS_SAMPLING = 20_000
REPS_NULL = 400_000
PRECISION = 12                             # do not let rounding move the answer


def x_limits(y, precision=PRECISION):
    """Ask the library for the X-chart limits on this series.

    precision=12 on every call. At the default of 3 the limits are rounded
    before the width is taken, which moves section 1 at these scales.
    """
    df = pd.DataFrame({"t": range(len(y)), "y": np.asarray(y, float)})
    st = pb.formulate(df, response="y", time="t", precision=precision)
    r = st.execute(chart="X", companion=True)
    s = r.get_statistics("X")
    return st, s


def section_1():
    print("1. LIMIT WIDTH vs SERIES LENGTH")
    print("   one stable process, mean %.2f sd %.2f, true limit span %.2f\n"
          % (MEAN, SD, TRUE_WIDTH))
    y = np.random.default_rng(7).normal(MEAN, SD, max(LENGTHS))
    print("   %4s %12s %14s %6s %12s" % ("T", "limit width", "share of true", "ADS", "recommended"))
    for T in LENGTHS:
        st, s = x_limits(y[:T])
        w = float(s["upl"]) - float(s["lpl"])
        print("   %4d %12.2f %13.0f%% %6d %12s"
              % (T, w, 100 * w / TRUE_WIDTH,
                 int(st.analytical_design_state.sds), str(st.recommended_chart)))
    print()


def mrbar_over_d2(y):
    return np.abs(np.diff(y)).mean() / D2_N2


def section_2():
    print("2. SAMPLING DISTRIBUTION OF MRbar/d2, sigma fixed at 1.0")
    print("   %d replicates per length\n" % REPS_SAMPLING)
    rng = np.random.default_rng(20260903)
    print("   %4s %8s %8s %8s %9s %6s" % ("n", "median", "p10", "p90", "p90/p10", "RSD"))
    for n in NS:
        draws = rng.standard_normal((REPS_SAMPLING, n))
        v = np.abs(np.diff(draws, axis=1)).mean(axis=1) / D2_N2
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        print("   %4d %8.3f %8.3f %8.3f %8.2fx %5.0f%%"
              % (n, p50, p10, p90, p90 / p10, 100 * v.std(ddof=1) / v.mean()))
    print()


def drift_share(y):
    d = np.diff(np.asarray(y, float))
    return abs(d.mean()) / np.abs(d).mean()


def section_3():
    print("3. THE DRIFT TEST'S NULL")
    print("   %d replicates of T iid standard normal draws." % REPS_NULL)
    print("   For T iid observations from a continuous distribution, all T!")
    print("   orderings are equally likely, so P(monotone) = 2/T!\n")
    rng = np.random.default_rng(20260903)
    print("   %4s %12s %14s %14s" % ("T", "P(monotone)", "theory 2/T!", "95th pct share"))
    for T in (4, 5, 6):
        z = rng.standard_normal((REPS_NULL, T))
        d = np.diff(z, axis=1)
        mono = ((d > 0).all(axis=1) | (d < 0).all(axis=1)).mean()
        share = np.abs(d.mean(axis=1)) / np.abs(d).mean(axis=1)
        theory = 2.0 / __import__("math").factorial(T)
        print("   %4d %11.4f %14.4f %14.3f"
              % (T, mono, theory, np.percentile(share, 95)))
    print("\n   At T=4, P(monotone)=1/12=8.3% > 5%, so the 95th percentile of the")
    print("   null is 1.000 -- the maximum attainable. No four-point series can")
    print("   exceed it, so the test cannot fire.\n")


if __name__ == "__main__":
    print("processbehavior", pb.__version__, "| numpy", np.__version__,
          "| pandas", pd.__version__, "\n")
    section_1(); section_2(); section_3()
