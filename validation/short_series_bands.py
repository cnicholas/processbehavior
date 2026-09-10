"""Sampling distribution of MRbar/d2 for every n from 3 to 30.

Companion to short_series_sampling.py, for issue #114. Self-contained: numpy
and processbehavior. No data files.

    python short_series_bands.py

WHY THIS IS A SEPARATE SCRIPT WITH A DIFFERENT RNG DESIGN
---------------------------------------------------------
`short_series_sampling.py` draws all its lengths from ONE stream, so its rows
depend on the order of the loop -- which is why the n=5 row has to stay in it.
That is fine for reproducing a fixed table and bad for a table someone will cut
thresholds out of.

Here each length gets its OWN independent stream, `default_rng([SEED, n])`, so
every row is reproducible on its own and no row depends on which other lengths
were computed. Rows therefore agree with the original table to within Monte
Carlo error rather than exactly; the agreement is checked and printed below.

Replicates raised to 100,000 so the band edges are not chosen off noise. The
Monte Carlo standard error of each reported quantile is given, because a
threshold placed where two adjacent n differ by less than their MC error is a
threshold placed on nothing.

NO CRITERION IS PROPOSED HERE. Quantiles and spreads only. Which functional of
this distribution should govern an adequacy band -- p90/p10, RSD, a coverage
probability, something else -- and where the cut falls are design decisions for
the maintainers. This supplies the arithmetic once that choice is made.
"""
import numpy as np
from processbehavior.spc_constants import D2_N2

SEED = 20260903
REPS = 100_000
LENGTHS = range(3, 31)
QS = [5, 10, 25, 50, 75, 90, 95]


def sweep(n, reps=REPS, seed=SEED):
    rng = np.random.default_rng([seed, n])
    z = rng.standard_normal((reps, n))
    return np.abs(np.diff(z, axis=1)).mean(axis=1) / D2_N2


def q_se(v, q):
    """MC standard error of the q-th percentile, via the density at that point."""
    p = q / 100.0
    x = np.percentile(v, q)
    h = 1.06 * v.std(ddof=1) * len(v) ** -0.2          # Silverman bandwidth
    dens = np.mean(np.abs(v - x) < h) / (2 * h)
    return np.sqrt(p * (1 - p) / len(v)) / dens if dens > 0 else float("nan")


if __name__ == "__main__":
    import processbehavior as pb
    print("processbehavior %s | numpy %s" % (pb.__version__, np.__version__))
    print("MRbar/d2 under an iid standard normal process, sigma = 1.0")
    print("%d replicates per length, independent stream per n "
          "(default_rng([%d, n]))\n" % (REPS, SEED))

    hdr = "%4s" % "n" + "".join("%8s" % ("p%d" % q) for q in QS) + \
          "%9s%8s%10s%10s" % ("p90/p10", "RSD", "se(p10)", "se(p90)")
    print(hdr); print("-" * len(hdr))
    rows = {}
    for n in LENGTHS:
        v = sweep(n)
        qv = np.percentile(v, QS)
        rsd = v.std(ddof=1) / v.mean()
        rows[n] = dict(zip(QS, qv))
        print("%4d" % n + "".join("%8.3f" % x for x in qv) +
              "%8.2fx%8.0f%%%10.4f%10.4f"
              % (qv[QS.index(90)] / qv[QS.index(10)], 100 * rsd,
                 q_se(v, 10), q_se(v, 90)))

    print("\nAGREEMENT WITH THE TABLE IN THE ISSUE (single-stream, 20k reps)")
    pub = {3: (0.889, 0.337, 1.797), 4: (0.925, 0.429, 1.689),
           5: (0.942, 0.482, 1.585), 8: (0.969, 0.592, 1.454),
           12: (0.976, 0.671, 1.351), 25: (0.989, 0.769, 1.245)}
    print("%4s %22s %22s %10s" % ("n", "published (p50/p10/p90)",
                                  "here (p50/p10/p90)", "max |diff|"))
    for n, (m, a, b) in pub.items():
        h = (rows[n][50], rows[n][10], rows[n][90])
        print("%4d %22s %22s %10.4f"
              % (n, "%.3f / %.3f / %.3f" % (m, a, b),
                 "%.3f / %.3f / %.3f" % h,
                 max(abs(h[0] - m), abs(h[1] - a), abs(h[2] - b))))
    print("\nDifferences are Monte Carlo, not a discrepancy: the two scripts use\n"
          "different streams by design.")
