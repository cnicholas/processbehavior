"""
Named Calibration value objects — frozen mean/sigma for control-chart limits.

A ``Calibration`` lets an analyst pin a chart's limits to a known
("standards-given", in Wheeler's terms) mean and within-subgroup individual
standard deviation instead of estimating them from the data on hand. It mirrors
the value-object style of :class:`~processbehavior.capability.SpecLimits`, but
where ``SpecLimits`` feeds the capability histogram, a ``Calibration`` feeds the
control-chart limit machinery via ``study.execute(calibration=...)``.

Semantics (applied *forward* from the given individual sigma; the sigma is
never run back through c4/d2/b3/b4 to "recover" a process sigma):

- **Location charts** (X individuals, raw response, recentered residual, plain
  residual) place limits constant-free in sigma: ``center ± n_sigma·sigma``
  (Xbar divides the half-width by ``√N`` because it plots subgroup means).
- **Dispersion-statistic charts** (S, mR) necessarily carry their
  sampling-distribution constants, because their y-axis is a dispersion
  statistic, not sigma: S centers at ``c4(N)·sigma``; mR at ``d2(2)·sigma``.

References
----------
Wheeler, D. J. *Understanding Statistical Process Control* — standards-given
(known mean/sigma) limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .exceptions import ValidationError


@dataclass(frozen=True)
class Calibration:
    """A named, frozen ``(mean, sigma)`` pair for standards-given control limits.

    Parameters
    ----------
    label : str
        Non-empty identifier used to key the calibration on a ``Study`` and to
        select it by name in ``study.execute(calibration=...)``.
    mean : float
        The known process mean (center line) for location charts. Ignored for a
        plain residual chart, whose center is fixed at 0.
    sigma : float
        The known within-subgroup standard deviation of **individual** values
        (``> 0``). Used as-is; each chart shows the sampling distribution of the
        statistic it plots, so an Xbar band is narrower by ``√N`` and an S chart
        centers at ``c4(N)·sigma`` — both correct, neither a bug.

    Examples
    --------
    >>> cal = Calibration(label='line-3 baseline', mean=10.0, sigma=0.5)
    >>> cal.sigma
    0.5
    """

    label: str
    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise ValidationError('Calibration.label must be a non-empty string.')
        if not math.isfinite(self.mean):
            raise ValidationError(f'Calibration.mean must be finite, got {self.mean!r}.')
        if not (math.isfinite(self.sigma) and self.sigma > 0):
            raise ValidationError(f'Calibration.sigma must be finite and > 0, got {self.sigma!r}.')
