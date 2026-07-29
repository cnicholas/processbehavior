"""
Derived Variables — serializable specifications for transforms and binning.

A :class:`Derivation` is a *spec* (data, not code) that describes a new column
derived from an existing one: a continuous→continuous **transform** (``log``,
``sqrt``, ``arcsin(√x)``, ``zscore``, …) or a continuous→categorical **bin**
(``equal_freq``/``equal_width``/``breaks``/``sd``). Specs are evaluated against a
source column by :func:`evaluate`, which is the single primitive shared by the
fluent ``ProcessBehavior`` verbs (attach path) and the application's attach-free
live preview.

Design notes
------------
- The spec carries an opaque ``id`` (identity, distinct from the editable
  ``label``) and round-trips to/from a plain dict via :meth:`Derivation.to_dict`
  / :meth:`Derivation.from_dict`.
- Data-dependent fits (bin edges, z-score μ/σ) are resolved by ``evaluate`` and
  written into ``fitted``; they freeze when a study is formulated.
- ``evaluate`` / :func:`validate` never raise on routine user or domain states —
  domain violations are returned as structured data. Exceptions are reserved for
  programmer error and the explicit ``on_invalid='error'`` commit at formulation.

Box–Cox is intentionally out of scope for v1 (it would need scipy); the transform
registry leaves a clean slot to add it behind an optional import later.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from .exceptions import ValidationError

# ============================================================================
# Enums
# ============================================================================

# Canonical transform functions (v1). ``ln`` is an accepted alias for ``log``.
TRANSFORM_FUNCTIONS: tuple[str, ...] = (
    'log', 'log10', 'sqrt', 'arcsin', 'inverse', 'square', 'power', 'zscore',
)
BIN_METHODS: tuple[str, ...] = ('equal_freq', 'equal_width', 'breaks', 'sd')
BIN_LABEL_STYLES: tuple[str, ...] = ('range', 'ordinal', 'number')

# Closed-boundary clamp tolerance: float noise this close to a *closed* domain
# boundary is clamped to it (valid), not flagged as a domain violation.
_CLAMP_EPS = 1e-9

# Curated ordinal labels keyed on the *fitted* bin count.
_ORDINAL_LABELS: dict[int, list[str]] = {
    2: ['Low', 'High'],
    3: ['Low', 'Medium', 'High'],
    4: ['Low', 'Medium-Low', 'Medium-High', 'High'],
    5: ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
}


# ============================================================================
# JSON-safe (de)serialization for params / fitted (handles ±inf, nan)
# ============================================================================


def _jsonify(obj):
    """Recursively make a value JSON-safe, encoding non-finite floats as tags."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isinf(obj):
            return 'Infinity' if obj > 0 else '-Infinity'
        if math.isnan(obj):
            return 'NaN'
        return obj
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _dejsonify(obj):
    """Inverse of :func:`_jsonify`."""
    if obj == 'Infinity':
        return math.inf
    if obj == '-Infinity':
        return -math.inf
    if obj == 'NaN':
        return math.nan
    if isinstance(obj, dict):
        return {k: _dejsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dejsonify(v) for v in obj]
    return obj


def _ascending(seq) -> bool:
    return all(a < b for a, b in zip(seq, seq[1:]))


# ============================================================================
# Result objects
# ============================================================================


@dataclass
class EvalResult:
    """Outcome of evaluating one :class:`Derivation` against a column.

    Attributes
    ----------
    values : pandas.Series
        The derived column (numeric for transforms, ordered Categorical for bins).
    n_invalid : int
        Count of domain violations among non-NA inputs (never counts NA).
    invalid_index : pandas.Index
        Row labels of the violating inputs (contains no NA-row labels).
    fitted : dict
        Resolved data-dependent params to write back onto the spec
        (e.g. ``{'mu', 'sigma'}`` for z-score, ``{'edges', 'n_bins', …}`` for bins).
    message : str or None
        Human-readable note (e.g. a qcut bin-count drop), or None.
    """

    values: pd.Series
    n_invalid: int
    invalid_index: pd.Index
    fitted: dict
    message: str | None = None


@dataclass
class ValidationResult:
    """Structured pre-commit check (no exceptions for routine states)."""

    ok: bool
    issues: list[dict]

    def summary(self) -> str:
        return '; '.join(i.get('message', i.get('code', '')) for i in self.issues)


# ============================================================================
# Derivation spec
# ============================================================================


@dataclass(frozen=True)
class Derivation:
    """A serializable derived-variable specification.

    Construct via the :meth:`transform` / :meth:`bin` factories rather than
    directly. Equality is by content (``family``/``column``/``function``/
    ``label``/``params``); ``id`` and ``fitted`` are excluded from equality —
    ``id`` is opaque identity, ``fitted`` is resolved state.
    """

    family: str
    column: str
    function: str
    label: str | None = None
    params: dict = field(default_factory=dict)
    fitted: dict = field(default_factory=dict, compare=False)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], compare=False)

    # -- validation (branches on family) -----------------------------------
    def __post_init__(self) -> None:
        if not self.column or not isinstance(self.column, str):
            raise ValidationError('Derivation.column must be a non-empty string.')

        if self.family == 'transform':
            if self.function not in TRANSFORM_FUNCTIONS:
                raise ValidationError(
                    f"Unknown transform function {self.function!r}. "
                    f'Supported: {list(TRANSFORM_FUNCTIONS)} (ln is an alias for log).'
                )
            if self.function == 'power' and 'exponent' not in self.params:
                raise ValidationError("transform 'power' requires an 'exponent' param.")

        elif self.family == 'bin':
            if self.function != 'bin':
                raise ValidationError(
                    f"bin derivations must have function='bin', got {self.function!r}."
                )
            method = self.params.get('method')
            if method not in BIN_METHODS:
                raise ValidationError(
                    f"Unknown bin method {method!r}. Supported: {list(BIN_METHODS)}."
                )
            if method == 'breaks':
                breaks = self.params.get('breaks')
                if not breaks or len(breaks) < 1 or not _ascending(breaks):
                    raise ValidationError(
                        "bin method 'breaks' requires an ascending list of cut points."
                    )
            elif method in ('equal_freq', 'equal_width'):
                n = self.params.get('n')
                if not (isinstance(n, int) and n > 0):
                    raise ValidationError(
                        f"bin method {method!r} requires an integer n > 0, got {n!r}."
                    )
            bl = self.params.get('bin_labels', 'range')
            if isinstance(bl, str) and bl not in BIN_LABEL_STYLES:
                raise ValidationError(
                    f"Unknown bin_labels style {bl!r}. Use one of {list(BIN_LABEL_STYLES)} "
                    'or an explicit list of names.'
                )
        else:
            raise ValidationError(
                f"Derivation.family must be 'transform' or 'bin', got {self.family!r}."
            )

    # -- factories ---------------------------------------------------------
    @classmethod
    def transform(
        cls,
        column: str,
        function: str,
        *,
        label: str | None = None,
        shift: float | None = None,
        exponent: float | None = None,
        on_invalid: str = 'error',
    ) -> Derivation:
        """Build a transform spec. ``ln`` canonicalizes to ``log``."""
        fn = 'log' if function == 'ln' else function
        params: dict = {'on_invalid': on_invalid}
        if shift is not None:
            params['shift'] = shift
        if exponent is not None:
            params['exponent'] = exponent
        return cls(family='transform', column=column, function=fn, label=label, params=params)

    @classmethod
    def bin(
        cls,
        column: str,
        *,
        method: str = 'equal_freq',
        n: int = 4,
        breaks: list[float] | None = None,
        bin_labels='range',
        label: str | None = None,
        right: bool = False,
    ) -> Derivation:
        """Build a binning spec."""
        params: dict = {'method': method, 'bin_labels': bin_labels, 'right': right}
        if method == 'breaks':
            params['breaks'] = list(breaks) if breaks is not None else None
        else:
            params['n'] = n
        return cls(family='bin', column=column, function='bin', label=label, params=params)

    # -- naming / serialization -------------------------------------------
    @property
    def output_name(self) -> str:
        """Output column name — ``label`` or ``{column}_{function}`` (bins → ``{column}_bin``)."""
        return self.label or f'{self.column}_{self.function}'

    def to_dict(self) -> dict:
        """Plain JSON-safe dict (includes ``id`` and frozen ``fitted``)."""
        return {
            'id': self.id,
            'family': self.family,
            'column': self.column,
            'function': self.function,
            'label': self.label,
            'params': _jsonify(self.params),
            'fitted': _jsonify(self.fitted),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Derivation:
        """Reconstruct from :meth:`to_dict`. Takes ``id`` from the dict (never mints)."""
        return cls(
            family=d['family'],
            column=d['column'],
            function=d['function'],
            label=d.get('label'),
            params=_dejsonify(d.get('params', {})),
            fitted=_dejsonify(d.get('fitted', {})),
            id=d['id'],
        )

    def with_fitted(self, fitted: dict) -> Derivation:
        """Return a copy with resolved ``fitted`` values frozen on (same ``id``)."""
        return replace(self, fitted=dict(fitted))


# ============================================================================
# Transform registry
# ============================================================================


def _bounded_domain(lower=None, lower_closed=True, upper=None, upper_closed=True):
    """Return a ``(x, present) -> (clamped, violation)`` domain checker.

    Closed boundaries clamp float-noise within ``_CLAMP_EPS`` to the boundary
    (valid, not counted); open boundaries flag the boundary value itself.
    NA is never folded into the violation mask (``present`` gates every term).
    """

    def check(x: pd.Series, present: pd.Series):
        clamped = x.copy()
        violation = pd.Series(False, index=x.index)
        if lower is not None:
            if lower_closed:
                near = present & (x < lower) & (x >= lower - _CLAMP_EPS)
                clamped = clamped.mask(near, lower)
                violation = violation | (present & (x < lower - _CLAMP_EPS))
            else:
                violation = violation | (present & (x <= lower))
        if upper is not None:
            if upper_closed:
                near = present & (x > upper) & (x <= upper + _CLAMP_EPS)
                clamped = clamped.mask(near, upper)
                violation = violation | (present & (x > upper + _CLAMP_EPS))
            else:
                violation = violation | (present & (x >= upper))
        return clamped, violation

    return check


# name -> (numpy fn applied to clamped input, domain checker)
_TRANSFORM_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    'log': (np.log, _bounded_domain(lower=0.0, lower_closed=False)),
    'log10': (np.log10, _bounded_domain(lower=0.0, lower_closed=False)),
    'sqrt': (np.sqrt, _bounded_domain(lower=0.0, lower_closed=True)),
    'arcsin': (
        lambda a: np.arcsin(np.sqrt(a)),
        _bounded_domain(lower=0.0, lower_closed=True, upper=1.0, upper_closed=True),
    ),
    'square': (lambda a: np.power(a, 2), _bounded_domain()),
    'inverse': (lambda a: 1.0 / a, lambda x, present: (x, present & (x == 0))),
}


def _evaluate_transform(spec: Derivation, col: pd.Series) -> EvalResult:
    x = pd.to_numeric(col, errors='coerce').astype('float64')
    present = x.notna()
    params = spec.params
    shift = params.get('shift')
    if shift is not None:
        x = x + shift

    fitted: dict = {}
    message: str | None = None

    if spec.function == 'zscore':
        clamped, violation = x, pd.Series(False, index=x.index)
        vals = x[present]
        mu = float(vals.mean()) if present.any() else math.nan
        sigma = float(vals.std(ddof=1)) if present.sum() > 1 else math.nan
        fitted = {'mu': mu, 'sigma': sigma}
        with np.errstate(all='ignore'):
            y = (x - mu) / sigma
        if not (sigma and math.isfinite(sigma) and sigma > 0):
            message = 'zero or undefined variance; zscore is undefined'

    elif spec.function == 'power':
        exponent = params['exponent']
        with np.errstate(all='ignore'):
            y = pd.Series(np.power(x.to_numpy(), exponent), index=x.index)
        # A power is a domain violation where it cannot be represented as a
        # finite real (negative base to a fractional power, 0 to a negative power).
        violation = present & ~np.isfinite(y)

    else:
        fn, domain = _TRANSFORM_REGISTRY[spec.function]
        clamped, violation = domain(x, present)
        with np.errstate(all='ignore'):
            y = pd.Series(fn(clamped.to_numpy()), index=x.index)

    # Violations -> NaN in the output (NA inputs already produce NaN).
    y = y.where(~violation)
    n_invalid = int(violation.sum())
    invalid_index = x.index[violation]
    return EvalResult(
        values=y, n_invalid=n_invalid, invalid_index=invalid_index, fitted=fitted, message=message
    )


# ============================================================================
# Binning
# ============================================================================


def _fmt(v: float) -> str:
    return f'{v:g}'


def _range_labels(edges, right: bool) -> list[str]:
    labels = []
    count = len(edges) - 1
    for i in range(count):
        a, b = edges[i], edges[i + 1]
        if math.isinf(a):
            labels.append(f'< {_fmt(b)}' if not right else f'<= {_fmt(b)}')
        elif math.isinf(b):
            labels.append(f'>= {_fmt(a)}' if not right else f'> {_fmt(a)}')
        else:
            left = '[' if not right else '('
            rb = ')' if not right else ']'
            # final (right=False) / first (right=True) interval is closed on the
            # bounded side so the extreme value is included.
            if not right and i == count - 1:
                rb = ']'
            if right and i == 0:
                left = '['
            labels.append(f'{left}{_fmt(a)}, {_fmt(b)}{rb}')
    return labels


def _bin_label_names(bin_labels, edges, right: bool):
    """Resolve the ordered label names for the fitted edges. Returns (labels, message)."""
    count = len(edges) - 1
    if isinstance(bin_labels, (list, tuple)):
        if len(bin_labels) != count:
            return _range_labels(edges, right), (
                f'{len(bin_labels)} labels supplied but {count} bins fitted; using range labels'
            )
        return list(bin_labels), None
    if bin_labels == 'ordinal':
        if count in _ORDINAL_LABELS:
            return list(_ORDINAL_LABELS[count]), None
        return [f'Bin {i + 1}' for i in range(count)], None  # fall back to number
    if bin_labels == 'number':
        return [f'Bin {i + 1}' for i in range(count)], None
    return _range_labels(edges, right), None


def _fit_edges(spec: Derivation, present_vals: pd.Series):
    """Return (edges, message) for the requested method, fitted on non-NA values."""
    params = spec.params
    method = params['method']
    message = None

    if method == 'breaks':
        edges = [-math.inf, *[float(b) for b in params['breaks']], math.inf]
    elif method == 'sd':
        mu = float(present_vals.mean())
        sigma = float(present_vals.std(ddof=1)) if len(present_vals) > 1 else math.nan
        edges = [-math.inf, mu - 2 * sigma, mu - sigma, mu + sigma, mu + 2 * sigma, math.inf]
    elif method == 'equal_width':
        n = params['n']
        lo, hi = float(present_vals.min()), float(present_vals.max())
        edges = list(np.linspace(lo, hi, n + 1))
    else:  # equal_freq
        n = params['n']
        qs = np.linspace(0.0, 1.0, n + 1)
        edges = list(np.unique(np.quantile(present_vals, qs)))
        if len(edges) - 1 != n:
            message = f'requested {n} bins, ties produced {len(edges) - 1}'
    return edges, message


def _evaluate_bin(spec: Derivation, col: pd.Series) -> EvalResult:
    x = pd.to_numeric(col, errors='coerce').astype('float64')
    present = x.notna()
    params = spec.params
    right = params.get('right', False)
    empty_index = x.index[[]]

    if present.sum() == 0:
        return EvalResult(
            values=pd.Series(pd.Categorical([np.nan] * len(x)), index=x.index),
            n_invalid=0, invalid_index=empty_index, fitted={}, message='no non-NA values to bin',
        )

    edges, fit_msg = _fit_edges(spec, x[present])

    # Degenerate fit (no spread) — cannot bin.
    if len(edges) < 2 or any(not math.isfinite(e) for e in edges[1:-1]):
        return EvalResult(
            values=pd.Series(pd.Categorical([np.nan] * len(x)), index=x.index),
            n_invalid=0, invalid_index=empty_index, fitted={'method': params['method']},
            message='column has no spread; cannot bin',
        )

    labels, label_msg = _bin_label_names(params.get('bin_labels', 'range'), edges, right)

    # Close the top edge so the maximum lands in a bin — which is what _range_labels already
    # promises by rendering the final interval with a closed bracket. `include_lowest` is
    # pandas' equivalent for the *bottom* edge and covers right=True; nothing covered the
    # top, so under right=False (the default) every observation equal to the maximum binned
    # to NaN and silently left the analysis. Only right=False needs this: the minimum was
    # always included, by include_lowest for right=True and by the half-open interval
    # otherwise. nextafter is the minimal correct widening — no epsilon to tune. `edges`
    # itself is untouched, so fitted['edges'] and the labels keep the true cut points.
    cut_edges = list(edges)
    if not right and math.isfinite(cut_edges[-1]):
        cut_edges[-1] = float(np.nextafter(cut_edges[-1], math.inf))

    cats = pd.cut(x, bins=cut_edges, right=right, labels=labels, include_lowest=True, ordered=True)
    values = pd.Series(cats, index=x.index)

    fitted = {
        'method': params['method'],
        'edges': [float(e) for e in edges],
        'n_bins': len(edges) - 1,
        'right': right,
        'labels': list(labels),
    }
    if params['method'] == 'sd':
        fitted['mu'] = float(x[present].mean())
        fitted['sigma'] = float(x[present].std(ddof=1)) if present.sum() > 1 else math.nan

    message = '; '.join(m for m in (fit_msg, label_msg) if m) or None
    return EvalResult(
        values=values, n_invalid=0, invalid_index=empty_index, fitted=fitted, message=message
    )


# ============================================================================
# Public engine
# ============================================================================


def evaluate(spec: Derivation, column: pd.Series) -> EvalResult:
    """Evaluate one derivation against a column — the shared preview/attach primitive.

    Pre-existing NA passes through as NA; ``n_invalid`` counts only domain
    violations among non-NA values. Never raises.
    """
    if spec.family == 'transform':
        return _evaluate_transform(spec, column)
    return _evaluate_bin(spec, column)


def validate(spec: Derivation, dataset: pd.DataFrame, existing_names=None) -> ValidationResult:
    """Structured pre-commit check (label collision, dtype, bin-label count, breaks order).

    Evaluates the spec against ``dataset[spec.column]`` so a bin's explicit-label
    count is checked against the **fitted** bin count (post-qcut), not the
    requested ``n``. Domain violations are *not* failures — they live in the
    :class:`EvalResult` data channel. Never raises.
    """
    issues: list[dict] = []
    existing = set(existing_names or set()) | set(dataset.columns)

    if spec.column not in dataset.columns:
        issues.append({
            'code': 'column_not_found',
            'message': f"Column '{spec.column}' not found. Available: {list(dataset.columns)}",
        })
        return ValidationResult(ok=False, issues=issues)

    if not pd.api.types.is_numeric_dtype(dataset[spec.column]):
        issues.append({
            'code': 'not_numeric',
            'message': f"Column '{spec.column}' is not numeric; cannot derive from it.",
        })

    name = spec.output_name
    if name in existing:
        issues.append({
            'code': 'label_collision',
            'message': f"Output name '{name}' already exists; choose a different label.",
        })

    if spec.family == 'bin' and spec.params.get('method') == 'breaks':
        breaks = spec.params.get('breaks') or []
        if not _ascending(breaks):
            issues.append({'code': 'breaks_order', 'message': 'breaks must be strictly ascending.'})

    # Evaluate to check bin-label count against the *fitted* bin count.
    if not any(i['code'] in ('not_numeric',) for i in issues):
        res = evaluate(spec, dataset[spec.column])
        if spec.family == 'bin':
            bl = spec.params.get('bin_labels')
            fitted_n = res.fitted.get('n_bins')
            if isinstance(bl, (list, tuple)) and fitted_n is not None and len(bl) != fitted_n:
                issues.append({
                    'code': 'label_count',
                    'message': f'{len(bl)} labels supplied but {fitted_n} bins fitted.',
                })

    return ValidationResult(ok=len(issues) == 0, issues=issues)


# ============================================================================
# Free-function conveniences (delegate to ProcessBehavior methods)
# ============================================================================


def derivations(pb) -> tuple:
    """Return the pending derivation specs attached to ``pb``."""
    return pb.derivations


def remove_derived(pb, id: str):
    """Return a new ProcessBehavior with the derivation ``id`` removed."""
    return pb.remove_derived(id)


def replace_derived(pb, id: str, spec: Derivation):
    """Return a new ProcessBehavior with the derivation ``id`` replaced by ``spec``."""
    return pb.replace_derived(id, spec)
