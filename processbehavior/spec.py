from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnalysisSpec:
    """Declarative spec for SPC analyses.

    Keep this small and stable. Add optional sections over time (rules, residuals, changepoints).
    """

    response_var: str
    time_var: str | None = None
    grouping: list[str] = field(default_factory=list)  # rational subgroup keys
    analyses: list[dict[str, Any]] = field(default_factory=lambda: [{'type': 'XBAR_S'}])
    residuals: dict[str, Any] | None = None
    rules: dict[str, Any] | None = None
    rounding: int = 4
    spec_version: str = '1.0'

    def __post_init__(self):
        if not self.response_var:
            raise ValueError('response_var is required')
        if any(not g for g in self.grouping):
            raise ValueError('grouping contains empty values')
        for a in self.analyses:
            if 'type' not in a:
                raise ValueError("each analysis dict must include a 'type' key")
