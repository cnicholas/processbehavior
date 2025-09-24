from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnalysisType(Enum):
    """Supported analysis chart types."""

    XBAR = 'Xbar'
    S = 'S'
    IMR = 'Imr'
    R = 'R'


class TimeUnit(Enum):
    """Supported time grouping units."""

    YEAR = 'Year'
    QUARTER = 'Quarter'
    MONTH = 'Month'
    WEEK = 'Week'


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


@dataclass
class AnalysisSpecification:
    """Specification class for SPC analysis configuration.

    Validates and processes analysis specifications for process behavior charts.
    Supports Xbar, S, Imr, and R chart types with flexible grouping and time options.
    """

    analysis_type: AnalysisType | str  # Accept both enum and string for backward compatibility
    response_var: str
    rsg_vars: list[str] | None = None
    rsg_var_name: str = 'rsg'
    rsg_var_delim: str = '_'
    time_var: str | None = None
    time_unit: TimeUnit | str | None = None  # Accept both enum and string
    round_to: int = 3

    def __post_init__(self):
        # Normalize enum inputs to their values for backward compatibility
        if isinstance(self.analysis_type, AnalysisType):
            object.__setattr__(self, 'analysis_type', self.analysis_type.value)
        if isinstance(self.time_unit, TimeUnit):
            object.__setattr__(self, 'time_unit', self.time_unit.value)

        # Initialize computed properties
        self._has_grouping = self.rsg_vars is not None
        self._has_time = self.time_var is not None
        self._requires_sort = self._has_time
        self._grouping_cols = [self.rsg_var_name] if self._has_grouping else []

        # Validate required parameters
        self._validate_inputs()

        # Build derived collections
        self._build_time_grouping_units()
        self._time_grouping_cols = self._build_time_unit_cols()
        self._sort_cols = self._build_sort_cols()
        self._data_prep_output_cols = self._build_data_prep_cols()
        self._analysis_output_cols = self._build_analysis_output_cols()

    @property
    def supported_analysis_types(self) -> list[str]:
        """Get list of supported analysis types."""
        return [t.value for t in AnalysisType]

    @property
    def valid_time_units(self) -> list[str]:
        """Get list of valid time units."""
        return [t.value for t in TimeUnit]

    @property
    def grouped_analyses(self) -> list[str]:
        """Get list of analysis types that require grouping."""
        return [AnalysisType.XBAR.value, AnalysisType.S.value]

    @classmethod
    def from_dict(cls, analysis_type: AnalysisType | str, analysis_specification: dict):
        """Create AnalysisSpecification from dictionary (legacy compatibility)."""
        response_var = analysis_specification.get('response_var')
        if response_var is None:
            raise ValueError('response_var is required in analysis_specification dictionary')

        return cls(
            analysis_type=analysis_type,
            response_var=response_var,
            rsg_vars=analysis_specification.get('rsg_vars'),
            rsg_var_name=analysis_specification.get('rsg_var_name', 'rsg'),
            rsg_var_delim=analysis_specification.get('rsg_var_delim', '_'),
            time_var=analysis_specification.get('time_var'),
            time_unit=analysis_specification.get('time_unit'),
            round_to=analysis_specification.get('round_to', 3),
        )

    @classmethod
    def for_xbar_chart(cls, response_var: str, rsg_vars: list[str], **kwargs):
        """Factory method for Xbar charts."""
        return cls(
            analysis_type=AnalysisType.XBAR, response_var=response_var, rsg_vars=rsg_vars, **kwargs
        )

    @classmethod
    def for_imr_chart(cls, response_var: str, **kwargs):
        """Factory method for IMR charts."""
        return cls(analysis_type=AnalysisType.IMR, response_var=response_var, **kwargs)

    def _validate_inputs(self) -> None:
        """Validate all input parameters."""
        if self.analysis_type not in self.supported_analysis_types:
            raise ValueError(
                f'Analysis type: {self.analysis_type} is not supported, '
                f'specify one of: {self.supported_analysis_types}!'
            )

        if self.analysis_type in self.grouped_analyses and self.rsg_vars is None:
            raise ValueError(
                f'A grouping variable is required to produce a {self.analysis_type} analysis!'
            )

        if self.response_var is None:
            raise ValueError(
                f'A response variable is required to produce a {self.analysis_type} analysis!'
            )

        if self.time_unit is not None:
            if not self._has_time:
                raise ValueError('A time variable is required when a time_unit is provided!')
            if self.time_unit not in self.valid_time_units:
                raise ValueError(f'Time unit must be one: {self.valid_time_units}!')

    @property
    def data_prep_output_cols(self) -> list:
        return self._data_prep_output_cols

    @property
    def analysis_output_cols(self) -> list:
        return self._analysis_output_cols

    @property
    def sort_cols(self) -> list:
        return self._sort_cols

    @property
    def time_grouping_units(self) -> list:
        return self._time_grouping_units

    @property
    def time_grouping_cols(self) -> dict:
        return self._time_grouping_cols

    @property
    def has_grouping(self) -> bool:
        return self._has_grouping

    @property
    def grouping_cols(self) -> list:
        return self._grouping_cols

    @property
    def has_time(self) -> bool:
        return self._has_time

    @property
    def requires_sort(self) -> bool:
        return self._requires_sort

    def _build_time_grouping_units(self) -> None:
        """Build time grouping units list."""
        if self.time_unit is None:
            self._time_grouping_units = []
        elif self.time_unit == 'Year':
            self._time_grouping_units = ['Year']
        else:
            self._time_grouping_units = ['Year', self.time_unit]

    def _build_time_unit_cols(self) -> dict:
        """Build time unit column mappings."""
        return (
            {unit: f'{self.time_var}_{unit.lower()}' for unit in self._time_grouping_units}
            if self.time_var
            else {}
        )

    def _build_sort_cols(self) -> list:
        """Build list of columns for sorting."""
        if not self._has_time:
            return []
        if self._has_grouping:
            return [self.rsg_var_name, self.time_var]
        return [self.time_var]

    def _build_data_prep_cols(self) -> list:
        """Build list of columns to keep after data preparation."""
        cols = [self.response_var]

        # Add time unit columns
        cols.extend(self._time_grouping_cols.values())

        # Add grouping columns
        if self._has_grouping:
            cols = [self.rsg_var_name, 'n'] + cols

        # Add time column
        if self._has_time:
            cols = [self.time_var] + cols

        return cols

    def _build_analysis_output_cols(self) -> list:
        """Build list of columns for analysis output."""
        cols = [self.response_var, 'mean', 'lcl', 'ucl', 'beyond_limits']

        if self._has_grouping:
            cols.insert(0, self.rsg_var_name)

        if self._has_time:
            cols.insert(0, self.time_var)
        else:
            cols.insert(0, 'x')

        return cols
