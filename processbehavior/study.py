"""
Study class for process behavior analysis formulation.

The Study object represents a formulated analysis - it knows what data structure
you have (SDS), what charts are valid, and guides you toward correct analysis.

This is the "teaching" layer of the API that helps users understand their data
before running calculations.

Design Philosophy (Pythonic Hadley):
- Human-first: Rich __repr__ teaches users about their data
- Pit of success: Valid charts shown, invalid charts explained
- Composability: study.execute() returns AnalysisResult for chaining
- Immutable: Frozen dataclass, different formulations create new objects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .exceptions import ChartNotAvailableError

if TYPE_CHECKING:
    import pandas as pd

    from .analysis_dataset import AnalysisDataSet
    from .analysis_result import AnalysisResult
    from .analysis_specification import DataPrepConfig
    from .process_behavior import ProcessBehavior
    from .sds_detector import SDSAnalysisPlan


@dataclass
class DesignReport:
    """
    Compares sampling plan to observed data.

    Returned by study.design(). Provides insight into the experimental
    design structure and any mismatches between plan and observation.

    Attributes
    ----------
    factors : pd.DataFrame
        Factor-level summary table with columns: factor, planned, observed,
        missing_levels, extra_levels
    missing_levels : dict[str, list]
        Levels in plan but not observed, per factor
    extra_levels : dict[str, list]
        Levels observed but not in plan, per factor

    Examples
    --------
    >>> design = study.design()
    >>> design
    DesignReport(2 factors, 1 missing_levels, 0 extra_levels)
      Lane: planned=[1,2,3,4], observed=[1,2,3,4]
      Phase: planned=[1,2,3], observed=[1,2], missing=[3]

    >>> design.factors
       factor     planned      observed  missing_levels  extra_levels
    0    Lane  [1,2,3,4]    [1,2,3,4]              []            []
    1   Phase    [1,2,3]        [1,2]             [3]            []

    >>> design.missing_levels
    {'Lane': [], 'Phase': [3]}
    """

    _sampling_plan: dict[str, list] | None
    _observed_levels: dict[str, list]
    _factors: list[str]

    @property
    def factors(self) -> pd.DataFrame:
        """
        Factor-level summary table.

        Returns DataFrame with columns:
        - factor: Factor column name
        - planned: Levels in plan (or observed if no plan)
        - observed: Levels actually in data
        - missing_levels: Levels in plan but not in data
        - extra_levels: Levels in data but not in plan
        """
        import pandas as pd

        rows = []
        for factor in self._factors:
            observed = self._observed_levels.get(factor, [])
            if self._sampling_plan is not None:
                planned = self._sampling_plan.get(factor, observed)
            else:
                planned = observed

            planned_set = set(planned)
            observed_set = set(observed)

            missing = self._safe_sort(list(planned_set - observed_set))
            extra = self._safe_sort(list(observed_set - planned_set))

            rows.append({
                'factor': factor,
                'planned': planned,
                'observed': observed,
                'missing_levels': missing,
                'extra_levels': extra
            })

        return pd.DataFrame(rows)

    @property
    def missing_levels(self) -> dict[str, list]:
        """Levels in plan but not observed, per factor."""
        result: dict[str, list] = {}
        for factor in self._factors:
            observed = set(self._observed_levels.get(factor, []))
            if self._sampling_plan is not None:
                planned = set(self._sampling_plan.get(factor, []))
            else:
                planned = observed
            result[factor] = self._safe_sort(list(planned - observed))
        return result

    @property
    def extra_levels(self) -> dict[str, list]:
        """Levels observed but not in plan, per factor."""
        result: dict[str, list] = {}
        for factor in self._factors:
            observed = set(self._observed_levels.get(factor, []))
            if self._sampling_plan is not None:
                planned = set(self._sampling_plan.get(factor, []))
            else:
                planned = observed
            result[factor] = self._safe_sort(list(observed - planned))
        return result

    @property
    def has_plan(self) -> bool:
        """Whether a sampling plan was provided."""
        return self._sampling_plan is not None

    @staticmethod
    def _safe_sort(items: list) -> list:
        """Sort items safely, handling mixed types."""
        try:
            return sorted(items)
        except TypeError:
            return list(items)

    def __repr__(self) -> str:
        """Nice summary showing plan vs observed per factor."""
        # Count total missing and extra
        total_missing = sum(len(v) for v in self.missing_levels.values())
        total_extra = sum(len(v) for v in self.extra_levels.values())

        plan_status = "with plan" if self.has_plan else "observed only"
        header = f"DesignReport({len(self._factors)} factors, {plan_status})"

        if total_missing > 0 or total_extra > 0:
            header += f" [{total_missing} missing, {total_extra} extra]"

        lines = [header]

        for factor in self._factors:
            observed = self._observed_levels.get(factor, [])
            if self._sampling_plan is not None:
                planned = self._sampling_plan.get(factor, observed)
            else:
                planned = observed

            planned_set = set(planned)
            observed_set = set(observed)
            missing = self._safe_sort(list(planned_set - observed_set))

            line = f"  {factor}: observed={observed}"
            if self.has_plan:
                line = f"  {factor}: planned={planned}, observed={observed}"
            if missing:
                line += f", missing={missing}"

            lines.append(line)

        return '\n'.join(lines)


class StudyChartAccessor:
    """
    Provides IDE auto-completion for valid chart types in a Study.

    This class dynamically creates attributes for each valid chart type,
    enabling IDE auto-completion and preventing invalid chart selections.

    Usage:
        study = pb.formulate(response='weight', factors=['lane'])

        # IDE auto-completes valid charts
        result = study.execute(chart=study.charts.Xbar)

    Attributes are set dynamically based on SDS-specific valid charts.
    """

    def __init__(self, valid_charts: list[str], residual_charts: list[str] | None = None):
        """
        Initialize accessor with valid chart types.

        Parameters
        ----------
        valid_charts : list of str
            Primary chart types (Xbar, S, Imr, etc.)
        residual_charts : list of str, optional
            Residual chart types (R2_S, R3_Imr, etc.)
        """
        self._valid_charts = valid_charts
        self._residual_charts = residual_charts or []
        self._all_charts = valid_charts + self._residual_charts

        # Dynamically add each valid chart as an attribute
        for chart in self._all_charts:
            # Convert chart names to valid Python identifiers
            attr_name = chart.replace(':', '_').replace('-', '_')
            setattr(self, attr_name, chart)

    def __repr__(self) -> str:
        """Display available chart types."""
        parts = [f"Primary: {', '.join(self._valid_charts)}"]
        if self._residual_charts:
            parts.append(f"Residual: {', '.join(self._residual_charts)}")
        return f"StudyChartAccessor({'; '.join(parts)})"

    def __dir__(self) -> list[str]:
        """Support for tab-completion in IPython/Jupyter."""
        return [c.replace(':', '_').replace('-', '_') for c in self._all_charts]


@dataclass(frozen=True)
class Study:
    """
    A Study formulation for process behavior analysis.

    The Study object represents a complete formulation of how to analyze
    process behavior data. It encapsulates:

    - The data structure (Sampling Design State)
    - Valid and recommended chart types
    - Available residual analyses
    - Guidance on methodology
    - Pre-calculated dataset with residuals (R1-R5, RCR1-RCR5)

    This is an immutable object - to change the formulation, create a new Study.

    Parameters
    ----------
    _pdf : ProcessBehavior
        Reference to the data source
    _spec : AnalysisSpecification
        Internal specification (parameter mapping)
    _plan : SDSAnalysisPlan
        Analysis plan based on detected SDS
    _ads : AnalysisDataSet
        Pre-calculated AnalysisDataSet with rsg, means, and residuals (R1-R5)

    Examples
    --------
    Create a study formulation:

    >>> pb = ProcessBehavior(df)
    >>> study = pb.formulate(response='weight', factors=['lane'], time='pull')
    >>> print(study)  # Rich display of formulation

    Check what's available:

    >>> study.sds  # 1-6
    >>> study.valid_charts  # ['Xbar', 'S', ...]
    >>> study.recommended_chart  # 'Xbar'

    Access the prepared dataset:

    >>> study.dataset  # DataFrame with rsg, R1-R5, RCR1-RCR5

    Run the analysis:

    >>> result = study.execute()  # Uses recommended chart
    >>> result = study.execute(chart='Xbar')  # Explicit chart
    >>> result = study.execute(chart=study.charts.Xbar)  # Via accessor

    See Also
    --------
    ProcessBehavior.formulate : Create a Study from data
    AnalysisResult : Result of study.execute()
    """
    _pdf: ProcessBehavior
    _spec: DataPrepConfig
    _plan: SDSAnalysisPlan
    _ads: AnalysisDataSet
    _sampling_plan: dict[str, list] | None = None
    _factor_order: list[str] | None = None

    # =========================================================================
    # User-Facing Properties (Clean Names)
    # =========================================================================

    @property
    def response(self) -> str:
        """
        The response variable being analyzed.

        This is the measurement or outcome variable that will be charted.
        """
        return self._spec.response_var

    @property
    def factors(self) -> list[str] | None:
        """
        Grouping factors defining rational subgroups.

        These are the categorical variables (like Lane, Operator, Machine)
        that define how observations are grouped. Returns None if no
        factors are specified.
        """
        rsg_vars = self._spec.rsg_vars
        return list(rsg_vars) if rsg_vars else None

    @property
    def time(self) -> str | None:
        """
        Time variable for ordering observations.

        This defines the sequence of measurements (like Pull, Day, Hour).
        Returns None if no time variable is specified.
        """
        return self._spec.time_var

    @property
    def precision(self) -> int:
        """
        Decimal precision for output values.

        Statistics and chart values will be rounded to this many decimal places.
        """
        return self._spec.round_to

    @property
    def dataset(self) -> pd.DataFrame:
        """
        Full analysis dataset with means and residuals.

        This is the pre-calculated dataset produced during formulate().
        It includes:
        - rsg: Rational subgroup identifier
        - Ybar, Ybar_k, Ybar_t, Ybar_kt: Hierarchical means
        - R1-R5: VAS residuals (where applicable for the SDS)
        - RCR1-RCR5: Re-centered residuals (Y reconstructed from components)

        Returns a copy to preserve immutability.

        Returns
        -------
        pd.DataFrame
            Copy of the analysis dataset

        Examples
        --------
        >>> study = pdf.formulate(response='weight', factors=['lane'], time='pull')
        >>> df = study.dataset
        >>> df[['rsg', 'weight', 'R1', 'R2', 'R3', 'R4', 'R5']].head()
        """
        return self._ads.analysis_dataset.copy()

    # =========================================================================
    # SDS Properties (Sampling Design State)
    # =========================================================================

    @property
    def sds(self) -> int:
        """
        Sampling Design State (0-6).

        The SDS classifies your data structure based on:
        - Whether you have grouping factors
        - Whether you have a time variable
        - Whether you have replication (multiple observations per cell)

        SDS determines which charts are valid and how residuals are calculated.

        Returns
        -------
        int
            SDS value from 0 to 6

        See Also
        --------
        sds_name : Human-readable name for the SDS
        sds_description : Detailed explanation
        """
        return self._plan.sds

    @property
    def sds_name(self) -> str:
        """
        Human-readable name for the detected Sampling Design State.

        Examples: "Full Factorial with Replication", "Time Series Only"
        """
        return self._plan.name

    @property
    def sds_description(self) -> str:
        """
        Detailed description of the detected data structure.

        Explains what the SDS means in terms of your data structure
        and what analysis approaches are appropriate.
        """
        return self._plan.description

    # =========================================================================
    # Chart Properties
    # =========================================================================

    @property
    def valid_charts(self) -> list[str]:
        """
        Chart types that are valid for this data structure.

        These are the primary control charts that can be created
        based on the detected SDS.

        Returns
        -------
        list of str
            Valid chart types (e.g., ['Xbar', 'S', 'Imr'])
        """
        return self._plan.valid_charts

    @property
    def recommended_chart(self) -> str:
        """
        The recommended chart type for this data structure.

        This is the chart type that best suits your data based on
        Wheeler & Bishop methodology.
        """
        return self._plan.recommended_chart

    @property
    def residual_charts(self) -> list[str]:
        """
        Available residual chart types for VAS analysis.

        Residual charts help diagnose sources of variation:
        - R2: Within-subgroup variation (measurement noise)
        - R3: Interaction effects (factor × time)
        - R4: Time effects (trends, shifts over time)
        - R5: Factor effects (differences between levels)

        Returns
        -------
        list of str
            Available residual chart types (e.g., ['R2_S', 'R3_Imr'])
        """
        return self._plan.residual_charts

    @property
    def charts(self) -> StudyChartAccessor:
        """
        Accessor for IDE auto-completion of valid chart types.

        Use this for IDE-assisted chart selection:

            study.charts.Xbar  # Auto-completes valid charts
            study.charts.R2_S  # Residual charts too

        Returns
        -------
        StudyChartAccessor
            Object with chart types as attributes
        """
        return StudyChartAccessor(self.valid_charts, self.residual_charts)

    @property
    def support(self) -> pd.DataFrame:
        """
        Chart support matrix for this study.

        Returns a DataFrame with one row per chart type showing availability,
        recommendations, and explanations. This is the single source of truth
        for chart capabilities.

        Returns
        -------
        pd.DataFrame
            Columns: chart, category, available, recommended, reason, question

        Examples
        --------
        >>> study.support
               chart  category  available  recommended  ...
        0       Xbar   primary       True         True  ...
        1          S   primary       True        False  ...

        >>> study.support[study.support['available']]  # Filter to available
        >>> study.support.query("category == 'residual'")  # Residual charts
        """
        import pandas as pd

        from .sds_detector import SDSAnalysisPlan

        rows = []

        # All possible primary charts
        ALL_PRIMARY = ['Xbar', 'S', 'Imr', 'R']

        # Build invalid_reasons dict from _plan.invalid_charts
        invalid_reasons = self._parse_invalid_charts()

        for chart in ALL_PRIMARY:
            rows.append({
                'chart': chart,
                'category': 'primary',
                'available': chart in self.valid_charts,
                'recommended': chart == self.recommended_chart,
                'reason': invalid_reasons.get(chart),
                'question': SDSAnalysisPlan.CHART_QUESTIONS.get(chart, '')
            })

        # All possible residual charts
        ALL_RESIDUALS = [
            'R2_S', 'R2_Imr',
            'R3_Xbar', 'R3_S', 'R3_Imr',
            'R4_Xbar', 'R4_S', 'R4_Imr',
            'R5_Xbar', 'R5_S', 'R5_Imr'
        ]

        for chart in ALL_RESIDUALS:
            available = chart in self.residual_charts
            rows.append({
                'chart': chart,
                'category': 'residual',
                'available': available,
                'recommended': False,
                'reason': None if available else 'Not available for this SDS',
                'question': SDSAnalysisPlan.CHART_QUESTIONS.get(chart, '')
            })

        return pd.DataFrame(rows)

    def _parse_invalid_charts(self) -> dict[str, str]:
        """
        Parse invalid_charts list into dict of chart → reason.

        The _plan.invalid_charts format is: ['S (requires n≥2 per subgroup)']
        This parses to: {'S': 'requires n≥2 per subgroup'}
        """
        result = {}
        for entry in self._plan.invalid_charts:
            # Format: 'ChartType (reason)'
            if '(' in entry and entry.endswith(')'):
                chart = entry.split('(')[0].strip()
                reason = entry[entry.index('(') + 1:-1]
                result[chart] = reason
        return result

    # =========================================================================
    # Guidance Methods
    # =========================================================================

    def why_not(self, chart: str) -> str:
        """
        Explain why a chart type is or isn't available for this study.

        This is a teaching method - it helps users understand the
        methodology by explaining constraints. Uses the support DataFrame
        as the single source of truth.

        Parameters
        ----------
        chart : str
            Chart type to check (e.g., 'Imr', 'S', 'R2_S')

        Returns
        -------
        str
            Explanation of availability with the question the chart answers

        Examples
        --------
        >>> study.why_not('S')
        "'S' unavailable: requires n≥2 per subgroup"

        >>> study.why_not('Xbar')
        "'Xbar' IS available. Are subgroup means stable over time?"
        """
        df = self.support
        row = df[df['chart'] == chart]

        if row.empty:
            return f"'{chart}' is not a recognized chart type. Use study.support to see all options."

        row = row.iloc[0]
        if row['available']:
            return f"'{chart}' IS available. {row['question']}"
        else:
            return f"'{chart}' unavailable: {row['reason']}"

    def design(self) -> DesignReport:
        """
        Get design report comparing sampling plan to observed data.

        Returns a DesignReport showing factors, levels, and any mismatches
        between planned and observed structure. Works with or without a
        sampling plan:

        - With plan: Shows planned vs observed, highlights missing/extra levels
        - Without plan: Shows observed structure only

        Note: Time Handling
        -------------------
        The sampling plan specifies factor levels only, not time levels.
        For SDS detection, observed unique time values are used as the
        "planned" time set. This means coverage detects missing factor
        combos within observed time blocks, but time points not in the
        data are not considered "missing."

        Returns
        -------
        DesignReport
            Object with:
            - factors: DataFrame with planned/observed/missing/extra per factor
            - missing_levels: dict of levels in plan but not observed
            - extra_levels: dict of levels observed but not in plan
            - has_plan: whether a sampling plan was provided

        Examples
        --------
        >>> study = pb.formulate(
        ...     response=pb.cols.Weight,
        ...     plan={pb.cols.Lane: [1,2,3,4], pb.cols.Phase: [1,2,3]}
        ... )
        >>> design = study.design()
        >>> design
        DesignReport(2 factors, with plan)
          Lane: planned=[1,2,3,4], observed=[1,2,3,4]
          Phase: planned=[1,2,3], observed=[1,2], missing=[3]

        >>> design.missing_levels
        {'Lane': [], 'Phase': [3]}

        Without a plan (shows observed structure):

        >>> study = pb.formulate(response='weight', factors=['lane', 'phase'])
        >>> study.design()
        DesignReport(2 factors, observed only)
          lane: observed=[1, 2, 3, 4]
          phase: observed=[1, 2]
        """
        # Get the factors - either from factor_order (plan) or spec
        if self._factor_order is not None:
            factors = self._factor_order
        elif self._spec.rsg_vars:
            factors = list(self._spec.rsg_vars)
        else:
            factors = []

        # Get observed levels from the data
        observed_levels: dict[str, list] = {}
        for factor in factors:
            if factor in self._pdf.data.columns:
                values = self._pdf.data[factor].dropna().unique()
                try:
                    observed_levels[factor] = sorted(values.tolist())
                except TypeError:
                    observed_levels[factor] = list(values)
            else:
                observed_levels[factor] = []

        return DesignReport(
            _sampling_plan=self._sampling_plan,
            _observed_levels=observed_levels,
            _factors=factors
        )

    def execute(
        self,
        chart: str | None = None,
        recentered: bool = False
    ) -> AnalysisResult:
        """
        Run the analysis and return results.

        This executes the formulated study using the specified chart type
        (or the recommended chart if none specified).

        Parameters
        ----------
        chart : str, optional
            Chart type to use. If None, uses recommended_chart.
            Can use study.charts.Xbar for IDE auto-completion.

            Primary charts: 'Xbar', 'S', 'Imr', 'R'
            Residual charts: 'R2_S', 'R2_Imr', 'R3_Imr', 'R4_Imr', 'R5_Imr'

        recentered : bool, default False
            For residual charts only. If True, uses re-centered residuals
            (RCR2, RCR3, etc.) which add back the appropriate mean for
            easier interpretation. See Tom Bishop Equation 80.

        Returns
        -------
        AnalysisResult
            Complete analysis results with charts, statistics, residuals.
            Use result.plot() to visualize or result.to_excel() to export.

        Raises
        ------
        ValueError
            If specified chart is not valid for this SDS

        Examples
        --------
        Use recommended chart:

        >>> result = study.execute()

        Specify chart explicitly:

        >>> result = study.execute(chart='Xbar')
        >>> result = study.execute(chart=study.charts.Xbar)

        Analyze residual charts (VAS):

        >>> result = study.execute(chart='R4_Imr')  # Time effects
        >>> result = study.execute(chart='R5_Imr')  # Factor effects
        >>> result = study.execute(chart='R4_Imr', recentered=True)  # Re-centered

        Chain to visualization:

        >>> study.execute().plot()
        """
        # Import here to avoid circular imports
        from .analysis import Analysis

        # Determine chart type
        chart_request = chart or self.recommended_chart

        # Check if this is a residual chart request
        is_residual_chart = self._is_residual_chart(chart_request)

        if is_residual_chart:
            # Parse residual chart name: 'R4_Imr' → residual='R4', base_chart='Imr'
            residual, base_chart = self._parse_residual_chart(chart_request)

            # Validate residual chart is available for this SDS
            if chart_request not in self.residual_charts:
                available_list = list(self.residual_charts) if self.residual_charts else []
                available_str = ', '.join(available_list) if available_list else 'None'
                raise ChartNotAvailableError(
                    f"Residual chart '{chart_request}' is not available for SDS {self.sds}.\n"
                    f"Available residual charts: {available_str}\n"
                    f"Use study.residual_charts to see available options.",
                    chart=chart_request,
                    available=available_list
                )

            # Build spec dict for residual chart
            spec_dict = {
                'analysis_type': base_chart,  # The underlying chart type (S or Imr)
                'response_var': self._spec.response_var,
                'time_var': self._spec.time_var,
                'rsg_vars': self._spec.rsg_vars,
                'rsg_var_name': self._spec.rsg_var_name,
                'rsg_var_delim': self._spec.rsg_var_delim,
                'round_to': self._spec.round_to,
                'zero_center': self._spec.zero_center,
                # Residual-specific parameters
                'residual': residual,  # e.g., 'R4'
                'residual_chart_type': base_chart,  # e.g., 'Imr'
                'recentered': recentered
            }
        else:
            # Primary chart validation
            if chart_request not in self.valid_charts:
                available_list = list(self.valid_charts)
                raise ChartNotAvailableError(
                    f"Chart type '{chart_request}' is not valid for SDS {self.sds}.\n"
                    f"Valid charts: {', '.join(available_list)}\n"
                    f"Recommended: {self.recommended_chart}\n"
                    f"Use study.why_not('{chart_request}') for explanation.",
                    chart=chart_request,
                    available=available_list
                )

            # Build spec dict for primary chart
            spec_dict = {
                'analysis_type': chart_request,
                'response_var': self._spec.response_var,
                'time_var': self._spec.time_var,
                'rsg_vars': self._spec.rsg_vars,
                'rsg_var_name': self._spec.rsg_var_name,
                'rsg_var_delim': self._spec.rsg_var_delim,
                'round_to': self._spec.round_to,
                'zero_center': self._spec.zero_center
            }

        # Create and run analysis using pre-calculated AnalysisDataSet
        # This makes execute() cheap - the expensive residual calculation was done in formulate()
        analysis = Analysis(self._pdf.data, spec_dict, analysis_dataset=self._ads)
        return analysis.calculate()

    def _is_residual_chart(self, chart: str) -> bool:
        """Check if chart name is a residual chart (e.g., 'R4_Imr')."""
        if not chart:
            return False
        # Residual charts start with R followed by digit and underscore
        return (
            len(chart) >= 4 and
            chart[0] == 'R' and
            chart[1].isdigit() and
            '_' in chart
        )

    def _parse_residual_chart(self, chart: str) -> tuple[str, str]:
        """
        Parse residual chart name into components.

        Parameters
        ----------
        chart : str
            Residual chart name (e.g., 'R4_Imr', 'R2_S')

        Returns
        -------
        tuple[str, str]
            (residual, base_chart) e.g., ('R4', 'Imr')
        """
        parts = chart.split('_', 1)
        residual = parts[0]  # e.g., 'R4'
        base_chart = parts[1] if len(parts) > 1 else 'Imr'  # e.g., 'Imr'
        return residual, base_chart

    # =========================================================================
    # Display Methods
    # =========================================================================

    def __repr__(self) -> str:
        """
        Concise study summary.

        Shows formulation, SDS, and available charts in minimal format.
        Use study.support for the full chart availability DataFrame.
        """
        # 1-line formulation summary
        factors_str = ', '.join(self.factors) if self.factors else 'None'
        time_str = self.time or 'None'

        # Get available charts from support DataFrame
        avail = self.support[self.support['available']]
        primary = avail[avail['category'] == 'primary']['chart'].tolist()
        residual = avail[avail['category'] == 'residual']['chart'].tolist()

        lines = [
            f"Study(response='{self.response}', factors=[{factors_str}], time='{time_str}', sds={self.sds})",
            f"  Valid: {', '.join(primary)} | Recommended: {self.recommended_chart}",
        ]
        if residual:
            lines.append(f"  Residuals: {', '.join(residual)}")
        lines.append("  → study.execute() or study.support for details")

        return '\n'.join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        factors_str = ', '.join(self.factors) if self.factors else 'None'
        time_str = self.time or 'None'

        # Get available charts from support DataFrame
        avail = self.support[self.support['available']]
        primary = avail[avail['category'] == 'primary']['chart'].tolist()
        residual = avail[avail['category'] == 'residual']['chart'].tolist()

        residual_html = f"<br><strong>Residuals:</strong> {', '.join(residual)}" if residual else ""

        style = "font-family: monospace; padding: 8px; border: 1px solid #ccc; background: #f9f9f9"
        html = f"""
        <div style="{style}">
            <code>Study(response='{self.response}', factors=[{factors_str}], time='{time_str}', sds={self.sds})</code>
            <br><strong>Valid:</strong> {', '.join(primary)} | <strong>Recommended:</strong> {self.recommended_chart}
            {residual_html}
            <br><em>→ study.execute() or study.support for details</em>
        </div>
        """
        return html
