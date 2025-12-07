"""
Study class for process behavior analysis formulation.

The Study object represents a formulated analysis - it knows what data structure
you have (SDS), what charts are valid, and guides you toward correct analysis.

This is the "teaching" layer of the API that helps users understand their data
before running calculations.

Design Philosophy (Pythonic Hadley):
- Human-first: Rich __repr__ teaches users about their data
- Pit of success: Valid charts shown, invalid charts explained
- Composability: study.analyze() returns AnalysisResult for chaining
- Immutable: Frozen dataclass, different formulations create new objects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from .analysis_dataset import AnalysisDataSet
    from .analysis_result import AnalysisResult
    from .analysis_specification import DataPrepConfig
    from .process_dataframe import ProcessDataFrame
    from .sds_detector import SDSAnalysisPlan


class StudyChartAccessor:
    """
    Provides IDE auto-completion for valid chart types in a Study.

    This class dynamically creates attributes for each valid chart type,
    enabling IDE auto-completion and preventing invalid chart selections.

    Usage:
        study = pdf.formulate(response='weight', factors=['lane'])

        # IDE auto-completes valid charts
        result = study.analyze(chart=study.charts.Xbar)

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
    _pdf : ProcessDataFrame
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

    >>> pdf = ProcessDataFrame(df)
    >>> study = pdf.formulate(response='weight', factors=['lane'], time='pull')
    >>> print(study)  # Rich display of formulation

    Check what's available:

    >>> study.sds  # 1-6
    >>> study.valid_charts  # ['Xbar', 'S', ...]
    >>> study.recommended_chart  # 'Xbar'

    Access the prepared dataset:

    >>> study.dataset  # DataFrame with rsg, R1-R5, RCR1-RCR5

    Run the analysis:

    >>> result = study.analyze()  # Uses recommended chart
    >>> result = study.analyze(chart='Xbar')  # Explicit chart
    >>> result = study.analyze(chart=study.charts.Xbar)  # Via accessor

    See Also
    --------
    ProcessDataFrame.formulate : Create a Study from data
    AnalysisResult : Result of study.analyze()
    """
    _pdf: 'ProcessDataFrame'
    _spec: 'DataPrepConfig'
    _plan: 'SDSAnalysisPlan'
    _ads: 'AnalysisDataSet'

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
    def dataset(self) -> 'pd.DataFrame':
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

    # =========================================================================
    # Guidance Methods
    # =========================================================================

    def why_not(self, chart: str) -> str:
        """
        Explain why a chart type is not valid for this study.

        This is a teaching method - it helps users understand the
        methodology by explaining constraints.

        Parameters
        ----------
        chart : str
            Chart type to check (e.g., 'Imr', 'S')

        Returns
        -------
        str
            Explanation of validity or invalidity

        Examples
        --------
        >>> study.why_not('S')
        "S chart requires n≥2 observations per subgroup. Your data has n=1."

        >>> study.why_not('Xbar')
        "Xbar IS valid for this study. Use study.analyze(chart='Xbar')"
        """
        if chart in self.valid_charts:
            return f"'{chart}' IS valid for this study. Use study.analyze(chart='{chart}')"

        # Check the invalid_charts list for the reason
        invalid_charts = self._plan.invalid_charts
        for invalid in invalid_charts:
            if chart in invalid:
                return invalid

        return f"'{chart}' is not a recognized chart type. Valid types: {self.valid_charts}"

    def analyze(
        self,
        chart: str | None = None,
        recentered: bool = False
    ) -> 'AnalysisResult':
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

        >>> result = study.analyze()

        Specify chart explicitly:

        >>> result = study.analyze(chart='Xbar')
        >>> result = study.analyze(chart=study.charts.Xbar)

        Analyze residual charts (VAS):

        >>> result = study.analyze(chart='R4_Imr')  # Time effects
        >>> result = study.analyze(chart='R5_Imr')  # Factor effects
        >>> result = study.analyze(chart='R4_Imr', recentered=True)  # Re-centered

        Chain to visualization:

        >>> study.analyze().plot()
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
                available = ', '.join(self.residual_charts) if self.residual_charts else 'None'
                raise ValueError(
                    f"Residual chart '{chart_request}' is not available for SDS {self.sds}.\n"
                    f"Available residual charts: {available}\n"
                    f"Use study.residual_charts to see available options."
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
                valid_list = ', '.join(self.valid_charts)
                raise ValueError(
                    f"Chart type '{chart_request}' is not valid for SDS {self.sds}.\n"
                    f"Valid charts: {valid_list}\n"
                    f"Recommended: {self.recommended_chart}\n"
                    f"Use study.why_not('{chart_request}') for explanation."
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
        # This makes analyze() cheap - the expensive residual calculation was done in formulate()
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
        Rich display of study formulation.

        Provides a teaching-focused view showing:
        - What was formulated
        - What SDS was detected
        - What charts are valid
        - What to do next
        """
        width = 66

        lines = []
        lines.append("╔" + "═" * width + "╗")
        lines.append("║" + "STUDY FORMULATION".center(width) + "║")
        lines.append("╠" + "═" * width + "╣")

        # Formulation details
        lines.append("║" + f"  Response: {self.response}".ljust(width) + "║")

        if self.factors:
            factors_str = ', '.join(self.factors)
            lines.append("║" + f"  Factors:  {factors_str}".ljust(width) + "║")

        if self.time:
            lines.append("║" + f"  Time:     {self.time}".ljust(width) + "║")

        lines.append("║" + f"  Precision: {self.precision} decimal places".ljust(width) + "║")

        lines.append("╠" + "═" * width + "╣")

        # SDS detection
        lines.append("║" + f"  Detected: SDS {self.sds} - {self.sds_name}".ljust(width) + "║")
        lines.append("║" + "".ljust(width) + "║")

        # Charts
        valid_str = ', '.join(self.valid_charts)
        lines.append("║" + f"  Valid Charts:  {valid_str}".ljust(width) + "║")
        lines.append("║" + f"  Recommended:   {self.recommended_chart}".ljust(width) + "║")

        if self.residual_charts:
            residual_str = ', '.join(self.residual_charts)
            lines.append("║" + f"  Residuals:     {residual_str}".ljust(width) + "║")

        lines.append("╠" + "═" * width + "╣")

        # Next steps
        lines.append("║" + f"  Next: study.analyze() or study.analyze(chart='Xbar')".ljust(width) + "║")

        lines.append("╚" + "═" * width + "╝")

        return '\n'.join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        # Simple HTML version - could be enhanced later
        html = f"""
        <div style="font-family: monospace; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
            <h3>Study Formulation</h3>
            <p><strong>Response:</strong> {self.response}</p>
            <p><strong>Factors:</strong> {', '.join(self.factors) if self.factors else 'None'}</p>
            <p><strong>Time:</strong> {self.time or 'None'}</p>
            <hr>
            <p><strong>SDS {self.sds}:</strong> {self.sds_name}</p>
            <p><strong>Valid Charts:</strong> {', '.join(self.valid_charts)}</p>
            <p><strong>Recommended:</strong> {self.recommended_chart}</p>
            <hr>
            <p><em>Next: study.analyze() or study.analyze(chart='{self.recommended_chart}')</em></p>
        </div>
        """
        return html
