# ============================================================================
# NEW: Refactored Analysis Class (replacing factory pattern)
# ============================================================================

class Analysis:
    """
    Unified analysis class handling all chart types via strategy pattern.

    This class replaces the AbstractFactory pattern with a simpler, more maintainable
    approach. All analysis types (Xbar, S, Imr, R) are handled through internal
    strategy methods.

    Usage:
        analysis = Analysis(df, specification)
        result = analysis.calculate()
    """

    def __init__(self, df: pd.DataFrame, specification: dict):
        """
        Initialize analysis with data and specification.

        Args:
            df: Input DataFrame with raw data
            specification: Dictionary containing analysis configuration including 'analysis_type'
        """
        self.raw_df = df
        self.analysis_type = specification['analysis_type']
        self.spec = AnalysisSpecification(self.analysis_type, specification)
        self.ads = AnalysisDataSet(df, self.spec)

    def calculate(self) -> pd.DataFrame:
        """
        Execute the appropriate analysis strategy and return results.

        Returns:
            DataFrame containing analysis results

        Raises:
            ValueError: If analysis_type is not supported
        """
        strategies = {
            'Xbar': self._calculate_xbar,
            'S': self._calculate_s,
            'Imr': self._calculate_imr,
            'R': self._calculate_r
        }

        if self.analysis_type not in strategies:
            raise ValueError(
                f'Analysis type {self.analysis_type} not supported! '
                f'Valid types: {list(strategies.keys())}'
            )

        return strategies[self.analysis_type]()

    def _calculate_xbar(self) -> pd.DataFrame:
        """
        Calculate Xbar (mean) chart statistics.

        Logic moved from Xbar.calculate_statistics()
        """
        df = self.ads.analysis_dataset
        spec = self.spec
        result = {}
        statistics = {}
        out = df.copy()

        print(f'\nIn calculate statistics XbarS...')
        print(f'\nDataframe has columns: {out.columns.to_list()}')
        print(f'\n{out.head(10)}')
        print(f'\nn.max={out["n"].max()}')

        if spec.zero_center:
            print(f'zero-centering data')
            zero_mean = out[spec.response_var].mean()
            out[spec.response_var] = out[spec.response_var] - zero_mean

        out = out.groupby(spec.rsg_var_name, as_index=False).agg(
            s=pd.NamedAgg(column=spec.response_var, aggfunc="std"),
            mean=pd.NamedAgg(column=spec.response_var, aggfunc="mean"),
            n=pd.NamedAgg(column='n', aggfunc="max")
        )

        # Handle case where no subgroups have >1 observation
        if out.shape[0] == 0:
            raise ValueError("All subgroups have 1 or less observations!")

        _Xbar = out["mean"].mean()
        out['Xbar'] = _Xbar
        _S = out["s"].mean()
        out['S'] = _S
        _N = out['n'].max()
        out['N'] = _N

        # if subgroup sizes are equal use N (limits will be same for all groups)
        n_max = _N
        n_to_use = "N" if out['n'].eq(n_max).all() else "n"
        print(f'Analysis is using: {n_to_use} for calculations!\nScenario: {1 if n_to_use=="N" else 2}')

        # CALCULATE XBAR
        xbar = out.copy()
        xbar[['lcl', 'ucl']] = xbar.apply(
            lambda row: obj.calculate_limits(
                mean=row['Xbar'],
                sd=row['S'],
                N=row[n_to_use],
                limits_type='Xbar',
                round_to=spec.round_to
            ), axis=1
        )

        xbar['beyond_limits'] = xbar.apply(
            lambda row: obj.detect_beyond_limits(
                x=row['mean'],
                ucl=row['ucl'],
                lcl=row['lcl']
            ), axis=1
        )

        xbar = xbar.round(spec.round_to)

        statistics['Mean'] = round(_Xbar, spec.round_to)
        if n_to_use == "N":
            statistics['N'] = _N
            statistics['ucl'] = xbar['ucl'].max()
            statistics['lcl'] = xbar['lcl'].max()
        else:
            variable_stats = 'Varies'
            statistics['N'] = variable_stats
            statistics['lcl'] = variable_stats
            statistics['ucl'] = variable_stats

        cols_to_keep = ['rsg', 'mean', 'Xbar', 'lcl', 'ucl', 'beyond_limits']
        xbar = xbar[cols_to_keep]
        result['Xbar'] = {'data': xbar, 'statistics': statistics}

        # CALCULATE S
        statistics = {}
        statistics['S'] = round(_S, spec.round_to)

        sbar = out.copy()
        sbar[['lcl', 'ucl']] = sbar.apply(
            lambda row: obj.calculate_limits(
                mean=0,
                sd=row['S'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        sbar['beyond_limits'] = sbar.apply(
            lambda row: obj.detect_beyond_limits(
                x=row['S'],
                ucl=row['ucl'],
                lcl=row['lcl']
            ), axis=1
        )

        sbar = sbar.round(spec.round_to)

        if n_to_use == "N":
            statistics['N'] = _N
            statistics['ucl'] = sbar['ucl'].max()
            statistics['lcl'] = sbar['lcl'].max()
        else:
            variable_stats = 'Varies'
            statistics['N'] = variable_stats
            statistics['lcl'] = variable_stats
            statistics['ucl'] = variable_stats

        cols_to_keep = ['rsg', 's', 'S', 'lcl', 'ucl', 'beyond_limits']
        sbar = sbar[cols_to_keep]
        result['Sbar'] = {'data': sbar, 'statistics': statistics}

        return result

    def _calculate_s(self) -> pd.DataFrame:
        """
        Calculate S (standard deviation) chart statistics.

        Logic moved from calculate_statistics_S()
        """
        df = self.raw_df
        spec = self.spec
        out = prepare_dataset(df=df, analysis_specification=spec)

        out = out.groupby(spec.rsg_var_name, as_index=False).agg(
            s=pd.NamedAgg(column=spec.response_var, aggfunc="std"),
            n=pd.NamedAgg(column=spec.rsg_var_name, aggfunc="count"),
        )

        # remove RSGs with a single observation
        mask = out['n'].eq(1)
        out = out[~mask]

        out['S'] = out["s"].mean()
        out['groups'] = out["n"].count()
        out['N'] = out['n'].max()

        # if subgroup sizes are equal use N (limits will be same for all groups)
        n_max = out['n'].max()
        n_to_use = "N" if (out['n'].eq(n_max).all()) else "n"

        # Add limits columns
        out[['lcl', 'ucl']] = out.apply(
            lambda row: obj.calculate_limits(
                mean=0,
                sd=row['S'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        out['beyond_limits'] = out.apply(
            lambda row: obj.detect_beyond_limits(
                x=row['S'],
                ucl=row['ucl'],
                lcl=row['lcl']
            ), axis=1
        )

        cols_to_keep = ['rsg', 's', 'S', 'lcl', 'ucl', 'beyond_limits']
        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        return out

    def _calculate_imr(self) -> pd.DataFrame:
        """
        Calculate IMR (Individual Moving Range) chart statistics.

        Logic moved from calculate_statistics_Imr()
        """
        df = self.raw_df
        spec = self.spec
        out = prepare_dataset(df=df, analysis_specification=spec)

        if spec.zero_center:
            print(f'zero-centering data')
            zero_mean = out[spec.response_var].mean()
            print(f'zero-mean:{zero_mean}')
            out[spec.response_var] = out[spec.response_var] - zero_mean

        print(f'\nIn calculate statistics IMR...')
        print(f'\nDataframe has columns: {out.columns.to_list()}')

        if spec.has_grouping:
            out['mr'] = abs(out.groupby(spec.rsg_var_name)[spec.response_var].diff())
            grouped = out.groupby(spec.rsg_var_name, as_index=False)
            grouped = grouped.agg(
                mean=pd.NamedAgg(spec.response_var, 'mean'),
                mR=pd.NamedAgg('mr', 'mean')
            )

            limits = grouped.apply(
                lambda row: obj.calculate_limits(
                    mean=row['mean'],
                    sd=0,
                    N=0,
                    mR=row.mR,
                    limits_type="Imr"
                ), axis=1
            )

            grouped = pd.merge(grouped, limits, left_index=True, right_index=True)
            out = pd.merge(out, grouped, how='left', on=spec.rsg_var_name)
        else:
            out['mR'] = abs(out[spec.response_var].diff()).mean()
            out['mean'] = out[spec.response_var].mean()
            mR = out['mR'].max()
            _mean = out['mean'].max()
            limits = obj.calculate_limits(
                mean=_mean,
                sd=0,
                N=0,
                mR=mR,
                limits_type="Imr",
                round_to=spec.round_to
            )
            out['lcl'] = limits['lcl']
            out['ucl'] = limits['ucl']

        out['beyond_limits'] = np.where(out[spec.response_var] < out['lcl'], -1, 0)
        out['beyond_limits'] = np.where(out[spec.response_var] > out['ucl'], 1, 0)

        cols_to_keep = [spec.response_var, 'mean', 'lcl', 'ucl', 'beyond_limits']

        if spec.has_time:
            if spec.has_grouping:
                cols_to_keep.insert(0, spec.rsg_var_name)
                cols_to_keep.insert(0, spec.time_var)
            else:
                cols_to_keep.insert(0, spec.time_var)
        else:
            if spec.has_grouping:
                out['x'] = out.groupby(spec.rsg_var_name).cumcount() + 1
                cols_to_keep.insert(0, spec.rsg_var_name)
                cols_to_keep.insert(0, 'x')
            else:
                out['x'] = out.index + 1
                cols_to_keep.insert(0, 'x')

        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        if spec.has_grouping:
            statistics = gather_analysis_statistics(
                df=out,
                statistics_to_collect=['mean', 'lcl', 'ucl'],
                grouping_var=spec.rsg_var_name
            )
            split_dict = split_df_by_group(df=out, grouping_var=spec.rsg_var_name)
            out = package_analysis(
                analysis_output=split_dict,
                summary_statistics_output=statistics
            )
        else:
            statistics = gather_analysis_statistics(
                df=out,
                statistics_to_collect=['mean', 'lcl', 'ucl']
            )
            _out = {'all': out}
            out = package_analysis(
                analysis_output=_out,
                summary_statistics_output=statistics
            )

        return out

    def _calculate_r(self) -> pd.DataFrame:
        """
        Calculate R (Range) chart statistics.

        Logic moved from calculate_statistics_R()
        """
        df = self.raw_df
        spec = self.spec
        out = prepare_dataset(df=df, analysis_specification=spec)

        if spec.zero_center:
            print(f'zero-centering data')
            zero_mean = out[spec.response_var].mean()
            out[spec.response_var] = out[spec.response_var] - zero_mean

        print(f'\nIn calculate statistics R...')
        print(f'\nDataframe has columns: {out.columns.to_list()}')

        if spec.has_grouping:
            out['mr'] = abs(out.groupby(spec.rsg_var_name)[spec.response_var].diff())
            grouped = out.groupby(spec.rsg_var_name, as_index=False)
            grouped = grouped.agg(mR=pd.NamedAgg('mr', 'mean'))

            limits = grouped.apply(
                lambda row: obj.calculate_limits(
                    mean=0,
                    sd=0,
                    N=0,
                    mR=row.mR,
                    limits_type="R",
                    round_to=spec.round_to
                ), axis=1
            )

            grouped = pd.merge(grouped, limits, left_index=True, right_index=True)
            out = pd.merge(out, grouped, how='left', on=spec.rsg_var_name)
        else:
            out['mr'] = abs(out[spec.response_var].diff())
            out['mR'] = out['mr'].mean()
            mR = out['mR'].max()
            limits = obj.calculate_limits(
                mean=0,
                sd=0,
                N=0,
                mR=mR,
                limits_type="R",
                round_to=spec.round_to
            )
            out['lcl'] = limits['lcl']
            out['ucl'] = limits["ucl"]

        # Drop NAs
        out = out.dropna()

        # Calculate Beyond Limits
        out['beyond_limits'] = np.where(out[spec.response_var] > out['lcl'], -1, 0)
        out['beyond_limits'] = np.where(out[spec.response_var] > out['ucl'], 1, 0)

        cols_to_keep = ['mr', 'mR', 'lcl', 'ucl', 'beyond_limits']

        if spec.has_time:
            if spec.has_grouping:
                cols_to_keep.insert(0, 'rsg')
                cols_to_keep.insert(0, spec.time_var)
            else:
                cols_to_keep.insert(0, spec.time_var)
        else:
            if spec.has_grouping:
                out['x'] = out.groupby(spec.rsg_var_name).cumcount() + 1
                cols_to_keep.insert(0, 'rsg')
                cols_to_keep.insert(0, 'x')
            else:
                out['x'] = out.index
                cols_to_keep.insert(0, 'x')

        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        if spec.has_grouping:
            statistics = gather_analysis_statistics(
                df=out,
                statistics_to_collect=['mR', 'lcl', 'ucl'],
                grouping_var=spec.rsg_var_name
            )
            split_dict = split_df_by_group(df=out, grouping_var=spec.rsg_var_name)
            out = package_analysis(
                analysis_output=split_dict,
                summary_statistics_output=statistics
            )
        else:
            statistics = gather_analysis_statistics(
                df=out,
                statistics_to_collect=['mR', 'lcl', 'ucl']
            )
            _out = {'all': out}
            out = package_analysis(
                analysis_output=_out,
                summary_statistics_output=statistics
            )

        return out


# ============================================================================
# Module-level Functions
# ============================================================================

def perform_analysis(df: pd.DataFrame, specification: dict):
    """
    Main entry point for performing statistical process control analysis.

    Args:
        df: Input DataFrame with raw data
        specification: Dictionary containing analysis configuration including 'analysis_type'

    Returns:
        DataFrame or dict containing analysis results

    Raises:
        ValueError: If analysis_type is not supported
    """
    # Use new unified Analysis class
    analysis = Analysis(df, specification)
    return analysis.calculate()
