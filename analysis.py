"""
Analysis - Main user-facing class for process behavior chart calculations.

This module provides the Analysis class which executes chart calculations using
the strategy pattern. It supports:
- Xbar and S charts (subgroup mean and variation)
- IMR charts (individual moving range)
- R charts (range)

The Analysis class coordinates between:
- AnalysisSpecification (configuration)
- AnalysisDataSet (data preparation and VAS calculations)  
- Chart calculation strategies
- AnalysisResult (unified result container)

Usage:
    spec = {
        'analysis_type': 'Xbar',
        'response_var': 'Height',
        'time_var': 'Time',
        'rsg_vars': ['Operator', 'Machine']
    }
    
    analysis = Analysis(df, spec)
    result = analysis.calculate()
    
    # Access charts
    xbar = result.get_chart('Xbar')
    s = result.get_chart('Sbar')
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from spc_constants import calculate_limits, detect_beyond_limits
from analysis_specification import AnalysisSpecification
from analysis_dataset import AnalysisDataSet, split_df_by_group, gather_analysis_statistics, package_analysis
from analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

