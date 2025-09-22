"""
Constants for SPC analysis configuration.

This module contains configuration constants used throughout the process behavior charts system.
"""

# Column naming constants
RSG_VARIABLE_NAME = 'rsg'
TIME_VARIABLE_NAME = 'time'
RESPONSE_VARIABLE_NAME = 'response'

# Data type constants for validation
NUMBER_COLUMN_TYPES = ['int64', 'float64', 'int32', 'Int64', 'Float64', 'Int32']
DATE_COLUMN_TYPES = ['int64', 'int32', 'Int64', 'Int32', 'datetime64[ns]', 'object']

# Analysis type constants
SUPPORTED_ANALYSIS_TYPES = ['Xbar', 'S', 'Imr', 'R']
GROUPED_ANALYSES = ['Xbar', 'S']

# Time unit constants
VALID_TIME_UNITS = ['Year', 'Quarter', 'Month', 'Week']
