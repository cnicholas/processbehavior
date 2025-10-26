"""
Signal detection framework for control charts.

Provides flexible, declarative API for detecting Western Electric rules
and custom patterns in process behavior data.
"""

from __future__ import annotations

from .config import RuleSet, SignalConfig, ZoneDefinition
from .detector import SignalDetector
from .result import SignalResult

__all__ = ['SignalConfig', 'ZoneDefinition', 'RuleSet', 'SignalDetector', 'SignalResult']
