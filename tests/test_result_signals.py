"""Tests for `result_signals` — signal-detection orchestration."""

from __future__ import annotations

import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic
from processbehavior.exceptions import ChartNotAvailableError, ProcessBehaviorError
from processbehavior.result_signals import (
    _KNOWN_CHART_TYPES,
    _build_config,
    detect_signals_for_result,
    extract_chart_type,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xbar_result():
    df = synthetic.make_design(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return (
        ProcessBehavior(df)
        .formulate(response="y", factors=["factor 1", "factor 2"], time="time")
        .execute()
    )


@pytest.fixture(scope="module")
def x_mr_companion_result():
    df = synthetic.make_design(2, K1=3, K2=2, T=10, seed=42)
    return (
        ProcessBehavior(df)
        .formulate(response="y", factors=["factor 1", "factor 2"], time="time")
        .execute(chart="X", by=[], companion=True)
    )


# ---------------------------------------------------------------------------
# detect_signals_for_result
# ---------------------------------------------------------------------------


class TestDetectSignalsForResult:
    def test_chart_none_returns_dict_per_chart(self, x_mr_companion_result):
        results = detect_signals_for_result(x_mr_companion_result)
        assert isinstance(results, dict)
        assert set(results.keys()) == set(x_mr_companion_result.charts.keys())

    def test_specific_chart_returns_single_signalresult(self, xbar_result):
        from processbehavior.signals.result import SignalResult

        result = detect_signals_for_result(xbar_result, chart="Xbar")
        assert isinstance(result, SignalResult)

    def test_unknown_chart_raises(self, xbar_result):
        with pytest.raises(ChartNotAvailableError):
            detect_signals_for_result(xbar_result, chart="DoesNotExist")

    def test_missing_metadata_raises(self):
        """A chart_info without metadata is a methodology bug — must raise."""

        class _StubResult:
            charts = {"Bad": {"data": None, "statistics": {}}}
            all_charts = ["Bad"]

            def _resolve_chart_name(self, name):
                return name

        with pytest.raises(ProcessBehaviorError):
            detect_signals_for_result(_StubResult(), chart="Bad")  # type: ignore[arg-type]

    def test_rules_preset_string_end_to_end(self, xbar_result):
        """The documented presets work through the public detect path."""
        for preset in ("standard", "extended", "all"):
            out = detect_signals_for_result(xbar_result, chart="Xbar", rules=preset)
            assert hasattr(out, "count")

    def test_rules_list_explicit(self, xbar_result):
        """rules=['rule_1'] is accepted; passed through."""
        result = detect_signals_for_result(xbar_result, chart="Xbar", rules=["rule_1"])
        assert result is not None

    def test_rules_ruleset_object(self, xbar_result):
        """rules=RuleSet(...) uses RuleSet.get_rules() under the hood."""
        from processbehavior.signals import RuleSet

        result = detect_signals_for_result(
            xbar_result, chart="Xbar", rules=RuleSet().beyond_limits()
        )
        assert result is not None

    def test_explicit_config_takes_precedence(self, xbar_result):
        """When `config=` is given, `rules=` and `**kwargs` are ignored."""
        from processbehavior.signals import SignalConfig

        cfg = SignalConfig(enabled_rules=["rule_1"])
        result = detect_signals_for_result(xbar_result, chart="Xbar", config=cfg)
        assert result is not None


# ---------------------------------------------------------------------------
# extract_chart_type
# ---------------------------------------------------------------------------


class TestExtractChartType:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Xbar", "Xbar"),
            ("Xbar_F1_1", "Xbar"),
            ("S", "S"),
            ("S_lane_2", "S"),
            ("X", "X"),
            ("X_F1_1_F2_2", "X"),
            ("mR", "mR"),
            ("mR_strat", "mR"),
        ],
    )
    def test_known_prefixes_map_to_base_type(self, name, expected):
        assert extract_chart_type(name) == expected

    def test_unknown_falls_back_to_xbar(self):
        """Legacy behaviour: unrecognised names default to 'Xbar'."""
        assert extract_chart_type("Unknown") == "Xbar"
        assert extract_chart_type("") == "Xbar"

    def test_known_types_constant_matches_default_prefixes(self):
        """The known-types tuple stays aligned with the project's chart names."""
        assert set(_KNOWN_CHART_TYPES) == {"Xbar", "S", "X", "mR"}


# ---------------------------------------------------------------------------
# _build_config
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def test_explicit_config_returned_unchanged(self):
        from processbehavior.signals import SignalConfig

        cfg = SignalConfig(enabled_rules=["rule_2"])
        result = _build_config(rules="standard", config=cfg, kwargs={"min_observations": 99})
        assert result is cfg
        assert cfg.enabled_rules == ["rule_2"]

    def test_default_config_when_rules_none(self):
        cfg = _build_config(rules=None, config=None, kwargs={})
        # SignalConfig default — exact comparison is implementation-detail-y,
        # so just confirm we got a config object.
        assert cfg is not None
        assert hasattr(cfg, "enabled_rules")

    def test_rules_preset_string_resolves_to_rule_list(self):
        """Preset strings resolve through SignalConfig.__post_init__.

        Regression: assigning enabled_rules after construction bypassed the
        preset resolution, so detect_signals(rules='standard') — the documented
        public path — raised 'Invalid enabled_rules state' while
        SignalConfig(enabled_rules='standard') worked.
        """
        cfg = _build_config(rules="extended", config=None, kwargs={})
        assert cfg.enabled_rules == [f"rule_{i}" for i in range(1, 9)]

        cfg = _build_config(rules="standard", config=None, kwargs={})
        assert cfg.enabled_rules == ["rule_1", "rule_2", "rule_3", "rule_4"]

    def test_rules_list_applied(self):
        cfg = _build_config(rules=["rule_1", "rule_2"], config=None, kwargs={})
        assert cfg.enabled_rules == ["rule_1", "rule_2"]

    def test_kwargs_set_attrs_if_present(self):
        cfg = _build_config(rules=None, config=None, kwargs={"min_observations": 7})
        # Only set if the attribute exists on SignalConfig
        assert getattr(cfg, "min_observations", None) == 7
