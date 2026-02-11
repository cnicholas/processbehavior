# Changelog

All notable changes to ProcessBehavior are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Jupyter Book documentation structure
- Comprehensive tutorials (IMR, Xbar-S, Stratified, Signal Detection)
- User guide for all major features
- API reference documentation
- Western Electric rules reference

### Changed
- Removed scipy dependency (pure Python implementations)
- Improved signal detection for S charts (rule-based minimum observations)

### Fixed
- R2_S chart signal detection no longer requires 20 observations
- Residual plot tests now properly exercise `plot_residuals()`

---

## [0.1.0] - 2025-XX-XX

### Added

#### Core Features
- `ProcessBehavior` class for data wrapping with IDE auto-completion
- `formulate()` method for study definition
- Automatic Sampling Design State (SDS 1-6) detection
- `Study` class with chart recommendations

#### Analysis
- `analyze()` method for chart calculation
- `AnalysisResult` class for unified result access
- VAS residual calculations (R1-R5)
- Support for Xbar, S, Imr, and R charts
- Stratified IMR analysis

#### Signal Detection
- Western Electric rules 1-8
- `detect_signals()` method
- `RuleSet` builder for custom configurations
- `SignalResult` class for result access

#### Visualization
- Interactive Plotly-based charts
- Zone shading (A, B, C zones)
- Signal highlighting
- Four built-in themes (processbehavior, minimal, dark, ggplot)
- Custom theme support via `ChartTheme`
- Faceted plots for stratified analysis

#### Export
- Excel export with `to_excel()`
- HTML export for interactive charts
- Chart image embedding (requires kaleido)

#### Data Handling
- Automatic garbage value cleaning
- Natural sorting for factor levels
- Configurable precision

### Dependencies
- pandas >= 2.0
- natsort >= 8.0
- plotly >= 5.18 (optional, for visualization)
- openpyxl >= 3.1 (optional, for Excel export)
- kaleido (optional, for image export)

---

## Version History

### Pre-release Development

The following features were developed during the pre-release phase:

#### Phase 1: Core Architecture
- ProcessBehavior wrapper
- Column auto-completion
- Basic data validation

#### Phase 2: SDS Detection
- Sampling Design State detection algorithm
- SDSAnalysisPlan specification
- Valid chart determination

#### Phase 3: VAS Implementation
- R1-R5 residual calculations
- Mean calculations (Y̅, Y̅_k, Y̅_t, Y̅_kt)
- Residual chart types

#### Phase 4: Visualization
- Plotly integration
- Theme system
- Zone and signal visualization

#### Phase 5: Signal Detection
- Western Electric rules
- RuleSet fluent API
- SignalResult class

#### Phase 6: Export
- Excel workbook generation
- Multi-sheet organization
- Chart image embedding

---

## Roadmap

### Planned for v0.2.0
- Additional chart types
- Performance optimizations for large datasets
- Enhanced effects analysis

### Planned for v0.3.0
- Real-time monitoring mode
- Database integration
- Custom rule definitions
- API stability guarantees

### Under Consideration
- R integration via reticulate
- Web dashboard
- Automated report generation
- Multi-language support

---

## Migration Notes

### From Pre-release to v0.1.0

If you were using a pre-release version, note these changes:

#### API Changes
```python
# Old (pre-release)
result = pb.analyze(response_var='weight', chart_type='Imr').calculate()

# New (v0.1.0)
study = pb.formulate(response=pb.cols.weight)
result = study.execute()
```

#### Terminology Changes
- `grouping_vars` → `factors`
- `time_var` → `time`
- `chart_type` → `chart`
- `.calculate()` removed (analysis is immediate)

#### Import Changes
```python
# Old
from processbehavior import ProcessBehavior

# New (same, but more exports available)
from processbehavior import (
    ProcessBehavior,
    Study,
    AnalysisResult,
    SignalDetector,
    RuleSet
)
```

---

## Contributing

See [CONTRIBUTING.md](https://github.com/cnicholas/processbehavior/blob/main/CONTRIBUTING.md) for guidelines.

## License

ProcessBehavior is released under the Apache 2.0 License.
