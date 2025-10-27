# Chart Schema Design: Where Should Value Column Logic Live?

## Context

During Phase 1 implementation of WECO Rules integration, we identified a fundamental design question:

**Where should "which column to plot/analyze for chart X" logic live?**

This logic is needed by:
- **Plotter**: Determining which column to plot on y-axis
- **SignalDetector**: Determining which column to analyze for violations
- **Future components**: Any system that needs to access chart values

Current situation:
- Plotter has `_get_value_column()` with explicit logic
- SignalDetector has no logic (defaults to 'mean' ❌)
- No shared implementation → duplication and bugs

---

## The Deeper Question

This isn't just about "where to put a utility function." It's about **architectural philosophy**:

1. **Is chart schema configuration or computation?**
2. **Should schema be defined once or emerge from implementation?**
3. **Who is the source of truth: SDS, Specification, or Chart Result?**

Key insight from discussion: "Residuals are SDS and chart specific" - suggesting schema might belong with SDS analysis plans.

---

## Option 1: Specification-Driven (AnalysisSpecification)

### Concept
The specification already defines what analysis to perform. Extend it to define the output column contract.

### Implementation
```python
# In AnalysisSpecification or related
class ChartOutputContract:
    """Defines the columns that each chart type produces."""

    CHART_SCHEMAS = {
        'Xbar': {
            'value_col': 'xbar',
            'center_col': 'center',
            'required_cols': ['rsg', 'xbar', 'center', 'lcl', 'ucl', 'beyond_limits']
        },
        'Sbar': {
            'value_col': 's',
            'center_col': 'center',
            'required_cols': ['rsg', 's', 'center', 'lcl', 'ucl', 'beyond_limits']
        },
        'R': {
            'value_col': 'mr',
            'center_col': 'center',
            'required_cols': ['mr', 'center', 'lcl', 'ucl', 'beyond_limits']
        },
        'Imr': {
            'value_col': '{response_var}',  # Dynamic
            'center_col': 'center',
            'required_cols': ['{response_var}', 'center', 'lcl', 'ucl', 'beyond_limits']
        }
    }

    @classmethod
    def get_value_column(cls, chart_type: str, response_var: Optional[str] = None) -> str:
        """Get the value column for a chart type."""
        schema = cls.CHART_SCHEMAS[chart_type]
        value_col = schema['value_col']

        # Handle dynamic columns (IMR)
        if '{response_var}' in value_col:
            if not response_var:
                raise ValueError("response_var required for IMR charts")
            return response_var

        return value_col
```

### Usage
```python
# In Plotter or SignalDetector
from processbehavior.analysis_specification import ChartOutputContract

value_col = ChartOutputContract.get_value_column('Xbar')
# Returns: 'xbar'
```

### Pros
- ✅ **Single source of truth** - Chart schema defined once
- ✅ **Validates contract** - Can check if chart data matches schema
- ✅ **Discoverable** - Schema is configuration, not hidden in code
- ✅ **Type-safe** - Could use dataclasses/TypedDict for schemas
- ✅ **Extensible** - Easy to add new chart types
- ✅ **Testable** - Schema is data, easy to validate

### Cons
- ⚠️ **Circular dependency risk** - Specification used early in pipeline, might need chart results
- ⚠️ **Coupling** - Ties specification to chart implementation details
- ⚠️ **Timing issue** - Specification created before charts, but schema describes chart outputs

### Best For
- Systems where chart schema needs to be validated/enforced
- When you want compile-time checking of column contracts
- Static schemas that don't vary by data

### Architectural Fit
**3/5** - Specification is for input/requirements, not output schema

---

## Option 2: SDS-Driven (SDSAnalysisPlan) ⭐⭐

### Concept
The SDS already knows what's possible for a given data structure. Extend it to define outputs.

Key insight: "Residuals are SDS and chart specific" - suggests schema belongs with SDS.

### Implementation
```python
# In sds_detector.py - extend SDSAnalysisPlan
@dataclass
class ChartSchema:
    """Schema for chart output columns."""
    value_col: str  # Can contain placeholders like {response_var}
    center_col: str = 'center'
    required_cols: list[str] = field(default_factory=list)

    def resolve_value_column(self, response_var: Optional[str] = None) -> str:
        """Resolve dynamic column names."""
        if '{response_var}' in self.value_col:
            if not response_var:
                raise ValueError("response_var required")
            return response_var
        return self.value_col


@dataclass
class SDSAnalysisPlan:
    """Analysis capabilities and outputs for a sampling design state."""

    # Existing fields
    sds: int
    name: str
    valid_charts: list[str]
    residual_methods: list[str]
    supports_effects: bool
    supports_interactions: bool

    # NEW: Chart output schemas
    chart_schemas: dict[str, ChartSchema] = field(default_factory=dict)

    def get_value_column(self, chart_type: str, response_var: Optional[str] = None) -> str:
        """Get the value column for a chart in this SDS."""
        schema = self.chart_schemas.get(chart_type)
        if not schema:
            raise ValueError(
                f"Chart type {chart_type} not valid for SDS {self.sds}.\n"
                f"Valid charts: {self.valid_charts}"
            )

        return schema.resolve_value_column(response_var)


# Usage in SDS definitions
SDS_PLANS = {
    1: SDSAnalysisPlan(
        sds=1,
        name="Subgrouped - Single Response",
        valid_charts=['Xbar', 'Sbar'],
        chart_schemas={
            'Xbar': ChartSchema(
                value_col='xbar',
                required_cols=['rsg', 'xbar', 'center', 'lcl', 'ucl', 'beyond_limits']
            ),
            'Sbar': ChartSchema(
                value_col='s',
                required_cols=['rsg', 's', 'center', 'lcl', 'ucl', 'beyond_limits']
            )
        },
        residual_methods=['exact'],
        supports_effects=True,
        supports_interactions=True
    ),
    5: SDSAnalysisPlan(
        sds=5,
        name="Individual - Single Response",
        valid_charts=['Imr', 'R'],
        chart_schemas={
            'Imr': ChartSchema(
                value_col='{response_var}',
                required_cols=['{response_var}', 'center', 'lcl', 'ucl', 'beyond_limits']
            ),
            'R': ChartSchema(
                value_col='mr',
                required_cols=['mr', 'center', 'lcl', 'ucl', 'beyond_limits']
            )
        },
        residual_methods=['moving_average'],
        supports_effects=False,
        supports_interactions=False
    )
}
```

### Usage
```python
# In AnalysisResult or SignalDetector
sds_plan = self.summary.get('sds_plan')  # or lookup by SDS number
value_col = sds_plan.get_value_column('Xbar', response_var='measurement')
```

### Pros
- ✅ **Architecturally consistent** - SDS already defines capabilities
- ✅ **Natural fit** - "What can this SDS produce?" includes output schema
- ✅ **Residuals connection** - Residuals are also SDS-specific, same pattern
- ✅ **Validation** - Can validate chart output matches SDS schema
- ✅ **Per-SDS customization** - Different SDS could have different schemas (future flexibility)
- ✅ **Discoverability** - SDS plan documents everything about that state
- ✅ **Centralized knowledge** - All SDS capabilities in one place

### Cons
- ⚠️ **More complex** - SDS detector becomes larger
- ⚠️ **Schema duplication** - Same chart schema repeated across multiple SDS plans
  - Example: Xbar schema in SDS 1, 2, 3, 4
- ⚠️ **Indirect access** - Need to know SDS to get schema
- ⚠️ **Maintenance burden** - Update all SDS plans when chart schema changes

### Best For
- When chart output schema truly varies by SDS (currently doesn't, but could)
- When you want "what can I do?" and "what will it produce?" in one place
- Systems where SDS is the primary organizing principle

### Architectural Fit
**5/5** - Perfect fit with existing SDS-driven architecture

---

## Option 3: Chart Result Metadata ⭐⭐⭐

### Concept
The chart calculation already returns `{'data': df, 'statistics': stats}`. Add schema metadata.

### Implementation
```python
# In Analysis class when building charts
def _calculate_xbar(self, ...):
    # ... existing calculation code ...

    result['Xbar'] = {
        'data': xbar,
        'statistics': statistics,
        'metadata': {  # NEW
            'chart_type': 'Xbar',
            'value_col': 'xbar',
            'center_col': 'center',
            'x_col': spec.time_var or 'x'
        }
    }
    return result

def _calculate_imr(self, ...):
    # ... existing calculation code ...

    result['Imr'] = {
        'data': out,
        'statistics': statistics,
        'metadata': {  # NEW
            'chart_type': 'Imr',
            'value_col': spec.response_var,
            'center_col': 'center',
            'x_col': spec.time_var or 'x'
        }
    }
    return result
```

### Usage
```python
# In Plotter
chart_info = self.charts['Xbar']
value_col = chart_info['metadata']['value_col']  # Direct lookup!
x_col = chart_info['metadata']['x_col']

# In SignalDetector (via AnalysisResult.detect_signals)
chart_info = self.charts[chart]
value_col = chart_info['metadata']['value_col']
detector.detect(data=chart_info['data'], value_col=value_col, ...)
```

### Pros
- ✅ **Self-documenting** - Chart data carries its own schema
- ✅ **No lookup needed** - Schema travels with data
- ✅ **Simple implementation** - Just add dict entry during calculation
- ✅ **Runtime accurate** - Schema reflects what was actually produced
- ✅ **No new classes** - Uses existing dict structure
- ✅ **Easy migration** - Add gradually, fall back to inference if missing
- ✅ **Minimal disruption** - Additive change, no refactoring

### Cons
- ⚠️ **Scattered definition** - Schema defined in multiple calculation methods
- ⚠️ **Duplication** - Each chart calculation repeats schema definition
- ⚠️ **No validation** - Schema could be wrong/inconsistent between charts
- ⚠️ **Runtime only** - Can't validate schema before analysis runs
- ⚠️ **No central registry** - Can't query "what schemas exist?"

### Best For
- Pragmatic solution when you want simple, direct access
- When schema is purely documentation, not enforcement
- Quick fix to unblock WECO integration
- When you trust chart calculation is correct

### Architectural Fit
**4/5** - Fits well with existing result structure, self-contained

---

## Option 4: ChartType Enum/Registry (Type-Driven)

### Concept
Make chart types first-class with associated metadata using Python Enum.

### Implementation
```python
# In new module: chart_types.py
from enum import Enum
from dataclasses import dataclass, field

@dataclass
class ChartTypeInfo:
    """Complete information about a chart type."""
    name: str
    value_col: str
    center_col: str = 'center'
    requires_response_var: bool = False
    valid_for_sds: list[int] = field(default_factory=list)
    residual_capable: bool = False
    required_columns: list[str] = field(default_factory=list)


class ChartType(Enum):
    """Registry of all chart types with their specifications."""

    XBAR = ChartTypeInfo(
        name='Xbar',
        value_col='xbar',
        center_col='center',
        valid_for_sds=[1, 2, 3, 4],
        residual_capable=True,
        required_columns=['rsg', 'xbar', 'center', 'lcl', 'ucl', 'beyond_limits']
    )

    SBAR = ChartTypeInfo(
        name='Sbar',
        value_col='s',
        center_col='center',
        valid_for_sds=[1, 2, 3, 4],
        residual_capable=False,
        required_columns=['rsg', 's', 'center', 'lcl', 'ucl', 'beyond_limits']
    )

    IMR = ChartTypeInfo(
        name='Imr',
        value_col='{response_var}',
        center_col='center',
        requires_response_var=True,
        valid_for_sds=[5, 6],
        residual_capable=True,
        required_columns=['{response_var}', 'center', 'lcl', 'ucl', 'beyond_limits']
    )

    R = ChartTypeInfo(
        name='R',
        value_col='mr',
        center_col='center',
        valid_for_sds=[5, 6],
        residual_capable=False,
        required_columns=['mr', 'center', 'lcl', 'ucl', 'beyond_limits']
    )

    @classmethod
    def get_value_column(cls, chart_name: str, response_var: Optional[str] = None) -> str:
        """Get value column for a chart type."""
        for chart_type in cls:
            if chart_type.value.name == chart_name:
                info = chart_type.value
                if info.requires_response_var:
                    if not response_var:
                        raise ValueError(f"{chart_name} requires response_var")
                    return response_var
                return info.value_col

        raise ValueError(f"Unknown chart type: {chart_name}")

    @classmethod
    def get_charts_for_sds(cls, sds: int) -> list[str]:
        """Get all valid chart types for an SDS."""
        return [ct.value.name for ct in cls if sds in ct.value.valid_for_sds]

    @classmethod
    def get_residual_capable_charts(cls) -> list[str]:
        """Get all charts that support residual plotting."""
        return [ct.value.name for ct in cls if ct.value.residual_capable]
```

### Usage
```python
# Direct lookup
value_col = ChartType.get_value_column('Xbar')

# Query capabilities
residual_charts = ChartType.get_residual_capable_charts()
# Returns: ['Xbar', 'Imr']

# Get charts for SDS
valid_charts = ChartType.get_charts_for_sds(sds=1)
# Returns: ['Xbar', 'Sbar']
```

### Pros
- ✅ **Type-safe** - Enum gives compile-time checking
- ✅ **Complete metadata** - All chart info in one place
- ✅ **Queryable** - Can ask "which charts support residuals?"
- ✅ **Extensible** - Add new chart types easily
- ✅ **Single source** - One definition per chart type
- ✅ **IDE support** - Autocomplete for chart types
- ✅ **Discoverability** - Can enumerate all chart types
- ✅ **Validation** - Can validate chart against schema

### Cons
- ⚠️ **New abstraction** - Adds significant complexity
- ⚠️ **String matching** - Still need to match chart_name strings from results
- ⚠️ **Maintenance** - Central registry needs updates for any change
- ⚠️ **Over-engineering** - Might be overkill for 4 chart types
- ⚠️ **Not Pythonic** - Enum pattern less common in data science code

### Best For
- When you want full type system for charts
- When chart metadata is queried frequently
- Larger systems with many chart types
- When you prioritize type safety over simplicity

### Architectural Fit
**3/5** - Solid pattern but adds abstraction layer not present elsewhere

---

## Option 5: Hybrid (SDS + Chart Result Metadata) ⭐⭐⭐⭐

### Concept
**Specification at the SDS level, implementation in chart results.**

SDS defines what SHOULD be produced (contract).
Chart result metadata describes what WAS produced (reality).
Validation ensures they match.

### Implementation
```python
# 1. SDS defines expected schemas (Option 2)
@dataclass
class SDSAnalysisPlan:
    sds: int
    valid_charts: list[str]
    chart_schemas: dict[str, ChartSchema]  # Expected output
    ...

# 2. Chart calculation produces metadata (Option 3)
def _calculate_xbar(self, ...):
    result['Xbar'] = {
        'data': xbar,
        'statistics': statistics,
        'metadata': {
            'chart_type': 'Xbar',
            'value_col': 'xbar',
            'center_col': 'center'
        }
    }

# 3. Validation layer (NEW)
def _validate_chart_output(
    chart_result: dict,
    expected_schema: ChartSchema,
    chart_name: str
) -> None:
    """Validate that chart output matches SDS schema."""
    actual_cols = set(chart_result['data'].columns)
    metadata = chart_result['metadata']

    # Check value column
    if metadata['value_col'] != expected_schema.value_col:
        raise ValueError(
            f"Schema mismatch for {chart_name}: "
            f"expected value_col='{expected_schema.value_col}', "
            f"got '{metadata['value_col']}'"
        )

    # Check required columns present
    for col in expected_schema.required_cols:
        if col not in actual_cols:
            raise ValueError(
                f"Missing required column '{col}' in {chart_name} output"
            )

# 4. Consumers use metadata (runtime truth)
chart_info = self.charts['Xbar']
value_col = chart_info['metadata']['value_col']  # Always use this

# 5. Optional validation can be enabled
if VALIDATE_SCHEMAS:
    sds_plan = get_sds_plan(sds)
    expected = sds_plan.chart_schemas['Xbar']
    _validate_chart_output(chart_info, expected, 'Xbar')
```

### Usage Patterns
```python
# Consumer perspective (Plotter, SignalDetector)
# Always use metadata - simple, direct
value_col = chart_info['metadata']['value_col']

# Developer perspective (debugging, validation)
# Can check against SDS schema when needed
sds_plan = result.summary['sds_plan']
expected_schema = sds_plan.chart_schemas['Xbar']
# Compare with actual metadata
```

### Pros
- ✅ **Best of both worlds** - Specification AND runtime data
- ✅ **Validation possible** - Can check output matches schema
- ✅ **Self-documenting** - Chart data carries metadata
- ✅ **Architecturally sound** - SDS defines contract, charts fulfill it
- ✅ **Debug friendly** - Easy to trace mismatches
- ✅ **Gradual adoption** - Can add validation later
- ✅ **Fail-safe** - Consumers use metadata (always works), validation optional

### Cons
- ⚠️ **Two places** - Schema exists in SDS and chart result
- ⚠️ **Duplication** - Same info in two locations
- ⚠️ **Most complex** - Most code to implement
- ⚠️ **Maintenance** - Keep SDS schema and metadata in sync

### Best For
- Production systems needing validation
- When you want to catch bugs early (schema mismatch)
- Long-term maintainability
- Systems that evolve over time

### Architectural Fit
**5/5** - Best architectural fit, supports validation and evolution

---

## Comparison Matrix

| Criteria | Option 1<br/>Specification | Option 2<br/>SDS-Driven | Option 3<br/>Metadata | Option 4<br/>Enum Registry | Option 5<br/>Hybrid |
|----------|---------------------------|-------------------------|----------------------|---------------------------|---------------------|
| **Simplicity** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Architectural Fit** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Validation** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Extensibility** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Discoverability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Implementation Time** | 3-4 hours | 4-6 hours | 2-3 hours | 5-7 hours | 6-8 hours |
| **Risk** | Low | Medium | Very Low | Medium | Medium |
| **Long-term Maintenance** | Medium | High | Low | High | High |

---

## Recommendation: Phased Approach

### Phase 1 (IMMEDIATE): Option 3 - Chart Result Metadata

**Goal**: Unblock WECO integration NOW

**Implementation**:
```python
# In Analysis._calculate_xbar, _calculate_sbar, _calculate_imr, _calculate_r
# Add metadata dict to each chart result

result['Xbar'] = {
    'data': xbar,
    'statistics': statistics,
    'metadata': {
        'chart_type': 'Xbar',
        'value_col': 'xbar',
        'center_col': 'center'
    }
}
```

**Changes needed**:
1. Add metadata to 4 chart calculation methods (Xbar, Sbar, IMR, R)
2. Update AnalysisResult.detect_signals() to use metadata
3. Update Plotter._get_value_column() to use metadata (with fallback)
4. Add integration tests

**Time**: 2-3 hours
**Risk**: Very Low (additive change)
**Benefit**: Immediate fix, WECO integration works

### Phase 2 (FUTURE): Option 5 - Add SDS Schema Validation

**Goal**: Add validation layer without disrupting working code

**Implementation**:
1. Add `chart_schemas` to SDSAnalysisPlan dataclass
2. Define expected schemas for each SDS
3. Add optional validation function
4. Enable validation in tests/debug mode

**Time**: 4-6 hours
**Risk**: Low (validation layer on top)
**Benefit**: Catch bugs, enforce contracts, better maintainability

### Why This Approach?

1. **Immediate value**: Phase 1 solves the problem NOW
2. **Low risk**: Additive changes, no breaking changes
3. **Flexibility**: Can always add validation later
4. **Practical**: Don't let perfect be the enemy of good
5. **Evolutionary**: Fits with "make it work, make it right, make it fast"

---

## Open Questions

### 1. Validation Importance
How important is catching schema mismatches early?
- **Low importance** → Stay with Option 3
- **High importance** → Go straight to Option 5

### 2. Residuals Pattern
Should residuals follow the same metadata pattern?
```python
result['R1'] = {
    'data': r1_data,
    'statistics': {...},
    'metadata': {
        'chart_type': 'Residual',
        'value_col': 'R1',
        'residual_level': 1,
        'is_residual': True
    }
}
```

### 3. Effects and Interactions
Should effects/interactions also have metadata?
```python
result['effects']['operator'] = {
    'data': effects_data,
    'metadata': {
        'value_col': 'effect',
        'factor': 'operator',
        'is_effect': True
    }
}
```

### 4. Future Chart Types
Planning to add new chart types? If yes:
- Option 2 or 5 better (centralized schema)
- Option 3 sufficient if rarely adding charts

### 5. Schema Evolution
How to handle backward compatibility if schemas change?
- Metadata versioning?
- Schema migration strategy?

---

## Decision

**Recommended**: Phase 1 with Option 3 (Chart Result Metadata)

**Rationale**:
- ✅ Simplest solution that solves the problem
- ✅ Minimal disruption to existing code
- ✅ Self-documenting charts
- ✅ Easy to implement and test
- ✅ Sets up for future validation layer (Option 5)
- ✅ No premature optimization

**Next Steps**:
1. Implement Phase 1 (Option 3) to unblock WECO integration
2. Evaluate Phase 2 (Option 5) after Phase 1 is working
3. Consider validation layer based on bug frequency
4. Apply same pattern to residuals/effects when plotting those

---

## References

- **WECO_RULES_INTEGRATION_PLAN.md**: Phase 1 requirements
- **ARCHITECTURE_ASSESSMENT_2025.md**: Architectural principles
- **processbehavior/plotting/plotter.py:417-496**: Current _get_value_column() implementation
- **processbehavior/sds_detector.py**: SDSAnalysisPlan structure
