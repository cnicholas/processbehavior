"""The API reference must cover the public API — and only APIs that exist.

docs/reference/api.md is hand-written (mystmd has no mature Python autodoc), so
nothing structural stops it drifting from the code. This file is the drift
guard, at name-presence level by design:

- every symbol in ``processbehavior.__all__`` must appear in the reference, so
  adding an export without documenting it fails CI naming the symbol;
- a curated list of load-bearing vocabulary must appear (kwargs and types the
  reference exists to teach);
- known fictions and removed spellings must NOT appear, so a stale revert or a
  careless paste of pre-0.2.0 docs fails loudly.

Deliberately not here: signature parsing. Keeping the check to names makes it
cheap, deterministic, and free of false positives from prose formatting.
"""

import re
from pathlib import Path

import pytest

import processbehavior

API_MD = Path(__file__).resolve().parents[1] / 'docs' / 'reference' / 'api.md'


@pytest.fixture(scope='module')
def api_text() -> str:
    return API_MD.read_text(encoding='utf-8')


@pytest.mark.parametrize('name', sorted(processbehavior.__all__))
def test_every_export_is_documented(api_text, name):
    """Each top-level export appears in the reference by name."""
    assert re.search(rf'\b{re.escape(name)}\b', api_text), (
        f'{name!r} is exported from processbehavior but never appears in '
        f'docs/reference/api.md — document it (or remove the export).'
    )


#: Kwargs and types the reference exists to teach; their absence means a
#: rewrite dropped load-bearing content even if every export name survived.
LOAD_BEARING = [
    'companion',
    'stratify_by',
    'calibration',
    'FocusedAnalysisResult',
    'lpl',
    'upl',
    'limits_vary',
]


@pytest.mark.parametrize('term', LOAD_BEARING)
def test_load_bearing_vocabulary_present(api_text, term):
    assert re.search(rf'\b{re.escape(term)}\b', api_text), (
        f'{term!r} is part of the documented contract and must appear in api.md.'
    )


#: APIs that do not exist (or spellings removed in 0.2.0). Their reappearance
#: means stale content came back. 'execute(paired' rather than bare 'paired='
#: because CapabilityResult.plot() has a real paired= kwarg.
FICTIONS = [
    'get_c4',
    'get_A3',
    'get_B3',
    'get_B4',
    'get_d2',
    'get_D3',
    'get_D4',
    'DataPrepConfig',
    'SDSAnalysisPlan',
    'ResidualCalculator',
    'execute(paired',
    'R2_Imr',
    "chart='Imr'",
    'sds_name',
    'available_residuals',
]


@pytest.mark.parametrize('fiction', FICTIONS)
def test_no_documented_fictions(api_text, fiction):
    assert fiction not in api_text, (
        f'{fiction!r} appears in api.md but does not exist in the library '
        f'(or was removed in 0.2.0).'
    )
