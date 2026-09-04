"""The Bishop reference artifacts must never change quietly.

These files ARE the ground truth: the T100 validation database and the three
Minitab reference-analysis fixtures that e2e_bishop_report.py's 280 assertions
check against. Every number the library claims to validate traces back to them,
so an edit here is an edit to what "correct" means for the whole package.

A hash mismatch is not necessarily a bug — it means the reference data changed.
If that change is intentional (e.g., Bishop supplies updated reference output),
recompute the hash and update it IN THE SAME reviewed diff:

    python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <path>

The test exists so that change can never happen as a side effect, a stray
regeneration, or a quiet line in a large diff. (The inline EXPECTED_* dicts in
e2e_bishop_report.py are guarded differently: the script exits nonzero when any
assertion fails, so mis-editing them breaks the build directly.)
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_ARTIFACTS = [
    (
        'validation/PBTESTDATABASE_T100.csv',
        '7b6e407cf33dbd84380bb0894a31dc5e5863a27f3f101fe958e3c8ef94989328',
    ),
    (
        'tests/fixtures/bishop_analyses/vassds1analysis.json',
        'e34f276446584b6ce2c765dfc7f8a154fa74bd01e82e6335b13e2af1b85e8c26',
    ),
    (
        'tests/fixtures/bishop_analyses/vassds2analysis.json',
        'aeaebaf82cdef7c121ffc73dad9419ef0ac723b1f8369a451d8bc6a3eb1b62cb',
    ),
    (
        'tests/fixtures/bishop_analyses/vassds3analysis.json',
        'b5a34385cf3d3184f1a016854c62901fe1ab44111897ca80e1ca8fae6c2f3440',
    ),
]


@pytest.mark.parametrize(('relpath', 'expected_sha256'), REFERENCE_ARTIFACTS)
def test_reference_artifact_unchanged(relpath, expected_sha256):
    actual = hashlib.sha256((ROOT / relpath).read_bytes()).hexdigest()
    assert actual == expected_sha256, (
        f'{relpath} no longer matches its pinned SHA-256.\n'
        f'  pinned: {expected_sha256}\n'
        f'  actual: {actual}\n'
        f'This file is Bishop reference ground truth. If the change is intentional, '
        f'update the pin in this file in the same reviewed commit.'
    )
