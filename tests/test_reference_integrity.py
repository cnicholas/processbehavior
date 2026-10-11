"""The Bishop reference artifacts must never change quietly.

These files ARE the ground truth: the T100 validation database and the three
Minitab reference-analysis fixtures that e2e_bishop_report.py's assertions
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
        'validation/PBTESTKNOWNEFFECTS_T100.csv',
        '29704adcfa16d4bf06ab46b9fa5ff5e570ab9e0261c3103a53fbd7312e8975bf',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_1.json',
        '78c0f9c40239726d7ca56f775c0aa4b22655ab23796d32ad16210c1ce66fcab7',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_2.json',
        '9b3cc7ec7954da8d9490cb7da9519c11439cdf83febd4eed87e778842b9317f8',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_3.json',
        '70811e5f59fcd9995e9dc4437aec1aec08868add3996d95b3b245ed02e07eb7c',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_4.json',
        'f556d81251ff35706d8f5eddd5ad9b8f46b4af08520cfc19185ed223139f665c',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_5.json',
        '60c205dc4a9dc49398cfa00621272651c11be64f1cfb620115e4a928edab70aa',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_6.json',
        '4247d11e987aefb85c75cc0df5ec4aa3dd477eba6d0f6bfc2ee6e53acecc5443',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_1.json',
        '32e31db625f6bbeb2d9b9759b5e635b31d0ec4f5ba0235fd7d8c616d72f06f1b',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_2.json',
        '096cad6602731fdfc536c8bed89c6d87427496c61879cdae53d6715f6eb56eca',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_3.json',
        'eab2e47b7781871a82c48a51dc0b3ad156cc33580fe1b3d6074b1341afd86035',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_4.json',
        '50069462af1eaac4161153be3d35814342521f5976ab80dfe7e9fe7cadb6123b',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_5.json',
        'aabf897651880dccde74b8aef2a21e034e23878e4ac25adc6f10f19f3291a66b',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_6.json',
        '5c5d4a8c88b77d017863a564c653e90d9f38010656342504b5269b8e65ef01cf',
    ),
    (
        'tests/fixtures/bishop_analyses/known_effects_sds_1.json',
        '4d4cbd86327c4c4b820df72cc736b15cc4d4cd82f4e6c8da14f6670e64a49978',
    ),
    (
        'validation/aco_per_capita_expenditure.csv',
        '4c91829f13f3774eaf748be3064c9c9a9e2e288c6e5fa4946271e75fed77ac94',
    ),
    (
        'tests/fixtures/bishop_analyses/aco_medicare.json',
        '9494ee2e7ca31ce508347f8bd2e216de41252bed332e7d1bfdea78da677fd793',
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
