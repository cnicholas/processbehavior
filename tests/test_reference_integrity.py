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
        '18bc3172fd7ce5378dd2a1204151f79d8390fa9ed68564c3fc5b91cc53705058',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_2.json',
        '867937df46f602fb5f0c49559a24c5bdcf1dcb6d815fa7ae8d556c40b03ce947',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_3.json',
        'd0c7dd31e4f1caa4c078021ea5974b50964c67420120b83d91caf51a2ebe477a',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_4.json',
        '9ad9da957b1504272cecebd6552c98f365a25952503d6c71d1d20c678d785146',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_5.json',
        'df307bc4fc7603268bd4ce537095a2a10a3d89be0e5e931cb6ed28ff56894cda',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_6.json',
        '38fad86476a04be122739091e3b464152096ccc3978296ee921c3c065ad91e6c',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_1.json',
        '15b8579f6d18abfb917b7217d4c7792289e087e029f17387bf112f55db6d6f81',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_2.json',
        '38c216a433bb98660b4252b8f62a30babb2763ac1150fffbcfd8ed2f6dabad7e',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_3.json',
        'e6e0e1828866965a9bc8d4f34d8a8088ff0e5ade4a7f509d56ea1c6f3d38d7ac',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_4.json',
        '8cfe623f54467c66f38e8615f13b2ed97fdb1f2aa1225d4e03ef440f3ae8f437',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_5.json',
        'ab56f4ec6e23dfcf12ff0a4b4a2cf898f9420303ad25f52e54ad721cb6d47276',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_6.json',
        'd5374b63c7e84f7170cebe6d77524b0ee74c71593a9818e6173e76dac50584b0',
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
