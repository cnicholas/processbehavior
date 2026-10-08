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
        '94266cd323ca7047163ffdb71bc587116c52be7e71b340df4e62a8aa843358cd',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_2.json',
        'dbee3923677d6f9c2d1abf645c368a7c33c4195ea0e7ee31aad70aed634e2326',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_3.json',
        '6b6e20858a68aa2d8f12a4bf94b46fc6773b91f2303f3e0aefcc2d9024209d9f',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_4.json',
        '65faa6baf8d1e611de39d31db79a3e065b644b2fcab35e10679c3d1295fdb8fd',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_5.json',
        '2c306ea4ebf17d7dd77c70aaa78910afc5ab08d66dab7af617ae4463a384f5a5',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_6.json',
        'e2388419b178238216b6a11f7a21a3484e18aabb6f100959e5d1f1c9a34f8684',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_1.json',
        '5b4af30c6904e1a837a8f392e8239552f8aa41ce92f195cabe509138c2ce3002',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_2.json',
        '096cad6602731fdfc536c8bed89c6d87427496c61879cdae53d6715f6eb56eca',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_3.json',
        'c324689071a5af0ab4a5fb55fee6d5948e08140d8ccb91b147ebde5167cd691f',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_4.json',
        'b28cf80db8f2f6aa1e41ef16d12c9607f004a0cd4460de75c787c11a6967832d',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_5.json',
        'df2a0bf40c732f5bdcd21b64a89c0df8fd850374ea773389405fc2212f23b2dc',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_6.json',
        '79dd31e929a3899e6af03163f77e96c317f650b5ca518e626df45cf8225ac793',
    ),
    (
        'tests/fixtures/bishop_analyses/known_effects_sds_1.json',
        '82b8231ae0cd04d62f56787b24bb64ff76c3c84747c2f15bf3d0c35a671f16b0',
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
