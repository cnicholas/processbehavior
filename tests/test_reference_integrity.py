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
        '5bcb65168ea86c7d202f8b9dedd188f134673a884f5805c8605c4d676e7e8692',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_4.json',
        '03b7102e640485a861dd5bf665e1c697d126e7133e20e775fb1cf2c60a071ad2',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_5.json',
        'f3c605b89199e036b0b3e9a700fa4f319034ece6e6b52fb36ad49c452023e280',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_sds_6.json',
        '86d63d2d1c77f3444dd6cafd584d13c8c570b89ffca195ff35de4a858bfceb38',
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
        '7dbd2f7b2175a184fd4950ea442cdd5dd8a983b393616485cd42a05b35e15778',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_4.json',
        '680a357c882d82bebdeece19cdd2a7f211d7c3d9f148c4ce7a2c42102f6896ca',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_5.json',
        'c4e3b19492c482fb6e06e7bb01c3d79b26ea545e1df21c03cc4c1329c0517345',
    ),
    (
        'tests/fixtures/bishop_analyses/pm_inert_sds_6.json',
        '942a1c805d8862c478a5599c70f9dd96b23bc6b1cafd76a67c2cd4b49f515e63',
    ),
    (
        'tests/fixtures/bishop_analyses/known_effects_sds_1.json',
        '82b8231ae0cd04d62f56787b24bb64ff76c3c84747c2f15bf3d0c35a671f16b0',
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
