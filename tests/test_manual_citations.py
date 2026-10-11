"""Equation citations name Bishop's VAS documentation manual in one form, so they can be checked.

The library cites the manual by edition and equation, e.g. "(10-1 manual Eq 14-8)" or "Eqs 14-4..14-13".
The private manual repository reads these citations and checks every number against the manual's
equations, and reports the ones a new edition changes; that only works if the citations use the
chapter-number form. These tests keep the form: no dotted numbers ("Eq 12.4", an older manual's style),
and every file that cites an equation names the edition it means.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITE = re.compile(r'\bEq(?:uation)?s?\.?\s*\d{1,2}-\d{1,3}')
DOTTED = re.compile(r'\bEq(?:uation)?s?\.?\s*\d{1,2}\.\d{1,3}')
EDITION = re.compile(r'\b\d{1,2}-\d{1,2}(?:-\d{4})? manual\b|\bmanual dated \d{1,2}-\d{1,2}-\d{4}\b')


def _sources() -> dict[str, str]:
    files = subprocess.run(
        ['git', 'ls-files', 'processbehavior', 'docs', 'tests', 'validation', 'README.md', 'CONTRIBUTING.md'],
        capture_output=True, text=True, cwd=ROOT, check=True,
    ).stdout.split()
    out = {}
    for f in files:
        # released history keeps the numbers it was written with
        if not f.endswith(('.py', '.md', '.ipynb')) or Path(f).name.lower() == 'changelog.md':
            continue
        text = (ROOT / f).read_text(encoding='utf-8', errors='replace')
        if f.endswith('.ipynb'):
            text = '\n'.join(''.join(c.get('source', [])) for c in json.loads(text).get('cells', []))
        out[f] = text
    return out


def test_citations_use_chapter_number_form():
    dotted = {f: DOTTED.findall(t) for f, t in _sources().items() if DOTTED.search(t)}
    dotted.pop('tests/test_manual_citations.py', None)          # this file names the form it forbids
    assert dotted == {}


def test_every_citing_file_names_the_edition():
    sources = _sources()
    citing = {f for f, t in sources.items() if CITE.search(t)}
    assert len(citing) >= 20, 'too few citations found: the scan is broken'
    assert sorted(f for f in citing if not EDITION.search(sources[f])) == []
