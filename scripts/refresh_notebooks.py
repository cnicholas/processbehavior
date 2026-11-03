"""Re-execute the docs notebooks so their saved outputs match the library.

The docs site shows each notebook's saved outputs; it does not run them. After a change to the
library, run this to bring them up to date. CI runs it with --check, which executes every docs
notebook and fails if what a reader would see differs from what is saved.

    python scripts/refresh_notebooks.py               # refresh every docs notebook in place
    python scripts/refresh_notebooks.py PATH ...      # refresh some
    python scripts/refresh_notebooks.py --check       # fail if a fresh run differs from the saved outputs

What is compared is what a reader sees: printed text, errors, tables, and each figure's data and
labels. Plotly templates and styling are left out, so a plotly upgrade alone is not drift. Two
things that differ on every run are ignored: the temporary path in a warning's location, and the
random id of a derived variable.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import pprint
import re
import sys
from pathlib import Path

import nbformat
import numpy as np
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted([*ROOT.glob('docs/getting-started/*.ipynb'), *ROOT.glob('docs/tutorials/*.ipynb')])
PLOTLY = 'application/vnd.plotly.v1+json'
TEMP_PATH = re.compile(r'\S*ipykernel_\d+[/\\]\d+\.py')  # where a warning was raised; differs every run
DERIVATION_ID = re.compile(r'\bid=[0-9a-f]{12}\b')  # Derivation.id is a random uuid by design
AXES = ('xaxis', 'yaxis')


def execute(path: Path) -> nbformat.NotebookNode:
    """Run a notebook top to bottom in its own folder, as a reader would."""
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb, timeout=900, kernel_name='python3', record_timing=False, resources={'metadata': {'path': str(path.parent)}}
    ).execute()
    return nb


def _values(v):
    """A trace's x or y as plain values: decode plotly's base64 typed arrays and round floats.

    A missing value becomes None, whether it was saved as null or as NaN, because NaN never
    equals itself and would make identical figures compare as different.
    """
    if isinstance(v, dict) and 'bdata' in v:
        arr = np.frombuffer(base64.b64decode(v['bdata']), dtype=np.dtype(v['dtype']))
        if v.get('shape'):
            arr = arr.reshape([int(n) for n in str(v['shape']).split(',')])
        v = arr.tolist()
    if isinstance(v, (list, tuple)):
        return [_values(x) for x in v]
    if isinstance(v, float):
        return None if v != v else float(f'{v:.10g}')
    return v


def _label(title):
    return title.get('text') if isinstance(title, dict) else title


def _figure(fig: dict) -> dict:
    layout = fig.get('layout', {})
    return {
        'traces': [
            {'type': t.get('type'), 'name': t.get('name'), 'x': _values(t.get('x')), 'y': _values(t.get('y'))}
            for t in fig.get('data', [])
        ],
        'title': _label(layout.get('title')),
        'axes': {k: _label(v.get('title')) for k, v in sorted(layout.items()) if k[:5] in AXES and isinstance(v, dict)},
        'annotations': [a.get('text') for a in layout.get('annotations', [])],
        'shapes': [_values([s.get(k) for k in ('x0', 'x1', 'y0', 'y1')]) for s in layout.get('shapes', [])],
    }


def seen(nb: nbformat.NotebookNode) -> list[list]:
    """What a reader sees in each code cell: printed text, errors, tables and figures."""
    cells = []
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        out: list = []
        for o in cell.get('outputs', []):
            kind = o.get('output_type')
            if kind == 'stream':
                text = DERIVATION_ID.sub('id=<random>', TEMP_PATH.sub('<cell>', o.get('text', '')))
                if out and out[-1][:2] == ('stream', o.get('name')):
                    # The kernel splits printed text into chunks by timing; a reader sees one block.
                    out[-1] = ('stream', o.get('name'), out[-1][2] + text)
                else:
                    out.append(('stream', o.get('name'), text))
            elif kind == 'error':
                out.append(('error', o.get('ename'), o.get('evalue')))
            elif 'data' in o:
                if PLOTLY in o['data']:
                    out.append(('figure', _figure(o['data'][PLOTLY])))
                elif 'text/plain' in o['data']:
                    out.append(('text', o['data']['text/plain']))
        cells.append(out)
    return cells


def _first_difference(saved: list, fresh: list, limit: int = 20) -> str:
    before, after = (pprint.pformat(x, width=110).splitlines() for x in (saved, fresh))
    lines = difflib.unified_diff(before, after, 'saved', 'fresh', n=1, lineterm='')
    return '\n'.join(list(lines)[:limit])


def _keep_unchanged(saved, fresh, before: list, after: list) -> None:
    """Keep the saved bytes of every cell a reader would see unchanged, so float noise is not churn."""
    code = [(s, f) for s, f in zip(saved.cells, fresh.cells, strict=True) if s.cell_type == 'code']
    for (s_cell, f_cell), a, b in zip(code, before, after, strict=True):
        if a == b:
            f_cell.outputs, f_cell.execution_count = s_cell.outputs, s_cell.execution_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('paths', nargs='*', type=Path, help='notebooks to refresh (default: every docs notebook)')
    parser.add_argument('--check', action='store_true', help='fail if a fresh run differs from the saved outputs')
    args = parser.parse_args(argv)
    paths = [p.resolve() for p in args.paths] or NOTEBOOKS

    drifted = []
    for path in paths:
        name = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        saved = nbformat.read(path, as_version=4)
        fresh = execute(path)
        before, after = seen(saved), seen(fresh)
        if before == after:
            print(f'{name}: current')
            continue
        changed = [i + 1 for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
        if args.check:
            drifted.append(name)
            first = changed[0] - 1
            print(f'{name}: code cells {changed} differ from a fresh run')
            print(_first_difference(before[first], after[first]))
        else:
            fresh.metadata = saved.metadata  # keep the kernel and Python version the notebook was saved with
            _keep_unchanged(saved, fresh, before, after)
            nbformat.write(fresh, path)
            print(f'{name}: refreshed (code cells {changed})')

    if drifted:
        print(f'\n{len(drifted)} notebook(s) no longer match the library. Refresh them with:')
        print('  python scripts/refresh_notebooks.py ' + ' '.join(str(n) for n in drifted))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
