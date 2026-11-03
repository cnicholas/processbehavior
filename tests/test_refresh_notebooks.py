"""
scripts/refresh_notebooks.py: what counts as a docs notebook drifting from the library.

CI runs the script with --check, so seen() and same() must ignore what differs between runs and
machines (warning paths, random derivation ids, how the kernel chunks printed text, plotly styling
and number encoding, float noise in the last digits) and must still catch a changed number in text
or in a figure.
"""

import base64
import importlib.util
from pathlib import Path

import nbformat
import numpy as np
from nbformat.v4 import new_code_cell, new_notebook, new_output

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'refresh_notebooks.py'
_spec = importlib.util.spec_from_file_location('refresh_notebooks', SCRIPT)
refresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh)

PLOTLY = 'application/vnd.plotly.v1+json'


def _nb(*outputs) -> nbformat.NotebookNode:
    return new_notebook(cells=[new_code_cell('x', outputs=list(outputs))])


def _stream(text, name='stdout'):
    return new_output('stream', name=name, text=text)


def _figure(y, template='plotly', title='X chart'):
    fig = {
        'data': [{'type': 'scatter', 'name': 'X', 'x': list(range(1, 4)), 'y': y}],
        'layout': {'template': {'layout': {'font': template}}, 'title': {'text': title}},
    }
    return new_output('display_data', data={PLOTLY: fig, 'text/plain': 'Figure()'})


def _bdata(values):
    return {'dtype': 'f8', 'bdata': base64.b64encode(np.array(values, dtype='f8').tobytes()).decode()}


def _alike(saved, fresh) -> bool:
    """Whether the check would call these two single-cell notebooks the same."""
    return refresh.same(refresh.seen(saved), refresh.seen(fresh))


def test_docs_notebooks_are_found():
    names = {p.name for p in refresh.NOTEBOOKS}
    assert {'quickstart.ipynb', 'signal-detection.ipynb', 'sds-validation.ipynb'} <= names


def test_warning_location_is_ignored():
    a = _stream('/var/folders/x/T/ipykernel_123/456.py:4: ProcessBehaviorWarning: blank values', 'stderr')
    b = _stream('/tmp/ipykernel_9/789.py:4: ProcessBehaviorWarning: blank values', 'stderr')
    assert _alike(_nb(a), _nb(b))


def test_random_derivation_id_is_ignored():
    a = _stream('temp_bin  family=bin  id=565862afc3e4\n')
    b = _stream('temp_bin  family=bin  id=5251aeb49f5f\n')
    assert _alike(_nb(a), _nb(b))


def test_chunked_printing_reads_as_one_block():
    assert _alike(_nb(_stream('Observed: 3'), _stream('\nValid: X\n')), _nb(_stream('Observed: 3\nValid: X\n')))


def test_plotly_styling_and_encoding_are_ignored():
    saved = _nb(_figure([1.0, 2.5, float('nan')], template='plotly'))
    fresh = _nb(_figure(_bdata([1.0, 2.5, np.nan]), template='plotly_white'))
    assert _alike(saved, fresh)


def test_float_noise_in_a_figure_is_ignored():
    """Sums run in a different order on another machine; the last digits are not drift."""
    assert _alike(_nb(_figure([0.006711006926])), _nb(_figure([0.006711006927])))


def test_a_changed_number_in_text_is_drift():
    a = new_output('execute_result', data={'text/plain': "{'center': 72.379, 'upl': 76.392}"}, execution_count=1)
    b = new_output('execute_result', data={'text/plain': "{'center': 72.379, 'upl': 76.391}"}, execution_count=1)
    assert not _alike(_nb(a), _nb(b))


def test_a_changed_figure_value_is_drift():
    assert not _alike(_nb(_figure([1.0, 2.5, 3.0])), _nb(_figure([1.0, 2.5, 3.1])))


def test_a_changed_figure_label_is_drift():
    assert not _alike(_nb(_figure([1.0], title='UPL = 4.930')), _nb(_figure([1.0], title='UPL = 4.929')))


def test_reordered_traces_are_drift():
    """Trace order sets the legend and each line's colour, so a reorder is something a reader sees."""

    def two(first, second):
        traces = [{'type': 'scatter', 'name': first, 'y': [1.0]}, {'type': 'scatter', 'name': second, 'y': [2.0]}]
        return _nb(new_output('display_data', data={PLOTLY: {'data': traces}}))

    assert not _alike(two('A_1', 'A_2'), two('A_2', 'A_1'))
