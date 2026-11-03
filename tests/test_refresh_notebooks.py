"""
scripts/refresh_notebooks.py: what counts as a docs notebook drifting from the library.

CI runs the script with --check, so seen() must ignore what differs on every run (warning paths,
random derivation ids, how the kernel chunks printed text, plotly styling and number encoding) and
must still catch a changed number in text or in a figure.
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
    fig = {'data': [{'type': 'scatter', 'name': 'X', 'x': [1, 2, 3], 'y': y}],
           'layout': {'template': {'layout': {'font': template}}, 'title': {'text': title}}}
    return new_output('display_data', data={PLOTLY: fig, 'text/plain': 'Figure()'})


def _bdata(values):
    return {'dtype': 'f8', 'bdata': base64.b64encode(np.array(values, dtype='f8').tobytes()).decode()}


def test_docs_notebooks_are_found():
    names = {p.name for p in refresh.NOTEBOOKS}
    assert {'quickstart.ipynb', 'signal-detection.ipynb', 'sds-validation.ipynb'} <= names


def test_warning_location_is_ignored():
    a = _stream('/var/folders/x/T/ipykernel_123/456.py:4: ProcessBehaviorWarning: blank values', 'stderr')
    b = _stream('/tmp/ipykernel_9/789.py:4: ProcessBehaviorWarning: blank values', 'stderr')
    assert refresh.seen(_nb(a)) == refresh.seen(_nb(b))


def test_random_derivation_id_is_ignored():
    assert refresh.seen(_nb(_stream('temp_bin  family=bin  id=565862afc3e4\n'))) == refresh.seen(
        _nb(_stream('temp_bin  family=bin  id=5251aeb49f5f\n'))
    )


def test_chunked_printing_reads_as_one_block():
    assert refresh.seen(_nb(_stream('Observed: 3'), _stream('\nValid: X\n'))) == refresh.seen(
        _nb(_stream('Observed: 3\nValid: X\n'))
    )


def test_plotly_styling_and_encoding_are_ignored():
    saved = _nb(_figure([1.0, 2.5, float('nan')], template='plotly'))
    fresh = _nb(_figure(_bdata([1.0, 2.5, np.nan]), template='plotly_white'))
    assert refresh.seen(saved) == refresh.seen(fresh)


def test_a_changed_number_in_text_is_drift():
    a = new_output('execute_result', data={'text/plain': "{'center': 72.379, 'upl': 76.392}"}, execution_count=1)
    b = new_output('execute_result', data={'text/plain': "{'center': 72.379, 'upl': 76.391}"}, execution_count=1)
    assert refresh.seen(_nb(a)) != refresh.seen(_nb(b))


def test_a_changed_figure_value_is_drift():
    assert refresh.seen(_nb(_figure([1.0, 2.5, 3.0]))) != refresh.seen(_nb(_figure([1.0, 2.5, 3.1])))


def test_a_changed_figure_label_is_drift():
    saved, fresh = _nb(_figure([1.0], title='UPL = 4.930')), _nb(_figure([1.0], title='UPL = 4.929'))
    assert refresh.seen(saved) != refresh.seen(fresh)
