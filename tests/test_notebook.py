import re
"""The notebook must be in sync with the package and run top-to-bottom offline WITHOUT the package."""
import os, shutil, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks' / 'tago_v38_t4-agent_graph_orchestration.ipynb'

def test_notebook_in_sync():
    assert subprocess.run([sys.executable, 'tools/build_notebook.py', '--check'], cwd=ROOT).returncode == 0

def test_notebook_executes_offline_without_package(tmp_path, monkeypatch):
    nbformat = pytest.importorskip('nbformat'); nbclient = pytest.importorskip('nbclient')
    shutil.copytree(ROOT / 'results', tmp_path / 'results')
    nb = nbformat.read(NOTEBOOK, 4)
    monkeypatch.setenv('TAGO_RUN_LIVE', '0')
    nbclient.NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(tmp_path)}}).execute()
    text = '\n'.join(o.get('text', '') for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []))
    assert '"passed": true' in text and re.search(r'Ran \d+ tests', text)
    assert not [o for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.get('output_type') == 'error']

def test_every_code_cell_is_explained():
    """Each code cell must follow a markdown cell that says what it does (the user-facing contract)."""
    nbformat = pytest.importorskip('nbformat')
    nb = nbformat.read(NOTEBOOK, 4); last_md = ''; unexplained = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown': last_md = cell.source; continue
        if 'What this' not in last_md and 'What the next cell does' not in last_md and 'What these' not in last_md:
            unexplained.append((index, cell.source.splitlines()[0][:60]))
    assert not unexplained, unexplained
