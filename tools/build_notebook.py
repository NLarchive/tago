"""Build the self-contained Colab notebook from the tago/ package and notebooks/narrative.md.

The package is the single source of truth (tested with pytest). The notebook inlines the
package code cell by cell, removing `from tago... import` lines, so it still runs with
"upload -> Run all" and needs no pip install of this repository.

    python tools/build_notebook.py          # write notebooks/tago_v38_t4-agent_graph_orchestration.ipynb
    python tools/build_notebook.py --check  # exit 1 if the committed notebook is stale
"""
import hashlib, re, sys
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'tago'
NARRATIVE = ROOT / 'notebooks' / 'narrative.md'
OUTPUT = ROOT / 'notebooks' / 'tago_v38_t4-agent_graph_orchestration.ipynb'
EXCLUDED = {'__init__.py', '__main__.py', 'project_tasks.py'}
MARKER = re.compile(r'^# %% cell: (\S+)\s*$')
TAGO_IMPORT = re.compile(r'^\s*(from tago[\w.]* import|import tago)')

def strip_tago_imports(lines):
    out, skipping = [], False
    for line in lines:
        if skipping:
            skipping = ')' not in line; continue
        if TAGO_IMPORT.match(line):
            skipping = '(' in line and ')' not in line; continue
        out.append(line)
    return out

def chunks():
    found = {}
    for path in sorted(PACKAGE.rglob('*.py')):
        if path.name in EXCLUDED: continue
        rel = path.relative_to(PACKAGE).as_posix(); current = None
        for line in path.read_text(encoding='utf-8').splitlines():
            match = MARKER.match(line)
            if match:
                current = f'{rel}#{match.group(1)}'
                if current in found: raise SystemExit(f'Duplicate cell id {current}')
                found[current] = []; continue
            if current: found[current].append(line)
        if current is None: raise SystemExit(f'{rel} has no "# %% cell:" markers')
    return {k: '\n'.join(strip_tago_imports(v)).strip('\n') for k, v in found.items()}

def generated(name):
    sys.path.insert(0, str(ROOT))
    if name == 'v37-results':
        from tago.workflows.analysis import run_summary, markdown_report
        return markdown_report([run_summary(ROOT / 'results' / 'v37_colab_t4_qwen2.5-7b')])
    if name == 'v381-results':
        from tago.workflows.analysis import run_summary, markdown_report, markdown_synthetic
        path = ROOT / 'results' / 'v38_1_colab_t4_qwen2.5-7b'
        return markdown_report([run_summary(path)]) + '\n\n**Synthetic sweep (same run):**\n\n' + markdown_synthetic(path)
    if name == 'v382-results':
        from tago.workflows.analysis import run_summary, markdown_report, markdown_synthetic
        path = ROOT / 'results' / 'v38_2_colab_t4_qwen2.5-7b'
        return markdown_report([run_summary(path)]) + '\n\n**Synthetic sweep (same run):**\n\n' + markdown_synthetic(path)
    if name == 'v383-results':
        from tago.workflows.analysis import run_summary, markdown_report, markdown_synthetic
        path = ROOT / 'results' / 'v38_3_colab_t4_qwen2.5-7b'
        return markdown_report([run_summary(path)]) + '\n\n**Synthetic sweep (same run):**\n\n' + markdown_synthetic(path)
    if name == 'v384-results':
        from tago.workflows.analysis import run_summary, markdown_report, markdown_synthetic
        path = ROOT / 'results' / 'v38_4_colab_t4_qwen2.5-7b'
        return markdown_report([run_summary(path)]) + '\n\n**Synthetic sweep (same run):**\n\n' + markdown_synthetic(path)
    if name == 'v384-campaign':
        return (ROOT / 'results' / 'v38_4_campaign_colab_t4' / 'campaign_report.md').read_text(encoding='utf-8').rstrip()
    raise SystemExit('Unknown generated block ' + name)

def cell_id(kind, text):
    return hashlib.sha1((kind + text).encode('utf-8')).hexdigest()[:12]

def build():
    code = chunks(); used = set(); cells = []; buffer = []
    def flush():
        text = '\n'.join(buffer).strip('\n'); buffer.clear()
        if text: cells.append(nbformat.v4.new_markdown_cell(text, id=cell_id('md', text + str(len(cells)))))
    lines = NARRATIVE.read_text(encoding='utf-8').splitlines(); i = 0
    while i < len(lines):
        line = lines[i]
        include = re.match(r'^<!-- code: (\S+) -->$', line.strip())
        gen = re.match(r'^<!-- generated: (\S+) -->$', line.strip())
        if include:
            key = include.group(1)
            if key not in code: raise SystemExit('Unknown code cell ' + key)
            if key in used: raise SystemExit('Code cell used twice ' + key)
            flush(); used.add(key); cells.append(nbformat.v4.new_code_cell(code[key], id=cell_id('code', key)))
        elif gen:
            buffer.append(generated(gen.group(1)))
        elif line.startswith('```python'):
            flush(); body = []; i += 1
            while not lines[i].startswith('```'): body.append(lines[i]); i += 1
            text = '\n'.join(body); cells.append(nbformat.v4.new_code_cell(text, id=cell_id('inline', text)))
        elif line.startswith('#') and buffer and any(b.strip() for b in buffer):
            flush(); buffer.append(line)
        else:
            buffer.append(line)
        i += 1
    flush()
    unused = sorted(set(code) - used)
    if unused: raise SystemExit('Package cells missing from narrative: ' + ', '.join(unused))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                   'colab': {'name': OUTPUT.name, 'provenance': [], 'gpuType': 'T4'}}
    nbformat.validate(nb)
    return nbformat.writes(nb) + '\n'

if __name__ == '__main__':
    text = build()
    if '--check' in sys.argv:
        current = OUTPUT.read_text(encoding='utf-8') if OUTPUT.exists() else ''
        if current != text: print('Notebook is stale: run python tools/build_notebook.py'); sys.exit(1)
        print('Notebook is in sync with the package and narrative.'); sys.exit(0)
    OUTPUT.write_text(text, encoding='utf-8'); print('Wrote', OUTPUT.relative_to(ROOT))
