"""The node.tasks.json graphs are valid, and the validator really catches broken graphs."""
import copy, json
from pathlib import Path
from tago.gateway.project_tasks import validate_graph, graph_files, main

ROOT = Path(__file__).resolve().parents[1]

def test_repository_graphs_are_valid():
    files = graph_files(ROOT / 'node.tasks.json')
    assert len(files) == 4
    assert [e for f in files for e in validate_graph(f, ROOT)] == []

def reopen_a_predecessor(d):
    """Make a Done task depend on an unfinished one, whatever the graph's current state."""
    task = next(t for t in d['tasks'] if t['status'] == 'Done' and t['dependencies'])
    pred = next(p for p in d['tasks'] if p['task_id'] == task['dependencies'][0]['predecessor_task_id'])
    pred.update(status='Not Started', progress_percentage=0)

def make_a_cycle(d):
    """Make an existing predecessor depend on its own successor."""
    task = next(t for t in d['tasks'] if t['dependencies'])
    pred = next(p for p in d['tasks'] if p['task_id'] == task['dependencies'][0]['predecessor_task_id'])
    pred['dependencies'].append({'predecessor_task_id': task['task_id'], 'type': 'FS', 'lag_days': 0})

def test_validator_rejects_inconsistent_graphs(tmp_path):
    base = json.loads((ROOT / 'results' / 'node.tasks.json').read_text(encoding='utf-8'))
    base['project'].pop('parent_graph'); base['navigation']['rootModule'] = base['navigation']['modules'][0]['path'] = 'g.json'
    broken = {
        'done_before_predecessor': lambda d: reopen_a_predecessor(d),
        'missing_verification': lambda d: d['tasks'][0].update(verification={}),
        'cycle': lambda d: make_a_cycle(d),
        'blocked_without_reason': lambda d: d['tasks'][0].update(status='Blocked', blocker_reason=''),
        'unknown_predecessor': lambda d: d['tasks'][0]['dependencies'].append({'predecessor_task_id': 99, 'type': 'FS', 'lag_days': 0}),
    }
    for name, mutate in broken.items():
        data = copy.deepcopy(base); mutate(data)
        path = tmp_path / 'g.json'; path.write_text(json.dumps(data))
        assert validate_graph(path, tmp_path), name

def test_gate_is_closed_until_live_results_exist():
    assert main(['gate', str(ROOT / 'node.tasks.json'), '--no-run']) == 1
