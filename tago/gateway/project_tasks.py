"""Gateway/CLI support: validate and verify the node.tasks.json project graphs.

The task graphs are the project's definition of done. This module makes them testable:

* ``validate`` - structure, enums, unique ids, existing dependencies, acyclic graph, status
  consistency (a Done task cannot depend on an unfinished one), a verification command on
  every task, and two-way links between a parent task and its sub-graph file.
* ``verify``   - re-runs the verification commands of every Done task that can run locally.
* ``gate``     - the project end condition: every task in every graph is Done/Cancelled and
  every graph_end check passes.

Commands run from the repository root (the folder holding the root node.tasks.json).
"""
import json, os, subprocess, sys
from pathlib import Path

STATUS = {'Not Started', 'In Progress', 'Blocked', 'In Review', 'Done', 'Cancelled'}
PRIORITY = {'Critical', 'High', 'Medium', 'Low'}
COMPLEXITY = {'', 'Very Low', 'Low', 'Medium', 'High', 'Very High'}
DEP_TYPES = {'FS', 'SS', 'FF', 'SF'}
LOCAL = {None, 'local'}

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def graph_files(root):
    """Root graph plus every sub-graph reachable through task links of type 'subgraph'."""
    root = Path(root); seen = []; queue = [root]
    while queue:
        path = queue.pop(0)
        if path in seen or not path.exists(): continue
        seen.append(path)
        for task in load(path).get('tasks', []):
            for link in task.get('links', []):
                if link.get('type') == 'subgraph': queue.append((path.parent / link['path']).resolve())
    return seen

def _cycle(tasks):
    deps = {t['task_id']: [d['predecessor_task_id'] for d in t.get('dependencies', [])] for t in tasks}
    state = {}
    def visit(node):
        if state.get(node) == 1: return True
        if state.get(node) == 2: return False
        state[node] = 1
        if any(visit(d) for d in deps.get(node, []) if d in deps): return True
        state[node] = 2; return False
    return any(visit(n) for n in deps)

def validate_graph(path, repo_root):
    data = load(path); errors = []; path = Path(path).resolve(); where = path.relative_to(repo_root)
    for key in ['name', 'start_date', 'end_date', 'status', 'timezone', 'currency', 'project_manager']:
        if key not in data.get('project', {}): errors.append(f'{where}: project.{key} missing')
    tasks = data.get('tasks', [])
    ids = [t.get('task_id') for t in tasks]
    if len(ids) != len(set(ids)): errors.append(f'{where}: duplicate task_id')
    by_id = {t['task_id']: t for t in tasks}
    for t in tasks:
        tag = f"{where} task {t.get('task_id')} '{t.get('task_name')}'"
        for key in ['task_id', 'task_name', 'start_date', 'end_date', 'due_date', 'priority', 'status', 'progress_percentage']:
            if key not in t: errors.append(f'{tag}: {key} missing')
        if t.get('status') not in STATUS: errors.append(f'{tag}: bad status {t.get("status")}')
        if t.get('priority') not in PRIORITY: errors.append(f'{tag}: bad priority')
        if t.get('complexity', '') not in COMPLEXITY: errors.append(f'{tag}: bad complexity')
        if t.get('status') == 'Done' and t.get('progress_percentage') != 100: errors.append(f'{tag}: Done requires progress 100')
        if t.get('status') == 'Not Started' and t.get('progress_percentage') != 0: errors.append(f'{tag}: Not Started requires progress 0')
        if t.get('status') == 'Blocked' and not t.get('blocker_reason'): errors.append(f'{tag}: Blocked requires blocker_reason')
        verification = t.get('verification') or {}
        if not verification.get('commands') or not verification.get('success_signal'):
            errors.append(f'{tag}: verification.commands and success_signal are required (verifiable goal)')
        for dep in t.get('dependencies', []):
            if dep.get('type') not in DEP_TYPES: errors.append(f'{tag}: bad dependency type')
            pred = by_id.get(dep.get('predecessor_task_id'))
            if pred is None: errors.append(f'{tag}: unknown predecessor {dep.get("predecessor_task_id")}')
            elif t.get('status') == 'Done' and dep.get('type') == 'FS' and pred.get('status') not in {'Done', 'Cancelled'}:
                errors.append(f'{tag}: Done but FS predecessor {pred["task_id"]} is {pred["status"]}')
        for link in t.get('links', []):
            if link.get('type') != 'subgraph': continue
            child_path = (path.parent / link['path']).resolve()
            if not child_path.exists(): errors.append(f'{tag}: subgraph {link["path"]} not found'); continue
            parent = load(child_path).get('project', {}).get('parent_graph', {})
            if (child_path.parent / parent.get('path', '')).resolve() != path:
                errors.append(f'{tag}: subgraph {link["path"]} does not point back to {where}')
            if parent.get('task_id') != t['task_id']: errors.append(f'{tag}: subgraph parent_graph.task_id mismatch')
            child_status = {c['status'] for c in load(child_path).get('tasks', [])}
            if t.get('status') == 'Done' and not child_status <= {'Done', 'Cancelled'}:
                errors.append(f'{tag}: Done but its subgraph still has open tasks')
    if _cycle(tasks): errors.append(f'{where}: dependency cycle')
    nav = data.get('navigation', {})
    for module in nav.get('modules', []):
        if module.get('path') == nav.get('rootModule'): known = set(by_id)
        else:
            target = (repo_root / module['path'])
            if not target.exists(): errors.append(f"{where}: navigation module {module['path']} not found"); continue
            known = {t['task_id'] for t in load(target).get('tasks', [])}
        missing = set(module.get('taskIds', [])) - known
        if missing: errors.append(f'{where}: navigation lists unknown task ids {sorted(missing)}')
    for unit in data.get('unit_registry', []):
        if unit.get('status') in {'stable', 'active_development'} and not (repo_root / unit['path']).exists():
            errors.append(f"{where}: unit {unit['name']} path {unit['path']} missing")
    end = data.get('project', {}).get('graph_end', {})
    if not end.get('verification', {}).get('commands'): errors.append(f'{where}: project.graph_end.verification.commands missing')
    return errors

def run_commands(commands, repo_root):
    results = []
    for command in commands:
        env = dict(os.environ, TAGO_TASKS_VERIFYING='1')   # nested verify/gate calls become no-ops (no recursion)
        proc = subprocess.run(command, shell=True, cwd=repo_root, capture_output=True, text=True, env=env)
        results.append({'command': command, 'returncode': proc.returncode, 'tail': (proc.stdout + proc.stderr)[-400:]})
    return results

def status_report(root):
    root = Path(root).resolve(); rows = []
    for path in graph_files(root):
        for t in load(path).get('tasks', []):
            rows.append({'graph': str(path.relative_to(root.parent)), 'id': t['task_id'], 'name': t['task_name'], 'status': t['status'],
                         'progress': t.get('progress_percentage'), 'requires': (t.get('verification') or {}).get('requires', 'local'),
                         'blocker': t.get('blocker_reason', '')})
    return rows

def main(argv):
    if not argv: print(__doc__); return 2
    command = argv[0]; root = Path(argv[1] if len(argv) > 1 and not argv[1].startswith('-') else 'node.tasks.json').resolve()
    repo_root = root.parent
    if command == 'validate':
        errors = [e for path in graph_files(root) for e in validate_graph(path, repo_root)]
        for e in errors: print('ERROR', e)
        print(f'{len(graph_files(root))} graph(s), {len(errors)} error(s)'); return 1 if errors else 0
    if command == 'status':
        for r in status_report(root):
            print(f"{r['graph']:<28} {r['id']:>3}  {r['status']:<12} {str(r['progress']):>4}%  [{r['requires']}]  {r['name']}"
                  + (f"  -- blocked: {r['blocker']}" if r['blocker'] else ''))
        return 0
    if command in {'verify', 'gate'}:
        run = '--no-run' not in argv and not os.environ.get('TAGO_TASKS_VERIFYING')
        failures = 0; open_tasks = 0
        for path in graph_files(root):
            data = load(path)
            for t in data['tasks']:
                v = t.get('verification') or {}
                if t['status'] not in {'Done', 'Cancelled'}: open_tasks += 1
                if not run or t['status'] != 'Done' or v.get('requires') not in LOCAL: continue
                for r in run_commands(v['commands'], repo_root):
                    ok = r['returncode'] == 0; failures += not ok
                    print(('PASS' if ok else 'FAIL'), f"{path.relative_to(repo_root)}#{t['task_id']}", r['command'])
                    if not ok: print('   ', r['tail'].replace('\n', '\n    '))
            if command == 'gate' and run:
                for r in run_commands(data['project']['graph_end']['verification']['commands'], repo_root):
                    ok = r['returncode'] == 0; failures += not ok
                    print(('PASS' if ok else 'FAIL'), f'{path.relative_to(repo_root)}#graph_end', r['command'])
        if command == 'gate':
            print(f'open tasks: {open_tasks}, failing checks: {failures}')
            return 0 if failures == 0 and open_tasks == 0 else 1
        print(f'failing checks: {failures}'); return 1 if failures else 0
    print('Unknown command'); return 2

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
