"""Command line entry point (gateway ring).

    python -m tago test                       # offline contract suite (no model, no network)
    python -m tago analyze RUN_DIR [RUN_DIR]  # markdown comparison of saved run folders
    python -m tago golden MODEL [MODEL ...]   # live golden cases (needs Ollama or API keys)
    python -m tago results check RUN_DIR [--version 3.8] [--workflows E01,E03,E06] [--synthetic]
    python -m tago tasks validate|status|verify|gate [node.tasks.json]
"""
import json, sys

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {'-h', '--help'}: print(__doc__); return 0
    command, rest = argv[0], argv[1:]
    if command == 'test':
        from tago.workflows.contracts import run_offline_contracts
        report = run_offline_contracts(verbosity=2); print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1
    if command == 'analyze':
        from tago.workflows.analysis import run_summary, markdown_report
        print(markdown_report([run_summary(d) for d in rest])); return 0
    if command == 'golden':
        from tago.config import SELECTED_WORKFLOWS, RETRIEVAL_MODE
        from tago.application.packs import PACKS
        from tago.infrastructure.tools import RetrievalAdapter
        from tago.workflows.experiments import run_model_matrix
        result = run_model_matrix(rest, [PACKS[w] for w in SELECTED_WORKFLOWS], RetrievalAdapter(RETRIEVAL_MODE))
        print(json.dumps(result, indent=2)); return 0
    if command == 'results' and rest[:1] == ['check']:
        return check_results(rest[1:])
    if command == 'tasks':
        from tago.gateway.project_tasks import main as tasks_main
        return tasks_main(rest)
    print(__doc__); return 2

def check_results(args):
    """Verify that a committed live-run folder is complete and from the expected version."""
    from pathlib import Path
    from tago.workflows.analysis import run_summary
    opts = {'--version': '3.8', '--workflows': 'E01,E03,E06'}
    paths, items = [], iter(args)
    for a in items:
        if a in opts: opts[a] = next(items, opts[a])
        elif not a.startswith('--'): paths.append(a)
    problems = []
    for path in paths:
        folder = Path(path)
        if not folder.is_dir(): problems.append(f'{path}: missing'); continue
        summary = run_summary(folder); runtime = summary['runtime']
        if str(runtime.get('version')) != opts['--version']: problems.append(f"{path}: runtime.version={runtime.get('version')}")
        cases = {c['workflow_id'] for c in summary['cases']}
        wanted = set(opts['--workflows'].split(','))
        if not wanted <= cases: problems.append(f'{path}: missing golden cases {sorted(wanted - cases)}')
        if not summary['orchestrator_calls']: problems.append(f'{path}: no model call records')
        if any(c['failure_category'] == 'harness_error' for c in summary['cases']): problems.append(f'{path}: harness errors')
        if '--synthetic' in args and not (folder / 'synthetic' / 'summary.json').exists():
            problems.append(f'{path}: no synthetic/summary.json (unzip the synthetic bundle into {path}/synthetic/)')
        print(f"{path}: {summary['passed']}/{summary['total']} golden passed, {summary['orchestrator_calls']} calls, model {summary['models']}")
    for p in problems: print('PROBLEM', p)
    return 1 if problems or not paths else 0

if __name__ == '__main__':
    sys.exit(main())
