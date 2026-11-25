"""Ring 4 - workflows: read saved run folders (v37 or v38 format) and compare them.

Used to produce the "real results" tables in the README from committed artifacts,
so numbers in the documentation are regenerated from data, never typed by hand.
"""

# %% cell: analysis
import json, statistics
from pathlib import Path
from tago.application.evaluation import classify_failure, duplicate_calls

def load_run(directory):
    path = Path(directory)
    read = lambda name: json.loads((path / name).read_text(encoding='utf-8')) if (path / name).exists() else None
    return {'path': str(path), 'golden': read('golden_results.json') or [], 'calls': read('model_calls.json') or [],
            'runtime': read('runtime.json') or {}}

def _latency(calls):
    values = [c['latency_sec'] for c in calls if c.get('latency_sec') is not None]
    if not values: return {}
    warm = values[1:] or values
    return {'first_call_sec': round(values[0], 2), 'median_warm_sec': round(statistics.median(warm), 2),
            'total_sec': round(sum(values), 1)}

def run_summary(directory):
    run = load_run(directory)
    calls = [c for c in run['calls'] if c.get('role', 'orchestrator') == 'orchestrator']
    models = sorted({c.get('model') for c in calls if c.get('model')}) or [run['runtime'].get('model')]
    rows = []
    for row in run['golden']:
        state = row.get('state') or {}
        evaluation = row.get('evaluation') or {}
        observations = state.get('observations', [])
        category = None if evaluation.get('ok') else (evaluation.get('failure_category') or (classify_failure(state, evaluation.get('checks')) if state else 'harness_error'))
        rows.append({'workflow_id': row['workflow_id'], 'ok': bool(evaluation.get('ok')), 'status': state.get('status'),
                     'failure_category': category,
                     'model_calls': state.get('model_calls', sum(1 for t in state.get('trace', []) if t.get('to') == 'validate')),
                     'tool_calls_ok': len(observations), 'duplicate_calls': duplicate_calls(observations),
                     'repairs': sum(1 for t in state.get('trace', []) if t.get('to') == 'repair'),
                     'tools_used': [o['tool'] for o in observations], 'last_error': (state.get('error') or '')[:160]})
    prompt = [c.get('prompt_tokens') for c in calls if c.get('prompt_tokens')]
    completion = [c.get('completion_tokens') for c in calls if c.get('completion_tokens')]
    return {'path': run['path'], 'models': models, 'runtime': run['runtime'], 'cases': rows,
            'passed': sum(r['ok'] for r in rows), 'total': len(rows), 'orchestrator_calls': len(calls),
            'latency': _latency(calls), 'prompt_tokens_range': [min(prompt), max(prompt)] if prompt else None,
            'completion_tokens_mean': round(statistics.mean(completion), 1) if completion else None,
            'raw_tool_requests': [json.loads(c['raw']).get('tool') for c in calls if c.get('raw') and '"tool"' in c['raw']]}

def markdown_report(summaries):
    lines = ['| Run | Model | Passed | Calls | First call s | Median warm s |', '|---|---|---|---|---|---|']
    for s in summaries:
        lat = s['latency']
        lines.append(f"| `{Path(s['path']).name}` | {', '.join(map(str, s['models']))} | {s['passed']}/{s['total']} | "
                     f"{s['orchestrator_calls']} | {lat.get('first_call_sec', '-')} | {lat.get('median_warm_sec', '-')} |")
    lines += ['', '| Run | Case | Result | Failure category | Model calls | Tool calls OK | Duplicate calls | Repairs | Tools used |',
              '|---|---|---|---|---|---|---|---|---|']
    for s in summaries:
        for c in s['cases']:
            lines.append(f"| `{Path(s['path']).name}` | {c['workflow_id']} | {'PASS' if c['ok'] else c['status']} | "
                         f"{c['failure_category'] or '-'} | {c['model_calls']} | {c['tool_calls_ok']} | {c['duplicate_calls']} | "
                         f"{c['repairs']} | {', '.join(c['tools_used']) or '-'} |")
    return '\n'.join(lines)


def synthetic_summary(directory):
    """Per (workflow, mode) outcomes of a synthetic sweep, read from <run>/synthetic/summary.json."""
    path = Path(directory) / 'synthetic' / 'summary.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

def markdown_synthetic(directory):
    summary = synthetic_summary(directory)
    if not summary: return '_No synthetic sweep in this run._'
    lines = ['| Workflow | Mode | Attempted | Reached the orchestrator | Passed | Outcomes | Failure categories |', '|---|---|---|---|---|---|---|']
    for g in summary['groups']:
        lines.append(f"| {g['workflow_id']} | {g['mode']} | {g['attempted']} | {g['eligible']} | {g['passed']} | "
                     f"{json.dumps(g['outcomes'])} | {json.dumps(g['failure_categories']) if g['failure_categories'] else '-'} |")
    return '\n'.join(lines)
