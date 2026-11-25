"""Ring 4 - workflows: runtime provisioning, golden/synthetic/matrix experiments, freeze, replay, export."""

# %% cell: runtime
import copy, hashlib, json, os, subprocess, sys, time, uuid
from pathlib import Path
import requests
from tago.config import (STATE_CONDITIONED, THINK, GENERATION_ATTEMPTS, VERSION, PROMPT_VERSION, BASE_SEED, SPLIT, OUTPUT_DIR, EMBEDDING_MODEL, MAX_CALLS_PER_ROLE,
    MAX_SECONDS_PER_ROLE, REQUEST_TIMEOUT_SEC, WARMUP_TIMEOUT_SEC, OLLAMA_URL, GUIDED_PROGRESS, CONSTRAINED_SCHEMA,
    MAX_REPAIRS, MAX_GRAPH_STEPS, RETRIEVAL_MODE, GROUNDED_QUOTES)
from tago.core.domain import canonical, digest, BudgetError
from tago.infrastructure.backends import make_backend
from tago.infrastructure.profiles import resolve_profile
from tago.application.packs import PACKS
from tago.application.evaluation import summarize, RunStore, duplicate_calls
from tago.application.orchestrator import AgentGraphOrchestrator
from tago.application.synthetic import SyntheticCaseAgent, DatasetAuditor

def prepare_runtime(plugin, url, models, optional_models=()):
    """Only this function installs Ollama, starts its daemon or pulls models.

    `models` are required (a failed pull aborts). `optional_models` are comparison-matrix candidates: a failed pull
    (wrong tag, no disk) is recorded in `pull_errors` and that model is skipped later, so one bad tag cannot waste the run.
    """
    import shutil, urllib.request
    if plugin in {'existing_ollama', 'api'}:
        return {'plugin': plugin, 'url': url}
    if plugin != 'colab_ollama': raise ValueError('Unknown runtime plugin')
    if not Path('/content').exists() or sys.platform != 'linux':
        raise RuntimeError('colab_ollama requires Colab Linux; choose existing_ollama elsewhere')
    if url.rstrip('/') != 'http://127.0.0.1:11434':
        raise ValueError('Colab provisioning targets localhost only; use existing_ollama for remote URLs')
    print('[runtime] Preparing Ollama for Colab...', flush=True)
    binary = shutil.which('ollama')
    if binary is None:
        if shutil.which('zstd') is None or shutil.which('curl') is None:
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'zstd', 'curl', 'ca-certificates'], check=True)
        installer = Path('/tmp/tago_ollama_install.sh')
        print('[runtime] Downloading the Ollama installer...', flush=True)
        urllib.request.urlretrieve('https://ollama.com/install.sh', installer)
        subprocess.run(['sh', str(installer)], check=True)
        binary = shutil.which('ollama')
    if binary is None: raise RuntimeError('Ollama installation failed')
    def available():
        try: return requests.get(url + '/api/tags', timeout=2).ok
        except requests.RequestException: return False
    if not available():
        print('[runtime] Starting the Ollama server...', flush=True)
        env = os.environ.copy(); env['OLLAMA_HOST'] = '127.0.0.1:11434'
        with open('/tmp/tago_ollama.log', 'a') as log:
            subprocess.Popen([binary, 'serve'], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        for _ in range(60):
            if available(): break
            time.sleep(1)
        else: raise RuntimeError('Ollama did not start; inspect /tmp/tago_ollama.log')
    env = os.environ.copy(); env['OLLAMA_HOST'] = url
    installed = {m['name'] for m in requests.get(url + '/api/tags', timeout=5).json()['models']}
    pull_errors = {}
    for model in sorted(set(models) | set(optional_models)):
        if model not in installed and model + ':latest' not in installed:
            print('[runtime] Downloading model:', model, flush=True)
            done = subprocess.run([binary, 'pull', model], env=env, check=False)
            if done.returncode != 0:
                if model in models: raise RuntimeError(f'Required model could not be pulled: {model}')
                pull_errors[model] = f'ollama pull exited with {done.returncode}'; print('[runtime] SKIP (pull failed):', model, flush=True)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip() if shutil.which('nvidia-smi') else 'none'
    print('[runtime] Ollama and models ready. GPU:', gpu, flush=True)
    return {'plugin': plugin, 'url': url, 'gpu': gpu, 'pull_errors': pull_errors,
            'ollama_version': subprocess.check_output([binary, '--version'], text=True).strip()}

def live_backend(spec, url=OLLAMA_URL, warm=True):
    """Construct, health-check and warm a backend under the configured budgets."""
    backend = make_backend(spec, base_url=url, max_calls=MAX_CALLS_PER_ROLE, max_seconds=MAX_SECONDS_PER_ROLE,
                           timeout=REQUEST_TIMEOUT_SEC)
    if backend.provider == 'ollama':
        backend.healthcheck()
        if warm:
            print(f'[runtime] Warming {spec} (first load can take minutes on a T4)...', flush=True)
            print('[runtime] Warm-up:', backend.warmup(WARMUP_TIMEOUT_SEC), flush=True)
    return backend

# %% cell: golden-harness
def execute_case(pack, case, orchestrator, retriever, seed):
    case = pack.validate_case(case)
    host = pack.tools(case['world'], retriever)
    start = len(orchestrator.backend.records)
    began = time.monotonic()
    state = orchestrator.run(pack, case['request'], host, seed)
    state['wall_sec'] = round(time.monotonic() - began, 3)
    state['duplicate_calls'] = duplicate_calls(state['observations'])
    state['call_records'] = list(range(start, len(orchestrator.backend.records)))
    return state, pack.evaluate(case, state)

def run_golden(packs, orchestrator, retriever, seed=BASE_SEED):
    rows = []
    for pack in packs:
        case = {'request': pack.golden_request, 'world': copy.deepcopy(pack.golden_world)}
        try:
            state, evaluation = execute_case(pack, case, orchestrator, retriever, seed)
            rows.append({'workflow_id': pack.workflow_id, 'case': case, 'state': state, 'evaluation': evaluation})
        except Exception as error:
            rows.append({'workflow_id': pack.workflow_id, 'case': case, 'evaluation': {'ok': False, 'failure_category': 'harness_error'},
                         'error': f'{type(error).__name__}: {error}'})
        ev = rows[-1]['evaluation']
        print('GOLDEN', pack.workflow_id, 'PASS' if ev['ok'] else 'FAIL ' + str(ev.get('failure_category')), flush=True)
    return rows

def save_golden(rows, backend, runtime_info, directory=OUTPUT_DIR, label=None, settings=None):
    store = RunStore(Path(directory) / ('golden-' + (label or '') + uuid.uuid4().hex[:12]))
    store.write('golden_results.json', rows)
    store.write('model_calls.json', backend.records)
    store.write('runtime.json', {**(runtime_info or {}), 'version': VERSION, 'prompt_version': PROMPT_VERSION,
                                 'model': backend.model, 'provider': getattr(backend, 'provider', None),
                                 'warmup': getattr(backend, 'warmup_record', None), 'base_seed': BASE_SEED,
                                 'settings': {'constrained_schema': CONSTRAINED_SCHEMA, 'guided_progress': GUIDED_PROGRESS,
                                              'state_conditioned': STATE_CONDITIONED, 'grounded_quotes': GROUNDED_QUOTES,
                                              'think': getattr(backend, 'think', THINK), 'generation_attempts': GENERATION_ATTEMPTS,
                                              'max_repairs': MAX_REPAIRS, 'max_graph_steps': MAX_GRAPH_STEPS,
                                              'retrieval': RETRIEVAL_MODE, 'timeout_sec': REQUEST_TIMEOUT_SEC, **(settings or {})}})
    return str(store.path.resolve())

# %% cell: synthetic-harness
def run_synthetic(packs, generator, auditor, orchestrator, retriever, n_clean=2, n_noisy=2, directory=OUTPUT_DIR):
    if any(type(n) is not int or not 0 <= n <= 100 for n in [n_clean, n_noisy]): raise ValueError('Counts must be integers 0..100')
    if SPLIT not in {'development', 'evaluation'}: raise ValueError('Unknown split')
    store = RunStore(Path(directory) / ('synthetic-' + uuid.uuid4().hex[:12]))
    roles = [generator.backend, auditor.backend, orchestrator.backend]
    backends = list({id(b): b for b in roles}.values())   # roles may share one backend object
    offsets = [len(b.records) for b in backends]
    manifest = {'version': VERSION, 'prompt_version': PROMPT_VERSION, 'base_seed': BASE_SEED, 'split': SPLIT,
        'models': {r: b.model for r, b in zip(['generator', 'auditor', 'orchestrator'], roles)},
        'retrieval': retriever.mode, 'embedding_model': EMBEDDING_MODEL if retriever.mode == 'hybrid' else None,
        'workflows': {p.workflow_id: {'objective': p.objective, 'schema': p.case_schema.model_json_schema(),
            'required_tools': p.required_tools, 'finding_keys': p.finding_keys, 'action': p.action, 'status': p.status} for p in packs},
        'n_clean': n_clean, 'n_noisy': n_noisy, 'note': 'Replay stored cases; provider seeds are best effort.'}
    store.write('manifest.json', manifest); rows = []; stop = False
    for mode, count in [('clean', n_clean), ('noisy', n_noisy)]:
        for batch in range(count):
            for pack in packs:
                seed = int(digest([BASE_SEED, SPLIT, mode, batch, pack.workflow_id])[:8], 16) % (2**31)
                row = {'case_id': uuid.uuid4().hex, 'workflow_id': pack.workflow_id, 'mode': mode, 'seed': seed,
                       'case': None, 'audit': {}, 'evaluation': {}, 'outcome': 'generation_error'}
                stage = 'generation'
                try:
                    row['case'] = generator.generate(pack, mode, seed)
                    stage = 'audit'; row['audit'] = auditor.audit(pack, row['case'], mode, seed)
                    if not row['audit']['accepted']: row['outcome'] = 'dataset_invalid'
                    else:
                        stage = 'orchestrator'
                        row['state'], row['evaluation'] = execute_case(pack, row['case'], orchestrator, retriever, seed)
                        row['outcome'] = 'pass' if row['evaluation']['ok'] else 'orchestrator_failure'
                        if 'BudgetError' in (row['state'].get('error') or ''): stop = True
                except BudgetError as error: row.update(outcome='budget_exhausted', error=str(error)); stop = True
                except Exception as error: row.update(outcome=stage + '_error', error=f'{type(error).__name__}: {error}')
                row['generation_attempts'] = len(generator.attempt_errors) + (1 if row['case'] else 0); row['generation_errors'] = list(generator.attempt_errors)
                rows.append(row); store.append(row)
                store.write('model_calls.json', [r for b, offset in zip(backends, offsets) for r in b.records[offset:]])
                print(mode, batch + 1, pack.workflow_id, row['outcome'], flush=True)
                if stop: break
            if stop: break
        if stop: break
    report = {'path': str(store.path.resolve()), 'requested': len(packs) * (n_clean + n_noisy),
              'attempted': len(rows), 'stopped_early': stop, 'groups': summarize(rows)}
    store.write('summary.json', report)
    return rows, report

# %% cell: model-matrix
def run_model_matrix(specs, packs, retriever, runtime_info=None, directory=OUTPUT_DIR):
    """Run the same golden cases on several models (local and/or API) under identical rules.

    Missing API keys or unregistered models are recorded as 'skipped', never silently dropped.
    """
    results = {}
    for spec in specs:
        try:
            backend = live_backend(spec)
        except Exception as error:
            results[spec] = {'skipped': f'{type(error).__name__}: {error}'}; print('MATRIX skip', spec, error); continue
        try:
            rows = run_golden(packs, AgentGraphOrchestrator(backend), retriever)
            path = save_golden(rows, backend, runtime_info, directory, label=spec.replace(':', '_').replace('/', '_') + '-')
            categories = {}
            for r in rows:
                if not r['evaluation']['ok']: categories[r['evaluation'].get('failure_category') or 'unclassified'] = categories.get(r['evaluation'].get('failure_category') or 'unclassified', 0) + 1
            results[spec] = {'path': path, 'passed': sum(r['evaluation']['ok'] for r in rows), 'total': len(rows),
                             'failure_categories': categories, 'thinking': backend.profile.thinking if hasattr(backend, 'profile') else None}
            print('MATRIX', spec, results[spec]['passed'], '/', results[spec]['total'], categories, flush=True)
        except Exception as error:   # one broken model (bad tokenizer id, OOM, timeout) must not stop the comparison
            results[spec] = {'error': f'{type(error).__name__}: {error}'}; print('MATRIX error', spec, error, flush=True)
        finally:
            if hasattr(backend, 'unload'):
                try: backend.unload()
                except Exception: pass
    return results

# %% cell: campaign
import shutil

# Each rung adds one switch. Rung 3 is the shipped configuration; rung 0 reproduces the v3.7 behaviour.
ABLATION_LADDER = [
    ('rung0-v37-behaviour', dict(constrained=False, guided=False, state_conditioned=False, grounded_quotes=False)),
    ('rung1-constrained-guided', dict(constrained=True, guided=True, state_conditioned=False, grounded_quotes=False)),
    ('rung2-state-conditioned', dict(constrained=True, guided=True, state_conditioned=True, grounded_quotes=False)),
    ('rung3-grounded-quotes', dict(constrained=True, guided=True, state_conditioned=True, grounded_quotes=True)),
]

def orchestrator_settings(orchestrator):
    """The switches an orchestrator really used (the constructor can switch dependent ones off)."""
    return {'constrained_schema': orchestrator.constrained, 'guided_progress': orchestrator.guided,
            'state_conditioned': orchestrator.state_conditioned, 'grounded_quotes': orchestrator.grounded_quotes}

def ollama_pull(spec, url=OLLAMA_URL):
    """Download a model if it is missing. Returns True when this call downloaded it, False when it was already there."""
    binary = shutil.which('ollama')
    if binary is None: raise RuntimeError('ollama is not installed')
    names = {m['name'] for m in requests.get(url + '/api/tags', timeout=5).json()['models']}
    if spec in names or spec + ':latest' in names: return False
    env = os.environ.copy(); env['OLLAMA_HOST'] = url
    if subprocess.run([binary, 'pull', spec], env=env, check=False).returncode != 0: raise RuntimeError(f'ollama pull failed for {spec}')
    return True

def ollama_remove(spec, url=OLLAMA_URL):
    """Delete a model this campaign downloaded, so many models fit on the Colab disk one after another."""
    binary = shutil.which('ollama')
    if binary:
        env = os.environ.copy(); env['OLLAMA_HOST'] = url
        subprocess.run([binary, 'rm', spec], env=env, check=False, capture_output=True)

def campaign_markdown(items):
    """One table for every campaign item, whatever its kind."""
    lines = ['| Item | Kind | Model | Think | Result | Failure categories | First call s | Minutes |', '|---|---|---|---|---|---|---|---|']
    for i in items:
        result = i.get('skipped') or i.get('error') or f"{i['passed']}/{i['total']}"
        if i.get('skipped'): result = 'skipped: ' + i['skipped']
        if i.get('error'): result = 'error: ' + i['error'][:80]
        lines.append(f"| {i['name']} | {i['kind']} | {i.get('model', '-')} | {i.get('think', '-')} | {result} | "
                     f"{json.dumps(i.get('failure_categories') or {})} | {i.get('first_call_sec', '-')} | {i.get('minutes', '-')} |")
    return '\n'.join(lines)

def run_campaign(packs, retriever, runtime_info, model, ablation=True, models=(), thinking_models=(), auditor=None,
                 n_clean=1, n_noisy=1, max_minutes=240, directory=OUTPUT_DIR, backend_factory=None, pull=None, remove=None):
    """Run every extra measurement in one session, reusing the loaded notebook state.

    Order: ablation ladder on `model` (already warm), different-family auditor study, other models, thinking models.
    One broken item (bad tag, out of memory, timeout) is recorded and the campaign continues. No item starts after
    `max_minutes`; it is recorded as skipped. Every golden run is saved in the same format as the main run, so
    `python -m tago analyze <folder>` reads it.
    """
    backend_factory = backend_factory or (lambda spec, warm=True: live_backend(spec, warm=warm))
    pull = pull or ollama_pull; remove = remove or ollama_remove
    root = Path(directory) / ('campaign-' + uuid.uuid4().hex[:12]); root.mkdir(parents=True)
    began = time.monotonic(); items = []

    def save():
        (root / 'campaign_summary.json').write_text(json.dumps({'version': VERSION, 'prompt_version': PROMPT_VERSION, 'base_seed': BASE_SEED,
            'main_model': model, 'items': items}, indent=2), encoding='utf-8')
        (root / 'campaign_report.md').write_text(campaign_markdown(items) + '\n', encoding='utf-8')
    def over(): return (time.monotonic() - began) / 60 >= max_minutes
    def golden(name, kind, spec, backend, orchestrator, think='-'):
        start = time.monotonic()
        rows = run_golden(packs, orchestrator, retriever)
        path = save_golden(rows, backend, runtime_info, root, label=name + '-', settings=orchestrator_settings(orchestrator))
        categories = {}
        for r in rows:
            if not r['evaluation']['ok']:
                key = r['evaluation'].get('failure_category') or 'unclassified'; categories[key] = categories.get(key, 0) + 1
        first = backend.records[0].get('latency_sec') if backend.records else None
        item = {'name': name, 'kind': kind, 'model': spec, 'think': think, 'path': path, 'passed': sum(r['evaluation']['ok'] for r in rows),
                'total': len(rows), 'failure_categories': categories, 'first_call_sec': None if first is None else round(first, 2),
                'minutes': round((time.monotonic() - start) / 60, 1), 'settings': orchestrator_settings(orchestrator)}
        print('CAMPAIGN', name, f"{item['passed']}/{item['total']}", categories, flush=True)
        return item
    def guarded(name, kind, spec, work):
        if over(): items.append({'name': name, 'kind': kind, 'model': spec, 'skipped': 'time budget'}); save(); return
        try: items.append(work())
        except Exception as error:
            items.append({'name': name, 'kind': kind, 'model': spec, 'error': f'{type(error).__name__}: {error}'}); print('CAMPAIGN error', name, error, flush=True)
        save()

    if ablation:
        for name, switches in ABLATION_LADDER:
            def rung(name=name, switches=switches):
                backend = backend_factory(model, warm=False)
                return golden(name, 'ablation', model, backend, AgentGraphOrchestrator(backend, **switches))
            guarded(name, 'ablation', model, rung)
    if auditor:
        def study():
            start = time.monotonic(); downloaded = pull(auditor)
            try:
                main, other = backend_factory(model, warm=False), backend_factory(auditor)
                rows, report = run_synthetic(packs, SyntheticCaseAgent(main), DatasetAuditor(other), AgentGraphOrchestrator(main), retriever, n_clean, n_noisy, root)
                RunStore(report['path']).write('runtime.json', runtime_info)
                passed = sum(r['outcome'] == 'pass' for r in rows)
                categories = {}
                for r in rows:
                    if r['outcome'] != 'pass': key = (r.get('evaluation') or {}).get('failure_category') or r['outcome']; categories[key] = categories.get(key, 0) + 1
                print('CAMPAIGN auditor', auditor, f'{passed}/{len(rows)}', categories, flush=True)
                return {'name': 'auditor-' + auditor.replace(':', '_'), 'kind': 'auditor', 'model': f'generator/orchestrator {model}, auditor {auditor}',
                        'path': report['path'], 'passed': passed, 'total': len(rows), 'failure_categories': categories,
                        'minutes': round((time.monotonic() - start) / 60, 1)}
            finally:
                if downloaded: remove(auditor)
        guarded('auditor-' + auditor.replace(':', '_'), 'auditor', auditor, study)
    first_model, pulled = True, set()
    for spec, kind in [(m, 'model') for m in models] + [(m, 'thinking') for m in thinking_models]:
        variants = ['default', 'off'] if kind == 'thinking' else ['default']
        for think in variants:
            name = spec.replace(':', '_') + ('' if kind == 'model' else '-think-' + think)
            def run(spec=spec, kind=kind, think=think, name=name):
                nonlocal first_model
                if first_model:
                    first_model = False
                    try: backend_factory(model, warm=False).unload()   # free the T4 before a 9-14B model
                    except Exception: pass
                if pull(spec): pulled.add(spec)
                backend = None
                try:
                    backend = backend_factory(spec)
                    backend.think = think
                    return golden(name, kind, spec, backend, AgentGraphOrchestrator(backend), think)
                finally:
                    if hasattr(backend, 'unload'):
                        try: backend.unload()
                        except Exception: pass
                    if spec in pulled and think == variants[-1]: remove(spec); pulled.discard(spec)
            guarded(name, kind, spec, run)
    save()
    return {'path': str(root.resolve()), 'items': items}

# %% cell: freeze-replay-export
def freeze_dataset(rows, directory):
    cases = [{key: r[key] for key in ['case_id', 'workflow_id', 'mode', 'seed', 'case']} for r in rows if r.get('audit', {}).get('accepted') and r.get('case')]
    if not cases: raise ValueError('No accepted cases')
    path = Path(directory); path.mkdir(parents=True, exist_ok=False)
    payload = ''.join(canonical(case) + '\n' for case in cases)
    (path / 'evaluation.jsonl').write_text(payload, encoding='utf-8')
    manifest = {'version': VERSION, 'prompt_version': PROMPT_VERSION, 'count': len(cases),
                'sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest(),
                'schemas': {wid: digest(PACKS[wid].case_schema.model_json_schema()) for wid in {r['workflow_id'] for r in cases}}}
    (path / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return str(path.resolve())

def replay_frozen(directory, orchestrator, retriever):
    path = Path(directory); manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    payload = (path / 'evaluation.jsonl').read_text(encoding='utf-8')
    if hashlib.sha256(payload.encode('utf-8')).hexdigest() != manifest['sha256']: raise ValueError('Dataset hash changed')
    for wid, signature in manifest['schemas'].items():
        if digest(PACKS[wid].case_schema.model_json_schema()) != signature: raise ValueError('Workflow schema changed')
    result = []
    for line in payload.splitlines():
        if not line.strip(): continue
        row = json.loads(line); pack = PACKS[row['workflow_id']]
        state, evaluation = execute_case(pack, row['case'], orchestrator, retriever, row['seed'])
        result.append({'case_id': row['case_id'], 'workflow_id': pack.workflow_id, 'state': state, 'evaluation': evaluation})
    return result

def export_bundle(directory):
    import shutil
    path = Path(directory)
    archive = shutil.make_archive(str(path), 'zip', root_dir=path)
    print('Artifact bundle:', archive)
    return archive
