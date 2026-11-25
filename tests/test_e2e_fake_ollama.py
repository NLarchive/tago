"""End-to-end plumbing test over real HTTP against a fake Ollama server.

The fake server plays a *scripted policy* (not a model): it calls the next uncalled tool and
then finishes with fixture answers. This exercises the real OllamaBackend transport, warm-up,
golden + synthetic harnesses, run storage, analysis, freeze and replay - everything the
Colab run uses except the actual GPU model.
"""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from tago.application.packs import PACKS, STATED_FIELDS
from tago.application.orchestrator import AgentGraphOrchestrator
from tago.application.synthetic import SyntheticCaseAgent, DatasetAuditor
from tago.infrastructure.backends import OllamaBackend
from tago.infrastructure.profiles import ModelProfile, TokenProfiler
from tago.infrastructure.tools import RetrievalAdapter
from tago.workflows.contracts import FakeTokenizer, tool_args, offline_final
from tago.workflows.experiments import run_golden, save_golden, run_synthetic, freeze_dataset, replay_frozen
from tago.workflows.analysis import run_summary

def respond(body):
    schema = body['format']; title = schema.get('title', '')
    payload = json.loads(body['messages'][1]['content'])
    if title == 'AuditVerdict':
        out = {'valid': True, 'request_matches_world': True, 'style_matches': True, 'reason': 'fake audit'}
    elif title.endswith('Case'):
        pack = PACKS[title[:-4]]; out = {'request': pack.golden_request + ' Figures: ' + ' '.join(str(pack.golden_world[f]) for f in STATED_FIELDS.get(pack.workflow_id, ())), 'world': pack.golden_world}
    else:
        pack = PACKS[title.split('_')[0]]
        remaining = payload['progress']['required_tools_not_yet_called']
        out = ({'kind': 'tool', 'tool': {'name': remaining[0], 'arguments': tool_args(pack, remaining[0])}}
               if remaining else offline_final(pack.workflow_id)(payload))
    return {'message': {'content': json.dumps(out)}, 'prompt_eval_count': 30, 'eval_count': 12}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def _send(self, data):
        raw = json.dumps(data).encode(); self.send_response(200)
        self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(raw)))
        self.end_headers(); self.wfile.write(raw)
    def do_GET(self): self._send({'models': [{'name': 'fake:1b'}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
        self._send({'template': 'fake'} if self.path == '/api/show' else respond(body) if self.path == '/api/chat' else {})

@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()

def test_full_pipeline_over_http(server, tmp_path):
    profile = ModelProfile('fake:1b', 'fake', context_tokens=8192)
    backend = OllamaBackend('fake:1b', server, profile=profile, profiler=TokenProfiler(profile, FakeTokenizer()))
    assert backend.healthcheck() and backend.warmup(timeout=10)['latency_sec'] >= 0
    retriever = RetrievalAdapter('lexical'); orchestrator = AgentGraphOrchestrator(backend)
    rows = run_golden(list(PACKS.values()), orchestrator, retriever)
    assert [r['evaluation']['ok'] for r in rows] == [True] * 10
    path = save_golden(rows, backend, {'plugin': 'fake'}, tmp_path)
    summary = run_summary(path)
    assert (summary['passed'], summary['total']) == (10, 10) and all(c['duplicate_calls'] == 0 for c in summary['cases'])
    packs = [PACKS[w] for w in ['E01', 'E03', 'E06']]
    synthetic, report = run_synthetic(packs, SyntheticCaseAgent(backend), DatasetAuditor(backend), orchestrator, retriever, 1, 1, tmp_path)
    assert [r['outcome'] for r in synthetic] == ['pass'] * 6 and report['attempted'] == 6
    frozen = freeze_dataset(synthetic, tmp_path / 'frozen')
    assert all(r['evaluation']['ok'] for r in replay_frozen(frozen, orchestrator, retriever))
