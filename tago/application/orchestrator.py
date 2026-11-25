"""Ring 2 - application: the bounded agent/tool observation graph.

start -> infer -> validate -> tool -> infer ... -> finish | repair -> quarantine | failed
"""

# %% cell: graph-guard
import copy
import requests
from tago.core.domain import BudgetError, ContextError, AgentStep, canonical, tool_request_of
from tago.application.packs import build_step_schema, build_state_schema
from tago.application.evaluation import EVIDENCE_REQUIREMENTS, classify_failure
from tago.config import MAX_GRAPH_STEPS, MAX_REPAIRS, GUIDED_PROGRESS, CONSTRAINED_SCHEMA, STATE_CONDITIONED, GROUNDED_QUOTES

EDGES = {'start': {'infer'}, 'infer': {'validate', 'repair', 'failed'},
         'validate': {'tool', 'finish', 'repair'}, 'tool': {'infer', 'repair', 'failed'},
         'repair': {'infer', 'quarantine'}, 'finish': set(), 'failed': set(), 'quarantine': set()}

def allowed_tools(pack, host):
    """Legal tool names in the current graph state; empty means the only legal step is `final`."""
    called = {o['tool'] for o in host.observations}
    remaining = tuple(t for t in pack.required_tools if t not in called)
    if remaining: return remaining
    if pack.workflow_id == 'E06':   # the candidate must be repaired until code_check passes (see validate_final)
        last = next((o for o in reversed(host.observations) if o['tool'] == 'code_check'), None)
        if not last or not last['data']['tests_passed']: return ('code_check',)
    return ()

def validate_final(step, host, pack):
    """Evidence guard: a final answer must rest on observed tool results with literal citations."""
    if step.kind != 'final': raise ValueError('Expected final')
    observed = {o['id']: o for o in host.observations}
    squash = lambda text: ' '.join(str(text).split())   # whitespace-insensitive, otherwise literal (case and punctuation count)
    missing = set(pack.required_tools) - {o['tool'] for o in host.observations}
    if missing: raise ValueError(f'Required tool evidence missing: call {sorted(missing)} before finishing')
    cited = set()
    for c in step.citations:
        if c.source_id not in observed:
            raise ValueError(f'Citation source {c.source_id} does not exist; observed ids: {sorted(observed)}')
        if squash(c.quote) not in squash(observed[c.source_id]['content']):
            raise ValueError(f'Citation is not an exact observed-source substring: {c.source_id}. Copy a fragment verbatim from: '
                             + observed[c.source_id]['content'][:500])
        cited.add(c.source_id)
    keys = [f.key for f in step.findings]
    if len(keys) != len(set(keys)) or set(keys) != set(pack.finding_keys):
        raise ValueError(f'Use each required finding key exactly once: {list(pack.finding_keys)}')
    for f in step.findings:
        if not set(f.source_ids) <= cited: raise ValueError(f'Every finding needs cited observed evidence: {f.key} names {sorted(set(f.source_ids) - cited)} which has no citation; cite every observation you name in a finding')
    if pack.workflow_id == 'E06':
        last = next((o for o in reversed(host.observations) if o['tool'] == 'code_check'), None)
        if not last or not last['data']['tests_passed']: raise ValueError('Repair the candidate until code_check succeeds')

# %% cell: graph-runner
class AgentGraphOrchestrator:
    """Generic runner: receives a public pack and a tool host; never reads the world or oracle."""
    def __init__(self, backend, max_steps=MAX_GRAPH_STEPS, max_repairs=MAX_REPAIRS, guided=GUIDED_PROGRESS,
                 constrained=CONSTRAINED_SCHEMA, state_conditioned=STATE_CONDITIONED, grounded_quotes=GROUNDED_QUOTES):
        self.backend, self.max_steps, self.max_repairs = backend, max_steps, max_repairs
        self.guided, self.constrained = guided, constrained   # ablation switches (v37 behaviour = False, False)
        self.state_conditioned = state_conditioned and constrained   # v38.2 ablation level 3
        self.grounded_quotes = grounded_quotes and self.state_conditioned   # v38.3 ablation level 4
    def _payload(self, pack, request, host, feedback):
        payload = {'objective': pack.objective, 'request': request, 'required_finding_keys': list(pack.finding_keys),
                   'required_evidence_by_finding': {k: sorted(v) for k, v in EVIDENCE_REQUIREMENTS[pack.workflow_id].items()},
                   'available_proposal_action': pack.action, 'required_status': pack.status,
                   'available_tools': host.registry.describe(pack.required_tools),
                   # v38: send each observation once (v37 sent both `data` and `content`, doubling tokens).
                   'observations': [{k: o[k] for k in ('id', 'tool', 'arguments', 'content')} for o in host.observations],
                   'repair_feedback': feedback}
        if self.guided:
            called = [o['tool'] for o in host.observations]
            payload['progress'] = {'tools_called': called,
                                   'next_step': ('call one of: ' + ', '.join(allowed_tools(pack, host))) if allowed_tools(pack, host) else 'all evidence collected: emit kind=final now',
                                   'required_tools_not_yet_called': [t for t in pack.required_tools if t not in called]}
        return payload
    def run(self, pack, request, host, seed):
        schema = build_step_schema(pack, host.registry) if self.constrained else AgentStep
        step_schemas = []   # names of the schema used per model call (state-conditioned runs vary it)
        state = {'status': 'RUNNING', 'node': 'start', 'trace': [], 'decision': None, 'observations': [],
                 'error': None, 'repairs': 0, 'repair_log': [], 'approval_request': None, 'executed': False,
                 'model_calls': 0, 'schema': schema.__name__}
        feedback = ''; candidate = None; consecutive = 0
        def move(node):
            previous = state['node']
            if node not in EDGES[previous]: raise RuntimeError(f'Invalid edge {previous}->{node}')
            state['trace'].append({'from': previous, 'to': node}); state['node'] = node
        move('infer')
        for _ in range(self.max_steps):
            node = state['node']
            if node in {'finish', 'failed', 'quarantine'}: break
            try:
                if node == 'infer':
                    state['model_calls'] += 1
                    call_schema = build_state_schema(pack, host.registry, allowed_tools(pack, host), host.observations if self.grounded_quotes else None, request) if self.state_conditioned else schema
                    step_schemas.append(call_schema.__name__)
                    candidate = self.backend.chat('orchestrator', pack.worker_system,
                                                  self._payload(pack, request, host, feedback), call_schema, seed)
                    move('validate')
                elif node == 'validate':
                    if candidate.kind == 'tool': move('tool')
                    else:
                        pack.validate_final(candidate, host)
                        state['decision'] = candidate.model_dump(); state['status'] = candidate.status
                        if candidate.status == 'PENDING_APPROVAL':
                            state['approval_request'] = {'action': candidate.action, 'executed': False, 'summary': candidate.summary}
                        move('finish')
                elif node == 'tool':
                    host.call(tool_request_of(candidate)); feedback = ''; consecutive = 0; move('infer')
                elif node == 'repair':
                    if consecutive > self.max_repairs:
                        state.update(status='QUARANTINED', error=feedback); move('quarantine')
                    else: move('infer')
            except (BudgetError, ContextError, requests.RequestException) as e:
                state.update(status='FAILED', error=f'{type(e).__name__}: {e}')
                state['trace'].append({'from': state['node'], 'to': 'failed', 'reason': 'infrastructure'})
                state['node'] = 'failed'; break
            except Exception as e:
                # v38: tell the model what it sent and why it was rejected (v37 repeated identical bad calls).
                attempted = canonical(candidate.model_dump()) if candidate is not None and state['node'] != 'infer' else None
                feedback = f'{type(e).__name__}: {e}' + (f' | rejected step: {attempted[:600]}' if attempted else '')
                state['repairs'] += 1; consecutive += 1
                state['repair_log'].append({'node': state['node'], 'error': f'{type(e).__name__}: {str(e)[:300]}'})
                move('repair')
        if state['node'] not in {'finish', 'failed', 'quarantine'}:
            state['trace'].append({'from': state['node'], 'to': 'quarantine', 'reason': 'step_budget'})
            state.update(node='quarantine', status='QUARANTINED', error='step_budget')
        state['step_schemas'] = step_schemas
        state['observations'] = copy.deepcopy(host.observations)
        state['denied_tool_calls'] = copy.deepcopy(getattr(host, 'denied', []))
        if state['status'] in {'QUARANTINED', 'FAILED'}: state['failure_category'] = classify_failure(state)
        return state
