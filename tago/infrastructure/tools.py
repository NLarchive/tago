"""Ring 2 - infrastructure: deterministic sandbox tools and the per-case tool host.

Every tool is a local algorithm over the generated (fictional) case facts. Nothing here
performs an external write, opens a network connection or executes model-written code.
"""

# %% cell: retrieval
import ast, copy, re
from collections import defaultdict, deque
from tago.core.domain import load_public_model, canonical, norm, ToolError, ToolRequest, arithmetic_value
from tago.core.ports import HOST_FACTORIES, TOOL_REGISTRIES
from tago.config import EMBEDDING_MODEL

class RetrievalAdapter:
    """Lexical overlap, optionally blended 50/50 with CPU sentence-embedding cosine similarity."""
    def __init__(self, mode='lexical', model_name=EMBEDDING_MODEL):
        if mode not in {'lexical', 'hybrid'}: raise ValueError('Unknown retrieval mode')
        self.mode, self.encoder = mode, None
        if mode == 'hybrid':
            from sentence_transformers import SentenceTransformer
            self.encoder = load_public_model(lambda: SentenceTransformer(model_name, device='cpu'))
    def search(self, query, documents, k=3):
        words = set(re.findall(r'\w+', norm(query)))
        scores = [len(words & set(re.findall(r'\w+', norm(d['text'])))) / max(1, len(words)) for d in documents]
        if self.encoder is not None:
            matrix = self.encoder.encode([query] + [d['text'] for d in documents], normalize_embeddings=True)
            scores = [.5 * score + .5 * float(matrix[0] @ vector) for score, vector in zip(scores, matrix[1:])]
        ranked = sorted(zip(documents, scores), key=lambda item: (-item[1], item[0]['id']))[:k]
        return [{'document_id': d['id'], 'text': d['text'], 'score': round(score, 5)} for d, score in ranked]

# %% cell: graph-traversal
def graph_traverse(edges, start, max_depth=3):
    """Breadth-first traversal with a depth bound; returns the real edges walked."""
    adjacency = defaultdict(list)
    for source, relation, target in edges: adjacency[source].append((relation, target))
    queue = deque([(start, 0)]); seen = {start}; reached = []
    while queue:
        source, depth = queue.popleft()
        if depth >= max_depth: continue
        for relation, target in adjacency[source]:
            reached.append({'source': source, 'relation': relation, 'target': target})
            if target not in seen: seen.add(target); queue.append((target, depth + 1))
    return {'start': start, 'edges': reached, 'nodes': sorted(seen)}

# %% cell: tool-contracts
# v38: argument descriptions are explicit. They are copied into the constrained JSON schema,
# so the model sees the exact key names and the decoder cannot emit other keys.
REFERENCE_ARG = 'The exact case reference identifier quoted in the request.'
TOOL_ARGUMENTS = {
    'ticket_read': {'reference': REFERENCE_ARG}, 'crm_read': {'reference': REFERENCE_ARG},
    'invoice_read': {'reference': REFERENCE_ARG}, 'receipt_read': {'reference': REFERENCE_ARG},
    'reconcile': {'reference': REFERENCE_ARG}, 'identity_read': {'reference': REFERENCE_ARG},
    'code_read': {'reference': REFERENCE_ARG}, 'inventory_read': {'reference': REFERENCE_ARG},
    'vendors_read': {'reference': REFERENCE_ARG}, 'alert_read': {'reference': REFERENCE_ARG},
    'calendar_read': {'reference': REFERENCE_ARG},
    'code_check': {'expression': 'Arithmetic expression in n using + - * // and integer constants no larger than 100.'},
    'retrieve': {'query': 'Your own short search words for the policy or decision document (a few keywords).'},
    'graph_walk': {'start': 'Exact node id to start from (case project/entity id or alert source host).'},
    'browser_open': {'url': 'Must be exactly sandbox://portal'},
    'browser_click': {'selector': 'A selector observed on the opened page, e.g. #offers'},
}
TOOL_PURPOSE = {
    'ticket_read': 'Read the support ticket record.', 'crm_read': 'Read the CRM account record.',
    'invoice_read': 'Read billed quantity and unit price.', 'receipt_read': 'Read received quantity.',
    'reconcile': 'Compute billed, payable and overcharge cents (needs invoice_read and receipt_read first).',
    'identity_read': 'Read employee access state.', 'code_read': 'Read the defective code record.',
    'code_check': 'Run checks for a candidate expression (needs code_read first).',
    'inventory_read': 'Read stock levels and lead-time limit.', 'vendors_read': 'Read vendor offers.',
    'alert_read': 'Read the security alert.', 'calendar_read': 'Read busy hours and budget.',
    'retrieve': 'Search the case policy/decision documents.', 'graph_walk': 'Traverse relationships from a node.',
    'browser_open': 'Open the sandbox portal home page.', 'browser_click': 'Click a link on the current sandbox page.',
}

# %% cell: tool-registry
class ToolRegistry:
    """Explicit registration and per-pack authorization; local adapter, not MCP transport."""
    def __init__(self): self.entries = {}
    def register(self, name, arguments, handler):
        if name in self.entries: raise ValueError('Tool already registered: ' + name)
        if not callable(handler): raise TypeError('Handler must be callable')
        self.entries[name] = {'arguments': dict(arguments), 'handler': handler}
    def describe(self, names):
        return {name: {'purpose': TOOL_PURPOSE.get(name, ''), 'arguments': dict(self.entries[name]['arguments'])} for name in names}
    def invoke(self, host, request):
        name, args = request.name, request.arguments
        if name not in host.pack.required_tools or name not in self.entries:
            raise ToolError(f'Tool {name!r} is not authorized for this workflow; allowed: {list(host.pack.required_tools)}')
        entry = self.entries[name]
        if set(args) != set(entry['arguments']) or any(not isinstance(v, str) or not v.strip() for v in args.values()):
            raise ToolError(f'Arguments must match documented nonempty string fields: {name} expects '
                            f'{sorted(entry["arguments"])} but received {sorted(args)}')
        if 'reference' in args and args['reference'].strip().strip('\'"').lstrip('#') != host.world[host.pack.reference_key]:
            raise ToolError('Wrong case reference: use the exact identifier from the request')
        return entry['handler'](host, args)

TOOL_REGISTRY = ToolRegistry()
TOOL_REGISTRIES['default'] = TOOL_REGISTRY

# %% cell: tool-handlers
def handle_retrieve(self, args):
    return {'mode': self.retriever.mode, 'hits': self.retriever.search(args['query'], self.documents())}

def handle_graph_walk(self, args):
    w = self.world
    permitted = w.get('project_id', w.get('entity_id', w.get('source_host')))
    if args['start'] != permitted: raise ToolError('Use this case entity/project/source host as graph start')
    return graph_traverse(self.edges(), args['start'])

def fields(*keys):
    """Handler factory for plain record reads that expose only the listed world fields."""
    def handler(self, args): return {k: self.world[k] for k in keys}
    return handler

def handle_reconcile(self, args):
    w = self.world
    if not {'invoice_read', 'receipt_read'} <= {o['tool'] for o in self.observations}:
        raise ToolError('Read invoice and receipt first')
    billed = w['billed_units'] * w['unit_price_cents']; received = w['received_units'] * w['unit_price_cents']
    return {'billed_cents': billed, 'payable_cents': received, 'overcharge_cents': billed - received}

def handle_code_check(self, args):
    w = self.world
    if not any(o['tool'] == 'code_read' for o in self.observations): raise ToolError('Read code first')
    expression = args['expression']
    cases = [{'n': n, 'expected': sum(range(n + 1)), 'actual': arithmetic_value(expression, n)} for n in [0, 1, w['maximum_n']]]
    return {'expression': expression, 'checks': cases, 'tests_passed': all(c['expected'] == c['actual'] for c in cases)}

def handle_browser_open(self, args):
    if args['url'] != 'sandbox://portal': raise ToolError('Only sandbox://portal is allowed')
    self.page = 'home'
    return {'url': 'sandbox://portal', 'html': '<main><a id="offers" href="sandbox://offers">View offers</a></main>',
            'links': [{'selector': '#offers', 'url': 'sandbox://offers'}]}

def handle_browser_click(self, args):
    if self.page != 'home' or args['selector'] != '#offers': raise ToolError('Open home and click its observed #offers link')
    self.page = 'offers'
    return {'url': 'sandbox://offers', 'offers': self.world['offers'], 'read_only': True}

for _name, _handler in {
    'retrieve': handle_retrieve, 'graph_walk': handle_graph_walk,
    'ticket_read': fields('ticket_id', 'customer_id', 'refund_cents'),
    'crm_read': fields('account_id', 'company_name', 'deal_cents'),
    'invoice_read': fields('invoice_id', 'billed_units', 'unit_price_cents'),
    'receipt_read': fields('invoice_id', 'received_units'), 'reconcile': handle_reconcile,
    'identity_read': fields('employee_id', 'application', 'privileged', 'manager_approved'),
    'code_read': fields('issue_id', 'operation', 'buggy_expression', 'maximum_n'), 'code_check': handle_code_check,
    'inventory_read': fields('sku', 'on_hand', 'target_stock', 'maximum_lead_days'),
    'vendors_read': fields('vendors'),
    # Kept identical to v37 for comparability. Known issue (task T-R07): the alert also exposes
    # sensitive_asset, so graph_walk is required by provenance rules but not needed to know the answer.
    'alert_read': fields('alert_id', 'source_host', 'sensitive_asset', 'failed_logins', 'threshold'),
    'calendar_read': fields('trip_id', 'busy_hours', 'budget_cents'),
    'browser_open': handle_browser_open, 'browser_click': handle_browser_click}.items():
    TOOL_REGISTRY.register(_name, TOOL_ARGUMENTS[_name], _handler)

# %% cell: sandbox-host
class EnterpriseSandbox:
    """Per-case tool host: owns the private world, the observation log and browser page state."""
    def __init__(self, pack, world, retriever, registry=None):
        self.registry = registry or TOOL_REGISTRY
        self.pack, self.world, self.retriever = pack, world, retriever
        self.observations = []; self.page = None; self.denied = []
    def documents(self):
        w = self.world; wid = self.pack.workflow_id
        facts = {
            'E01': {'self_service_limit_cents': w.get('self_service_limit_cents'), 'above_limit': 'human approval required'},
            'E02': {'company_name': w.get('company_name'), 'next_step': w.get('next_step')},
            'E04': {'privileged': True, 'manager_approval_required': True},
            'E05': {'chosen_database': w.get('chosen_database'), 'rationale': w.get('rationale')},
            'E08': {'jurisdiction': w.get('jurisdiction'), 'rule_threshold': w.get('rule_threshold'),
                    'rule': 'applies when employee_count >= rule_threshold'}}
        reference = {self.pack.reference_key: w[self.pack.reference_key]}
        if wid == 'E08': reference['employee_count'] = w['employee_count']
        return [{'id': 'policy-or-decision', 'text': self.pack.title + ' ' + canonical(facts.get(wid, {}))},
                {'id': 'case-reference', 'text': canonical(reference)},
                {'id': 'distractor', 'text': 'Unrelated fictional cafeteria opening hours: 08:00 to 17:00.'}]
    def edges(self):
        w = self.world; wid = self.pack.workflow_id
        if wid == 'E05': return [(w['project_id'], 'selected', w['chosen_database']), (w['chosen_database'], 'because', w['rationale'])]
        if wid == 'E08': return [(w['entity_id'], 'governed_by', w['jurisdiction']), (w['jurisdiction'], 'rule', 'SYNTHETIC_THRESHOLD_RULE')]
        if wid == 'E09': return [(w['source_host'], 'connects_to', 'SYNTH_GATEWAY'), ('SYNTH_GATEWAY', 'connects_to', w['sensitive_asset'])]
        return []
    def call(self, request):
        # v38 duplicate guard: an identical successful call adds no evidence and caused v37's E01 loop.
        for o in self.observations:
            if o['tool'] == request.name and o['arguments'] == request.arguments:
                self.denied.append({'tool': request.name, 'reason': 'duplicate'})
                raise ToolError(f'Duplicate call: {request.name} with these arguments already returned {o["id"]}. '
                                'Use that observation and call a different tool or finish.')
        try:
            result = self.registry.invoke(self, request)
        except ToolError as error:
            self.denied.append({'tool': request.name, 'reason': str(error)}); raise
        observation = {'id': f'obs-{len(self.observations) + 1}', 'tool': request.name,
                       'arguments': copy.deepcopy(request.arguments), 'data': result,
                       'content': canonical(result), 'sandbox': True}
        self.observations.append(observation)
        return observation

HOST_FACTORIES['sandbox'] = EnterpriseSandbox
