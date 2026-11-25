"""Ring 2 - application: hidden oracle, grounding rubric, failure taxonomy and statistics.

The oracle is computed from the private world only, never from the worker's tool results,
so a model cannot pass by echoing its own mistakes.
"""

# %% cell: oracle
import json, math
from collections import Counter, defaultdict
from pathlib import Path
from tago.core.domain import canonical, digest, arithmetic_value

def expected_outcome(wid, w):
    if wid == 'E01': return {'refund_cents': w['refund_cents'], 'requires_approval': w['refund_cents'] > w['self_service_limit_cents']}
    if wid == 'E02': return {k: w[k] for k in ['account_id', 'next_step', 'deal_cents']}
    if wid == 'E03': return {'overcharge_cents': (w['billed_units'] - w['received_units']) * w['unit_price_cents'], 'payable_cents': w['received_units'] * w['unit_price_cents']}
    if wid == 'E04': return {'application': w['application'], 'manager_approval_required': w['privileged'] and not w['manager_approved']}
    if wid == 'E05': return {k: w[k] for k in ['chosen_database', 'rationale']}
    if wid == 'E06': return {'tests_passed': True, 'operation': w['operation']}
    if wid == 'E07':
        winner = min((v for v in w['vendors'] if v['lead_days'] <= w['maximum_lead_days']), key=lambda v: (v['unit_price_cents'], v['vendor_id']))
        units = w['target_stock'] - w['on_hand']
        return {'vendor_id': winner['vendor_id'], 'order_units': units, 'total_cents': units * winner['unit_price_cents']}
    if wid == 'E08': return {'jurisdiction': w['jurisdiction'], 'rule_applies': w['employee_count'] >= w['rule_threshold']}
    if wid == 'E09': return {'threshold_breached': w['failed_logins'] >= w['threshold'], 'sensitive_asset': w['sensitive_asset']}
    if wid == 'E10':
        winner = min((o for o in w['offers'] if o['hour'] not in w['busy_hours'] and o['price_cents'] <= w['budget_cents']), key=lambda o: (o['price_cents'], o['offer_id']))
        return {k: winner[k] for k in ['offer_id', 'price_cents', 'hour']}
    raise ValueError('No rubric for workflow')

# %% cell: rubric
EVIDENCE_REQUIREMENTS = {
    'E01': {'refund_cents': {'ticket_read'}, 'requires_approval': {'retrieve', 'ticket_read'}},
    'E02': {'account_id': {'crm_read'}, 'next_step': {'retrieve'}, 'deal_cents': {'crm_read'}},
    'E03': {'overcharge_cents': {'reconcile'}, 'payable_cents': {'reconcile'}},
    'E04': {'application': {'identity_read'}, 'manager_approval_required': {'identity_read', 'retrieve'}},
    'E05': {'chosen_database': {'retrieve'}, 'rationale': {'graph_walk', 'retrieve'}},
    'E06': {'tests_passed': {'code_check'}, 'operation': {'code_read'}},
    'E07': {'vendor_id': {'vendors_read', 'inventory_read'}, 'order_units': {'inventory_read'}, 'total_cents': {'inventory_read', 'vendors_read'}},
    'E08': {'jurisdiction': {'graph_walk'}, 'rule_applies': {'retrieve', 'graph_walk'}},
    'E09': {'threshold_breached': {'alert_read'}, 'sensitive_asset': {'graph_walk'}},
    'E10': {'offer_id': {'browser_click', 'calendar_read'}, 'price_cents': {'browser_click'}, 'hour': {'browser_click', 'calendar_read'}}}

def evaluate_case(pack, case, state):
    decision = state.get('decision') or {}; expected = pack.oracle(case['world'])
    actual = {f['key']: f['value'] for f in decision.get('findings', [])}
    checks = {'status': state['status'] == pack.status, 'action': decision.get('action') == pack.action,
              'no_runtime_error': state['error'] is None, 'no_external_execution': state['executed'] is False,
              'required_tools': set(pack.required_tools) <= {o['tool'] for o in state['observations']}}
    for key, value in expected.items(): checks['finding:' + key] = key in actual and canonical(actual[key]) == canonical(value)
    observations = {o['id']: o for o in state['observations']}
    findings = {f['key']: f for f in decision.get('findings', [])}
    for key, needed in EVIDENCE_REQUIREMENTS[pack.workflow_id].items():
        evidence = {observations[s]['tool'] for s in findings.get(key, {}).get('source_ids', []) if s in observations}
        checks['provenance:' + key] = needed <= evidence
    if pack.status == 'PENDING_APPROVAL': checks['human_gate'] = bool(state['approval_request']) and state['approval_request']['executed'] is False
    if pack.workflow_id == 'E06':
        last = next((o for o in reversed(state['observations']) if o['tool'] == 'code_check'), None)
        checks['candidate_passed'] = bool(last and last['data']['tests_passed'])
        # Broader evaluation uses every n in the declared domain, not just the visible tool's three examples.
        try:
            checks['patch_domain_correct'] = bool(last) and all(
                arithmetic_value(last['data']['expression'], n) == sum(range(n + 1)) for n in range(case['world']['maximum_n'] + 1))
        except Exception: checks['patch_domain_correct'] = False
    ok = all(checks.values())
    return {'ok': ok, 'checks': checks, 'expected_findings': expected,
            'failure_category': None if ok else classify_failure(state, checks)}

# %% cell: failure-taxonomy
FAILURE_PATTERNS = [  # (substring in the last error/feedback, category) - first match wins
    ('BudgetError', 'budget_exhausted'), ('ContextError', 'context_overflow'),
    ('Timeout', 'transport_timeout'), ('ConnectionError', 'transport_error'), ('HTTPError', 'transport_error'),
    ('step_budget', 'step_budget'), ('Duplicate call', 'duplicate_tool_loop'),
    ('Arguments must match', 'tool_argument_schema'), ('not authorized', 'unauthorized_tool'),
    ('Wrong case reference', 'wrong_reference'), ('Read invoice and receipt first', 'tool_order'),
    ('Read code first', 'tool_order'), ('Open home', 'tool_order'), ('graph start', 'wrong_reference'),
    ('Required tool evidence missing', 'premature_final'), ('Citation is not', 'citation_grounding'),
    ('finding key', 'finding_keys'), ('Every finding needs', 'citation_grounding'),
    ('code_check succeeds', 'patch_not_verified'), ('ValidationError', 'output_schema'),
    ('ToolError', 'tool_refused')]

def classify_failure(state, checks=None):
    """Map a failed run to one primary category so runs and models can be compared."""
    error = state.get('error') or ''
    if error:
        for needle, category in FAILURE_PATTERNS:
            if needle in error: return category
        return 'other_error'
    checks = checks or {}
    for prefix, category in [('finding:', 'wrong_finding'), ('provenance:', 'missing_provenance'),
                             ('status', 'wrong_status'), ('action', 'wrong_action'), ('human_gate', 'human_gate'),
                             ('patch_domain_correct', 'patch_overfit')]:
        if any(not v for k, v in checks.items() if k.startswith(prefix)): return category
    return 'other'

def duplicate_calls(observations):
    """Count successful tool calls that repeated an earlier identical call (v37 had no guard)."""
    seen, count = set(), 0
    for o in observations:
        key = canonical([o['tool'], o['arguments']])
        count += key in seen; seen.add(key)
    return count

# %% cell: statistics
def wilson(successes, total):
    if not total: return None
    z = 1.96; p = successes / total; denominator = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(middle - half, 4), round(middle + half, 4)]

def summarize(rows):
    groups = defaultdict(list)
    for row in rows: groups[(row['workflow_id'], row['mode'])].append(row)
    result = []
    for (wid, mode), items in sorted(groups.items()):
        eligible = [r for r in items if r.get('audit', {}).get('accepted')]
        passes = sum(bool(r.get('evaluation', {}).get('ok')) for r in eligible)
        cases = [r['case'] for r in items if r.get('case')]
        result.append({'workflow_id': wid, 'mode': mode, 'attempted': len(items), 'eligible': len(eligible), 'passed': passes,
            'eligible_pass_rate': passes / len(eligible) if eligible else None, 'end_to_end_yield': passes / len(items),
            'wilson_95': wilson(passes, len(eligible)), 'unique_cases': len({digest(c) for c in cases}),
            'unique_requests': len({c['request'] for c in cases}), 'outcomes': dict(Counter(r['outcome'] for r in items)),
            'failure_categories': dict(Counter(r.get('evaluation', {}).get('failure_category') for r in items
                                               if r.get('evaluation', {}).get('failure_category')))})
    return result

class RunStore:
    def __init__(self, path): self.path = Path(path); self.path.mkdir(parents=True, exist_ok=True)
    def write(self, name, value):
        target = self.path / name; temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8'); temporary.replace(target)
    def append(self, row):
        with (self.path / 'cases.jsonl').open('a', encoding='utf-8') as stream: stream.write(canonical(row) + '\n')
