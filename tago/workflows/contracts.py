"""Ring 4 - workflows: offline contract suite (no LLM, GPU, network or tokenizer download).

These tests verify software behaviour with a scripted test double. They say nothing about
real-model quality; that is measured only by the live labs.
"""

# %% cell: offline-fixtures
import copy, json, unittest
from dataclasses import replace
from unittest.mock import patch
import requests
from tago.core.domain import (quote_fragments, ToolError, CaseError, ContextError, ToolRequest, AgentStep, AuditVerdict, FinanceWorld, canonical)
from tago.infrastructure.profiles import ModelProfile, TokenProfiler, ByteUpperBoundProfiler, resolve_profile, api_profile
from tago.infrastructure.backends import OllamaBackend, OpenAICompatibleBackend, AnthropicBackend, ScriptedBackend
from tago.infrastructure.tools import RetrievalAdapter, ToolRegistry, EnterpriseSandbox, arithmetic_value, graph_traverse
from tago.application.packs import PACKS, build_step_schema, build_state_schema
from tago.application.evaluation import EVIDENCE_REQUIREMENTS, evaluate_case, summarize, wilson, classify_failure
from tago.application.orchestrator import AgentGraphOrchestrator, EDGES, validate_final, allowed_tools
from tago.application.synthetic import SyntheticCaseAgent, DatasetAuditor

# Explicit fixture answers, independent of expected_outcome().
OFFLINE_ANSWERS = {
 'E01': ('PENDING_APPROVAL', 'draft_refund', {'refund_cents': 9000, 'requires_approval': True}),
 'E02': ('PROPOSED', 'draft_crm_update', {'account_id': 'A-201', 'next_step': 'schedule_demo', 'deal_cents': 120000}),
 'E03': ('PENDING_APPROVAL', 'draft_invoice_dispute', {'overcharge_cents': 5000, 'payable_cents': 25000}),
 'E04': ('PENDING_APPROVAL', 'request_access_approval', {'application': 'SyntheticAdmin', 'manager_approval_required': True}),
 'E05': ('ANSWERED', 'answer_with_evidence', {'chosen_database': 'PostgreSQL', 'rationale': 'transaction_integrity'}),
 'E06': ('PROPOSED', 'propose_patch', {'tests_passed': True, 'operation': 'inclusive_sum'}),
 'E07': ('PENDING_APPROVAL', 'draft_purchase', {'vendor_id': 'VEN-A', 'order_units': 16, 'total_cents': 16000}),
 'E08': ('PENDING_APPROVAL', 'request_legal_review', {'jurisdiction': 'SYNTH_A', 'rule_applies': True}),
 'E09': ('PENDING_APPROVAL', 'draft_containment_review', {'threshold_breached': True, 'sensitive_asset': 'DB-SECRET'}),
 'E10': ('PENDING_APPROVAL', 'draft_booking', {'offer_id': 'OFF-B', 'price_cents': 22000, 'hour': 11}),
}

def tool_args(pack, name):
    w = pack.golden_world
    if name == 'retrieve': return {'query': pack.title + ' ' + pack.objective}
    if name == 'graph_walk': return {'start': w.get('project_id', w.get('entity_id', w.get('source_host')))}
    if name == 'code_check': return {'expression': 'n * (n + 1) // 2'}
    if name == 'browser_open': return {'url': 'sandbox://portal'}
    if name == 'browser_click': return {'selector': '#offers'}
    return {'reference': w[pack.reference_key]}

def offline_tool_steps(pack):
    return [{'kind': 'tool', 'tool': {'name': name, 'arguments': tool_args(pack, name)}} for name in pack.required_tools]

def offline_final(wid, values=None):
    status, action, answers = OFFLINE_ANSWERS[wid]
    def make(payload):
        observations = payload['observations']
        findings = [{'key': key, 'value': value, 'source_ids': [o['id'] for o in observations if o['tool'] in EVIDENCE_REQUIREMENTS[wid][key]]}
                    for key, value in (values if values is not None else answers).items()]
        return {'kind': 'final', 'status': status, 'action': action, 'findings': findings,
                'citations': [{'source_id': o['id'], 'quote': quote_fragments(o['content'])[0]} for o in observations],
                'summary': 'Scripted contract fixture; not LLM evidence.'}
    return make

def offline_run(wid, steps=None, **orchestrator_options):
    pack = PACKS[wid]
    backend = ScriptedBackend(steps if steps is not None else offline_tool_steps(pack) + [offline_final(wid)])
    state = AgentGraphOrchestrator(backend, **orchestrator_options).run(pack, pack.golden_request,
        pack.tools(copy.deepcopy(pack.golden_world), RetrievalAdapter('lexical')), 0)
    return state, backend

class FakeTokenizer:
    chat_template = 'fake-template'; init_kwargs = {}
    def __init__(self, n=20): self.n = n
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=False):
        return list(range(max(self.n, sum(len(m['content']) for m in messages) // 4)))   # roughly 4 characters per token

class FakeResponse:
    def __init__(self, data): self.data = data; self.ok = True
    def raise_for_status(self): pass
    def json(self): return self.data

# %% cell: contracts-v37
class OfflineContractTests(unittest.TestCase):
    """v37 contract suite, ported unchanged in intent."""
    def test_all_ten_golden_contracts(self):
        for wid in PACKS:
            with self.subTest(workflow=wid):
                state, _ = offline_run(wid)
                result = evaluate_case(PACKS[wid], {'world': PACKS[wid].golden_world}, state)
                self.assertTrue(result['ok'], result)
                self.assertTrue(all(t['to'] in EDGES[t['from']] for t in state['trace']))
    def test_world_constraints_reject_invalid_invoice(self):
        with self.assertRaises(ValueError): FinanceWorld(invoice_id='INV-1', billed_units=1, received_units=2, unit_price_cents=3)
    def test_reference_mismatch_rejected(self):
        p = PACKS['E01']
        with self.assertRaises(CaseError): p.validate_case({'request': 'This request omits the reference', 'world': p.golden_world})
    def test_arithmetic_rejects_calls_attributes_and_powers(self):
        for expression in ['__import__("os")', 'n.real', 'n ** 100', '[n][0]', 'n +']:
            with self.subTest(expression=expression), self.assertRaises(ToolError): arithmetic_value(expression, 3)
    def test_arithmetic_patch_and_bug(self):
        self.assertEqual(arithmetic_value('n * (n + 1) // 2', 5), 15)
        self.assertEqual(arithmetic_value('n * (n - 1) // 2', 5), 10)
    def test_graph_cycle_bounded(self):
        self.assertEqual(graph_traverse([('A', 'to', 'B'), ('B', 'to', 'A'), ('B', 'to', 'C')], 'A', 1)['nodes'], ['A', 'B'])
    def test_lexical_retrieval_ranks_relevant_document(self):
        hits = RetrievalAdapter('lexical').search('invoice', [{'id': 'a', 'text': 'weather'}, {'id': 'b', 'text': 'invoice amount'}], 1)
        self.assertEqual(hits[0]['document_id'], 'b')
    def test_unknown_and_cross_pack_tool_denied(self):
        p = PACKS['E01']; host = p.tools(p.golden_world, RetrievalAdapter('lexical'))
        for name in ['delete_database', 'calendar_read']:
            with self.assertRaises(ToolError): host.call(ToolRequest(name=name, arguments={'reference': 'T-101'}))
        self.assertEqual(host.observations, [])
    def test_wrong_reference_and_extra_arguments_denied(self):
        p = PACKS['E01']; host = p.tools(p.golden_world, RetrievalAdapter('lexical'))
        for args in [{'reference': 'wrong'}, {'reference': 'T-101', 'extra': 'x'}]:
            with self.assertRaises(ToolError): host.call(ToolRequest(name='ticket_read', arguments=args))
    def test_registry_extension_without_host_edit(self):
        registry = ToolRegistry(); registry.register('echo', {'value': 'string'}, lambda host, args: {'echo': args['value']})
        p = replace(PACKS['E01'], required_tools=('echo',))
        host = EnterpriseSandbox(p, p.golden_world, RetrievalAdapter('lexical'), registry)
        self.assertEqual(host.call(ToolRequest(name='echo', arguments={'value': 'hello'}))['data'], {'echo': 'hello'})
        with self.assertRaises(ValueError): registry.register('echo', {}, lambda h, a: {})
    def test_reconcile_requires_prior_reads(self):
        p = PACKS['E03']; host = p.tools(p.golden_world, RetrievalAdapter('lexical'))
        with self.assertRaises(ToolError): host.call(ToolRequest(name='reconcile', arguments={'reference': 'INV-301'}))
    def test_browser_requires_observed_navigation(self):
        p = PACKS['E10']; host = p.tools(p.golden_world, RetrievalAdapter('lexical'))
        with self.assertRaises(ToolError): host.call(ToolRequest(name='browser_click', arguments={'selector': '#offers'}))
        with self.assertRaises(ToolError): host.call(ToolRequest(name='browser_open', arguments={'url': 'https://example.com'}))
    def test_wrong_amount_fails_independent_rubric(self):
        p = PACKS['E03']; state, _ = offline_run('E03', offline_tool_steps(p) + [offline_final('E03', {'overcharge_cents': 1, 'payable_cents': 25000})])
        result = evaluate_case(p, {'world': p.golden_world}, state)
        self.assertFalse(result['checks']['finding:overcharge_cents']); self.assertEqual(result['failure_category'], 'wrong_finding')
    def test_provenance_omission_fails(self):
        state, _ = offline_run('E01')
        state['decision']['findings'][1]['source_ids'] = []
        self.assertFalse(evaluate_case(PACKS['E01'], {'world': PACKS['E01'].golden_world}, state)['checks']['provenance:requires_approval'])
    def test_nonliteral_citation_fails(self):
        p = PACKS['E01']; host = p.tools(p.golden_world, RetrievalAdapter('lexical'))
        for step in offline_tool_steps(p): host.call(ToolRequest.model_validate(step['tool']))
        final = offline_final('E01')({'observations': host.observations})
        final['citations'][0]['quote'] = 'THIS WAS NEVER OBSERVED'
        with self.assertRaises(ValueError): validate_final(AgentStep.model_validate(final), host, p)
    def test_schema_repair_recovers(self):
        p = PACKS['E01']; state, _ = offline_run('E01', [{'invalid': True}] + offline_tool_steps(p) + [offline_final('E01')])
        self.assertEqual(state['status'], 'PENDING_APPROVAL'); self.assertEqual(state['repairs'], 1)
    def test_repair_exhaustion_quarantines(self):
        state, _ = offline_run('E01', [{'invalid': True}] * 3)
        self.assertEqual(state['status'], 'QUARANTINED'); self.assertEqual(state['failure_category'], 'output_schema')
    def test_transport_error_fails_without_schema_retry(self):
        state, _ = offline_run('E01', [requests.Timeout('fixture')])
        self.assertEqual(state['status'], 'FAILED'); self.assertEqual(state['repairs'], 0)
        self.assertEqual(state['failure_category'], 'transport_timeout')
    def test_step_budget_interrupts(self):
        p = PACKS['E01']; backend = ScriptedBackend(offline_tool_steps(p))
        state = AgentGraphOrchestrator(backend, max_steps=1).run(p, p.golden_request, p.tools(p.golden_world, RetrievalAdapter('lexical')), 0)
        self.assertEqual(state['error'], 'step_budget')
    def test_pending_human_action_never_executes(self):
        state, _ = offline_run('E04')
        self.assertFalse(state['executed']); self.assertFalse(state['approval_request']['executed'])
    def test_specialist_prompts_are_distinct(self):
        self.assertEqual(len({p.persona for p in PACKS.values()}), 10)
        self.assertIn('code_check', PACKS['E06'].worker_system); self.assertNotIn('code_check', PACKS['E01'].worker_system)
    def test_unknown_model_requires_explicit_profile(self):
        with self.assertRaises(ValueError): OllamaBackend('unregistered-model')
    def test_token_reserve_and_preflight_rejection(self):
        profile = ModelProfile('fake', 'fake', context_tokens=40, output_tokens=10, safety_tokens=5)
        self.assertEqual(TokenProfiler(profile, FakeTokenizer()).measure([])['total_reserved'], 35)
        small = replace(profile, context_tokens=30)
        backend = OllamaBackend('fake', profile=small, profiler=TokenProfiler(small, FakeTokenizer()))
        with patch.object(requests, 'post', side_effect=AssertionError('Network must not be called')):
            with self.assertRaises(ContextError): backend.chat('test', 'system', {}, AuditVerdict)
        self.assertEqual(backend.calls, 0)
    def test_invalid_generation_is_not_a_workflow_pass(self):
        backend = ScriptedBackend([{'request': 'missing identifier', 'world': PACKS['E01'].golden_world}])
        with self.assertRaises(CaseError): SyntheticCaseAgent(backend, attempts=1).generate(PACKS['E01'], 'clean', 0)
    def test_audit_rejection_record(self):
        p = PACKS['E01']; b = ScriptedBackend([{'valid': False, 'request_matches_world': False, 'style_matches': True, 'reason': 'contradiction'}])
        self.assertFalse(DatasetAuditor(b).audit(p, {'request': p.golden_request, 'world': p.golden_world}, 'clean', 0)['accepted'])
    def test_statistics_separate_invalid_dataset(self):
        rows = [{'workflow_id': 'E01', 'mode': 'clean', 'outcome': 'dataset_invalid', 'audit': {}},
                {'workflow_id': 'E01', 'mode': 'clean', 'outcome': 'pass', 'audit': {'accepted': True}, 'evaluation': {'ok': True}}]
        result = summarize(rows)[0]
        self.assertEqual(result['eligible'], 1); self.assertEqual(result['end_to_end_yield'], .5)
        self.assertIsNone(wilson(0, 0)); self.assertLess(wilson(16, 16)[0], .81)

# %% cell: contracts-v38
class V38RegressionTests(unittest.TestCase):
    """Each test pins one fix for a failure actually observed in the v37 Colab T4 run."""
    def test_constrained_schema_rejects_v37_argument_errors(self):
        # Observed v37 raw outputs: retrieve {} (E01), invoice_read {"invoice_id": ...} (E03), code_read {} (E06).
        for wid, bad in [('E01', {'name': 'retrieve', 'arguments': {}}),
                         ('E03', {'name': 'invoice_read', 'arguments': {'invoice_id': 'INV-301'}}),
                         ('E06', {'name': 'code_read', 'arguments': {}}),
                         ('E01', {'name': 'calendar_read', 'arguments': {'reference': 'T-101'}})]:
            with self.subTest(workflow=wid, call=bad), self.assertRaises(Exception):
                build_step_schema(PACKS[wid]).model_validate_json(json.dumps({'kind': 'tool', 'tool': bad}))
        ok = build_step_schema(PACKS['E03']).model_validate_json(json.dumps(
            {'kind': 'tool', 'tool': {'name': 'invoice_read', 'arguments': {'reference': 'INV-301'}}}))
        self.assertEqual(ok.tool.name, 'invoice_read')
    def test_constrained_json_schema_lists_exact_tools_and_keys(self):
        text = json.dumps(build_step_schema(PACKS['E03']).model_json_schema())
        for name in ['invoice_read', 'receipt_read', 'reconcile', '"reference"', 'overcharge_cents']: self.assertIn(name, text)
        self.assertNotIn('ticket_read', text); self.assertIn('"additionalProperties": false', text)
    def test_duplicate_call_denied_without_new_observation(self):
        p = PACKS['E01']; host = p.tools(p.golden_world, RetrievalAdapter('lexical'))
        host.call(ToolRequest(name='ticket_read', arguments={'reference': 'T-101'}))
        with self.assertRaisesRegex(ToolError, 'Duplicate call'): host.call(ToolRequest(name='ticket_read', arguments={'reference': 'T-101'}))
        self.assertEqual(len(host.observations), 1)
    def test_v37_e01_loop_now_recovers(self):
        p = PACKS['E01']; read = offline_tool_steps(p)[0]
        state, backend = offline_run('E01', [read, read, read] + offline_tool_steps(p)[1:] + [offline_final('E01')], state_conditioned=False)
        self.assertEqual(state['status'], 'PENDING_APPROVAL'); self.assertEqual(len(state['observations']), 2)
        self.assertIn('Duplicate call', backend.payloads[2]['repair_feedback'])
    def test_duplicate_loop_is_still_bounded(self):
        read = offline_tool_steps(PACKS['E01'])[0]
        state, _ = offline_run('E01', [read] * 6, state_conditioned=False)
        self.assertEqual(state['status'], 'QUARANTINED'); self.assertEqual(state['failure_category'], 'duplicate_tool_loop')
    def test_state_conditioning_makes_duplicates_undecodable(self):
        p = PACKS['E01']; read = offline_tool_steps(p)[0]
        state, _ = offline_run('E01', [read, read] + offline_tool_steps(p)[1:] + [offline_final('E01')])
        self.assertEqual(state['status'], 'PENDING_APPROVAL')
        self.assertEqual([o['tool'] for o in state['observations']], ['ticket_read', 'retrieve'])
        self.assertEqual(state['repair_log'][0]['node'], 'infer')   # the duplicate failed at the schema, never reached the host
        self.assertEqual(state['denied_tool_calls'], [])
        self.assertIn('ticket_read', state['step_schemas'][0]); self.assertIn('retrieve', state['step_schemas'][1])
        self.assertIn('_final_grounded_step', state['step_schemas'][-1])
    def test_real_v381_colab_envelope_failures_are_undecodable(self):
        # Raw outputs from results/v38-colab (T4, qwen2.5:7b): calls 5, 9, 11 and 18 of the golden run.
        e01_finish_as_tool = {'kind': 'tool', 'tool': None, 'findings': [{'key': 'refund_cents', 'value': 9000, 'source_ids': ['obs-1']}],
                              'citations': [{'source_id': 'obs-1', 'quote': '{"refund_cents":9000}'}], 'summary': 's'}
        e03_empty_tool = {'kind': 'tool', 'tool': None}
        e03_final_without_status = {'kind': 'final', 'tool': None, 'findings': [{'key': 'overcharge_cents', 'value': 5000, 'source_ids': ['obs-3']},
            {'key': 'payable_cents', 'value': 25000, 'source_ids': ['obs-3']}], 'citations': [{'source_id': 'obs-3', 'quote': 'reconcile result'}], 'summary': 's'}
        tool_only = build_state_schema(PACKS['E03'], None, ('invoice_read', 'receipt_read', 'reconcile'))
        final_only = build_state_schema(PACKS['E03'], None, ())
        flat = build_step_schema(PACKS['E03'])
        for bad in (e01_finish_as_tool, e03_empty_tool):
            with self.subTest(schema='tool_only', bad=bad['kind']), self.assertRaises(Exception): tool_only.model_validate_json(json.dumps(bad))
        for bad in (e03_empty_tool, e03_final_without_status):
            with self.subTest(schema='final_only'), self.assertRaises(Exception): final_only.model_validate_json(json.dumps(bad))
        with self.assertRaises(Exception): flat.model_validate_json(json.dumps(e03_final_without_status))
        good = dict(e03_final_without_status, status='PENDING_APPROVAL', action=PACKS['E03'].action)
        self.assertEqual(final_only.model_validate_json(json.dumps(good)).status, 'PENDING_APPROVAL')
    def test_state_schema_offers_only_uncalled_tools_and_final_only_when_done(self):
        p = PACKS['E01']; host = p.tools(copy.deepcopy(p.golden_world), RetrievalAdapter('lexical'))
        self.assertEqual(allowed_tools(p, host), ('ticket_read', 'retrieve'))
        host.call(ToolRequest(name='ticket_read', arguments={'reference': 'T-101'}))
        self.assertEqual(allowed_tools(p, host), ('retrieve',))
        text = json.dumps(build_state_schema(p, host.registry, allowed_tools(p, host)).model_json_schema())
        self.assertNotIn('ticket_read', text); self.assertIn('retrieve', text)
        host.call(ToolRequest(name='retrieve', arguments={'query': 'refund policy'}))
        self.assertEqual(allowed_tools(p, host), ())
        self.assertNotIn('"tool"', json.dumps(build_state_schema(p, host.registry, ()).model_json_schema()['properties']))
    def test_e06_may_retry_code_check_until_it_passes(self):
        p = PACKS['E06']; host = p.tools(copy.deepcopy(p.golden_world), RetrievalAdapter('lexical'))
        host.call(ToolRequest(name='code_read', arguments={'reference': p.golden_world['issue_id']}))
        host.call(ToolRequest(name='code_check', arguments={'expression': 'n * (n - 1) // 2'}))
        self.assertEqual(allowed_tools(p, host), ('code_check',))
        host.call(ToolRequest(name='code_check', arguments={'expression': 'n * (n + 1) // 2'}))
        self.assertEqual(allowed_tools(p, host), ())
    def test_decoding_schema_has_no_prose_to_copy_and_identifiers_are_patterned(self):
        # Real v38.1 failure: reference = 'The exact case reference identifier quoted in the request.'
        schema = build_state_schema(PACKS['E03'], None, ('invoice_read',))
        text = json.dumps(schema.model_json_schema())
        self.assertNotIn('exact case reference', text); self.assertNotIn('description', text)
        copied = {'kind': 'tool', 'tool': {'name': 'invoice_read', 'arguments': {'reference': 'The exact case reference identifier quoted in the request.'}}}
        with self.assertRaises(Exception): schema.model_validate_json(json.dumps(copied))
        for ref in ('INV-301', '987XYZ', '#calc123', 'ticket_1234567890'):
            schema.model_validate_json(json.dumps({'kind': 'tool', 'tool': {'name': 'invoice_read', 'arguments': {'reference': ref}}}))
        self.assertIn('exact case reference', json.dumps(PACKS['E03'].tools(PACKS['E03'].golden_world, RetrievalAdapter('lexical')).registry.describe(('invoice_read',))))
    def test_warmup_uses_the_chat_context_size(self):
        profile = ModelProfile('fake', 'fake', context_tokens=8192)
        backend = OllamaBackend('fake', profile=profile, profiler=TokenProfiler(profile, FakeTokenizer()))
        with patch.object(requests, 'post', return_value=FakeResponse({})) as post: backend.warmup()
        self.assertEqual(post.call_args.kwargs['json']['options']['num_ctx'], 8192)
    def test_hash_prefixed_reference_is_normalised_but_wrong_one_is_not(self):
        p = PACKS['E06']; host = p.tools(copy.deepcopy(p.golden_world), RetrievalAdapter('lexical'))
        host.call(ToolRequest(name='code_read', arguments={'reference': '#' + p.golden_world['issue_id']}))
        with self.assertRaisesRegex(ToolError, 'Wrong case reference'):
            host.call(ToolRequest(name='code_read', arguments={'reference': 'BUG-999'}))
    def test_case_reference_must_be_a_whole_token(self):
        p = PACKS['E06']; world = dict(p.golden_world, issue_id='calc123')
        for request, ok in [('Please fix issue calc123 and check it', True), ('Issue #calc123 needs a fix please', True),
                            ('Fix issue xcalc123 now please', False), ('Fix calc1234 now please okay', False), ('Fix the issue please soon', False)]:
            with self.subTest(request=request):
                if ok: p.validate_case({'request': request, 'world': world})
                else:
                    with self.assertRaises(CaseError): p.validate_case({'request': request, 'world': world})
    def test_synthetic_case_rules_from_the_real_v382_run(self):
        p = PACKS['E01']; w = dict(p.golden_world)
        placeholder = {'request': 'For #ticket_id the customer asked a refund of 9000 cents, limit 5000 cents', 'world': dict(w, ticket_id='#ticket_id')}
        mismatch = {'request': f"Ticket {w['ticket_id']}: refund of 10000 cents, above the 5000 cents limit", 'world': dict(w, refund_cents=100000, self_service_limit_cents=50000)}
        for name, case in (('placeholder reference', placeholder), ('request contradicts world', mismatch)):
            with self.subTest(name), self.assertRaises(CaseError): p.validate_synthetic(case)
        ok = {'request': f"Ticket {w['ticket_id']}: refund of {w['refund_cents']} cents, limit {w['self_service_limit_cents']} cents", 'world': w}
        p.validate_synthetic(ok)
        formatted = {'request': f"Ticket {w['ticket_id']}: refund of {w['refund_cents']:,} cents, limit {w['self_service_limit_cents']:,} cents", 'world': w}
        p.validate_synthetic(formatted)   # thousands separators are fine
        p.validate_case({'request': p.golden_request, 'world': w})   # golden requests are exempt from the stated-figures rule
    def test_generator_retries_with_the_validation_error(self):
        p = PACKS['E01']; good = {'request': p.golden_request + ' Refund 9000 cents, limit 5000 cents.', 'world': p.golden_world}
        bad = {'request': 'Customer with ID 1234567890 wants a refund, please escalate it', 'world': p.golden_world}
        backend = ScriptedBackend([bad, good]); agent = SyntheticCaseAgent(backend, attempts=3)
        self.assertEqual(agent.generate(p, 'clean', 5)['world'], p.golden_world)
        self.assertEqual(len(agent.attempt_errors), 1); self.assertIn('verbatim', backend.payloads[1]['previous_draft_was_rejected_because'])
        self.assertNotIn('previous_draft_was_rejected_because', backend.payloads[0])
        with self.assertRaises(CaseError): SyntheticCaseAgent(ScriptedBackend([bad, bad]), attempts=2).generate(p, 'clean', 5)
    def test_citation_error_shows_the_text_to_copy_and_ignores_whitespace(self):
        p = PACKS['E03']; steps = offline_tool_steps(p)
        def paraphrased(payload):
            step = offline_final('E03')(payload)
            step['citations'] = [{'source_id': 'obs-3', 'quote': 'The reconciliation of invoice INV-301 shows an overcharge of 5000 cents.'}]
            return step
        state, backend = offline_run('E03', steps + [paraphrased, offline_final('E03')], grounded_quotes=False)
        self.assertEqual(state['status'], 'PENDING_APPROVAL')
        self.assertIn('Copy a fragment verbatim from', backend.payloads[-1]['repair_feedback'])
        self.assertIn('overcharge_cents', backend.payloads[-1]['repair_feedback'])
        # whitespace is not part of the evidence: a model that re-wraps a sentence is still quoting it
        host = type('H', (), {'observations': [{'id': 'obs-1', 'tool': t, 'content': 'Refund   limit is\n 5000 cents.', 'data': {}} for t in p.required_tools]})()
        step = AgentStep(kind='final', status='PENDING_APPROVAL', action=p.action, findings=[{'key': k, 'value': 1, 'source_ids': ['obs-1']} for k in p.finding_keys],
                         citations=[{'source_id': 'obs-1', 'quote': 'Refund limit is 5000 cents.'}])
        validate_final(step, host, p)
        step.citations[0].quote = 'refund limit is 5000 cents.'   # case still counts
        with self.assertRaises(ValueError): validate_final(step, host, p)
    def test_real_v382_paraphrased_quotes_are_undecodable_but_fragments_are_not(self):
        # Raw v3.8.2 Colab outputs (results/v38_2_colab_t4_qwen2.5-7b): the model re-formatted JSON in every final.
        p = PACKS['E01']; host = p.tools(copy.deepcopy(p.golden_world), RetrievalAdapter('lexical'))
        for name, args in (('ticket_read', {'reference': 'T-101'}), ('retrieve', {'query': 'refund policy'})): host.call(ToolRequest(name=name, arguments=args))
        schema = build_state_schema(p, host.registry, (), host.observations)
        def final(quote, source='obs-1'):
            return json.dumps({'kind': 'final', 'status': 'PENDING_APPROVAL', 'action': p.action,
                'findings': [{'key': 'refund_cents', 'value': 9000, 'source_ids': ['obs-1']}, {'key': 'requires_approval', 'value': True, 'source_ids': ['obs-2']}],
                'citations': [{'source_id': source, 'quote': quote}, {'source_id': 'obs-2', 'quote': quote_fragments(host.observations[1]['content'])[0]}], 'summary': 's'})
        for bad in ('Customer_id: C-101, refund_cents: 9000, ticket_id: T-101', '"customer_id": "C-101", "refund_cents": 9000, "ticket_id": "T-101"'):
            with self.subTest(quote=bad), self.assertRaises(Exception): schema.model_validate_json(final(bad))
        good = quote_fragments(host.observations[0]['content'])
        self.assertIn('"refund_cents":9000', good)
        step = schema.model_validate_json(final('"refund_cents":9000')); validate_final(step, host, p)
        with self.assertRaises(Exception): schema.model_validate_json(final('"refund_cents":9000', source='obs-2'))   # a fragment of obs-1 cannot be attributed to obs-2
        with self.assertRaises(Exception): schema.model_validate_json(final('x').replace('PENDING_APPROVAL', 'PROPOSED'))   # status is the workflow contract
    def test_every_fragment_is_a_literal_substring(self):
        for text in ('{"a":1,"b":{"c":2}}', 'Customer support {"above_limit":"human approval required","limit":5000}', 'line one\nline two', ''):
            for fragment in quote_fragments(text): self.assertIn(fragment, text)
        self.assertEqual(quote_fragments('{"a":1,"b":2}')[1:], ['"a":1', '"b":2'])
    def test_token_count_handles_batch_encoding_and_rejects_implausible_counts(self):
        profile = ModelProfile('fake', 'fake', context_tokens=8192)
        class Batch(dict): pass
        class Tok:
            chat_template = 't'; init_kwargs = {}
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=False):
                return Batch(input_ids=list(range(300)), attention_mask=[1] * 300) if self.mode == 'dict' else [list(range(300))]
        tok = Tok(); tok.mode = 'dict'; messages = [{'role': 'user', 'content': 'x' * 1200}]
        self.assertEqual(TokenProfiler(profile, tok).measure(messages)['prompt_tokens_estimated'], 300)
        tok.mode = 'nested'; self.assertEqual(TokenProfiler(profile, tok).measure(messages)['prompt_tokens_estimated'], 300)
        class Broken(Tok):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=False): return Batch(input_ids=[1], attention_mask=[1])
        # v3.8.x Colab: len(BatchEncoding) == 2 slipped through as a 2-token prompt
        class Legacy(Tok):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=False): return {'a': [1], 'b': [1]}
        with self.assertRaises(ContextError): TokenProfiler(profile, Legacy()).measure([{'role': 'user', 'content': 'x' * 4000}])
    def test_arithmetic_error_lists_what_is_allowed(self):
        with self.assertRaisesRegex(ToolError, 'absolute value <= 100'): arithmetic_value('n * (n - 1) // 2 + 1000', 5)
    def test_repair_feedback_names_expected_arguments_in_v37_mode(self):
        bad = {'kind': 'tool', 'tool': {'name': 'invoice_read', 'arguments': {'invoice_id': 'INV-301'}}}
        state, backend = offline_run('E03', [bad, bad, bad], constrained=False, guided=False)
        feedback = backend.payloads[1]['repair_feedback']
        self.assertIn("expects ['reference']", feedback); self.assertIn('rejected step', feedback)
        self.assertEqual(state['failure_category'], 'tool_argument_schema')
    def test_progress_hint_lists_missing_tools(self):
        _, backend = offline_run('E03')
        self.assertEqual(backend.payloads[0]['progress']['required_tools_not_yet_called'], ['invoice_read', 'receipt_read', 'reconcile'])
        self.assertEqual(backend.payloads[2]['progress']['required_tools_not_yet_called'], ['reconcile'])
        _, plain = offline_run('E03', guided=False)
        self.assertNotIn('progress', plain.payloads[0])
    def test_observations_sent_once(self):
        _, backend = offline_run('E01')
        self.assertNotIn('data', backend.payloads[1]['observations'][0])
    def test_consecutive_repairs_reset_after_successful_tool(self):
        p = PACKS['E03']; steps = offline_tool_steps(p)
        script = [{'invalid': 1}, {'invalid': 1}, steps[0], {'invalid': 1}, {'invalid': 1}, steps[1], steps[2], offline_final('E03')]
        state, _ = offline_run('E03', script)
        self.assertEqual(state['status'], 'PENDING_APPROVAL'); self.assertEqual(state['repairs'], 4)
    def test_v37_errors_classify(self):
        self.assertEqual(classify_failure({'error': 'ToolError: Arguments must match documented nonempty string fields'}), 'tool_argument_schema')
        self.assertEqual(classify_failure({'error': 'step_budget'}), 'step_budget')
    def test_byte_upper_bound_profiler(self):
        profile = api_profile('anthropic:test-model')
        result = ByteUpperBoundProfiler(profile).measure([{'role': 'user', 'content': 'héllo'}])
        self.assertEqual(result['prompt_tokens_upper_bound'], 6 + 64)
        with self.assertRaises(ContextError): ByteUpperBoundProfiler(replace(profile, context_tokens=100)).measure([{'role': 'user', 'content': 'x'}])
    def test_profile_resolution(self):
        self.assertEqual(resolve_profile('qwen2.5:7b').tokenizer_id, 'Qwen/Qwen2.5-7B-Instruct')
        self.assertEqual(resolve_profile('openai:some-model').provider, 'openai')
        with self.assertRaises(ValueError): resolve_profile('mystery:7b')
    def test_ollama_request_uses_constrained_schema_and_keep_alive(self):
        profile = ModelProfile('fake', 'fake', context_tokens=8192)
        backend = OllamaBackend('fake', profile=profile, profiler=TokenProfiler(profile, FakeTokenizer()))
        schema = build_step_schema(PACKS['E01'])
        reply = {'message': {'content': json.dumps({'kind': 'tool', 'tool': {'name': 'ticket_read', 'arguments': {'reference': 'T-101'}}})},
                 'prompt_eval_count': 25, 'eval_count': 9}
        with patch.object(requests, 'post', return_value=FakeResponse(reply)) as post:
            step = backend.chat('orchestrator', 'sys', {'x': 1}, schema)
        body = post.call_args.kwargs['json']
        self.assertEqual(body['format'], schema.model_json_schema()); self.assertEqual(body['keep_alive'], '30m')
        self.assertEqual(step.tool.name, 'ticket_read'); self.assertEqual(backend.records[0]['prompt_token_delta'], 25 - backend.records[0]['preflight']['prompt_tokens_estimated'])
    def test_openai_and_anthropic_adapters_parse_structured_output(self):
        content = {'valid': True, 'request_matches_world': True, 'style_matches': True, 'reason': 'ok'}
        openai = OpenAICompatibleBackend(api_profile('openai:test-model'), api_key='test')
        reply = {'choices': [{'message': {'content': json.dumps(content)}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        with patch.object(requests, 'post', return_value=FakeResponse(reply)) as post:
            self.assertTrue(openai.chat('auditor', 'sys', {}, AuditVerdict).valid)
        self.assertEqual(post.call_args.kwargs['json']['response_format']['type'], 'json_schema')
        anthropic = AnthropicBackend(api_profile('anthropic:test-model'), api_key='test')
        reply = {'content': [{'type': 'tool_use', 'name': 'emit_step', 'input': content}], 'usage': {'input_tokens': 10, 'output_tokens': 5}}
        with patch.object(requests, 'post', return_value=FakeResponse(reply)) as post:
            self.assertTrue(anthropic.chat('auditor', 'sys', {}, AuditVerdict).valid)
        self.assertEqual(post.call_args.kwargs['json']['tool_choice'], {'type': 'tool', 'name': 'emit_step'})
        self.assertEqual(anthropic.records[0]['completion_tokens'], 5)

# %% cell: run-contracts
def run_offline_contracts(verbosity=1):
    suite = unittest.TestSuite()
    for case in (OfflineContractTests, V38RegressionTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return {'kind': 'offline_scripted_contracts_not_model_quality', 'tests': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors), 'passed': result.wasSuccessful()}
