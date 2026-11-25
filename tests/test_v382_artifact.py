"""Pins the real prompt-v38.2 Colab T4 run (results/v38_2_colab_t4_qwen2.5-7b) to the v38.3 fixes.

The model outputs are real; the assertions state which of them the new schemas and validators make impossible.
"""
import copy, json
from pathlib import Path
import pytest
from tago.core.domain import CaseError, ToolRequest, quote_fragments
from tago.application.packs import PACKS, build_state_schema
from tago.application.orchestrator import validate_final
from tago.infrastructure.tools import RetrievalAdapter
from tago.workflows.analysis import run_summary

RUN = Path(__file__).resolve().parents[1] / 'results' / 'v38_2_colab_t4_qwen2.5-7b'
pytestmark = pytest.mark.skipif(not RUN.exists(), reason='artifact not committed')

def golden(): return json.loads((RUN / 'golden_results.json').read_text(encoding='utf-8'))
def calls(): return json.loads((RUN / 'model_calls.json').read_text(encoding='utf-8'))
def cases(): return [json.loads(l) for l in (RUN / 'synthetic' / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]

def test_run_summary_matches_the_documented_result():
    s = run_summary(RUN)
    assert (s['passed'], s['total']) == (0, 3)
    assert {c['failure_category'] for c in s['cases']} == {'citation_grounding'}
    assert all(c['tool_calls_ok'] >= 2 and c['duplicate_calls'] == 0 for c in s['cases'])   # tools worked; the earlier failure classes are gone
    assert all(r['state']['model_calls'] <= 7 for r in golden())

def test_every_golden_finding_value_was_correct():
    for row in golden():
        checks = row['evaluation']['checks']
        assert checks['required_tools'] and checks['no_external_execution']
    finals = [json.loads(c['raw']) for c in calls() if json.loads(c['raw']).get('kind') == 'final']
    values = {(f['key'], json.dumps(f['value'])) for step in finals for f in step['findings']}
    assert ('overcharge_cents', '5000') in values and ('payable_cents', '25000') in values and ('refund_cents', '9000') in values

def test_all_nine_rejected_finals_are_undecodable_when_quotes_are_grounded():
    finals = [json.loads(c['raw']) for c in calls() if json.loads(c['raw']).get('kind') == 'final']
    assert len(finals) == 9                                    # model_calls.json is ordered by workflow: 3 finals each
    rows = golden(); rejected = 0
    for i, step in enumerate(finals):
        row = rows[i // 3]; observations = row['state']['observations']
        schema = build_state_schema(PACKS[row['workflow_id']], None, (), observations)
        try: schema.model_validate_json(json.dumps(dict(step, tool=None)))
        except Exception: rejected += 1
    assert rejected == 9

def test_a_grounded_final_for_the_same_observations_is_accepted():
    row = next(r for r in golden() if r['workflow_id'] == 'E03'); pack = PACKS['E03']
    host = pack.tools(copy.deepcopy(pack.golden_world), RetrievalAdapter('lexical'))
    for name in pack.required_tools: host.call(ToolRequest(name=name, arguments={'reference': pack.golden_world['invoice_id']}))
    quote = '"overcharge_cents":5000'
    assert quote in quote_fragments(host.observations[2]['content'])
    schema = build_state_schema(pack, host.registry, (), host.observations)
    step = schema.model_validate_json(json.dumps({'kind': 'final', 'status': pack.status, 'action': pack.action,
        'findings': [{'key': 'overcharge_cents', 'value': 5000, 'source_ids': ['obs-3']}, {'key': 'payable_cents', 'value': 25000, 'source_ids': ['obs-3']}],
        'citations': [{'source_id': f'obs-{i}', 'quote': quote_fragments(host.observations[i - 1]['content'])[0]} for i in (1, 2)] + [{'source_id': 'obs-3', 'quote': quote}], 'summary': 'ok'}))
    validate_final(step, host, pack)

def test_synthetic_sweep_reached_the_orchestrator_for_every_case():
    rows = cases()
    assert len(rows) == 6 and all(r['outcome'] == 'orchestrator_failure' for r in rows)
    assert {r['evaluation']['failure_category'] for r in rows} == {'wrong_reference', 'wrong_status', 'citation_grounding', 'tool_refused'}

def test_the_two_invalid_cases_the_auditor_accepted_are_now_rejected():
    e01 = next(r for r in cases() if r['workflow_id'] == 'E01' and r['mode'] == 'clean')
    assert e01['audit']['accepted'] is True                     # the same-model auditor passed a placeholder id and contradictory figures
    with pytest.raises(CaseError): PACKS['E01'].validate_synthetic(e01['case'])

def test_wrong_status_cannot_be_decoded():
    row = next(r for r in cases() if r['workflow_id'] == 'E01' and r['mode'] == 'noisy')
    assert row['state']['decision']['status'] == 'PROPOSED'    # every value, action and citation was right; the status was not
    pack = PACKS['E01']
    host = pack.tools(copy.deepcopy(pack.golden_world), RetrievalAdapter('lexical'))
    for name, args in (('ticket_read', {'reference': 'T-101'}), ('retrieve', {'query': 'refund policy'})): host.call(ToolRequest(name=name, arguments=args))
    schema = build_state_schema(pack, host.registry, (), host.observations)
    decision = dict(row['state']['decision'], citations=[{'source_id': 'obs-1', 'quote': '"refund_cents":9000'}], findings=[
        {'key': 'refund_cents', 'value': 9000, 'source_ids': ['obs-1']}, {'key': 'requires_approval', 'value': True, 'source_ids': ['obs-2']}])
    with pytest.raises(Exception): schema.model_validate_json(json.dumps(decision))
    decision['status'] = 'PENDING_APPROVAL'
    assert schema.model_validate_json(json.dumps(dict(decision, citations=decision['citations'] + [{'source_id': 'obs-2', 'quote': quote_fragments(host.observations[1]['content'])[0]}]))).status == 'PENDING_APPROVAL'

def test_preflight_bug_is_visible_in_the_artifact():
    # prompt_tokens_estimated was 2 for every call: len() of a BatchEncoding counted its keys.
    assert {c['preflight']['prompt_tokens_estimated'] for c in calls()} == {2}
    assert all(c['prompt_tokens'] > 700 for c in calls())

def test_first_call_latency_after_warmup_is_recorded():
    runtime = json.loads((RUN / 'runtime.json').read_text(encoding='utf-8'))
    assert runtime['prompt_version'] == 'v38.2' and runtime['warmup']['latency_sec'] > 60
    assert calls()[0]['latency_sec'] > 60      # still slow after the context-matched warm-up: motivates the chat warm-up step
