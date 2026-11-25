"""Pins the real prompt-v38.3 Colab T4 run (results/v38_3_colab_t4_qwen2.5-7b): the first run with passing cases."""
import copy, json
from pathlib import Path
import pytest
from tago.core.domain import ToolRequest, quote_fragments
from tago.application.packs import PACKS, build_state_schema, reference_candidates
from tago.infrastructure.tools import RetrievalAdapter, TOOL_ARGUMENTS
from tago.workflows.analysis import run_summary

RUN = Path(__file__).resolve().parents[1] / 'results' / 'v38_3_colab_t4_qwen2.5-7b'
pytestmark = pytest.mark.skipif(not RUN.exists(), reason='artifact not committed')
load = lambda p: json.loads((RUN / p).read_text(encoding='utf-8'))
cases = lambda: [json.loads(l) for l in (RUN / 'synthetic' / 'cases.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

def test_golden_two_of_three_pass_and_the_failure_is_one_missing_citation():
    s = run_summary(RUN)
    assert (s['passed'], s['total']) == (2, 3)
    assert [(c['workflow_id'], c['failure_category']) for c in s['cases'] if not c['ok']] == [('E06', 'citation_grounding')]
    e06 = load('golden_results.json')[2]['state']
    assert 'cited observed evidence: operation' in e06['error'] and e06['repairs'] == 3

def test_run_d_first_call_is_fast_and_the_preflight_is_real():
    runtime, calls = load('runtime.json'), load('model_calls.json')
    assert runtime['prompt_version'] == 'v38.3' and runtime['warmup']['chat_warmup_error'] is None
    assert calls[0]['latency_sec'] < 5 and runtime['warmup']['latency_sec'] > 100   # the slow first request now happens inside warm-up
    assert all(c['preflight']['prompt_tokens_estimated'] == c['prompt_tokens'] for c in calls)   # preflight bug fixed (was 2)

def test_the_rejected_e06_final_is_undecodable_under_the_one_citation_per_observation_schema():
    row = load('golden_results.json')[2]; observations = row['state']['observations']
    final = next(json.loads(c['raw']) for c in load('model_calls.json') if json.loads(c['raw']).get('kind') == 'final' and 'inclusive_sum' in c['raw'])
    assert [c['source_id'] for c in final['citations']] == ['obs-3']                    # obs-1 was named by a finding but never cited
    schema = build_state_schema(PACKS['E06'], None, (), observations, row['case']['request'])
    with pytest.raises(Exception): schema.model_validate_json(json.dumps(final))
    fixed = dict(final, citations=[{'source_id': o['id'], 'quote': quote_fragments(o['content'])[0]} for o in observations])
    assert schema.model_validate_json(json.dumps(fixed)).status == 'PROPOSED'

def test_the_wrong_reference_could_not_be_decoded_with_request_identifiers():
    row = next(r for r in cases() if r['workflow_id'] == 'E03' and r['mode'] == 'noisy')
    assert row['evaluation']['failure_category'] == 'wrong_reference'
    request = row['case']['request']
    assert reference_candidates(request) == ('12345', '8500', '6000', '1500')
    schema = build_state_schema(PACKS['E03'], None, ('invoice_read', 'receipt_read', 'reconcile'), [], request)
    bad = {'kind': 'tool', 'tool': {'name': 'invoice_read', 'arguments': {'reference': 'invoice_id_12345'}}}
    with pytest.raises(Exception): schema.model_validate_json(json.dumps(bad))
    bad['tool']['arguments']['reference'] = '12345'
    assert schema.model_validate_json(json.dumps(bad)).tool.arguments.reference == '12345'

def test_the_tool_description_that_leaked_into_a_query_is_gone():
    leaked = next(json.loads(c['raw'])['tool']['arguments']['query'] for c in load('synthetic/model_calls.json') if 'Natural-language' in c['raw'])
    assert leaked.startswith('Natural-language search query') and 'Natural-language' not in TOOL_ARGUMENTS['retrieve']['query']

def test_synthetic_four_of_six_pass_and_the_two_failures_are_diagnosed():
    rows = cases()
    assert sorted((r['workflow_id'], r['mode']) for r in rows if r['outcome'] == 'pass') == [('E01', 'clean'), ('E01', 'noisy'), ('E03', 'clean'), ('E06', 'clean')]
    assert {(r['workflow_id'], r['evaluation']['failure_category']) for r in rows if r['outcome'] != 'pass'} == {('E03', 'wrong_reference'), ('E06', 'duplicate_tool_loop')}
    e06 = next(r for r in rows if r['workflow_id'] == 'E06' and r['mode'] == 'noisy')
    assert e06['generation_attempts'] == 2 and 'maximum_n=100' in e06['generation_errors'][0]   # the deterministic validator rejected the first draft
    assert len({c['raw'] for c in load('synthetic/model_calls.json') if c['call'] in (45, 46, 47, 48)}) == 1   # temperature 0: the same refused call, four times

def test_known_grading_weakness_a_meaningless_fragment_passes_as_evidence():
    # E01 passes while citing '"mode":"hybrid"' for requires_approval. The rubric only checks that a cited observation exists
    # and that the quote is verbatim (backlog V39-02: the quote should contain or imply the finding value).
    row = load('golden_results.json')[0]
    assert row['evaluation']['ok'] and {'source_id': 'obs-2', 'quote': '"mode":"hybrid"'} in row['state']['decision']['citations']
