"""Pins the real prompt-v38.4 Colab T4 run (results/v38_4_colab_t4_qwen2.5-7b): the first run with every golden case passing."""
import json
from pathlib import Path
import pytest
from tago.workflows.analysis import run_summary

RUN = Path(__file__).resolve().parents[1] / 'results' / 'v38_4_colab_t4_qwen2.5-7b'
pytestmark = pytest.mark.skipif(not RUN.exists(), reason='artifact not committed')
load = lambda p: json.loads((RUN / p).read_text(encoding='utf-8'))
cases = lambda: [json.loads(l) for l in (RUN / 'synthetic' / 'cases.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

def test_golden_three_of_three_pass_without_repairs():
    s = run_summary(RUN)
    assert (s['passed'], s['total']) == (3, 3)
    rows = load('golden_results.json')
    assert [r['state']['repairs'] for r in rows] == [0, 0, 0]
    assert [r['state']['status'] for r in rows] == ['PENDING_APPROVAL', 'PENDING_APPROVAL', 'PROPOSED']

def test_e06_final_cites_every_observation_it_names():
    row = load('golden_results.json')[2]['state']
    assert [c['source_id'] for c in row['decision']['citations']] == ['obs-1', 'obs-2', 'obs-3']   # Run D cited only obs-3

def test_first_call_is_fast_and_the_preflight_is_real():
    runtime, calls = load('runtime.json'), load('model_calls.json')
    assert runtime['prompt_version'] == 'v38.4' and runtime['warmup']['chat_warmup_error'] is None
    assert calls[0]['latency_sec'] < 5 and runtime['warmup']['latency_sec'] > 100
    assert all(c['preflight']['prompt_tokens_estimated'] == c['prompt_tokens'] for c in calls)

def test_e03_noisy_uses_the_reference_written_in_the_request():
    row = next(r for r in cases() if r['workflow_id'] == 'E03' and r['mode'] == 'noisy')
    assert row['outcome'] == 'pass'
    assert {o['arguments']['reference'] for o in row['state']['observations']} == {'12345'}   # Run D used invoice_id_12345

def test_synthetic_five_of_six_pass_and_the_failure_is_model_arithmetic():
    rows = cases()
    assert sorted((r['workflow_id'], r['mode']) for r in rows if r['outcome'] == 'pass') == \
        [('E01', 'clean'), ('E01', 'noisy'), ('E03', 'clean'), ('E03', 'noisy'), ('E06', 'clean')]
    failed = [r for r in rows if r['outcome'] != 'pass']
    assert [(r['workflow_id'], r['mode'], r['evaluation']['failure_category']) for r in failed] == [('E06', 'noisy', 'step_budget')]
    state = failed[0]['state']
    assert [o['arguments'].get('expression') for o in state['observations'][1:]] == ['n * (n - 1) // 2', 'n * (n - 1) // 2 + 100 - 100']
    assert state['repairs'] == 4 and state['status'] != 'PROPOSED'

def test_known_grading_weakness_a_meaningless_fragment_still_passes_as_evidence():
    # Backlog V39-02: E01 still passes while citing '"mode":"hybrid"' for requires_approval.
    row = load('golden_results.json')[0]
    assert row['evaluation']['ok'] and {'source_id': 'obs-2', 'quote': '"mode":"hybrid"'} in row['state']['decision']['citations']
