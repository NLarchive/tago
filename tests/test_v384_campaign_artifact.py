"""Pins the real experiment campaign (Run F): ablation ladder, other models, thinking models, different-family auditor, repeat."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'results'
CAMPAIGN, REPEAT = ROOT / 'v38_4_campaign_colab_t4', ROOT / 'v38_4_repeat_colab_t4_qwen2.5-7b'
pytestmark = pytest.mark.skipif(not CAMPAIGN.exists(), reason='artifact not committed')
items = lambda: {i['name']: i for i in json.loads((CAMPAIGN / 'campaign_summary.json').read_text(encoding='utf-8'))['items']}
score = lambda i: (i['passed'], i['total'])

def test_ablation_ladder_reproduces_the_four_earlier_runs_on_one_model():
    it = items()
    assert [score(it[n]) for n in ('rung0-v37-behaviour', 'rung1-constrained-guided', 'rung2-state-conditioned', 'rung3-grounded-quotes')] == [(0, 3), (0, 3), (2, 3), (3, 3)]
    assert it['rung0-v37-behaviour']['failure_categories'] == {'duplicate_tool_loop': 3}
    assert it['rung1-constrained-guided']['failure_categories'] == {'duplicate_tool_loop': 2, 'output_schema': 1}
    assert it['rung2-state-conditioned']['failure_categories'] == {'citation_grounding': 1}
    assert it['rung3-grounded-quotes']['failure_categories'] == {}

def test_each_rung_recorded_the_switches_it_used():
    for name, expected in {'rung0-v37-behaviour': (False,) * 4, 'rung1-constrained-guided': (True, True, False, False),
                           'rung2-state-conditioned': (True, True, True, False), 'rung3-grounded-quotes': (True,) * 4}.items():
        run = next(CAMPAIGN.glob(f'golden-{name}'))
        s = json.loads((run / 'runtime.json').read_text(encoding='utf-8'))['settings']
        assert (s['constrained_schema'], s['guided_progress'], s['state_conditioned'], s['grounded_quotes']) == expected

def test_other_local_models_all_fail_on_the_arithmetic_or_evidence_cases():
    it = items()
    assert {n: score(it[n]) for n in ('qwen2.5_3b', 'qwen2.5_14b', 'llama3.1_8b', 'granite4.1_8b', 'ministral-3_14b')} == \
        {'qwen2.5_3b': (2, 3), 'qwen2.5_14b': (2, 3), 'llama3.1_8b': (1, 3), 'granite4.1_8b': (2, 3), 'ministral-3_14b': (3, 3)}
    assert it['qwen2.5_14b']['failure_categories'] == {'duplicate_tool_loop': 1}          # E06: size alone does not fix it
    assert it['granite4.1_8b']['failure_categories'] == {'missing_provenance': 1}

def test_thinking_models_keep_the_constraint_and_thinking_costs_time():
    it = items()
    for name in ('qwen3.5_9b', 'gemma4_12b'):
        on, off = it[name + '-think-default'], it[name + '-think-off']
        assert score(on) == score(off) == (3, 3) and on['failure_categories'] == off['failure_categories'] == {}
        assert on['first_call_sec'] > 2 * off['first_call_sec'] and on['minutes'] > 2 * off['minutes']

def test_auditor_study_does_not_separate_auditors_at_this_sample_size():
    a = items()['auditor-granite4.1_8b']
    assert (a['passed'], a['total']) == (4, 6) and a['failure_categories'] == {'tool_refused': 1, 'step_budget': 1}
    assert not [i for i in items().values() if i.get('error') or i.get('skipped')]

def test_main_run_repeats_run_e():
    rows = json.loads((REPEAT / 'golden_results.json').read_text(encoding='utf-8'))
    assert [r['evaluation']['ok'] for r in rows] == [True] * 3
    cases = [json.loads(l) for l in (REPEAT / 'synthetic' / 'cases.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    assert [(c['workflow_id'], c['mode'], c['evaluation']['failure_category']) for c in cases if c['outcome'] != 'pass'] == [('E06', 'noisy', 'step_budget')]
