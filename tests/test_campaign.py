"""The experiment campaign (notebook section 11.6) runs offline with scripted backends: structure, isolation and settings."""
import json
from pathlib import Path
from tago.application.packs import PACKS
from tago.infrastructure.backends import ScriptedBackend
from tago.infrastructure.tools import RetrievalAdapter
from tago.workflows.experiments import run_campaign, campaign_markdown, ABLATION_LADDER

def make(tmp_path, **kw):
    log = {'made': [], 'pulled': [], 'removed': []}
    def factory(spec, warm=True):
        log['made'].append(spec)
        if spec == 'broken:1b': raise RuntimeError('out of memory')
        return ScriptedBackend([])
    args = dict(packs=[PACKS['E01']], retriever=RetrievalAdapter('lexical'), runtime_info={}, model='qwen2.5:7b', ablation=True, models=(), thinking_models=(),
                auditor=None, directory=tmp_path, backend_factory=factory, pull=lambda s: log['pulled'].append(s) or True, remove=lambda s: log['removed'].append(s))
    args.update(kw)
    return run_campaign(**args), log

def test_ablation_ladder_records_the_switches_each_rung_really_used(tmp_path):
    result, _ = make(tmp_path)
    used = [i['settings'] for i in result['items']]
    assert [(u['constrained_schema'], u['guided_progress'], u['state_conditioned'], u['grounded_quotes']) for u in used] == \
        [(False, False, False, False), (True, True, False, False), (True, True, True, False), (True, True, True, True)]
    for item in result['items']:
        runtime = json.loads((Path(item['path']) / 'runtime.json').read_text(encoding='utf-8'))
        assert runtime['settings']['grounded_quotes'] == item['settings']['grounded_quotes']
    assert len(ABLATION_LADDER) == 4 and (Path(result['path']) / 'campaign_summary.json').exists()

def test_a_broken_model_is_recorded_and_the_campaign_continues(tmp_path):
    result, log = make(tmp_path, ablation=False, models=('broken:1b', 'qwen2.5:3b'))
    assert [i['name'] for i in result['items']] == ['broken_1b', 'qwen2.5_3b']
    assert 'out of memory' in result['items'][0]['error'] and result['items'][1]['total'] == 1
    assert log['removed'] == ['broken:1b', 'qwen2.5:3b']        # downloaded models are deleted again to keep the Colab disk free

def test_thinking_models_run_with_think_default_and_off(tmp_path):
    result, log = make(tmp_path, ablation=False, thinking_models=('qwen3.5:9b',))
    assert [(i['name'], i['think']) for i in result['items']] == [('qwen3.5_9b-think-default', 'default'), ('qwen3.5_9b-think-off', 'off')]
    assert log['removed'] == ['qwen3.5:9b']                     # removed once, after the last variant

def test_time_budget_skips_items_instead_of_starting_them(tmp_path):
    result, _ = make(tmp_path, models=('qwen2.5:3b',), max_minutes=0)
    assert all(i.get('skipped') == 'time budget' for i in result['items'])
    assert 'skipped: time budget' in campaign_markdown(result['items'])
