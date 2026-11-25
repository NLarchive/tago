"""Pins the real v3.8.1 Colab T4 run (results/v38_1_colab_t4_qwen2.5-7b) to the v3.8.2 fixes.

The raw model outputs are real; the assertions say which of them the new schemas make undecodable.
"""
import json, copy
from pathlib import Path
import pytest
from tago.application.packs import PACKS, build_state_schema
from tago.application.orchestrator import allowed_tools
from tago.infrastructure.tools import RetrievalAdapter

RUN = Path(__file__).resolve().parents[1] / 'results' / 'v38_1_colab_t4_qwen2.5-7b'
pytestmark = pytest.mark.skipif(not RUN.exists(), reason='artifact not committed')

def calls(name):
    return json.loads((RUN / name).read_text(encoding='utf-8'))

def schemas(pack):
    host = pack.tools(copy.deepcopy(pack.golden_world), RetrievalAdapter('lexical'))
    return (build_state_schema(pack, host.registry, tuple(pack.required_tools)), build_state_schema(pack, host.registry, ()))

def test_run_is_the_documented_0_of_3():
    rows = calls('golden_results.json')
    assert [r['evaluation']['ok'] for r in rows] == [False, False, False]
    assert {r['evaluation']['failure_category'] for r in rows} == {'output_schema'}
    assert all(r['evaluation']['checks']['required_tools'] for r in rows)   # v37's tool-argument failures are gone

def test_every_envelope_failure_is_undecodable_under_the_new_schemas():
    rows = calls('golden_results.json'); raw = [c['raw'] for c in calls('model_calls.json')]
    broken = [json.loads(r) for r in raw if json.loads(r).get('kind') == 'tool' and json.loads(r).get('tool') is None]
    broken += [json.loads(r) for r in raw if json.loads(r).get('kind') == 'final' and 'status' not in json.loads(r)]
    assert len(broken) == 4                      # E01 call 5, E03 calls 9 and 11, E06 call 18
    for pack in (PACKS['E01'], PACKS['E03'], PACKS['E06']):
        tool_only, final_only = schemas(pack)
        for step in broken:
            for schema in (tool_only, final_only):
                with pytest.raises(Exception): schema.model_validate_json(json.dumps(step))

def test_valid_v381_tool_calls_stay_valid():
    pack = PACKS['E03']; tool_only, _ = schemas(pack)
    ok = {'kind': 'tool', 'tool': {'name': 'invoice_read', 'arguments': {'reference': 'INV-301'}}}
    assert tool_only.model_validate_json(json.dumps(ok)).tool.name == 'invoice_read'

def test_description_leak_in_synthetic_run_is_undecodable():
    leaked = [json.loads(c['raw']) for c in calls_synth() if 'exact case reference identifier' in c['raw']]
    assert len(leaked) == 6                      # E03 clean (3 calls) and noisy (3 calls)
    tool_only, _ = schemas(PACKS['E03'])
    for step in leaked:
        with pytest.raises(Exception): tool_only.model_validate_json(json.dumps(step))

def calls_synth():
    return json.loads((RUN / 'synthetic' / 'model_calls.json').read_text(encoding='utf-8'))

def test_synthetic_generation_errors_were_case_validation_errors():
    rows = [json.loads(l) for l in (RUN / 'synthetic' / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    errors = [r['error'] for r in rows if r['outcome'] == 'generation_error']
    assert len(errors) == 3 and all(('CaseError' in e or 'ValidationError' in e) for e in errors)

def test_hash_prefixed_reference_from_the_run_is_now_accepted():
    # E06 noisy: request said '#calc123', world issue_id 'calc123'; v38.1 sent '#calc123' and was refused 3 times.
    rows = [json.loads(l) for l in (RUN / 'synthetic' / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    case = next(r['case'] for r in rows if r['workflow_id'] == 'E06' and r.get('case'))
    pack = PACKS['E06']; host = pack.tools(case['world'], RetrievalAdapter('lexical'))
    from tago.core.domain import ToolRequest
    assert host.call(ToolRequest(name='code_read', arguments={'reference': '#' + case['world']['issue_id']}))
