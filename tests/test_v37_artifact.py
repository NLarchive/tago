"""Pin the analysis of the real v3.7 Colab T4 run and prove v3.8's schema rejects its outputs."""
import json
from pathlib import Path
import pytest
from tago.workflows.analysis import run_summary
from tago.application.packs import PACKS, build_step_schema

RUN = Path(__file__).resolve().parents[1] / 'results' / 'v37_colab_t4_qwen2.5-7b'

def test_v37_summary_matches_artifact():
    s = run_summary(RUN)
    assert s['models'] == ['qwen2.5:7b'] and (s['passed'], s['total']) == (0, 3)
    assert s['orchestrator_calls'] == 13 and s['latency']['first_call_sec'] > 100
    assert {c['workflow_id']: c['failure_category'] for c in s['cases']} == {
        'E01': 'tool_argument_schema', 'E03': 'tool_argument_schema', 'E06': 'tool_argument_schema'}
    assert next(c for c in s['cases'] if c['workflow_id'] == 'E01')['duplicate_calls'] == 3

def test_every_rejected_v37_output_is_invalid_under_v38_schema():
    calls = json.loads((RUN / 'model_calls.json').read_text())
    order = ['E01'] * 7 + ['E03'] * 3 + ['E06'] * 3        # call order recorded in the artifact
    rejected = 0
    for wid, call in zip(order, calls):
        raw = json.loads(call['raw'])
        schema = build_step_schema(PACKS[wid])
        if raw['tool']['arguments'] in ({},) or 'invoice_id' in raw['tool']['arguments']:
            with pytest.raises(Exception): schema.model_validate_json(call['raw'])
            rejected += 1
        else:
            assert schema.model_validate_json(call['raw']).tool.name == 'ticket_read'
    assert rejected == 9   # 3x retrieve {}, 3x invoice_read{invoice_id}, 3x code_read {}
