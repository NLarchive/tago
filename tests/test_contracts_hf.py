import os
import pytest
from tago.core.domain import load_public_model

def test_expired_token_is_dropped_and_download_retried(monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'expired')
    calls = []
    def load():
        calls.append(os.environ.get('HF_TOKEN'))
        if os.environ.get('HF_TOKEN'): raise RuntimeError("401 Client Error ... User Access Token 'x' is expired")
        return 'model'
    assert load_public_model(load) == 'model' and calls == ['expired', None]

def test_unrelated_errors_are_not_swallowed():
    with pytest.raises(ValueError): load_public_model(lambda: (_ for _ in ()).throw(ValueError('bad shape')))

def test_still_failing_without_token_gives_clear_error(monkeypatch):
    def load(): raise RuntimeError('401 Unauthorized')
    with pytest.raises(RuntimeError, match='even without a token'): load_public_model(load)
