"""Ring 2 - infrastructure: ChatBackend adapters.

All adapters share one call path (budget check -> token preflight -> HTTP -> schema
validation -> call record) so local and API models are compared under identical rules.
"""

# %% cell: backend-base
import os, time
import requests
from tago.config import THINK
from tago.core.domain import canonical, digest, BudgetError, ContextError
from tago.infrastructure.profiles import ModelProfile, MODEL_PROFILES, TokenProfiler, ByteUpperBoundProfiler, resolve_profile

def secret(name):
    """Read an API key from the environment or, on Colab, from the Secrets panel."""
    value = os.environ.get(name)
    if value: return value
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None

class BaseBackend:
    provider = 'base'
    def __init__(self, profile, max_calls=12000, max_seconds=7200, timeout=180, profiler=None):
        self.profile = profile
        self.model = profile.model
        self.profiler = profiler or self._default_profiler()
        self.max_calls, self.max_seconds, self.timeout = max_calls, max_seconds, timeout
        self.calls, self.started, self.records = 0, time.monotonic(), []
        self.warmup_record = None
        self.server_template_hash = None
    def _default_profiler(self):
        return ByteUpperBoundProfiler(self.profile)
    def _messages(self, system, payload, schema_json):
        # The schema is included in the sent prompt, so its token cost is counted by preflight.
        return [{'role': 'system', 'content': system + '\nOUTPUT JSON SCHEMA:\n' + canonical(schema_json)},
                {'role': 'user', 'content': canonical(payload)}]
    def _send(self, messages, schema_json, seed, temperature):
        raise NotImplementedError
    def chat(self, role, system, payload, schema, seed=0, temperature=0):
        if self.calls >= self.max_calls or time.monotonic() - self.started >= self.max_seconds:
            raise BudgetError('Call/time budget exhausted')
        schema_json = schema.model_json_schema()
        messages = self._messages(system, payload, schema_json)
        preflight = self.profiler.measure(messages)
        self.calls += 1
        rec = {'role': role, 'provider': self.provider, 'model': self.model, 'seed': seed, 'call': self.calls,
               'preflight': preflight, 'prompt_hash': digest(messages), 'schema_hash': digest(schema_json),
               'server_template_hash': self.server_template_hash}
        start = time.monotonic()
        try:
            raw, usage = self._send(messages, schema_json, seed, temperature)
            rec.update(raw=raw, prompt_tokens=usage.get('prompt'), completion_tokens=usage.get('completion'))
            estimate = preflight.get('prompt_tokens_estimated')
            if rec['prompt_tokens'] is not None and estimate is not None:
                rec['prompt_token_delta'] = rec['prompt_tokens'] - estimate
            return schema.model_validate_json(raw)
        except Exception as error:
            rec['error'] = f'{type(error).__name__}: {error}'; raise
        finally:
            rec['latency_sec'] = time.monotonic() - start; self.records.append(rec)

# %% cell: backend-ollama
class OllamaBackend(BaseBackend):
    """Structured transport with explicit model profile and tokenizer preflight."""
    provider = 'ollama'
    def __init__(self, model, base_url='http://127.0.0.1:11434', max_calls=12000, max_seconds=7200,
                 timeout=180, profile=None, profiler=None):
        profile = profile or MODEL_PROFILES.get(model)
        if profile is None: raise ValueError(f'Register a ModelProfile for {model!r}; tokenizer mapping is not guessed')
        if profile.model != model: raise ValueError('Profile/model mismatch')
        self.base_url = base_url.rstrip('/')
        super().__init__(profile, max_calls, max_seconds, timeout, profiler)
        self.think = THINK   # per-backend so the campaign can compare 'default' and 'off' on one thinking model
    def _default_profiler(self):
        return TokenProfiler(self.profile)
    def healthcheck(self):
        r = requests.get(self.base_url + '/api/tags', timeout=5); r.raise_for_status()
        names = [m['name'] for m in r.json()['models']]
        if self.model not in names and self.model + ':latest' not in names: raise RuntimeError('Configured model is not installed')
        show = requests.post(self.base_url + '/api/show', json={'model': self.model}, timeout=30)
        show.raise_for_status(); self.server_template_hash = digest(show.json().get('template', ''))
        return True
    def warmup(self, timeout=900):
        """Load weights into GPU memory before timed calls. v37's first call took 119 s (timeout was 120 s)."""
        start = time.monotonic()
        # v38.2: load with the SAME num_ctx the chat calls use. The v38.1 Colab run warmed with Ollama's default context,
        # then the first chat call (num_ctx 8192) forced a reload: warm-up 68 s, first timed call still 79.6 s.
        r = requests.post(self.base_url + '/api/generate', json={'model': self.model, 'prompt': '', 'keep_alive': '30m',
                          'options': {'num_ctx': self.profile.context_tokens}}, timeout=timeout)
        r.raise_for_status()
        loaded = time.monotonic() - start
        # Second step: one tiny structured chat call. The v3.8.2 Colab run still paid 94 s on its first timed chat call after a
        # 71 s load, so the chat path (template, grammar, sampler setup) is exercised before any timed call.
        body = {'model': self.model, 'stream': False, 'keep_alive': '30m', 'format': {'type': 'object'},
                'messages': [{'role': 'user', 'content': 'Reply with an empty JSON object.'}],
                'options': {'temperature': 0, 'seed': 0, 'num_ctx': self.profile.context_tokens, 'num_predict': 16}}
        if self.think == 'off': body['think'] = False
        chat_error = None
        try: requests.post(self.base_url + '/api/chat', json=body, timeout=timeout).raise_for_status()
        except requests.RequestException as error: chat_error = f'{type(error).__name__}: {error}'   # best effort; real calls report their own errors
        self.warmup_record = {'model': self.model, 'latency_sec': time.monotonic() - start, 'load_sec': loaded, 'chat_warmup_error': chat_error}
        return self.warmup_record
    def unload(self):
        """Free GPU memory before the next model in a comparison matrix."""
        requests.post(self.base_url + '/api/generate', json={'model': self.model, 'keep_alive': 0}, timeout=60)
    def _send(self, messages, schema_json, seed, temperature):
        body = {'model': self.model, 'stream': False, 'keep_alive': '30m',
                'format': schema_json, 'messages': messages,
                'options': {'temperature': temperature, 'seed': seed, 'num_ctx': self.profile.context_tokens,
                            'num_predict': self.profile.output_tokens}}
        if self.think == 'off': body['think'] = False
        r = requests.post(self.base_url + '/api/chat', json=body, timeout=self.timeout)
        r.raise_for_status(); data = r.json()
        return data['message']['content'], {'prompt': data.get('prompt_eval_count'), 'completion': data.get('eval_count')}

# %% cell: backend-apis
class OpenAICompatibleBackend(BaseBackend):
    """OpenAI Chat Completions with json_schema response format.

    Works with OpenAI and compatible servers (vLLM, LM Studio, OpenRouter, Ollama's /v1) via OPENAI_BASE_URL.
    strict=False because the step schema uses optional fields that strict mode would reject.
    """
    provider = 'openai'
    def __init__(self, profile, api_key=None, **kwargs):
        self.api_key = api_key or secret('OPENAI_API_KEY')
        if not self.api_key: raise RuntimeError('OPENAI_API_KEY is not set')
        super().__init__(profile, **kwargs)
    def _send(self, messages, schema_json, seed, temperature):
        body = {'model': self.model, 'messages': messages, 'temperature': temperature, 'seed': seed,
                'max_tokens': self.profile.output_tokens,
                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'step', 'schema': schema_json, 'strict': False}}}
        r = requests.post(self.profile.base_url.rstrip('/') + '/chat/completions', json=body, timeout=self.timeout,
                          headers={'Authorization': 'Bearer ' + self.api_key})
        r.raise_for_status(); data = r.json(); usage = data.get('usage') or {}
        return data['choices'][0]['message']['content'], {'prompt': usage.get('prompt_tokens'), 'completion': usage.get('completion_tokens')}

class AnthropicBackend(BaseBackend):
    """Anthropic Messages API; structured output is forced through a single required tool."""
    provider = 'anthropic'
    def __init__(self, profile, api_key=None, **kwargs):
        self.api_key = api_key or secret('ANTHROPIC_API_KEY')
        if not self.api_key: raise RuntimeError('ANTHROPIC_API_KEY is not set')
        super().__init__(profile, **kwargs)
    def _send(self, messages, schema_json, seed, temperature):
        body = {'model': self.model, 'max_tokens': self.profile.output_tokens, 'temperature': temperature,
                'system': messages[0]['content'], 'messages': [{'role': 'user', 'content': messages[1]['content']}],
                'tools': [{'name': 'emit_step', 'description': 'Return the structured result.', 'input_schema': schema_json}],
                'tool_choice': {'type': 'tool', 'name': 'emit_step'}}
        r = requests.post(self.profile.base_url.rstrip('/') + '/messages', json=body, timeout=self.timeout,
                          headers={'x-api-key': self.api_key, 'anthropic-version': '2023-06-01'})
        r.raise_for_status(); data = r.json(); usage = data.get('usage') or {}
        block = next(b for b in data['content'] if b.get('type') == 'tool_use')
        return canonical(block['input']), {'prompt': usage.get('input_tokens'), 'completion': usage.get('output_tokens')}

def make_backend(spec, base_url=None, **budget):
    """Create a backend from a profile name ('qwen2.5:7b') or API spec ('anthropic:<model>')."""
    profile = resolve_profile(spec)
    if profile.provider == 'ollama':
        return OllamaBackend(profile.model, base_url or 'http://127.0.0.1:11434', profile=profile, **budget)
    if profile.provider == 'openai': return OpenAICompatibleBackend(profile, **budget)
    if profile.provider == 'anthropic': return AnthropicBackend(profile, **budget)
    raise ValueError('Unknown provider ' + profile.provider)

# %% cell: backend-scripted
class ScriptedBackend:
    """Deterministic test double: replays a queue of steps. It is NOT a language model."""
    model = 'SCRIPTED_TEST_DOUBLE_NOT_LLM'
    provider = 'scripted'
    def __init__(self, steps): self.steps = list(steps); self.records = []; self.payloads = []
    def chat(self, role, system, payload, schema, seed=0, temperature=0):
        self.payloads.append(payload)
        if not self.steps: raise AssertionError('Scripted queue exhausted')
        step = self.steps.pop(0)
        if isinstance(step, Exception): raise step
        if callable(step): step = step(payload)
        self.records.append({'role': role, 'provider': 'scripted', 'model': self.model, 'call': len(self.records) + 1,
                             'raw': canonical(step), 'latency_sec': 0.0})
        return schema.model_validate(step)
