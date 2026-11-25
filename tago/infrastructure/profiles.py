"""Ring 2 - infrastructure: model profiles and token preflight.

A profile is an explicit pairing of a served model with the way its prompt size is
bounded *before* the call. Local Ollama models use their real Hugging Face chat template.
API models use a provable UTF-8 byte upper bound (a byte-level BPE token is >= 1 byte),
because their tokenizers are not public; providers then report the real usage.
"""

# %% cell: model-profiles
import os
from dataclasses import dataclass, field, replace
from tago.core.domain import canonical, digest, ContextError, load_public_model

@dataclass(frozen=True)
class ModelProfile:
    """An explicit model/tokenizer pairing; unknown Ollama aliases must be registered."""
    model: str
    tokenizer_id: str | None = None
    tokenizer_revision: str = 'main'
    context_tokens: int = 8192
    output_tokens: int = 1600
    safety_tokens: int = 256
    provider: str = 'ollama'            # ollama | openai | anthropic
    base_url: str | None = None         # API base URL (OpenAI-compatible servers, proxies)
    t4_fit: str = ''                    # human note on whether it fits a 16 GB T4 at Q4
    thinking: str = 'none'              # 'none' | 'hybrid' (reasoning tokens count against output_tokens)
    status: str = 'registered'          # registered | candidate (tag verified on ollama.com, never run here) | measured
    notes: str = ''

MODEL_PROFILES = {
    'qwen2.5:7b': ModelProfile('qwen2.5:7b', 'Qwen/Qwen2.5-7B-Instruct', t4_fit='~4.7 GB Q4_K_M, fits',
                               status='measured', notes='v37 Colab T4 golden run: 0/3 (see results/).'),
    'qwen2.5:3b': ModelProfile('qwen2.5:3b', 'Qwen/Qwen2.5-3B-Instruct', t4_fit='~1.9 GB, fits'),
    'qwen2.5:14b': ModelProfile('qwen2.5:14b', 'Qwen/Qwen2.5-14B-Instruct', t4_fit='~9 GB Q4_K_M, fits; slower'),
    'llama3.1:8b': ModelProfile('llama3.1:8b', 'unsloth/Meta-Llama-3.1-8B-Instruct', t4_fit='~4.9 GB, fits',
                                notes='Ungated tokenizer mirror of meta-llama/Llama-3.1-8B-Instruct.'),
    'qwen3:8b': ModelProfile('qwen3:8b', 'Qwen/Qwen3-8B', t4_fit='~5.2 GB, fits', thinking='hybrid', output_tokens=3000,
                             notes='Hybrid thinking model; structured-output behaviour unverified here.'),
    # v38.2 candidates. Ollama tags were confirmed on ollama.com; the Hugging Face tokenizer ids were NOT opened from
    # the build environment. If an id is wrong, preflight raises ContextError on the first call (never a silent fallback).
    'ministral-3:8b': ModelProfile('ministral-3:8b', 'mistralai/Ministral-3-8B-Instruct-2512', status='candidate',
        t4_fit='6.0 GB Q4_K_M, fits (needs Ollama >= 0.13.1)', notes='Non-thinking instruct, vision-capable. Mistral may ship its template outside tokenizer_config.json.'),
    'ministral-3:14b': ModelProfile('ministral-3:14b', 'mistralai/Ministral-3-14B-Instruct-2512', status='candidate',
        t4_fit='~9 GB Q4_K_M, fits (needs Ollama >= 0.13.1)', notes='Non-thinking instruct.'),
    'granite4.1:8b': ModelProfile('granite4.1:8b', 'ibm-granite/granite-4.1-8b', status='candidate',
        t4_fit='5.3 GB Q4_K_M, fits', notes='Dense 8B, trained for RAG / tool calling / structured JSON; no thinking mode in 4.1.'),
    'phi4': ModelProfile('phi4', 'microsoft/phi-4', status='candidate', t4_fit='~9 GB Q4_K_M, fits', notes='Dense 14B, strong arithmetic.'),
    'gemma4:12b': ModelProfile('gemma4:12b', 'google/gemma-4-12B-it', status='candidate', thinking='hybrid', output_tokens=3000,
        t4_fit='7.6 GB, fits', notes='Thinks by default; with thinking off it still emits an empty thought block. Template may live in chat_template.jinja.'),
    'qwen3.5:9b': ModelProfile('qwen3.5:9b', 'Qwen/Qwen3.5-9B', status='candidate', thinking='hybrid', output_tokens=3000,
        t4_fit='~6.6 GB, fits', notes='Hybrid thinking; forcing think=false has been reported to disable format enforcement on some builds.'),
    'qwen3:14b': ModelProfile('qwen3:14b', 'Qwen/Qwen3-14B', status='candidate', thinking='hybrid', output_tokens=3000, t4_fit='~9.3 GB, fits'),
}

def api_profile(spec):
    """Build a profile for 'openai:<model>' or 'anthropic:<model>' specs (no local tokenizer)."""
    provider, _, model = spec.partition(':')
    if provider not in {'openai', 'anthropic'} or not model:
        raise ValueError(f'Register a ModelProfile for {spec!r}; tokenizer mapping is not guessed')
    base = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1') if provider == 'openai' else 'https://api.anthropic.com/v1'
    return ModelProfile(model, None, context_tokens=100000, output_tokens=2000, provider=provider, base_url=base,
                        notes='API model: byte upper-bound preflight; provider-reported token usage.')

def resolve_profile(spec):
    if isinstance(spec, ModelProfile): return spec
    if spec in MODEL_PROFILES: return MODEL_PROFILES[spec]
    return api_profile(spec)

# %% cell: token-preflight
class TokenProfiler:
    """Count the configured tokenizer's chat template, output reserve and safety margin.

    This is a tokenizer-based estimate of Ollama framing, not proof of byte-identical templates.
    No character-count fallback or silent truncation is permitted.
    """
    _cache = {}
    method = 'hf_chat_template'
    def __init__(self, profile, tokenizer=None):
        self.profile = profile
        if tokenizer is None:
            if not profile.tokenizer_id: raise ContextError('Profile has no tokenizer; use ByteUpperBoundProfiler')
            from transformers import AutoTokenizer
            key = (profile.tokenizer_id, profile.tokenizer_revision)
            if key not in self._cache:
                self._cache[key] = load_public_model(lambda: AutoTokenizer.from_pretrained(profile.tokenizer_id,
                    revision=profile.tokenizer_revision, trust_remote_code=False))
            tokenizer = self._cache[key]
        self.tokenizer = tokenizer
        if not getattr(tokenizer, 'chat_template', None): raise ContextError('Configured tokenizer has no chat template')
    def _count(self, messages):
        # Recent `transformers` versions return a BatchEncoding (a dict of input_ids, attention_mask) unless asked
        # otherwise; len() of that is the number of keys (2), which silently disabled this guard in the v3.8.x Colab runs.
        try: encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        except TypeError: encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(encoded, 'keys'):
            if 'input_ids' not in encoded: raise ContextError('Tokenizer returned an unrecognised structure for apply_chat_template')
            encoded = encoded['input_ids']
        if encoded and isinstance(encoded[0], (list, tuple)): encoded = encoded[0]
        count = len(encoded)
        floor = sum(len(m['content']) for m in messages) // 12   # a token is never shorter than ~1 character; 12 is very conservative
        if count < floor: raise ContextError(f'Token count {count} is implausibly low for {floor * 12}+ characters; tokenizer output format not understood')
        return count
    def measure(self, messages):
        prompt_tokens = self._count(messages)
        reserved = prompt_tokens + self.profile.output_tokens + self.profile.safety_tokens
        result = {'method': self.method, 'prompt_tokens_estimated': prompt_tokens,
                  'output_reserved': self.profile.output_tokens, 'safety_reserved': self.profile.safety_tokens,
                  'context_tokens': self.profile.context_tokens, 'total_reserved': reserved,
                  'tokenizer_id': self.profile.tokenizer_id, 'tokenizer_revision': self.profile.tokenizer_revision,
                  'tokenizer_commit': getattr(self.tokenizer, 'init_kwargs', {}).get('_commit_hash'),
                  'template_hash': digest(self.tokenizer.chat_template)}
        if reserved > self.profile.context_tokens: raise ContextError('Token budget exceeded: ' + canonical(result))
        return result

class ByteUpperBoundProfiler:
    """Provable upper bound for API models: tokens <= UTF-8 bytes + a per-message framing allowance."""
    method = 'utf8_byte_upper_bound'
    FRAMING_PER_MESSAGE = 64
    def __init__(self, profile): self.profile = profile
    def measure(self, messages):
        bound = sum(len(m['content'].encode('utf-8')) + self.FRAMING_PER_MESSAGE for m in messages)
        reserved = bound + self.profile.output_tokens + self.profile.safety_tokens
        result = {'method': self.method, 'prompt_tokens_upper_bound': bound,
                  'output_reserved': self.profile.output_tokens, 'safety_reserved': self.profile.safety_tokens,
                  'context_tokens': self.profile.context_tokens, 'total_reserved': reserved, 'tokenizer_id': None}
        if reserved > self.profile.context_tokens: raise ContextError('Token budget exceeded: ' + canonical(result))
        return result
