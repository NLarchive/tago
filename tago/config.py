"""Run switches, model choices and budgets.

Every value can be overridden with an environment variable, so the same code runs
in three places without edits:

* Google Colab ("Run all"): live mode is ON by default.
* A local/CI checkout (``pytest``, ``python -m tago``): live mode is OFF by default.
* The generated notebook: this cell is inlined and you can edit the values directly.
"""

# %% cell: config
import os, sys
from pathlib import Path

def env_flag(name, default):
    """Read a boolean switch such as TAGO_RUN_LIVE=1/0/true/false."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

IN_COLAB = Path('/content').exists() and 'google.colab' in sys.modules or bool(os.environ.get('COLAB_RELEASE_TAG'))

VERSION = '3.8'
PROMPT_VERSION = 'v38.4'

# --- What to run -------------------------------------------------------------
RUN_OFFLINE_TESTS = env_flag('TAGO_RUN_OFFLINE_TESTS', True)   # scripted contract suite, no model
REQUIRE_OFFLINE_PASS = env_flag('TAGO_REQUIRE_OFFLINE_PASS', True)
RUN_LIVE = env_flag('TAGO_RUN_LIVE', IN_COLAB)                  # real model calls (Colab default ON)
RUN_SYNTHETIC = env_flag('TAGO_RUN_SYNTHETIC', True)
REQUIRE_GOLDEN_PASS = env_flag('TAGO_REQUIRE_GOLDEN_PASS', False)  # v38: synthetic still runs after golden failures
GUIDED_PROGRESS = env_flag('TAGO_GUIDED_PROGRESS', True)        # v38: show called/not-yet-called tools to the worker
CONSTRAINED_SCHEMA = env_flag('TAGO_CONSTRAINED_SCHEMA', True)  # v38: per-pack tool/argument schema (False = v37 free dict)
THINK = os.environ.get('TAGO_THINK', 'default')   # default = do not send `think` (safest for `format`); off = send think=false (may disable format on some builds)
STATE_CONDITIONED = env_flag('TAGO_STATE_CONDITIONED', True)  # v38.2: only the legal step branch is decodable (False = v38.1 flat schema)
GROUNDED_QUOTES = env_flag('TAGO_GROUNDED_QUOTES', True)  # v38.3: a citation quote can only be a verbatim fragment of the cited observation (needs STATE_CONDITIONED)
GENERATION_ATTEMPTS = int(os.environ.get('TAGO_GENERATION_ATTEMPTS', '3'))  # v38.2: generator retries with validation feedback
N_CLEAN = int(os.environ.get('TAGO_N_CLEAN', '1'))
N_NOISY = int(os.environ.get('TAGO_N_NOISY', '1'))
SELECTED_WORKFLOWS = os.environ.get('TAGO_WORKFLOWS', 'E01,E03,E06').split(',')
BASE_SEED = int(os.environ.get('TAGO_SEED', '3800'))
SPLIT = 'evaluation'

# --- Runtime and models ------------------------------------------------------
RUNTIME_PLUGIN = os.environ.get('TAGO_RUNTIME', 'colab_ollama')  # colab_ollama | existing_ollama | api
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')
MODEL = os.environ.get('TAGO_MODEL', 'qwen2.5:7b')
GENERATOR_MODEL = os.environ.get('TAGO_GENERATOR_MODEL', MODEL)
AUDITOR_MODEL = os.environ.get('TAGO_AUDITOR_MODEL', MODEL)
ORCHESTRATOR_MODEL = os.environ.get('TAGO_ORCHESTRATOR_MODEL', MODEL)
RETRIEVAL_MODE = os.environ.get('TAGO_RETRIEVAL', 'hybrid')   # hybrid = lexical + CPU embeddings; lexical = none
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Optional comparison matrix (section 11.2). Each entry names a registered ModelProfile.
# API entries need OPENAI_API_KEY / ANTHROPIC_API_KEY (or Colab secrets) and are skipped otherwise.
RECOMMENDED_MATRIX = ['granite4.1:8b', 'ministral-3:14b', 'gemma4:12b', 'qwen3.5:9b']   # 3 non-thinking-ish + 2 hybrid-thinking families
_matrix = os.environ.get('TAGO_MODEL_MATRIX', '')
MODEL_MATRIX = RECOMMENDED_MATRIX if _matrix == 'recommended' else [m for m in _matrix.split(',') if m]

# --- Experiment campaign (section 11.6): the extra measurements, run after the main run in the same session ---------------
def _csv(name, default):
    return [m.strip() for m in os.environ.get(name, default).split(',') if m.strip()]
RUN_CAMPAIGN = env_flag('TAGO_RUN_CAMPAIGN', IN_COLAB)                 # on by default in Colab; off elsewhere and in CI
CAMPAIGN_ABLATION = env_flag('TAGO_CAMPAIGN_ABLATION', True)          # four rungs of the switch ladder on the main model
CAMPAIGN_MODELS = _csv('TAGO_CAMPAIGN_MODELS', 'qwen2.5:3b,qwen2.5:14b,llama3.1:8b,granite4.1:8b,ministral-3:14b')   # golden cases per model
CAMPAIGN_THINKING_MODELS = _csv('TAGO_CAMPAIGN_THINKING', 'qwen3.5:9b,gemma4:12b')   # run twice: think default and think off
CAMPAIGN_AUDITOR = os.environ.get('TAGO_CAMPAIGN_AUDITOR', 'granite4.1:8b')          # different-family auditor; empty string skips the study
CAMPAIGN_MAX_MINUTES = int(os.environ.get('TAGO_CAMPAIGN_MAX_MINUTES', '240'))      # no new item starts after this; the rest is recorded as skipped

# --- Budgets -----------------------------------------------------------------
MAX_CALLS_PER_ROLE = int(os.environ.get('TAGO_MAX_CALLS', '12000'))
MAX_SECONDS_PER_ROLE = int(os.environ.get('TAGO_MAX_SECONDS', '7200'))
REQUEST_TIMEOUT_SEC = int(os.environ.get('TAGO_TIMEOUT', '180'))
WARMUP_TIMEOUT_SEC = int(os.environ.get('TAGO_WARMUP_TIMEOUT', '900'))  # v37 cold start took 119 s of a 120 s timeout
MAX_GRAPH_STEPS = int(os.environ.get('TAGO_MAX_STEPS', '24'))
MAX_REPAIRS = int(os.environ.get('TAGO_MAX_REPAIRS', '2'))  # v38: consecutive repairs, reset after a successful tool call
OUTPUT_DIR = Path('/content/tago_v38_runs') if Path('/content').exists() else Path(os.environ.get('TAGO_OUTPUT_DIR', 'tago_v38_runs'))
