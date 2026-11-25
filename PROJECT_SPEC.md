# Project Specification: orchestration
Generated: 2026-11-23 19:39
<!-- To update this file, run: python D:\tools\context_compiler\map_project\app.py --root D:\agent\orchestration -->

## Folder structure

.github
.github/workflows
docs
notebooks
results
results/v37_colab_t4_qwen2.5-7b
tago
tago/application
tago/core
tago/gateway
tago/infrastructure
tago/workflows
tests
tools
.github/workflows/ci.yml
.gitignore
docs/v38_review.md
LICENSE
node.tasks.json
notebooks/narrative.md
notebooks/node.tasks.json
notebooks/tago_v38_agent_graph_lab.ipynb
pyproject.toml
README.md
results/node.tasks.json
results/v37_colab_t4_qwen2.5-7b/golden_results.json
results/v37_colab_t4_qwen2.5-7b/model_calls.json
results/v37_colab_t4_qwen2.5-7b/README.md
results/v37_colab_t4_qwen2.5-7b/runtime.json
tago/__init__.py
tago/__main__.py
tago/application/__init__.py
tago/application/evaluation.py
tago/application/orchestrator.py
tago/application/packs.py
tago/application/synthetic.py
tago/config.py
tago/core/__init__.py
tago/core/domain.py
tago/core/ports.py
tago/gateway/__init__.py
tago/gateway/project_tasks.py
tago/infrastructure/__init__.py
tago/infrastructure/backends.py
tago/infrastructure/profiles.py
tago/infrastructure/tools.py
tago/node.tasks.json
tago/workflows/__init__.py
tago/workflows/analysis.py
tago/workflows/contracts.py
tago/workflows/experiments.py
tests/test_architecture.py
tests/test_contracts.py
tests/test_docs.py
tests/test_e2e_fake_ollama.py
tests/test_notebook.py
tests/test_project_tasks.py
tests/test_v37_artifact.py
tools/build_notebook.py

---

## File Structures

structure from .gitignore:  (no extractable definitions)

structure from LICENSE:  (no extractable definitions)

structure from README.md:
    [file-summary] TAGO — Tool-Augmented Graph Orchestration lab
    [trust-moderate] [heading-1] # TAGO — Tool-Augmented Graph Orchestration lab
    [trust-moderate] [heading-2] ## What it measures
    [trust-moderate] [heading-2] ## Results
    [trust-moderate] [heading-3] ### Real (measured)
    [trust-moderate] [heading-3] ### What v3.8 changes
    [trust-moderate] [heading-3] ### Expected (hypotheses, not yet measured)
    [trust-moderate] [heading-2] ## Quick start
    [trust-moderate] [heading-2] ## Repository layout
    [trust-moderate] [heading-2] ## Project status and definition of done
    [trust-moderate] [heading-2] ## Limitations
    [trust-moderate] [heading-2] ## History
    [trust-moderate] [heading-2] ## License

structure from node.tasks.json:
    [file-summary] Hexagonal DDD atomic-unit project template. The task graph and the path graph are two views of the same architecture. Task dependency edges follow the same directional rules as the hexagonal rings. Ea
    [json-key] $schema: "https://json-schema.org/draft/2020-12/schema"
    [json-key] template_type: "project_task_template"
    [json-key] version: "3.3"
    [json-key] description: "Hexagonal DDD atomic-unit project template. The task graph a..."
    [json-key] enums: {status, project_status, priority, complexity, dependency_type, +15 more}
    [json-key] project: {name, description, start_date, end_date, status, +13 more}
    [json-key] project_structure: {root, architecture, architecture_guide, dependency_rules, atomic_unit_standard, +1 more}
    [json-key] unit_registry: [3 items]
    [json-key] categories: [6 items]
    [json-key] workers: [2 items]
    [json-key] tasks: [11 items]
    [json-key] required_fields: [15 items]
    [json-key] optional_fields: [40 items]
    [json-key] navigation: {rootModule, modules}

structure from pyproject.toml:  (no extractable definitions)

structure from .github/workflows/ci.yml:
    [file-summary] offline-contracts
    [trust-low] [note] [warning] PyYAML is unavailable; YAML extraction is running in regex fallback mode.
    [trust-low] [yaml-key] name:
    [trust-low] [yaml-key] on:
    [trust-low] [yaml-key] jobs:

structure from docs/v38_review.md:
    [file-summary] TAGO v3.7 review and v3.8 plan
    [trust-moderate] [heading-1] # TAGO v3.7 review and v3.8 plan
    [trust-moderate] [heading-2] ## 1. What the project is
    [trust-moderate] [heading-2] ## 2. Strengths worth keeping
    [trust-moderate] [heading-2] ## 3. Real results (v3.7, the only measurement)
    [trust-moderate] [heading-3] ### Root causes
    [trust-moderate] [heading-2] ## 4. Critique: issues found and their v3.8 status
    [trust-moderate] [heading-2] ## 5. Verification of the v3.8 changes (done here, offline)
    [trust-moderate] [heading-2] ## 6. What is still unknown

structure from notebooks/narrative.md:
    [file-summary] TAGO v3.8 — Tool-Augmented Graph Orchestration lab (Colab T4 ready)
    [trust-moderate] [heading-1] # TAGO v3.8 — Tool-Augmented Graph Orchestration lab (Colab T4 ready)
    [trust-moderate] [heading-2] ## Index
    [trust-moderate] [heading-2] ## 0. Settings and dependencies
    [trust-moderate] [heading-3] ### 0.1 Run switches, models and budgets
    [trust-moderate] [heading-3] ### 0.2 Install missing Python dependencies
    [trust-moderate] [heading-2] ## 1. Capability boundaries
    [trust-moderate] [heading-2] ## 2. Contracts, ports and model backends
    [trust-moderate] [heading-3] ### 2.1 Shared utilities and error types
    [trust-moderate] [heading-3] ### 2.2 The step contract
    [trust-moderate] [heading-3] ### 2.3 Ports
    [trust-moderate] [heading-3] ### 2.4 Model profiles
    [trust-moderate] [heading-3] ### 2.5 Token preflight
    [trust-moderate] [heading-3] ### 2.6 Shared backend call path
    [trust-moderate] [heading-3] ### 2.7 Ollama backend (local T4)
    [trust-moderate] [heading-3] ### 2.8 API backends (optional big models)
    [trust-moderate] [heading-3] ### 2.9 Scripted test double
    [trust-moderate] [heading-2] ## 3. Typed synthetic worlds
    [trust-moderate] [heading-2] ## 4. Workflow packs, prompts and the constrained step schema
    [trust-moderate] [heading-3] ### 4.1 Pack contract
    [trust-moderate] [heading-3] ### 4.2 Specialist personas
    [trust-moderate] [heading-3] ### 4.3 The ten registrations
    [trust-moderate] [heading-3] ### 4.4 Generator, auditor and worker prompts
    [trust-moderate] [heading-3] ### 4.5 Constrained step schema (the main v3.8 fix)
    [trust-moderate] [heading-2] ## 5. Sandbox tools
    [trust-moderate] [heading-3] ### 5.1 Retrieval
    [trust-moderate] [heading-3] ### 5.2 Graph traversal
    [trust-moderate] [heading-3] ### 5.3 Tool contracts
    [trust-moderate] [heading-3] ### 5.4 Registry and authorization
    [trust-moderate] [heading-3] ### 5.5 Tool handlers
    [trust-moderate] [heading-3] ### 5.6 Per-case tool host
    [trust-moderate] [heading-2] ## 6. Synthetic generation and audit agents
    [trust-moderate] [heading-2] ## 7. The bounded agent graph
    [trust-moderate] [heading-3] ### 7.1 Edges and the evidence guard
    [trust-moderate] [heading-3] ### 7.2 The graph runner
    [trust-moderate] [heading-2] ## 8. Hidden oracle, rubric and failure taxonomy
    [trust-moderate] [heading-3] ### 8.1 Oracle
    [trust-moderate] [heading-3] ### 8.2 Rubric
    [trust-moderate] [heading-3] ### 8.3 Failure taxonomy (new in v3.8)
    [trust-moderate] [heading-3] ### 8.4 Statistics and run store
    [trust-moderate] [heading-2] ## 9A. Offline contract suite — no model, GPU or network
    [trust-moderate] [heading-2] ## 10. Runtime provisioning
    [trust-moderate] [heading-3] ### 10.1 Runtime adapter and warmed backends
    [trust-moderate] [heading-3] ### 10.2 Assemble the live lab
    [trust-moderate] [heading-2] ## 11. Live labs and expected vs. real results
    [trust-moderate] [heading-3] ### 11.1 Golden harness
    [trust-moderate] [heading-3] ### 11.2 Synthetic harness
    [trust-moderate] [heading-3] ### 11.3 Model comparison matrix
    [trust-moderate] [heading-3] ### 11.4 Real results so far (v3.7, Colab T4, qwen2.5:7b)
    [trust-moderate] [heading-3] ### 11.5 Expected results (hypotheses to be measured, not results)
    [trust-moderate] [heading-2] ## 12. Analysis, frozen datasets and export
    [trust-moderate] [heading-3] ### 12.1 Analyse runs
    [trust-moderate] [heading-3] ### 12.2 Freeze, replay and export

structure from notebooks/node.tasks.json:
    [file-summary] Hexagonal DDD atomic-unit project template. The task graph and the path graph are two views of the same architecture. Task dependency edges follow the same directional rules as the hexagonal rings. Ea
    [json-key] $schema: "https://json-schema.org/draft/2020-12/schema"
    [json-key] template_type: "project_task_template"
    [json-key] version: "3.3"
    [json-key] description: "Hexagonal DDD atomic-unit project template. The task graph a..."
    [json-key] enums: {status, project_status, priority, complexity, dependency_type, +15 more}
    [json-key] project: {name, description, start_date, end_date, status, +14 more}
    [json-key] project_structure: {root, architecture, path_graph}
    [json-key] unit_registry: [1 items]
    [json-key] categories: [1 items]
    [json-key] workers: [2 items]
    [json-key] tasks: [4 items]
    [json-key] required_fields: [15 items]
    [json-key] optional_fields: [40 items]
    [json-key] navigation: {rootModule, modules}

structure from notebooks/tago_v38_agent_graph_lab.ipynb:
    IN_COLAB = ...  «docstring: none»
    VERSION = ...  «docstring: none»
    PROMPT_VERSION = ...  «docstring: none»
    RUN_OFFLINE_TESTS = ...  «docstring: none»
    REQUIRE_OFFLINE_PASS = ...  «docstring: none»
    RUN_LIVE = ...  «docstring: none»
    RUN_SYNTHETIC = ...  «docstring: none»
    REQUIRE_GOLDEN_PASS = ...  «docstring: none»
    GUIDED_PROGRESS = ...  «docstring: none»
    CONSTRAINED_SCHEMA = ...  «docstring: none»
    N_CLEAN = ...  «docstring: none»
    N_NOISY = ...  «docstring: none»
    SELECTED_WORKFLOWS = ...  «docstring: none»
    BASE_SEED = ...  «docstring: none»
    SPLIT = ...  «docstring: none»
    RUNTIME_PLUGIN = ...  «docstring: none»
    OLLAMA_URL = ...  «docstring: none»
    MODEL = ...  «docstring: none»
    GENERATOR_MODEL = ...  «docstring: none»
    AUDITOR_MODEL = ...  «docstring: none»
    ORCHESTRATOR_MODEL = ...  «docstring: none»
    RETRIEVAL_MODE = ...  «docstring: none»
    EMBEDDING_MODEL = ...  «docstring: none»
    MODEL_MATRIX = ...  «docstring: none»
    MAX_CALLS_PER_ROLE = ...  «docstring: none»
    MAX_SECONDS_PER_ROLE = ...  «docstring: none»
    REQUEST_TIMEOUT_SEC = ...  «docstring: none»
    WARMUP_TIMEOUT_SEC = ...  «docstring: none»
    MAX_GRAPH_STEPS = ...  «docstring: none»
    MAX_REPAIRS = ...  «docstring: none»
    OUTPUT_DIR = ...  «docstring: none»
    HOST_FACTORIES = ...  «docstring: none»
    TOOL_REGISTRIES = ...  «docstring: none»
    MODEL_PROFILES = ...  «docstring: none»
    SPECIALIST_PERSONAS = ...  «docstring: none»
    PACKS = ...  «docstring: none»
    GENERATOR_PROMPT = ...  «docstring: none»
    AUDITOR_PROMPT = ...  «docstring: none»
    COMMON_WORKER_PROMPT = ...  «docstring: none»
    _STEP_SCHEMAS = ...  «docstring: none»
    REFERENCE_ARG = ...  «docstring: none»
    TOOL_ARGUMENTS = ...  «docstring: none»
    TOOL_PURPOSE = ...  «docstring: none»
    TOOL_REGISTRY = ...  «docstring: none»
    SCENARIO_CONSTRAINTS = ...  «docstring: none»
    EDGES = ...  «docstring: none»
    EVIDENCE_REQUIREMENTS = ...  «docstring: none»
    FAILURE_PATTERNS = ...  «docstring: none»
    OFFLINE_ANSWERS = ...  «docstring: none»
    OFFLINE_REPORT = ...  «docstring: none»
    class Strict  «docstring: none»:

    class BudgetError  «docstring: none»:

    class ContextError  «docstring: none»:

    class CaseError  «docstring: none»:

    class ToolError  «docstring: none»:

    class AuditVerdict  «docstring: none»:

    class Citation  «docstring: none»:

    class Finding  «docstring: none»:

    class ToolRequest  «Transport-neutral tool call passed to the tool registry.»:

    class AgentStepBase  «Shared validation for the generic and the per-pack constrained step schemas.»:

    class AgentStep  «v37-compatible generic step. v38 workers use build_step_schema() instead (section 8.1).»:

    class ChatBackend  «Structured-output chat port. Must return an instance of `schema` or raise.»:

    class ToolHost  «Per-case tool host port. `call` returns an observation dict with a stable id.»:

    class ModelProfile  «An explicit model/tokenizer pairing; unknown Ollama aliases must be registered.»:

    class TokenProfiler  «Count the configured tokenizer's chat template, output reserve and safety margin. This is a tokenizer-based estimate of »:

    class ByteUpperBoundProfiler  «Provable upper bound for API models: tokens <= UTF-8 bytes + a per-message framing allowance.»:

    class BaseBackend  «docstring: none»:

    class OllamaBackend  «Structured transport with explicit model profile and tokenizer preflight.»:

    class OpenAICompatibleBackend  «OpenAI Chat Completions with json_schema response format. Works with OpenAI and compatible servers (vLLM, LM Studio, Ope»:

    class AnthropicBackend  «Anthropic Messages API; structured output is forced through a single required tool.»:

    class ScriptedBackend  «Deterministic test double: replays a queue of steps. It is NOT a language model.»:

    class SupportWorld  «docstring: none»:

    class SalesWorld  «docstring: none»:

    class FinanceWorld  «docstring: none»:

    class ITWorld  «docstring: none»:

    class KnowledgeWorld  «docstring: none»:

    class EngineeringWorld  «docstring: none»:

    class Vendor  «docstring: none»:

    class ProcurementWorld  «docstring: none»:

    class LegalWorld  «docstring: none»:

    class SecurityWorld  «docstring: none»:

    class Offer  «docstring: none»:

    class AssistantWorld  «docstring: none»:

    class WorkflowPack  «docstring: none»:

    class RetrievalAdapter  «Lexical overlap, optionally blended 50/50 with CPU sentence-embedding cosine similarity.»:

    class ToolRegistry  «Explicit registration and per-pack authorization; local adapter, not MCP transport.»:

    class EnterpriseSandbox  «Per-case tool host: owns the private world, the observation log and browser page state.»:

    class SyntheticCaseAgent  «Generates one fictional request + world. Invalid output is recorded, never replaced by a fixture.»:

    class DatasetAuditor  «Audits the generated input (not agent output). Same-model auditing is correlated, not independent.»:

    class AgentGraphOrchestrator  «Generic runner: receives a public pack and a tool host; never reads the world or oracle.»:

    class RunStore  «docstring: none»:

    class FakeTokenizer  «docstring: none»:

    class FakeResponse  «docstring: none»:

    class OfflineContractTests  «v37 contract suite, ported unchanged in intent.»:

    class V38RegressionTests  «Each test pins one fix for a failure actually observed in the v37 Colab T4 run.»:

    def env_flag(...)  «Read a boolean switch such as TAGO_RUN_LIVE=1/0/true/false.»
    def canonical(...)  «Deterministic JSON text: sorted keys, no whitespace. Used for hashing and exact citations.»
    def digest(...)  «docstring: none»
    def norm(...)  «Case/accent-insensitive normalisation for lexical retrieval.»
    def tool_request_of(...)  «Convert a generic or constrained tool step into a ToolRequest.»
    def api_profile(...)  «Build a profile for 'openai:<model>' or 'anthropic:<model>' specs (no local tokenizer).»
    def resolve_profile(...)  «docstring: none»
    def secret(...)  «Read an API key from the environment or, on Colab, from the Secrets panel.»
    def make_backend(...)  «Create a backend from a profile name ('qwen2.5:7b') or API spec ('anthropic:<model>').»
    def arithmetic_value(...)  «E06 domain rule. Evaluate n, small int constants, +, -, *, // only. Never calls eval().»
    def register(...)  «docstring: none»
    def build_step_schema(...)  «v38: a per-pack AgentStep whose tool field is a union of exact per-tool argument objects. With Ollama `format` / API jso»
    def graph_traverse(...)  «Breadth-first traversal with a depth bound; returns the real edges walked.»
    def handle_retrieve(...)  «docstring: none»
    def handle_graph_walk(...)  «docstring: none»
    def fields(...)  «Handler factory for plain record reads that expose only the listed world fields.»
    def handle_reconcile(...)  «docstring: none»
    def handle_code_check(...)  «docstring: none»
    def handle_browser_open(...)  «docstring: none»
    def handle_browser_click(...)  «docstring: none»
    def validate_final(...)  «Evidence guard: a final answer must rest on observed tool results with literal citations.»
    def expected_outcome(...)  «docstring: none»
    def evaluate_case(...)  «docstring: none»
    def classify_failure(...)  «Map a failed run to one primary category so runs and models can be compared.»
    def duplicate_calls(...)  «Count successful tool calls that repeated an earlier identical call (v37 had no guard).»
    def wilson(...)  «docstring: none»
    def summarize(...)  «docstring: none»
    def tool_args(...)  «docstring: none»
    def offline_tool_steps(...)  «docstring: none»
    def offline_final(...)  «docstring: none»
    def offline_run(...)  «docstring: none»
    def run_offline_contracts(...)  «docstring: none»
    def prepare_runtime(...)  «Only this function installs Ollama, starts its daemon or pulls models.»
    def live_backend(...)  «Construct, health-check and warm a backend under the configured budgets.»
    def execute_case(...)  «docstring: none»
    def run_golden(...)  «docstring: none»
    def save_golden(...)  «docstring: none»
    def run_synthetic(...)  «docstring: none»
    def run_model_matrix(...)  «Run the same golden cases on several models (local and/or API) under identical rules. Missing API keys or unregistered m»
    def load_run(...)  «docstring: none»
    def _latency(...)  «docstring: none»
    def run_summary(...)  «docstring: none»
    def markdown_report(...)  «docstring: none»
    def freeze_dataset(...)  «docstring: none»
    def replay_frozen(...)  «docstring: none»
    def export_bundle(...)  «docstring: none»
    [notebook-metadata] cells=121; kernel=python3  «Jupyter notebook metadata»
    [notebook-markdown-cell] # TAGO v3.8 — Tool-Augmented Graph Orchestration lab (Colab T4 ready) **What this notebook is.** A self-contained evaluation lab that asks a language model to w  «Markdown notebook cell»
    [heading-1] # TAGO v3.8 — Tool-Augmented Graph Orchestration lab (Colab T4 ready)
    [notebook-markdown-cell] ## Index - [0. Settings and dependencies](#s0) - [1. Capability boundaries](#s1) - [2. Contracts, ports and model backends](#s2) - [3. Typed synthetic worlds](#  «Markdown notebook cell»
    [heading-2] ## Index
    [notebook-markdown-cell] ## 0. Settings and dependencies  «Markdown notebook cell»
    [heading-2] ## 0. Settings and dependencies
    [notebook-markdown-cell] ### 0.1 Run switches, models and budgets **What this cell does.** It defines every setting the rest of the notebook reads. On Colab, `RUN_LIVE` is on automatica  «Markdown notebook cell»
    [heading-3] ### 0.1 Run switches, models and budgets
    [notebook-code-cell] import os, sys from pathlib import Path def env_flag(name, default): """Read a boolean switch such as TAGO_RUN_LIVE=1/0/true/false.""" value = os.environ.get(na  «Code notebook cell»
    [notebook-markdown-cell] ### 0.2 Install missing Python dependencies **What this cell does.** It pip-installs only what is missing. `transformers` is needed for token counting and `sent  «Markdown notebook cell»
    [heading-3] ### 0.2 Install missing Python dependencies
    [notebook-code-cell] import subprocess, sys print('[setup] Installing compatible Python dependencies...', flush=True) requirements = ['pydantic>=2.10,<3', 'requests>=2.31,<3'] if RU  «Code notebook cell»
    [notebook-markdown-cell] <a id="s1"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 1. Capability boundaries **What this section does.** It runs no code. It states what the lab really does, so nobody mistakes it for production automation. |   «Markdown notebook cell»
    [heading-2] ## 1. Capability boundaries
    [notebook-markdown-cell] ## 2. Contracts, ports and model backends The code is organised in hexagonal rings. The same layout is used in the `tago/` package on GitHub, and the notebook i  «Markdown notebook cell»
    [heading-2] ## 2. Contracts, ports and model backends
    [notebook-markdown-cell] ### 2.1 Shared utilities and error types **What this cell does.** It defines canonical JSON (stable hashes, exact citation matching) and four error types. Each   «Markdown notebook cell»
    [heading-3] ### 2.1 Shared utilities and error types
    [notebook-code-cell] import ast, copy, hashlib, json, math, re, time, uuid, unicodedata from collections import Counter, defaultdict, deque from dataclasses import dataclass, field,  «Code notebook cell»
    [notebook-markdown-cell] ### 2.2 The step contract **What this cell does.** It defines what one model turn must look like: either `kind=tool` (call one tool) or `kind=final` (status, ac  «Markdown notebook cell»
    [heading-3] ### 2.2 The step contract
    [notebook-code-cell] class AuditVerdict(Strict): valid: bool request_matches_world: bool style_matches: bool reason: str = Field(min_length=1, max_length=600) class Citation(Strict)  «Code notebook cell»
    [notebook-markdown-cell] ### 2.3 Ports **What this cell does.** It declares the two interfaces the orchestrator depends on: a structured-output `ChatBackend` and a per-case `ToolHost`.   «Markdown notebook cell»
    [heading-3] ### 2.3 Ports
    [notebook-code-cell] from typing import Protocol, runtime_checkable @runtime_checkable class ChatBackend(Protocol): """Structured-output chat port. Must return an instance of `schem  «Code notebook cell»
    [notebook-markdown-cell] ### 2.4 Model profiles **What this cell does.** It registers the models the lab knows how to size. Each local model is paired with its Hugging Face tokenizer, s  «Markdown notebook cell»
    [heading-3] ### 2.4 Model profiles
    [notebook-code-cell] import os from dataclasses import dataclass, field, replace @dataclass(frozen=True) class ModelProfile: """An explicit model/tokenizer pairing; unknown Ollama a  «Code notebook cell»
    [notebook-markdown-cell] ### 2.5 Token preflight **What this cell does.** Before any request is sent, it checks that *prompt + output reserve + safety margin ≤ context window*. If not,   «Markdown notebook cell»
    [heading-3] ### 2.5 Token preflight
    [notebook-code-cell] class TokenProfiler: """Count the configured tokenizer's chat template, output reserve and safety margin. This is a tokenizer-based estimate of Ollama framing,   «Code notebook cell»
    [notebook-markdown-cell] ### 2.6 Shared backend call path **What this cell does.** It runs every model call, local or API, through the same sequence: budget check → preflight → request   «Markdown notebook cell»
    [heading-3] ### 2.6 Shared backend call path
    [notebook-code-cell] import os, time import requests def secret(name): """Read an API key from the environment or, on Colab, from the Secrets panel.""" value = os.environ.get(name)   «Code notebook cell»
    [notebook-markdown-cell] ### 2.7 Ollama backend (local T4) **What this cell does.** It sends the JSON schema as Ollama's `format`, so decoding is grammar-constrained, and pins `num_ctx`  «Markdown notebook cell»
    [heading-3] ### 2.7 Ollama backend (local T4)
    [notebook-code-cell] class OllamaBackend(BaseBackend): """Structured transport with explicit model profile and tokenizer preflight.""" provider = 'ollama' def __init__(self, model,   «Code notebook cell»
    [notebook-markdown-cell] ### 2.8 API backends (optional big models) **What this cell does.** It adds OpenAI-compatible endpoints (OpenAI, vLLM, OpenRouter, LM Studio, Ollama `/v1`) and   «Markdown notebook cell»
    [heading-3] ### 2.8 API backends (optional big models)
    [notebook-code-cell] class OpenAICompatibleBackend(BaseBackend): """OpenAI Chat Completions with json_schema response format. Works with OpenAI and compatible servers (vLLM, LM Stud  «Code notebook cell»
    [notebook-markdown-cell] ### 2.9 Scripted test double **What this cell does.** It replays a fixed list of steps. The offline suite uses it to test the graph, tools and rubric without an  «Markdown notebook cell»
    [heading-3] ### 2.9 Scripted test double
    [notebook-code-cell] class ScriptedBackend: """Deterministic test double: replays a queue of steps. It is NOT a language model.""" model = 'SCRIPTED_TEST_DOUBLE_NOT_LLM' provider =   «Code notebook cell»
    [notebook-markdown-cell] <a id="s3"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 3. Typed synthetic worlds **What these cells do.** A *world* holds the private facts of one fictional case. The model never sees it; tools reveal parts of it  «Markdown notebook cell»
    [heading-2] ## 3. Typed synthetic worlds
    [notebook-code-cell] class SupportWorld(Strict): ticket_id: str = Field(min_length=3, max_length=40) customer_id: str = Field(min_length=3, max_length=40) refund_cents: int = Field(  «Code notebook cell»
    [notebook-code-cell] class FinanceWorld(Strict): invoice_id: str = Field(min_length=3, max_length=40) billed_units: int = Field(gt=0, le=10000) received_units: int = Field(ge=0, le=  «Code notebook cell»
    [notebook-markdown-cell] **What this cell does.** It defines the knowledge world (E05: the decision and its rationale must match) and the software world (E06). The software scenario is   «Markdown notebook cell»
    [notebook-code-cell] class KnowledgeWorld(Strict): project_id: str = Field(min_length=3, max_length=40) chosen_database: Literal['PostgreSQL', 'SQLite'] rationale: Literal['transact  «Code notebook cell»
    [notebook-markdown-cell] **What this cell does (E06 domain rule).** It defines the restricted arithmetic interpreter. It lives in the domain ring because both the `code_check` tool and   «Markdown notebook cell»
    [notebook-code-cell] def arithmetic_value(expression, n): """E06 domain rule. Evaluate n, small int constants, +, -, *, // only. Never calls eval().""" if len(expression) > 120: rai  «Code notebook cell»
    [notebook-markdown-cell] **What this cell does.** It defines the procurement world (E07: at least one vendor must meet the lead-time limit) and the fictional compliance world (E08: the   «Markdown notebook cell»
    [notebook-code-cell] class Vendor(Strict): vendor_id: str = Field(min_length=3, max_length=40) unit_price_cents: int = Field(gt=0, le=100000) lead_days: int = Field(ge=1, le=90) cla  «Code notebook cell»
    [notebook-markdown-cell] **What this cell does.** It defines the security world (E09: the threshold is breached and the source host differs from the asset) and the assistant world (E10:  «Markdown notebook cell»
    [notebook-code-cell] class SecurityWorld(Strict): alert_id: str = Field(min_length=3, max_length=40) source_host: str = Field(min_length=3, max_length=40) sensitive_asset: str = Fie  «Code notebook cell»
    [notebook-markdown-cell] <a id="s4"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 4. Workflow packs, prompts and the constrained step schema  «Markdown notebook cell»
    [heading-2] ## 4. Workflow packs, prompts and the constrained step schema
    [notebook-markdown-cell] ### 4.1 Pack contract **What this cell does.** A pack holds everything specific to one workflow. The graph runner gets the pack's public parts (objective, tool   «Markdown notebook cell»
    [heading-3] ### 4.1 Pack contract
    [notebook-code-cell] from dataclasses import dataclass from typing import Literal, Optional, Union from pydantic import BaseModel, Field, create_model @dataclass(frozen=True) class   «Code notebook cell»
    [notebook-markdown-cell] ### 4.2 Specialist personas **What this cell does.** It gives each workflow its own specialist instructions: what to read, what to cite and what never to execut  «Markdown notebook cell»
    [heading-3] ### 4.2 Specialist personas
    [notebook-code-cell] SPECIALIST_PERSONAS = { 'E01': 'You are a customer support escalation specialist. Read ticket and refund policy; cite both when deciding the approval gate. Draf  «Code notebook cell»
    [notebook-markdown-cell] ### 4.3 The ten registrations **What these cells do.** They register E01–E10, each with a small human-written golden case. Golden cases are the regression contr  «Markdown notebook cell»
    [heading-3] ### 4.3 The ten registrations
    [notebook-code-cell] register('E01', 'Customer support', SupportWorld, 'ticket_id', ['ticket_read', 'retrieve'], ['refund_cents', 'requires_approval'], 'draft_refund', 'PENDING_APPR  «Code notebook cell»
    [notebook-code-cell] register('E06', 'Software engineering', EngineeringWorld, 'issue_id', ['code_read', 'code_check'], ['tests_passed', 'operation'], 'propose_patch', 'PROPOSED', '  «Code notebook cell»
    [notebook-markdown-cell] ### 4.4 Generator, auditor and worker prompts **What this cell does.** It defines the three role prompts: the generator invents fictional cases, the auditor che  «Markdown notebook cell»
    [heading-3] ### 4.4 Generator, auditor and worker prompts
    [notebook-code-cell] GENERATOR_PROMPT = '''Generate one entirely fictional enterprise case as structured JSON: request and world. Invent fresh identifiers, facts and client wording   «Code notebook cell»
    [notebook-markdown-cell] ### 4.5 Constrained step schema (the main v3.8 fix) **What this cell does.** It builds a per-workflow JSON schema in which `tool` can only be one of that workfl  «Markdown notebook cell»
    [heading-3] ### 4.5 Constrained step schema (the main v3.8 fix)
    [notebook-code-cell] _STEP_SCHEMAS = {} def build_step_schema(pack, registry=None): """v38: a per-pack AgentStep whose tool field is a union of exact per-tool argument objects. With  «Code notebook cell»
    [notebook-markdown-cell] <a id="s5"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 5. Sandbox tools  «Markdown notebook cell»
    [heading-2] ## 5. Sandbox tools
    [notebook-markdown-cell] ### 5.1 Retrieval **What this cell does.** It ranks the case's three small documents (policy, case reference, distractor) by lexical overlap. In `hybrid` mode i  «Markdown notebook cell»
    [heading-3] ### 5.1 Retrieval
    [notebook-code-cell] import ast, copy, re from collections import defaultdict, deque class RetrievalAdapter: """Lexical overlap, optionally blended 50/50 with CPU sentence-embedding  «Code notebook cell»
    [notebook-markdown-cell] ### 5.2 Graph traversal **What this cell does.** It walks a relationship graph breadth-first up to a depth bound, and returns the real edges it walked. A cycle   «Markdown notebook cell»
    [heading-3] ### 5.2 Graph traversal
    [notebook-code-cell] def graph_traverse(edges, start, max_depth=3): """Breadth-first traversal with a depth bound; returns the real edges walked.""" adjacency = defaultdict(list) fo  «Code notebook cell»
    [notebook-markdown-cell] ### 5.3 Tool contracts **What this cell does.** It defines argument names, with descriptions, and a one-line purpose for each tool. Both are shown to the model   «Markdown notebook cell»
    [heading-3] ### 5.3 Tool contracts
    [notebook-code-cell] # v38: argument descriptions are explicit. They are copied into the constrained JSON schema, # so the model sees the exact key names and the decoder cannot emit  «Code notebook cell»
    [notebook-markdown-cell] ### 5.4 Registry and authorization **What this cell does.** It authorises each call against the pack's tool list, its exact argument keys and the case reference  «Markdown notebook cell»
    [heading-3] ### 5.4 Registry and authorization
    [notebook-code-cell] class ToolRegistry: """Explicit registration and per-pack authorization; local adapter, not MCP transport.""" def __init__(self): self.entries = {} def register  «Code notebook cell»
    [notebook-markdown-cell] ### 5.5 Tool handlers **What this cell does.** It implements each tool and registers it. Plain record reads expose only their listed fields. `reconcile` and `co  «Markdown notebook cell»
    [heading-3] ### 5.5 Tool handlers
    [notebook-code-cell] def handle_retrieve(self, args): return {'mode': self.retriever.mode, 'hits': self.retriever.search(args['query'], self.documents())} def handle_graph_walk(self  «Code notebook cell»
    [notebook-markdown-cell] ### 5.6 Per-case tool host **What this cell does.** It stores each successful call as an immutable observation (`obs-1`, `obs-2`, …), the only valid citation ta  «Markdown notebook cell»
    [heading-3] ### 5.6 Per-case tool host
    [notebook-code-cell] class EnterpriseSandbox: """Per-case tool host: owns the private world, the observation log and browser page state.""" def __init__(self, pack, world, retriever  «Code notebook cell»
    [notebook-markdown-cell] <a id="s6"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 6. Synthetic generation and audit agents **What this cell does.** The generator invents a new fictional case: clean English or noisy (typos, shorthand). A se  «Markdown notebook cell»
    [heading-2] ## 6. Synthetic generation and audit agents
    [notebook-code-cell] SCENARIO_CONSTRAINTS = { 'E01': 'Refund must exceed self-service cap.', 'E02': 'Next step is schedule_demo.', 'E03': 'Billed units exceed received units.', 'E04  «Code notebook cell»
    [notebook-markdown-cell] <a id="s7"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 7. The bounded agent graph  «Markdown notebook cell»
    [heading-2] ## 7. The bounded agent graph
    [notebook-markdown-cell] ### 7.1 Edges and the evidence guard **What this cell does.** It defines the allowed transitions and the checks a final answer must pass: every required tool wa  «Markdown notebook cell»
    [heading-3] ### 7.1 Edges and the evidence guard
    [notebook-code-cell] import copy import requests EDGES = {'start': {'infer'}, 'infer': {'validate', 'repair', 'failed'}, 'validate': {'tool', 'finish', 'repair'}, 'tool': {'infer',   «Code notebook cell»
    [notebook-markdown-cell] ### 7.2 The graph runner **What this cell does.** It defines the runner that executes one case. Each turn: `infer` asks the model for one step → `validate` chec  «Markdown notebook cell»
    [heading-3] ### 7.2 The graph runner
    [notebook-code-cell] class AgentGraphOrchestrator: """Generic runner: receives a public pack and a tool host; never reads the world or oracle.""" def __init__(self, backend, max_ste  «Code notebook cell»
    [notebook-markdown-cell] <a id="s8"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 8. Hidden oracle, rubric and failure taxonomy  «Markdown notebook cell»
    [heading-2] ## 8. Hidden oracle, rubric and failure taxonomy
    [notebook-markdown-cell] ### 8.1 Oracle **What this cell does.** It computes the correct answer from the private world alone, never from the model's tool results.  «Markdown notebook cell»
    [heading-3] ### 8.1 Oracle
    [notebook-code-cell] import json, math from collections import Counter, defaultdict from pathlib import Path def expected_outcome(wid, w): if wid == 'E01': return {'refund_cents': w  «Code notebook cell»
    [notebook-markdown-cell] ### 8.2 Rubric **What this cell does.** It grades a run on status, action, no runtime error, no external execution, required tools, every finding value, provena  «Markdown notebook cell»
    [heading-3] ### 8.2 Rubric
    [notebook-code-cell] EVIDENCE_REQUIREMENTS = { 'E01': {'refund_cents': {'ticket_read'}, 'requires_approval': {'retrieve', 'ticket_read'}}, 'E02': {'account_id': {'crm_read'}, 'next_  «Code notebook cell»
    [notebook-markdown-cell] ### 8.3 Failure taxonomy (new in v3.8) **What this cell does.** It assigns every failed run one primary category, such as `tool_argument_schema`, `duplicate_too  «Markdown notebook cell»
    [heading-3] ### 8.3 Failure taxonomy (new in v3.8)
    [notebook-code-cell] FAILURE_PATTERNS = [ # (substring in the last error/feedback, category) - first match wins ('BudgetError', 'budget_exhausted'), ('ContextError', 'context_overfl  «Code notebook cell»
    [notebook-markdown-cell] ### 8.4 Statistics and run store **What this cell does.** It reports results per workflow and mode. *Eligible pass rate* excludes invalid datasets. *End-to-end   «Markdown notebook cell»
    [heading-3] ### 8.4 Statistics and run store
    [notebook-code-cell] def wilson(successes, total): if not total: return None z = 1.96; p = successes / total; denominator = 1 + z * z / total middle = (p + z * z / (2 * total)) / de  «Code notebook cell»
    [notebook-markdown-cell] <a id="s9a"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 9A. Offline contract suite — no model, GPU or network **What these cells do.** They run 40 deterministic tests with the scripted double. 26 are the v3.7 cont  «Markdown notebook cell»
    [heading-2] ## 9A. Offline contract suite — no model, GPU or network
    [notebook-code-cell] import copy, json, unittest from dataclasses import replace from unittest.mock import patch import requests # Explicit fixture answers, independent of expected_  «Code notebook cell»
    [notebook-code-cell] class OfflineContractTests(unittest.TestCase): """v37 contract suite, ported unchanged in intent.""" def test_all_ten_golden_contracts(self): for wid in PACKS:   «Code notebook cell»
    [notebook-code-cell] class V38RegressionTests(unittest.TestCase): """Each test pins one fix for a failure actually observed in the v37 Colab T4 run.""" def test_constrained_schema_r  «Code notebook cell»
    [notebook-code-cell] def run_offline_contracts(verbosity=1): suite = unittest.TestSuite() for case in (OfflineContractTests, V38RegressionTests): suite.addTests(unittest.defaultTest  «Code notebook cell»
    [notebook-markdown-cell] **What the next cell does.** It runs the suite and blocks live work if anything fails. **What you should see:** `Ran 40 tests … OK` followed by `"passed": true`  «Markdown notebook cell»
    [notebook-code-cell] import json OFFLINE_REPORT = None if RUN_OFFLINE_TESTS: OFFLINE_REPORT = run_offline_contracts() print(json.dumps(OFFLINE_REPORT, indent=2)) if REQUIRE_OFFLINE_  «Code notebook cell»
    [notebook-markdown-cell] <a id="s10"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 10. Runtime provisioning  «Markdown notebook cell»
    [heading-2] ## 10. Runtime provisioning
    [notebook-markdown-cell] ### 10.1 Runtime adapter and warmed backends **What this cell does.** `prepare_runtime` is the only code that installs Ollama, starts its server or pulls models  «Markdown notebook cell»
    [heading-3] ### 10.1 Runtime adapter and warmed backends
    [notebook-code-cell] import copy, hashlib, json, os, subprocess, sys, time, uuid from pathlib import Path import requests def prepare_runtime(plugin, url, models): """Only this func  «Code notebook cell»
    [notebook-markdown-cell] ### 10.2 Assemble the live lab **What this cell does.** If `RUN_LIVE` is on, it provisions Ollama, loads the retriever and builds the three role backends (gener  «Markdown notebook cell»
    [heading-3] ### 10.2 Assemble the live lab
    [notebook-code-cell] runtime_info = generator = auditor = orchestrator = retriever = None selected_packs = [PACKS[w] for w in SELECTED_WORKFLOWS] if RUN_LIVE: if REQUIRE_OFFLINE_PAS  «Code notebook cell»
    [notebook-markdown-cell] <a id="s11"></a>  «Markdown notebook cell»
    [notebook-markdown-cell] ## 11. Live labs and expected vs. real results  «Markdown notebook cell»
    [heading-2] ## 11. Live labs and expected vs. real results
    [notebook-markdown-cell] ### 11.1 Golden harness **What this cell does.** It runs each selected pack's fixed golden case, saves the full state, the evaluation and every model-call recor  «Markdown notebook cell»
    [heading-3] ### 11.1 Golden harness
    [notebook-code-cell] def execute_case(pack, case, orchestrator, retriever, seed): case = pack.validate_case(case) host = pack.tools(case['world'], retriever) start = len(orchestrato  «Code notebook cell»
    [notebook-markdown-cell] ### 11.2 Synthetic harness **What this cell does.** For each mode (clean, noisy) and pack it runs: generate → audit → run if accepted → grade. Every case is app  «Markdown notebook cell»
    [heading-3] ### 11.2 Synthetic harness
    [notebook-code-cell] def run_synthetic(packs, generator, auditor, orchestrator, retriever, n_clean=2, n_noisy=2, directory=OUTPUT_DIR): if any(type(n) is not int or not 0 <= n <= 10  «Code notebook cell»
    [notebook-markdown-cell] ### 11.3 Model comparison matrix **What this cell does.** It runs the same golden cases on several models under identical rules (same schema, prompts, budgets a  «Markdown notebook cell»
    [heading-3] ### 11.3 Model comparison matrix
    [notebook-code-cell] def run_model_matrix(specs, packs, retriever, runtime_info=None, directory=OUTPUT_DIR): """Run the same golden cases on several models (local and/or API) under   «Code notebook cell»
    [notebook-markdown-cell] **What the next cell does.** It runs golden cases, then synthetic cases, then the optional matrix.  «Markdown notebook cell»
    [notebook-code-cell] golden_results, synthetic_rows, synthetic_report, golden_path, matrix_results = [], [], None, None, {} if RUN_LIVE: golden_results = run_golden(selected_packs,   «Code notebook cell»
    [notebook-markdown-cell] ### 11.4 Real results so far (v3.7, Colab T4, qwen2.5:7b) The table below is generated from the committed artifact `results/v37_colab_t4_qwen2.5-7b/` when the n  «Markdown notebook cell»
    [heading-3] ### 11.4 Real results so far (v3.7, Colab T4, qwen2.5:7b)
    [notebook-markdown-cell] ### 11.5 Expected results (hypotheses to be measured, not results) Nothing in this table has been measured for v3.8 yet. Each row says what the design predicts,  «Markdown notebook cell»
    [heading-3] ### 11.5 Expected results (hypotheses to be measured, not results)
    [notebook-markdown-cell] ## 12. Analysis, frozen datasets and export  «Markdown notebook cell»
    [heading-2] ## 12. Analysis, frozen datasets and export
    [notebook-markdown-cell] ### 12.1 Analyse runs **What this cell does.** It loads saved run folders (v3.7 or v3.8 format) and builds a comparison table: pass count, failure category, mod  «Markdown notebook cell»
    [heading-3] ### 12.1 Analyse runs
    [notebook-code-cell] import json, statistics from pathlib import Path def load_run(directory): path = Path(directory) read = lambda name: json.loads((path / name).read_text(encoding  «Code notebook cell»
    [notebook-markdown-cell] **What the next cell does.** It compares this run with the real v3.7 baseline, if the repository's `results/` folder is present, and prints every failed case in  «Markdown notebook cell»
    [notebook-code-cell] from pathlib import Path from IPython.display import Markdown, display runs = [p for p in [Path('results/v37_colab_t4_qwen2.5-7b'), Path(golden_path) if golden_  «Code notebook cell»
    [notebook-markdown-cell] ### 12.2 Freeze, replay and export **What this cell does.** `freeze_dataset` saves the accepted synthetic inputs (never the model's answers) with a SHA-256 mani  «Markdown notebook cell»
    [heading-3] ### 12.2 Freeze, replay and export
    [notebook-code-cell] def freeze_dataset(rows, directory): cases = [{key: r[key] for key in ['case_id', 'workflow_id', 'mode', 'seed', 'case']} for r in rows if r.get('audit', {}).ge  «Code notebook cell»
    [notebook-markdown-cell] **What the next cell does.** It zips every run folder produced in this session and shows download links. Colab storage is wiped when the session ends, so downlo  «Markdown notebook cell»
    [notebook-code-cell] artifact_bundles = [export_bundle(p) for p in [golden_path, synthetic_report and synthetic_report['path'], *[v['path'] for v in matrix_results.values() if 'path  «Code notebook cell»

structure from results/node.tasks.json:
    [file-summary] Hexagonal DDD atomic-unit project template. The task graph and the path graph are two views of the same architecture. Task dependency edges follow the same directional rules as the hexagonal rings. Ea
    [json-key] $schema: "https://json-schema.org/draft/2020-12/schema"
    [json-key] template_type: "project_task_template"
    [json-key] version: "3.3"
    [json-key] description: "Hexagonal DDD atomic-unit project template. The task graph a..."
    [json-key] enums: {status, project_status, priority, complexity, dependency_type, +15 more}
    [json-key] project: {name, description, start_date, end_date, status, +14 more}
    [json-key] project_structure: {root, architecture, path_graph}
    [json-key] unit_registry: [0 items]
    [json-key] categories: [1 items]
    [json-key] workers: [2 items]
    [json-key] tasks: [6 items]
    [json-key] required_fields: [15 items]
    [json-key] optional_fields: [40 items]
    [json-key] navigation: {rootModule, modules}

structure from results/v37_colab_t4_qwen2.5-7b/README.md:
    [file-summary] v3.7 golden run — Google Colab T4, Ollama, qwen2.5:7b
    [trust-moderate] [heading-1] # v3.7 golden run — Google Colab T4, Ollama, qwen2.5:7b

structure from results/v37_colab_t4_qwen2.5-7b/golden_results.json:
    [json-array] [3 items]

structure from results/v37_colab_t4_qwen2.5-7b/model_calls.json:
    [json-array] [13 items]

structure from results/v37_colab_t4_qwen2.5-7b/runtime.json:
    [json-key] plugin: "colab_ollama"
    [json-key] url: "http://127.0.0.1:11434"
    [json-key] ollama_version: "ollama version is 0.34.2"

structure from tago/__init__.py:
    [file-summary] TAGO v3.8 - Tool-Augmented Graph Orchestration evaluation lab. Importing the package is the composition root: it registers the sandbox tool host and the default tool registry into the ports-level regi

structure from tago/__main__.py:
    [file-summary] Command line entry point (gateway ring). python -m tago test # offline contract suite (no model, no network) python -m tago analyze RUN_DIR [RUN_DIR] # markdown comparison of saved run folders python 
    def main(argv)  «docstring: none»
    def check_results(args)  «Verify that a committed live-run folder is complete and from the expected version.»

structure from tago/config.py:
    [file-summary] Run switches, model choices and budgets. Every value can be overridden with an environment variable, so the same code runs in three places without edits: * Google Colab ("Run all"): live mode is ON by
    IN_COLAB = ...  «docstring: none»
    VERSION = ...  «docstring: none»
    PROMPT_VERSION = ...  «docstring: none»
    RUN_OFFLINE_TESTS = ...  «What to run»
    REQUIRE_OFFLINE_PASS = ...  «What to run»
    RUN_LIVE = ...  «What to run»
    RUN_SYNTHETIC = ...  «What to run»
    REQUIRE_GOLDEN_PASS = ...  «What to run»
    GUIDED_PROGRESS = ...  «What to run»
    CONSTRAINED_SCHEMA = ...  «What to run»
    N_CLEAN = ...  «What to run»
    N_NOISY = ...  «What to run»
    SELECTED_WORKFLOWS = ...  «What to run»
    BASE_SEED = ...  «What to run»
    SPLIT = ...  «What to run»
    RUNTIME_PLUGIN = ...  «Runtime and models»
    OLLAMA_URL = ...  «Runtime and models»
    MODEL = ...  «Runtime and models»
    GENERATOR_MODEL = ...  «Runtime and models»
    AUDITOR_MODEL = ...  «Runtime and models»
    ORCHESTRATOR_MODEL = ...  «Runtime and models»
    RETRIEVAL_MODE = ...  «Runtime and models»
    EMBEDDING_MODEL = ...  «Runtime and models»
    MODEL_MATRIX = ...  «Optional comparison matrix (section 11.2). Each entry names a registered ModelProfile. API entries need OPENAI_API_KEY /»
    MAX_CALLS_PER_ROLE = ...  «Budgets»
    MAX_SECONDS_PER_ROLE = ...  «Budgets»
    REQUEST_TIMEOUT_SEC = ...  «Budgets»
    WARMUP_TIMEOUT_SEC = ...  «Budgets»
    MAX_GRAPH_STEPS = ...  «Budgets»
    MAX_REPAIRS = ...  «Budgets»
    OUTPUT_DIR = ...  «Budgets»
    def env_flag(name, default)  «Read a boolean switch such as TAGO_RUN_LIVE=1/0/true/false.»

structure from tago/node.tasks.json:
    [file-summary] Hexagonal DDD atomic-unit project template. The task graph and the path graph are two views of the same architecture. Task dependency edges follow the same directional rules as the hexagonal rings. Ea
    [json-key] $schema: "https://json-schema.org/draft/2020-12/schema"
    [json-key] template_type: "project_task_template"
    [json-key] version: "3.3"
    [json-key] description: "Hexagonal DDD atomic-unit project template. The task graph a..."
    [json-key] enums: {status, project_status, priority, complexity, dependency_type, +15 more}
    [json-key] project: {name, description, start_date, end_date, status, +14 more}
    [json-key] project_structure: {root, architecture, architecture_guide, dependency_rules, atomic_unit_standard, +1 more}
    [json-key] unit_registry: [14 items]
    [json-key] categories: [6 items]
    [json-key] workers: [2 items]
    [json-key] tasks: [9 items]
    [json-key] required_fields: [15 items]
    [json-key] optional_fields: [40 items]
    [json-key] navigation: {rootModule, modules}

structure from tago/application/__init__.py:
    [file-summary] TAGO package.

structure from tago/application/evaluation.py:
    [file-summary] Ring 2 - application: hidden oracle, grounding rubric, failure taxonomy and statistics. The oracle is computed from the private world only, never from the worker's tool results, so a model cannot pass
    EVIDENCE_REQUIREMENTS = ...  «%% cell: rubric»
    FAILURE_PATTERNS = ...  «%% cell: failure-taxonomy»
    class RunStore  «docstring: none»:
        def __init__(self, path)  «docstring: none»
        def write(self, name, value)  «docstring: none»
        def append(self, row)  «docstring: none»

    def expected_outcome(wid, w)  «docstring: none»
    def evaluate_case(pack, case, state)  «docstring: none»
    def classify_failure(state, checks)  «Map a failed run to one primary category so runs and models can be compared.»
    def duplicate_calls(observations)  «Count successful tool calls that repeated an earlier identical call (v37 had no guard).»
    def wilson(successes, total)  «docstring: none»
    def summarize(rows)  «docstring: none»

structure from tago/application/orchestrator.py:
    [file-summary] Ring 2 - application: the bounded agent/tool observation graph. start -> infer -> validate -> tool -> infer ... -> finish | repair -> quarantine | failed
    EDGES = ...  «docstring: none»
    class AgentGraphOrchestrator  «Generic runner: receives a public pack and a tool host; never reads the world or oracle.»:
        def __init__(self, backend, max_steps, max_repairs, guided, constrained)  «docstring: none»
        def _payload(self, pack, request, host, feedback)  «docstring: none»
        def run(self, pack, request, host, seed)  «docstring: none»

    def validate_final(step, host, pack)  «Evidence guard: a final answer must rest on observed tool results with literal citations.»

structure from tago/application/packs.py:
    [file-summary] Ring 2 - application: workflow packs (domain injection), personas, prompts and step schemas. A pack is everything that is specific to one enterprise workflow: its world schema, allowed tools, required
    SPECIALIST_PERSONAS = ...  «%% cell: personas»
    PACKS = ...  «%% cell: personas»
    GENERATOR_PROMPT = ...  «%% cell: prompts»
    AUDITOR_PROMPT = ...  «%% cell: prompts»
    COMMON_WORKER_PROMPT = ...  «%% cell: prompts»
    _STEP_SCHEMAS = ...  «%% cell: step-schema»
    class WorkflowPack  «docstring: none»:
        def worker_system(self)  «docstring: none»
        def case_schema(self)  «docstring: none»
        def validate_case(self, case)  «docstring: none»
        def tools(self, world, retriever)  «docstring: none»
        def oracle(self, world)  «docstring: none»
        def validate_final(self, step, host)  «docstring: none»
        def evaluate(self, case, state)  «docstring: none»

    def register(wid, title, schema, ref, tools, keys, action, status, objective, world, request)  «docstring: none»
    def build_step_schema(pack, registry)  «v38: a per-pack AgentStep whose tool field is a union of exact per-tool argument objects.»

structure from tago/application/synthetic.py:
    [file-summary] Ring 2 - application: synthetic case generator and independent dataset auditor agents.
    SCENARIO_CONSTRAINTS = ...  «docstring: none»
    class SyntheticCaseAgent  «Generates one fictional request + world. Invalid output is recorded, never replaced by a fixture.»:
        def __init__(self, backend)  «docstring: none»
        def generate(self, pack, mode, seed)  «docstring: none»

    class DatasetAuditor  «Audits the generated input (not agent output). Same-model auditing is correlated, not independent.»:
        def __init__(self, backend)  «docstring: none»
        def audit(self, pack, case, mode, seed)  «docstring: none»


structure from tago/core/__init__.py:
    [file-summary] TAGO package.

structure from tago/core/domain.py:
    [file-summary] Ring 0 - domain: pure data contracts with zero infrastructure dependencies. Contains canonical hashing helpers, error types, the agent step contract and the ten typed synthetic "worlds" (the private f
    class Strict(BaseModel)  «docstring: none»:

    class BudgetError(RuntimeError)  «docstring: none»:

    class ContextError(RuntimeError)  «docstring: none»:

    class CaseError(ValueError)  «docstring: none»:

    class ToolError(ValueError)  «docstring: none»:

    class AuditVerdict(Strict)  «docstring: none»:

    class Citation(Strict)  «docstring: none»:

    class Finding(Strict)  «docstring: none»:

    class ToolRequest(Strict)  «Transport-neutral tool call passed to the tool registry.»:

    class AgentStepBase(Strict)  «Shared validation for the generic and the per-pack constrained step schemas.»:
        def consistent(self)  «docstring: none»

    class AgentStep(AgentStepBase)  «v37-compatible generic step. v38 workers use build_step_schema() instead (section 8.1).»:

    class SupportWorld(Strict)  «docstring: none»:
        def escalation_case(self)  «docstring: none»

    class SalesWorld(Strict)  «docstring: none»:

    class FinanceWorld(Strict)  «docstring: none»:
        def discrepancy(self)  «docstring: none»

    class ITWorld(Strict)  «docstring: none»:

    class KnowledgeWorld(Strict)  «docstring: none»:
        def decision_pair(self)  «docstring: none»

    class EngineeringWorld(Strict)  «docstring: none»:

    class Vendor(Strict)  «docstring: none»:

    class ProcurementWorld(Strict)  «docstring: none»:
        def feasible(self)  «docstring: none»

    class LegalWorld(Strict)  «docstring: none»:
        def applicable_case(self)  «docstring: none»

    class SecurityWorld(Strict)  «docstring: none»:
        def actionable(self)  «docstring: none»

    class Offer(Strict)  «docstring: none»:

    class AssistantWorld(Strict)  «docstring: none»:
        def feasible(self)  «docstring: none»

    def canonical(value)  «Deterministic JSON text: sorted keys, no whitespace. Used for hashing and exact citations.»
    def digest(value)  «docstring: none»
    def norm(text)  «Case/accent-insensitive normalisation for lexical retrieval.»
    def tool_request_of(step)  «Convert a generic or constrained tool step into a ToolRequest.»
    def arithmetic_value(expression, n)  «E06 domain rule. Evaluate n, small int constants, +, -, *, // only. Never calls eval().»

structure from tago/core/ports.py:
    [file-summary] Ring 1 - ports: the two interfaces the application layer depends on. The orchestrator never imports a concrete backend or tool host. Anything that satisfies these protocols (Ollama, an OpenAI-compatib
    HOST_FACTORIES = ...  «Composition registries: infrastructure registers concrete implementations here at import time, so the application ring c»
    TOOL_REGISTRIES = ...  «Composition registries: infrastructure registers concrete implementations here at import time, so the application ring c»
    class ChatBackend(Protocol)  «Structured-output chat port. Must return an instance of `schema` or raise.»:
        def chat(self, role, system, payload, schema, seed, temperature)  «docstring: none»

    class ToolHost(Protocol)  «Per-case tool host port. `call` returns an observation dict with a stable id.»:
        def call(self, request)  «docstring: none»


structure from tago/gateway/__init__.py:
    [file-summary] TAGO package.

structure from tago/gateway/project_tasks.py:
    [file-summary] Gateway/CLI support: validate and verify the node.tasks.json project graphs. The task graphs are the project's definition of done. This module makes them testable: * ``validate`` - structure, enums, u
    STATUS = ...  «docstring: none»
    PRIORITY = ...  «docstring: none»
    COMPLEXITY = ...  «docstring: none»
    DEP_TYPES = ...  «docstring: none»
    LOCAL = ...  «docstring: none»
    def load(path)  «docstring: none»
    def graph_files(root)  «Root graph plus every sub-graph reachable through task links of type 'subgraph'.»
    def _cycle(tasks)  «docstring: none»
    def validate_graph(path, repo_root)  «docstring: none»
    def run_commands(commands, repo_root)  «docstring: none»
    def status_report(root)  «docstring: none»
    def main(argv)  «docstring: none»

structure from tago/infrastructure/__init__.py:
    [file-summary] TAGO package.

structure from tago/infrastructure/backends.py:
    [file-summary] Ring 2 - infrastructure: ChatBackend adapters. All adapters share one call path (budget check -> token preflight -> HTTP -> schema validation -> call record) so local and API models are compared under
    class BaseBackend  «docstring: none»:
        def __init__(self, profile, max_calls, max_seconds, timeout, profiler)  «docstring: none»
        def _default_profiler(self)  «docstring: none»
        def _messages(self, system, payload, schema_json)  «docstring: none»
        def _send(self, messages, schema_json, seed, temperature)  «docstring: none»
        def chat(self, role, system, payload, schema, seed, temperature)  «docstring: none»

    class OllamaBackend(BaseBackend)  «Structured transport with explicit model profile and tokenizer preflight.»:
        def __init__(self, model, base_url, max_calls, max_seconds, timeout, profile, profiler)  «docstring: none»
        def _default_profiler(self)  «docstring: none»
        def healthcheck(self)  «docstring: none»
        def warmup(self, timeout)  «Load weights into GPU memory before timed calls. v37's first call took 119 s (timeout was 120 s).»
        def unload(self)  «Free GPU memory before the next model in a comparison matrix.»
        def _send(self, messages, schema_json, seed, temperature)  «docstring: none»

    class OpenAICompatibleBackend(BaseBackend)  «OpenAI Chat Completions with json_schema response format.»:
        def __init__(self, profile, api_key, **kwargs)  «docstring: none»
        def _send(self, messages, schema_json, seed, temperature)  «docstring: none»

    class AnthropicBackend(BaseBackend)  «Anthropic Messages API; structured output is forced through a single required tool.»:
        def __init__(self, profile, api_key, **kwargs)  «docstring: none»
        def _send(self, messages, schema_json, seed, temperature)  «docstring: none»

    class ScriptedBackend  «Deterministic test double: replays a queue of steps. It is NOT a language model.»:
        def __init__(self, steps)  «docstring: none»
        def chat(self, role, system, payload, schema, seed, temperature)  «docstring: none»

    def secret(name)  «Read an API key from the environment or, on Colab, from the Secrets panel.»
    def make_backend(spec, base_url, **budget)  «Create a backend from a profile name ('qwen2.5:7b') or API spec ('anthropic:<model>').»

structure from tago/infrastructure/profiles.py:
    [file-summary] Ring 2 - infrastructure: model profiles and token preflight. A profile is an explicit pairing of a served model with the way its prompt size is bounded *before* the call. Local Ollama models use their
    MODEL_PROFILES = ...  «docstring: none»
    class ModelProfile  «An explicit model/tokenizer pairing; unknown Ollama aliases must be registered.»:

    class TokenProfiler  «Count the configured tokenizer's chat template, output reserve and safety margin.»:
        def __init__(self, profile, tokenizer)  «docstring: none»
        def _count(self, messages)  «docstring: none»
        def measure(self, messages)  «docstring: none»

    class ByteUpperBoundProfiler  «Provable upper bound for API models: tokens <= UTF-8 bytes + a per-message framing allowance.»:
        def __init__(self, profile)  «docstring: none»
        def measure(self, messages)  «docstring: none»

    def api_profile(spec)  «Build a profile for 'openai:<model>' or 'anthropic:<model>' specs (no local tokenizer).»
    def resolve_profile(spec)  «docstring: none»

structure from tago/infrastructure/tools.py:
    [file-summary] Ring 2 - infrastructure: deterministic sandbox tools and the per-case tool host. Every tool is a local algorithm over the generated (fictional) case facts. Nothing here performs an external write, ope
    REFERENCE_ARG = ...  «%% cell: tool-contracts v38: argument descriptions are explicit. They are copied into the constrained JSON schema, so th»
    TOOL_ARGUMENTS = ...  «%% cell: tool-contracts v38: argument descriptions are explicit. They are copied into the constrained JSON schema, so th»
    TOOL_PURPOSE = ...  «%% cell: tool-contracts v38: argument descriptions are explicit. They are copied into the constrained JSON schema, so th»
    TOOL_REGISTRY = ...  «docstring: none»
    class RetrievalAdapter  «Lexical overlap, optionally blended 50/50 with CPU sentence-embedding cosine similarity.»:
        def __init__(self, mode, model_name)  «docstring: none»
        def search(self, query, documents, k)  «docstring: none»

    class ToolRegistry  «Explicit registration and per-pack authorization; local adapter, not MCP transport.»:
        def __init__(self)  «docstring: none»
        def register(self, name, arguments, handler)  «docstring: none»
        def describe(self, names)  «docstring: none»
        def invoke(self, host, request)  «docstring: none»

    class EnterpriseSandbox  «Per-case tool host: owns the private world, the observation log and browser page state.»:
        def __init__(self, pack, world, retriever, registry)  «docstring: none»
        def documents(self)  «docstring: none»
        def edges(self)  «docstring: none»
        def call(self, request)  «docstring: none»

    def graph_traverse(edges, start, max_depth)  «Breadth-first traversal with a depth bound; returns the real edges walked.»
    def handle_retrieve(self, args)  «docstring: none»
    def handle_graph_walk(self, args)  «docstring: none»
    def fields(*keys)  «Handler factory for plain record reads that expose only the listed world fields.»
    def handle_reconcile(self, args)  «docstring: none»
    def handle_code_check(self, args)  «docstring: none»
    def handle_browser_open(self, args)  «docstring: none»
    def handle_browser_click(self, args)  «docstring: none»

structure from tago/workflows/__init__.py:
    [file-summary] TAGO package.

structure from tago/workflows/analysis.py:
    [file-summary] Ring 4 - workflows: read saved run folders (v37 or v38 format) and compare them. Used to produce the "real results" tables in the README from committed artifacts, so numbers in the documentation are r
    def load_run(directory)  «docstring: none»
    def _latency(calls)  «docstring: none»
    def run_summary(directory)  «docstring: none»
    def markdown_report(summaries)  «docstring: none»

structure from tago/workflows/contracts.py:
    [file-summary] Ring 4 - workflows: offline contract suite (no LLM, GPU, network or tokenizer download). These tests verify software behaviour with a scripted test double. They say nothing about real-model quality; t
    OFFLINE_ANSWERS = ...  «Explicit fixture answers, independent of expected_outcome().»
    class FakeTokenizer  «docstring: none»:
        def __init__(self, n)  «docstring: none»
        def apply_chat_template(self, messages, tokenize, add_generation_prompt)  «docstring: none»

    class FakeResponse  «docstring: none»:
        def __init__(self, data)  «docstring: none»
        def raise_for_status(self)  «docstring: none»
        def json(self)  «docstring: none»

    class OfflineContractTests(TestCase)  «v37 contract suite, ported unchanged in intent.»:
        def test_all_ten_golden_contracts(self)  «docstring: none»
        def test_world_constraints_reject_invalid_invoice(self)  «docstring: none»
        def test_reference_mismatch_rejected(self)  «docstring: none»
        def test_arithmetic_rejects_calls_attributes_and_powers(self)  «docstring: none»
        def test_arithmetic_patch_and_bug(self)  «docstring: none»
        def test_graph_cycle_bounded(self)  «docstring: none»
        def test_lexical_retrieval_ranks_relevant_document(self)  «docstring: none»
        def test_unknown_and_cross_pack_tool_denied(self)  «docstring: none»
        def test_wrong_reference_and_extra_arguments_denied(self)  «docstring: none»
        def test_registry_extension_without_host_edit(self)  «docstring: none»
        def test_reconcile_requires_prior_reads(self)  «docstring: none»
        def test_browser_requires_observed_navigation(self)  «docstring: none»
        def test_wrong_amount_fails_independent_rubric(self)  «docstring: none»
        def test_provenance_omission_fails(self)  «docstring: none»
        def test_nonliteral_citation_fails(self)  «docstring: none»
        def test_schema_repair_recovers(self)  «docstring: none»
        def test_repair_exhaustion_quarantines(self)  «docstring: none»
        def test_transport_error_fails_without_schema_retry(self)  «docstring: none»
        def test_step_budget_interrupts(self)  «docstring: none»
        def test_pending_human_action_never_executes(self)  «docstring: none»
        def test_specialist_prompts_are_distinct(self)  «docstring: none»
        def test_unknown_model_requires_explicit_profile(self)  «docstring: none»
        def test_token_reserve_and_preflight_rejection(self)  «docstring: none»
        def test_invalid_generation_is_not_a_workflow_pass(self)  «docstring: none»
        def test_audit_rejection_record(self)  «docstring: none»
        def test_statistics_separate_invalid_dataset(self)  «docstring: none»

    class V38RegressionTests(TestCase)  «Each test pins one fix for a failure actually observed in the v37 Colab T4 run.»:
        def test_constrained_schema_rejects_v37_argument_errors(self)  «docstring: none»
        def test_constrained_json_schema_lists_exact_tools_and_keys(self)  «docstring: none»
        def test_duplicate_call_denied_without_new_observation(self)  «docstring: none»
        def test_v37_e01_loop_now_recovers(self)  «docstring: none»
        def test_duplicate_loop_is_still_bounded(self)  «docstring: none»
        def test_repair_feedback_names_expected_arguments_in_v37_mode(self)  «docstring: none»
        def test_progress_hint_lists_missing_tools(self)  «docstring: none»
        def test_observations_sent_once(self)  «docstring: none»
        def test_consecutive_repairs_reset_after_successful_tool(self)  «docstring: none»
        def test_v37_errors_classify(self)  «docstring: none»
        def test_byte_upper_bound_profiler(self)  «docstring: none»
        def test_profile_resolution(self)  «docstring: none»
        def test_ollama_request_uses_constrained_schema_and_keep_alive(self)  «docstring: none»
        def test_openai_and_anthropic_adapters_parse_structured_output(self)  «docstring: none»

    def tool_args(pack, name)  «docstring: none»
    def offline_tool_steps(pack)  «docstring: none»
    def offline_final(wid, values)  «docstring: none»
    def offline_run(wid, steps, **orchestrator_options)  «docstring: none»
    def run_offline_contracts(verbosity)  «docstring: none»

structure from tago/workflows/experiments.py:
    [file-summary] Ring 4 - workflows: runtime provisioning, golden/synthetic/matrix experiments, freeze, replay, export.
    def prepare_runtime(plugin, url, models)  «Only this function installs Ollama, starts its daemon or pulls models.»
    def live_backend(spec, url, warm)  «Construct, health-check and warm a backend under the configured budgets.»
    def execute_case(pack, case, orchestrator, retriever, seed)  «docstring: none»
    def run_golden(packs, orchestrator, retriever, seed)  «docstring: none»
    def save_golden(rows, backend, runtime_info, directory, label)  «docstring: none»
    def run_synthetic(packs, generator, auditor, orchestrator, retriever, n_clean, n_noisy, directory)  «docstring: none»
    def run_model_matrix(specs, packs, retriever, runtime_info, directory)  «Run the same golden cases on several models (local and/or API) under identical rules.»
    def freeze_dataset(rows, directory)  «docstring: none»
    def replay_frozen(directory, orchestrator, retriever)  «docstring: none»
    def export_bundle(directory)  «docstring: none»

structure from tests/test_architecture.py:
    [file-summary] Hexagonal ring rule: inner rings never import outer rings (checked on the source AST).
    PACKAGE = ...  «docstring: none»
    ALLOWED = ...  «docstring: none»
    def ring(module)  «docstring: none»
    def test_no_inner_ring_imports_outer_ring()  «docstring: none»

structure from tests/test_contracts.py:
    [file-summary] Collect the offline contract suite (the same one the notebook runs in section 9A).

structure from tests/test_docs.py:
    [file-summary] Documentation numbers must be regenerated from artifacts, never typed by hand.
    ROOT = ...  «docstring: none»
    def test_readme_results_table_matches_artifact()  «docstring: none»
    def test_readme_links_exist()  «docstring: none»
    def test_packaging_metadata()  «docstring: none»

structure from tests/test_e2e_fake_ollama.py:
    [file-summary] End-to-end plumbing test over real HTTP against a fake Ollama server. The fake server plays a *scripted policy* (not a model): it calls the next uncalled tool and then finishes with fixture answers. T
    class Handler(BaseHTTPRequestHandler)  «docstring: none»:
        def log_message(self, *args)  «docstring: none»
        def _send(self, data)  «docstring: none»
        def do_GET(self)  «docstring: none»
        def do_POST(self)  «docstring: none»

    def respond(body)  «docstring: none»
    def server()  «docstring: none»
    def test_full_pipeline_over_http(server, tmp_path)  «docstring: none»

structure from tests/test_notebook.py:
    [file-summary] The notebook must be in sync with the package and run top-to-bottom offline WITHOUT the package.
    ROOT = ...  «docstring: none»
    NOTEBOOK = ...  «docstring: none»
    def test_notebook_in_sync()  «docstring: none»
    def test_notebook_executes_offline_without_package(tmp_path, monkeypatch)  «docstring: none»
    def test_every_code_cell_is_explained()  «Each code cell must follow a markdown cell that says what it does (the user-facing contract).»

structure from tests/test_project_tasks.py:
    [file-summary] The node.tasks.json graphs are valid, and the validator really catches broken graphs.
    ROOT = ...  «docstring: none»
    def test_repository_graphs_are_valid()  «docstring: none»
    def test_validator_rejects_inconsistent_graphs(tmp_path)  «docstring: none»
    def test_gate_is_closed_until_live_results_exist()  «docstring: none»

structure from tests/test_v37_artifact.py:
    [file-summary] Pin the analysis of the real v3.7 Colab T4 run and prove v3.8's schema rejects its outputs.
    RUN = ...  «docstring: none»
    def test_v37_summary_matches_artifact()  «docstring: none»
    def test_every_rejected_v37_output_is_invalid_under_v38_schema()  «docstring: none»

structure from tools/build_notebook.py:
    [file-summary] Build the self-contained Colab notebook from the tago/ package and notebooks/narrative.md. The package is the single source of truth (tested with pytest). The notebook inlines the package code cell by
    ROOT = ...  «docstring: none»
    PACKAGE = ...  «docstring: none»
    NARRATIVE = ...  «docstring: none»
    OUTPUT = ...  «docstring: none»
    EXCLUDED = ...  «docstring: none»
    MARKER = ...  «docstring: none»
    TAGO_IMPORT = ...  «docstring: none»
    def strip_tago_imports(lines)  «docstring: none»
    def chunks()  «docstring: none»
    def generated(name)  «docstring: none»
    def cell_id(kind, text)  «docstring: none»
    def build()  «docstring: none»

---

## Relations structure

relations from tago/__init__.py:
    [import] tago.infrastructure.tools

relations from tago/__main__.py:
    [import] json
    [import] sys
    [import-from] pathlib
    [import-from] tago.workflows.analysis
    [import-from] tago.workflows.contracts
    [import-from] tago.config
    [import-from] tago.application.packs
    [import-from] tago.infrastructure.tools
    [import-from] tago.workflows.experiments
    [import-from] tago.gateway.project_tasks

relations from tago/config.py:
    [import] os
    [import] sys
    [import-from] pathlib

relations from tago/application/evaluation.py:
    [import] json
    [import] math
    [import-from] collections
    [import-from] pathlib
    [import-from] tago.core.domain

relations from tago/application/orchestrator.py:
    [import] copy
    [import] requests
    [import-from] tago.core.domain
    [import-from] tago.application.packs
    [import-from] tago.application.evaluation
    [import-from] tago.config

relations from tago/application/packs.py:
    [import-from] dataclasses
    [import-from] typing
    [import-from] pydantic
    [import-from] tago.core.domain
    [import-from] tago.core.ports
    [import-from] tago.application.evaluation
    [import-from] tago.application.orchestrator

relations from tago/application/synthetic.py:
    [import-from] tago.core.domain
    [import-from] tago.application.packs

relations from tago/core/domain.py:
    [import] ast
    [import] copy
    [import] hashlib
    [import] json
    [import] math
    [import] re
    [import] time
    [import] uuid
    [import] unicodedata
    [import-from] collections
    [import-from] dataclasses
    [import-from] typing
    [import-from] pydantic

relations from tago/core/ports.py:
    [import-from] typing

relations from tago/gateway/project_tasks.py:
    [import] json
    [import] os
    [import] subprocess
    [import] sys
    [import-from] pathlib

relations from tago/infrastructure/backends.py:
    [import] os
    [import] time
    [import] requests
    [import-from] tago.core.domain
    [import-from] tago.infrastructure.profiles
    [import-from] google.colab

relations from tago/infrastructure/profiles.py:
    [import] os
    [import-from] dataclasses
    [import-from] tago.core.domain
    [import-from] transformers

relations from tago/infrastructure/tools.py:
    [import] ast
    [import] copy
    [import] re
    [import-from] collections
    [import-from] tago.core.domain
    [import-from] tago.core.ports
    [import-from] tago.config
    [import-from] sentence_transformers

relations from tago/workflows/analysis.py:
    [import] json
    [import] statistics
    [import-from] pathlib
    [import-from] tago.application.evaluation

relations from tago/workflows/contracts.py:
    [import] copy
    [import] json
    [import] unittest
    [import-from] dataclasses
    [import-from] unittest.mock
    [import] requests
    [import-from] tago.core.domain
    [import-from] tago.infrastructure.profiles
    [import-from] tago.infrastructure.backends
    [import-from] tago.infrastructure.tools
    [import-from] tago.application.packs
    [import-from] tago.application.evaluation
    [import-from] tago.application.orchestrator
    [import-from] tago.application.synthetic

relations from tago/workflows/experiments.py:
    [import] copy
    [import] hashlib
    [import] json
    [import] os
    [import] subprocess
    [import] sys
    [import] time
    [import] uuid
    [import-from] pathlib
    [import] requests
    [import-from] tago.config
    [import-from] tago.core.domain
    [import-from] tago.infrastructure.backends
    [import-from] tago.infrastructure.profiles
    [import-from] tago.application.packs
    [import-from] tago.application.evaluation
    [import-from] tago.application.orchestrator
    [import] shutil
    [import] urllib.request

relations from tests/test_architecture.py:
    [import] ast
    [import-from] pathlib

relations from tests/test_contracts.py:
    [import-from] tago.workflows.contracts

relations from tests/test_docs.py:
    [import] re
    [import-from] pathlib
    [import-from] tago.workflows.analysis

relations from tests/test_e2e_fake_ollama.py:
    [import] json
    [import] threading
    [import-from] http.server
    [import] pytest
    [import-from] tago.application.packs
    [import-from] tago.application.orchestrator
    [import-from] tago.application.synthetic
    [import-from] tago.infrastructure.backends
    [import-from] tago.infrastructure.profiles
    [import-from] tago.infrastructure.tools
    [import-from] tago.workflows.contracts
    [import-from] tago.workflows.experiments
    [import-from] tago.workflows.analysis

relations from tests/test_notebook.py:
    [import] os
    [import] shutil
    [import] subprocess
    [import] sys
    [import-from] pathlib
    [import] pytest

relations from tests/test_project_tasks.py:
    [import] copy
    [import] json
    [import-from] pathlib
    [import-from] tago.gateway.project_tasks

relations from tests/test_v37_artifact.py:
    [import] json
    [import-from] pathlib
    [import] pytest
    [import-from] tago.workflows.analysis
    [import-from] tago.application.packs

relations from tools/build_notebook.py:
    [import] hashlib
    [import] re
    [import] sys
    [import-from] pathlib
    [import] nbformat
    [import-from] tago.workflows.analysis

---

## Flow structure

flow from notebooks/tago_v38_agent_graph_lab.ipynb:
    input -> transform -> state -> output
    [input] load_run
    [transform] build_step_schema, validate_final, summarize
    [state] save_golden
    [output] export_bundle

flow from tago/application/evaluation.py:
    transform -> state
    [transform] summarize
    [state] write

flow from tago/application/orchestrator.py:
    transform
    [transform] validate_final

flow from tago/application/packs.py:
    transform
    [transform] validate_case, validate_final, build_step_schema

flow from tago/application/synthetic.py:
    transform
    [transform] generate

flow from tago/gateway/project_tasks.py:
    input -> transform
    [input] load
    [transform] validate_graph

flow from tago/workflows/analysis.py:
    input
    [input] load_run

flow from tago/workflows/experiments.py:
    state -> output
    [state] save_golden
    [output] export_bundle

flow from tools/build_notebook.py:
    transform
    [transform] generated, build

---

## API endpoints

cli entry points:
    [cli] tago/__main__.py — script entry (__main__)
    [cli] tago/gateway/project_tasks.py — script entry (__main__)
    [cli] tools/build_notebook.py — script entry (__main__)

api surface:
    [no-http-routes] no HTTP route handlers detected

core surface candidates for API exposure:
    [candidate] tago/core/domain.py: canonical, digest, norm, consistent, tool_request_of (+3 more)
    [candidate] tago/infrastructure/tools.py: search, graph_traverse, register, describe, invoke (+3 more)
    [candidate] tago/workflows/experiments.py: prepare_runtime, live_backend, execute_case, run_golden, save_golden (+3 more)
    [candidate] tago/application/evaluation.py: expected_outcome, evaluate_case, classify_failure, duplicate_calls, wilson (+3 more)
    [candidate] tago/application/packs.py: worker_system, case_schema, validate_case, tools, oracle (+3 more)
    [candidate] tago/infrastructure/backends.py: secret, chat, healthcheck, warmup, unload (+1 more)

automation suggestions:
    [suggest-http] expose core functions via HTTP server; start with tago/core/domain.py
    [suggest-contract] add an OpenAPI/JSON schema to document the API surface

---
*45 files indexed · generated by map_project*