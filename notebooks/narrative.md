# TAGO v3.8 — T4 Agent Graph Orchestrator (Colab T4 ready)

**What this notebook is.** A self-contained evaluation lab that asks a language model to work through ten fictional enterprise workflows (refunds, invoice reconciliation, a code fix, procurement, and others). The model works step by step inside a bounded state graph. It can only learn facts by calling sandboxed tools, and it may only *propose* actions, never execute them. An independent hidden rubric then grades every answer for correctness, evidence and human-approval gating.

**What "TAGO" means.** **T**4 **A**gent **G**raph **O**rchestrator. **T4** is the GPU class the lab is sized for (a free Colab Tesla T4, about 15 GB of VRAM, an 8K context). **Agent** is a language model that decides one step at a time. **Graph** is the bounded state machine `start → infer → validate → tool → … → finish | repair → quarantine | failed` that runs it. **Orchestrator** is the generic runner that owns the budgets, the tool sandbox and the evidence rules, while workflow-specific knowledge is injected as packs.

**Quick start (Colab).**
1. Upload this notebook to Google Colab.
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*.

No API key or code edit is needed for the default run. The first run downloads Ollama (~1 GB), `qwen2.5:7b` (~4.7 GB), a tokenizer and a small embedding model, so allow 10–20 minutes before the first model answer.

**What happens, in order.**

| Step | Section | Needs GPU/network? | What you should see |
|---|---|---|---|
| 1 | 0. Settings and dependencies | network (pip) | `[setup] Dependencies ready.` |
| 2 | 1–9. Definitions | no | Nothing is printed. Cells only define code. |
| 3 | 9A. Offline contract suite | no | `Ran N tests … OK` and `"passed": true` |
| 4 | 10. Runtime | GPU + network | Ollama install, model pull, warm-up time (1–3 min on a T4) |
| 5 | 11.1 Golden cases | GPU | One `GOLDEN E0x PASS/FAIL <category>` line per workflow |
| 6 | 11.2 Synthetic cases | GPU | One line per generated case: `clean 1 E01 pass` and so on |
| 7 | 11.3 Model matrix (optional) | GPU and/or API keys | One golden score per model |
| 8 | 11.6 Experiment campaign | GPU + network | Ablation rungs, auditor study, other models and thinking models: one `CAMPAIGN <item> n/3` line each, then one table (about 2–3 h on a T4; `TAGO_RUN_CAMPAIGN=0` skips it) |
| 9 | 12. Analysis and export | no | A comparison table against the real v3.7 run, plus ZIP download links |

**Status of the measurements.** Five real Colab T4 runs exist (section 11.4), all with `qwen2.5:7b`. The first three passed no golden case: v3.7 failed on tool arguments, v3.8.1 on the step envelope, v3.8.2 on how quotes are copied. The fourth (v3.8.3) passed 2 of 3, and **the fifth (v3.8.4, the current code) passed 3 of 3** with 5 of 6 synthetic cases. Each fix is pinned by a test that replays the real model output. Everything in section 11.5 about other models and ablations is a hypothesis, labelled as such.

<a id="index"></a>
## Index

- [0. Settings and dependencies](#s0)
- [1. Capability boundaries](#s1)
- [2. Contracts, ports and model backends](#s2)
- [3. Typed synthetic worlds](#s3)
- [4. Workflow packs, prompts and the constrained step schema](#s4)
- [5. Sandbox tools](#s5)
- [6. Synthetic generation and audit agents](#s6)
- [7. The bounded agent graph](#s7)
- [8. Hidden oracle, rubric and failure taxonomy](#s8)
- [9A. Offline contract suite](#s9a)
- [10. Runtime provisioning](#s10)
- [11. Live labs and expected vs. real results](#s11)
- [12. Analysis, frozen datasets and export](#s12)

<a id="s0"></a>
## 0. Settings and dependencies

### 0.1 Run switches, models and budgets

**What this cell does.** It defines every setting the rest of the notebook reads. On Colab, `RUN_LIVE` is on automatically. On a laptop or in CI it is off, so the notebook only runs the offline checks. You can edit values here or set the matching `TAGO_*` environment variable.

The settings that matter most:
- `SELECTED_WORKFLOWS`: which of E01–E10 to run live. The default is the same three that v3.7 ran (E01, E03, E06), so the results can be compared directly.
- `MODEL` is the Ollama model for all three roles. `MODEL_MATRIX` is an optional list of extra models (local or `openai:<model>` / `anthropic:<model>`) for section 11.3.
- Four ablation switches stack: `CONSTRAINED_SCHEMA` (per-workflow tool and argument schema, v3.8), `GUIDED_PROGRESS` (tells the worker which tools are still missing, v3.8) `STATE_CONDITIONED` (v3.8.2: only the step that is legal *now* can be decoded) and `GROUNDED_QUOTES` (v3.8.3: citation quotes can only be exact fragments of the cited observation). Set all four to `False` to reproduce v3.7 behaviour.
- `GENERATION_ATTEMPTS` (v3.8.2): the synthetic generator may redraft a rejected case this many times, each time seeing the exact validation error. Every attempt is recorded.
- `THINK`: `default` never sends Ollama's `think` option (safest with `format`); `off` sends `think=false`, which has been reported to switch off format enforcement on some builds. Only touch it with a thinking model, and test one golden case first.
- `MODEL_MATRIX`: a comma-separated list, or the word `recommended` for four model families (granite4.1:8b, ministral-3:14b, gemma4:12b, qwen3.5:9b). Models whose download fails are skipped and recorded, not fatal. Allow 30+ extra minutes and ~30 GB of disk.
- **Experiment campaign (section 11.6), all in this one run.** `RUN_CAMPAIGN` is on by default in Colab. After the main run the notebook re-runs the golden cases under other settings and saves every result: `CAMPAIGN_ABLATION` (four switch rungs), `CAMPAIGN_MODELS` (other local models), `CAMPAIGN_THINKING_MODELS` (each run with `think` default and `off`), `CAMPAIGN_AUDITOR` (a synthetic sweep audited by a different model family) and `CAMPAIGN_MAX_MINUTES` (no new item starts after this). To run only the main experiment, set `TAGO_RUN_CAMPAIGN=0`. To change a list, set the matching `TAGO_CAMPAIGN_*` variable before this cell.
- `REQUIRE_GOLDEN_PASS` is now `False`. In v3.7 a golden failure blocked the synthetic sweep, so no synthetic data was ever collected.
- `WARMUP_TIMEOUT_SEC` controls the warm-up call. In v3.7 the first call took 119 s against a 120 s timeout.

<!-- code: config.py#config -->

### 0.2 Install missing Python dependencies

**What this cell does.** It pip-installs only what is missing. `transformers` is needed for token counting and `sentence-transformers` for hybrid retrieval, and both are installed only when live mode needs them. **What you should see:** `[setup] Dependencies ready.`

```python
import subprocess, sys
print('[setup] Installing compatible Python dependencies...', flush=True)
requirements = ['pydantic>=2.10,<3', 'requests>=2.31,<3']
if RUN_LIVE:
    requirements.append('transformers>=4.45')
    if RETRIEVAL_MODE == 'hybrid':
        requirements.append('sentence-transformers>=3')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *requirements], check=True)
print('[setup] Dependencies ready. Next: definitions and offline checks.', flush=True)
```

<a id="s1"></a>
## 1. Capability boundaries

**What this section does.** It runs no code. It states what the lab really does, so nobody mistakes it for production automation.

| Workflow | Execution path in this lab | Real system it stands in for (not connected) |
|---|---|---|
| E01 Customer support | Ticket read → policy retrieval → refund escalation draft | Ticketing, refund API |
| E02 Sales / RevOps | CRM read → research retrieval → CRM update draft | CRM, email |
| E03 Finance | Invoice + receipt reads → reconciliation → dispute draft | OCR, ERP, payments |
| E04 IT helpdesk | Identity read → access policy → approval request | Identity provider |
| E05 Knowledge Q&A | Retrieval → decision graph → cited answer | Enterprise search |
| E06 Software engineering | Code read → arithmetic patch checks → patch proposal | Repository, CI, PR API |
| E07 Procurement | Inventory → vendors → selection → purchase request | ERP, purchase orders |
| E08 Legal (fictional rules) | Rule retrieval → entity graph → review request | Legal corpus, expert review |
| E09 Security operations | Alert read → asset graph → containment review | SIEM / EDR |
| E10 Executive assistant | Calendar → sandbox portal open/click → booking proposal | Browser, calendar, payments |

Every tool is a deterministic local algorithm over generated fictional data. The "browser" is a fixed HTML fixture. The code checker evaluates only `n`, small integers, `+ - * //`. No tool writes anywhere. A passing score shows the model followed a grounded, gated procedure in a sandbox. It is not evidence of production readiness.

<a id="s2"></a>
## 2. Contracts, ports and model backends

The code is organised in hexagonal rings. The same layout is used in the `tago/` package on GitHub, and the notebook inlines it in this order: **domain** (pure data) → **ports** (interfaces) → **infrastructure** (model backends, tools) → **application** (packs, graph, evaluation) → **workflows** (experiments). Inner rings never depend on outer rings, except where a known exception is recorded as a task.

### 2.1 Shared utilities and error types

**What this cell does.** It defines canonical JSON (stable hashes, exact citation matching) four error types and `load_public_model`, which retries a Hugging Face download without a token when Colab's `HF_TOKEN` secret is expired (a public model then fails with `401 Unauthorized`). Each error type leads to a different graph outcome: `BudgetError` and `ContextError` end the run as **FAILED**, `ToolError` and validation errors go to **repair**, and `CaseError` marks an invalid *dataset* case, which is never counted as a model failure.

<!-- code: core/domain.py#utilities -->

### 2.2 The step contract

**What this cell does.** It defines what one model turn must look like: either `kind=tool` (call one tool) or `kind=final` (status, action, findings with `source_ids`, and literal citations). The generic `AgentStep` is the v3.7 contract. v3.8 workers use a stricter per-workflow version built in section 4.5.

<!-- code: core/domain.py#step-contract -->

**What the next cell does.** `quote_fragments` lists the verbatim pieces of a tool observation that may be cited: the whole text, each line, and every top-level `"key":value` pair of a JSON object. Each piece is a literal substring of the observation. The state-conditioned schema (section 4.5) offers exactly these pieces as the only decodable citation quotes.

**Why.** A 7B model asked to copy `{"refund_cents":9000}` into a free string writes `"refund_cents": 9000` (an extra space) or paraphrases it, and repeats the mistake after feedback. Choosing from a list of exact fragments turns a typing task into a selection task, and the evidence rule stays literal: a quote must be a substring of the observation it cites.

<!-- code: core/domain.py#quote-fragments -->

### 2.3 Ports

**What this cell does.** It declares the two interfaces the orchestrator depends on: a structured-output `ChatBackend` and a per-case `ToolHost`. Any implementation that meets them can be swapped in: Ollama, an API model, a scripted test double, or later an MCP tool host. Two small registries let infrastructure plug its concrete host and tool registry in, so the application ring never imports infrastructure.

<!-- code: core/ports.py#ports -->

### 2.4 Model profiles

**What this cell does.** It registers the models the lab knows how to size. Each local model is paired with its Hugging Face tokenizer, so prompt length is counted with the real chat template before every call. Unregistered Ollama aliases are refused on purpose rather than guessed. `t4_fit` notes whether the model fits a 16 GB T4 at 4-bit. Only `qwen2.5:7b` has `status='measured'`. The others are registered but have not been run yet.

<!-- code: infrastructure/profiles.py#model-profiles -->

### 2.5 Token preflight

**What this cell does.** Before any request is sent, it checks that *prompt + output reserve + safety margin ≤ context window*. If not, it raises `ContextError` and makes no network call, so nothing is silently truncated. API models have no public tokenizer, so they use a provable upper bound instead (UTF-8 bytes plus framing). Providers then report the real token counts.

<!-- code: infrastructure/profiles.py#token-preflight -->

### 2.6 Shared backend call path

**What this cell does.** It runs every model call, local or API, through the same sequence: budget check → preflight → request → schema validation → call record. The call record (prompt hash, schema hash, latency, tokens, raw output) is what section 12 analyses.

<!-- code: infrastructure/backends.py#backend-base -->

### 2.7 Ollama backend (local T4)

**What this cell does.** It sends the JSON schema as Ollama's `format`, so decoding is grammar-constrained, and pins `num_ctx`, `seed` and `temperature`. New in v3.8: `warmup()` loads the weights before timed calls, `keep_alive` keeps them loaded, and `unload()` frees the GPU between models in a comparison matrix.

<!-- code: infrastructure/backends.py#backend-ollama -->

### 2.8 API backends (optional big models)

**What this cell does.** It adds OpenAI-compatible endpoints (OpenAI, vLLM, OpenRouter, LM Studio, Ollama `/v1`) and the Anthropic Messages API, where structured output is forced through a single required tool. Keys are read from the environment or Colab *Secrets*. **Comparability caveat:** these providers validate against the schema but do not always decode under a grammar the way Ollama does. That is why the lab validates every output itself, whatever the provider.

<!-- code: infrastructure/backends.py#backend-apis -->

### 2.9 Scripted test double

**What this cell does.** It replays a fixed list of steps. The offline suite uses it to test the graph, tools and rubric without any model. Its results are never reported as model quality.

<!-- code: infrastructure/backends.py#backend-scripted -->

<a id="s3"></a>
## 3. Typed synthetic worlds

**What these cells do.** A *world* holds the private facts of one fictional case. The model never sees it; tools reveal parts of it. Validators enforce that each case is actually interesting (for example, the refund must exceed the self-service limit, so approval is required). All money is in integer cents.

<!-- code: core/domain.py#worlds-support-sales -->

<!-- code: core/domain.py#worlds-finance-it -->

**What this cell does.** It defines the knowledge world (E05: the decision and its rationale must match) and the software world (E06). The software scenario is deliberately narrow: fix an off-by-one error in the inclusive sum 1..n.

<!-- code: core/domain.py#worlds-knowledge-engineering -->

**What this cell does (E06 domain rule).** It defines the restricted arithmetic interpreter. It lives in the domain ring because both the `code_check` tool and the hidden rubric use it. It walks the syntax tree and supports only `n`, small integers, `+ - * //`. It never calls `eval`.

<!-- code: core/domain.py#arithmetic-rule -->

**What this cell does.** It defines the procurement world (E07: at least one vendor must meet the lead-time limit) and the fictional compliance world (E08: the rule must apply).

<!-- code: core/domain.py#worlds-procurement-legal -->

**What this cell does.** It defines the security world (E09: the threshold is breached and the source host differs from the asset) and the assistant world (E10: at least one offer is affordable at a free hour).

<!-- code: core/domain.py#worlds-security-assistant -->

<a id="s4"></a>
## 4. Workflow packs, prompts and the constrained step schema

### 4.1 Pack contract

**What this cell does.** A pack holds everything specific to one workflow. The graph runner gets the pack's public parts (objective, tool list, finding keys, proposal action) and never sees the world or the expected answers.

<!-- code: application/packs.py#pack-contract -->

### 4.2 Specialist personas

**What this cell does.** It gives each workflow its own specialist instructions: what to read, what to cite and what never to execute. These are appended to the common worker prompt.

<!-- code: application/packs.py#personas -->

### 4.3 The ten registrations

**What these cells do.** They register E01–E10, each with a small human-written golden case. Golden cases are the regression contract. Synthetic cases (section 11.2) are generated fresh and never copy them.

<!-- code: application/packs.py#register-E01-E05 -->

<!-- code: application/packs.py#register-E06-E10 -->

### 4.4 Generator, auditor and worker prompts

**What this cell does.** It defines the three role prompts: the generator invents fictional cases, the auditor checks them, and the worker solves them. **What changed in v3.8:** the worker prompt now says directly: never repeat a call whose result you already have, call every listed tool before finishing, and copy citation quotes character for character.

<!-- code: application/packs.py#prompts -->

### 4.5 Constrained and state-conditioned step schemas (the main v3.8 fixes)

**What this cell does.** It builds JSON schemas that Ollama compiles into a decoding grammar, so a malformed step cannot be generated in the first place.
- `build_step_schema` (v3.8): `tool` can only be one of the workflow's tools, each with **exactly** its documented argument keys; `findings[].key` can only be a required key.
- `build_state_schema` (v3.8.2): the schema depends on the graph state. While required tools are missing, the only legal step is `kind="tool"` over the tools **not yet called**. Once every required tool has been observed, the only legal step is `kind="final"` with mandatory `status`, `action`, `findings` and `citations`. E06 re-opens `code_check` while the last check has failed.
- `GROUNDED_QUOTES` (v3.8.3, extended in v3.8.4): in the final step, `status` and `action` are fixed to the pack's contract values and the citations are a fixed-length list with exactly one citation per observation, whose `quote` can only be one of that observation's `quote_fragments`. Tool `reference` arguments can only be an identifier written in the request.
- Argument *descriptions* are no longer in the decoding schema. Identifier arguments carry a no-space pattern instead.

**Why.** v3.7 failed on tool arguments (`retrieve {}`, `invoice_read {"invoice_id": …}`, `code_read {}`). v3.8.1 fixed that and then failed on the *envelope*: `kind:"tool"` with `tool:null` plus findings (the model meant to finish), a `final` with no `status`/`action`, and repeated reads of an already-observed ticket. In the synthetic sweep the model also copied the schema description "The exact case reference identifier quoted in the request." into the `reference` value, six times. See section 11.4.

**What this does *not* do.** It removes whole classes of malformed output, which means a pass under this schema measures less of the model's own planning than a pass without it. That is why the switches stay independent: the ablation in section 11.5 reports each level separately.

<!-- code: application/packs.py#step-schema -->

<a id="s5"></a>
## 5. Sandbox tools

### 5.1 Retrieval

**What this cell does.** It ranks the case's three small documents (policy, case reference, distractor) by lexical overlap. In `hybrid` mode it blends that 50/50 with CPU sentence-embedding similarity. This is a real ranking algorithm, but not a retrieval benchmark.

<!-- code: infrastructure/tools.py#retrieval -->

### 5.2 Graph traversal

**What this cell does.** It walks a relationship graph breadth-first up to a depth bound, and returns the real edges it walked. A cycle cannot loop forever.

<!-- code: infrastructure/tools.py#graph-traversal -->

### 5.3 Tool contracts

**What this cell does.** It defines argument names, with descriptions, and a one-line purpose for each tool. Both are shown to the model and copied into the constrained schema.

<!-- code: infrastructure/tools.py#tool-contracts -->

### 5.4 Registry and authorization

**What this cell does.** It authorises each call against the pack's tool list, its exact argument keys and the case reference. New in v3.8: the error message names the expected and the received keys, so the model can correct itself.

<!-- code: infrastructure/tools.py#tool-registry -->

### 5.5 Tool handlers

**What this cell does.** It implements each tool and registers it. Plain record reads expose only their listed fields. `reconcile` and `code_check` refuse to run before the reads they depend on. The browser can only open `sandbox://portal` and click a link it has actually seen.

<!-- code: infrastructure/tools.py#tool-handlers -->

### 5.6 Per-case tool host

**What this cell does.** It stores each successful call as an immutable observation (`obs-1`, `obs-2`, …), the only valid citation targets. New in v3.8: an identical repeated call is refused with a message pointing to the existing observation. In v3.7, E01 read the same ticket four times in a row.

<!-- code: infrastructure/tools.py#sandbox-host -->

<a id="s6"></a>
## 6. Synthetic generation and audit agents

**What this cell does.** The generator invents a new fictional case: clean English or noisy (typos, shorthand). A separate auditor call checks that the request and the world describe the same case. Rejected cases are recorded as `dataset_invalid` and never scored as model failures. Using the same model for generator, auditor and worker means their errors are correlated. That is a stated limitation, not independence.

<!-- code: application/synthetic.py#synthetic-agents -->

<a id="s7"></a>
## 7. The bounded agent graph

### 7.1 Edges and the evidence guard

**What this cell does.** It defines the allowed transitions and the checks a final answer must pass: every required tool was called, every citation quote is an exact substring of the cited observation, every required finding key appears exactly once, and every finding cites observed evidence. E06 additionally requires the last `code_check` to have passed.

<!-- code: application/orchestrator.py#graph-guard -->

### 7.2 The graph runner

**What this cell does.** It defines the runner that executes one case. Each turn: `infer` asks the model for one step → `validate` checks it → `tool` runs it and records an observation → back to `infer`. A rejected step goes to `repair`, which sends the reason back as `repair_feedback`. Too many consecutive repairs end in `quarantine`. Infrastructure errors (timeout, budget, context) go straight to `failed`.

**What changed in v3.8, and why:**

| Change | v3.7 evidence that motivated it |
|---|---|
| Constrained per-pack schema (`constrained=True`) | All three golden cases quarantined on malformed tool arguments |
| Repair feedback now includes the rejected step and the expected argument keys | The model returned byte-identical bad calls three times (same prompt, same output) |
| Duplicate-call guard plus a `progress` hint (`GUIDED_PROGRESS`) | E01 called `ticket_read` four times before moving on |
| Repair limit counts *consecutive* failures and resets after a successful tool call | A single limit for the whole run punished long workflows for early mistakes |
| Observations sent once (`content` only, not `data` and `content`) | Prompt grew from 770 to 1,126 tokens within one case |
| **v3.8.2** State-conditioned schema (`state_conditioned=True`) | v3.8.1 run: `kind:"tool"` with `tool:null` (3×), `final` without `status`/`action` (1×), `ticket_read`/`code_read` repeated 5× |
| **v3.8.2** Citation errors quote the observation text to copy; whitespace is ignored when comparing | v3.8.1 E03 final cited a paraphrase ("The reconciliation of invoice INV-301 shows…") that is not in any observation |
| **v3.8.2** Tool host strips a leading `#` and quotes from `reference` | v3.8.1 synthetic E06: the request said `#calc123`, the world id is `calc123`, the call was refused 3× |
| **v3.8.2** Warm-up uses the chat `num_ctx` | v3.8.1: warm-up 68 s, yet the first timed call still took 79.6 s (model reloaded for the new context size) |

All three switches can be turned off (`constrained=False, guided=False, state_conditioned=False`) to reproduce v3.7 behaviour for an ablation.

<!-- code: application/orchestrator.py#graph-runner -->

<a id="s8"></a>
## 8. Hidden oracle, rubric and failure taxonomy

### 8.1 Oracle

**What this cell does.** It computes the correct answer from the private world alone, never from the model's tool results.

<!-- code: application/evaluation.py#oracle -->

### 8.2 Rubric

**What this cell does.** It grades a run on status, action, no runtime error, no external execution, required tools, every finding value, provenance (each finding cites the right *kind* of tool), the human-approval gate and, for E06, the patch over the whole input domain.

<!-- code: application/evaluation.py#rubric -->

### 8.3 Failure taxonomy (new in v3.8)

**What this cell does.** It assigns every failed run one primary category, such as `tool_argument_schema`, `duplicate_tool_loop`, `citation_grounding`, `wrong_finding` or `transport_timeout`. Models and versions can then be compared by *why* they fail, not just how often.

<!-- code: application/evaluation.py#failure-taxonomy -->

### 8.4 Statistics and run store

**What this cell does.** It reports results per workflow and mode. *Eligible pass rate* excludes invalid datasets. *End-to-end yield* counts every attempted case. Wilson intervals are descriptive only, because synthetic cases are not independent samples.

<!-- code: application/evaluation.py#statistics -->

<a id="s9a"></a>
## 9A. Offline contract suite — no model, GPU or network

**What these cells do.** They run 40 deterministic tests with the scripted double. 26 are the v3.7 contracts. 14 are v3.8 regression tests, each pinned to a failure seen in the real v3.7 run. They test software behaviour, not model quality.

<!-- code: workflows/contracts.py#offline-fixtures -->

<!-- code: workflows/contracts.py#contracts-v37 -->

<!-- code: workflows/contracts.py#contracts-v38 -->

<!-- code: workflows/contracts.py#run-contracts -->

**What the next cell does.** It runs the suite and blocks live work if anything fails. **What you should see:** `Ran N tests … OK` followed by `"passed": true`, in about a second.

```python
import json
OFFLINE_REPORT = None
if RUN_OFFLINE_TESTS:
    OFFLINE_REPORT = run_offline_contracts()
    print(json.dumps(OFFLINE_REPORT, indent=2))
    if REQUIRE_OFFLINE_PASS and not OFFLINE_REPORT['passed']:
        raise RuntimeError('Offline contracts failed; inspect failures before live work')
else:
    print('Offline tests skipped by configuration.')
```

<a id="s10"></a>
## 10. Runtime provisioning

### 10.1 Runtime adapter and warmed backends

**What this cell does.** `prepare_runtime` is the only code that installs Ollama, starts its server or pulls models. `live_backend` builds a backend, checks that the model is installed and warms it up.

<!-- code: workflows/experiments.py#runtime -->

### 10.2 Assemble the live lab

**What this cell does.** If `RUN_LIVE` is on, it provisions Ollama, loads the retriever and builds the three role backends (generator, auditor, orchestrator). **What you should see on a T4:** the install and model download (several minutes the first time), then a warm-up record. v3.7 measured **119 s** for the first, cold call. After that, short tool-call turns took a median of **1.1 s**.

```python
runtime_info = generator = auditor = orchestrator = retriever = None
selected_packs = [PACKS[w] for w in SELECTED_WORKFLOWS]
if RUN_LIVE:
    if REQUIRE_OFFLINE_PASS and (not OFFLINE_REPORT or not OFFLINE_REPORT['passed']):
        raise RuntimeError('Run offline contracts first, or explicitly set REQUIRE_OFFLINE_PASS=False')
    roles = {GENERATOR_MODEL, AUDITOR_MODEL, ORCHESTRATOR_MODEL}
    local = [m for m in roles if resolve_profile(m).provider == 'ollama']
    matrix_local = [m for m in MODEL_MATRIX if m not in roles and resolve_profile(m).provider == 'ollama']
    runtime_info = prepare_runtime(RUNTIME_PLUGIN, OLLAMA_URL, local, matrix_local)
    print('[runtime] Loading retrieval adapter:', RETRIEVAL_MODE, flush=True)
    retriever = RetrievalAdapter(RETRIEVAL_MODE)
    print('[runtime] Loading tokenizer and role backends...', flush=True)
    orchestrator_backend = live_backend(ORCHESTRATOR_MODEL)
    shared = lambda spec: orchestrator_backend if spec == ORCHESTRATOR_MODEL else live_backend(spec)
    generator = SyntheticCaseAgent(shared(GENERATOR_MODEL))
    auditor = DatasetAuditor(shared(AUDITOR_MODEL))
    orchestrator = AgentGraphOrchestrator(orchestrator_backend)
    print('[runtime] Ready for live golden cases:', SELECTED_WORKFLOWS, flush=True)
else:
    print('Live models OFF (RUN_LIVE=False). Only the offline report above was produced.')
```

<a id="s11"></a>
## 11. Live labs and expected vs. real results

### 11.1 Golden harness

**What this cell does.** It runs each selected pack's fixed golden case, saves the full state, the evaluation and every model-call record, and prints one line per case.

<!-- code: workflows/experiments.py#golden-harness -->

### 11.2 Synthetic harness

**What this cell does.** For each mode (clean, noisy) and pack it runs: generate → audit → run if accepted → grade. Every case is appended to `cases.jsonl` as it finishes, so partial runs are kept.

<!-- code: workflows/experiments.py#synthetic-harness -->

### 11.3 Model comparison matrix

**What this cell does.** It runs the same golden cases on several models under identical rules (same schema, prompts, budgets and rubric) and saves one folder per model. A model with a missing key or an unregistered alias is recorded as `skipped`.

<!-- code: workflows/experiments.py#model-matrix -->

**What the next cell does.** It runs golden cases, then synthetic cases, then the optional matrix.

```python
golden_results, synthetic_rows, synthetic_report, golden_path, matrix_results = [], [], None, None, {}
if RUN_LIVE:
    golden_results = run_golden(selected_packs, orchestrator, retriever)
    golden_path = save_golden(golden_results, orchestrator.backend, runtime_info)
    print('Golden artifacts:', golden_path)
    golden_passed = all(r['evaluation']['ok'] for r in golden_results)
    if RUN_SYNTHETIC and (golden_passed or not REQUIRE_GOLDEN_PASS):
        synthetic_rows, synthetic_report = run_synthetic(selected_packs, generator, auditor, orchestrator, retriever, N_CLEAN, N_NOISY)
        RunStore(synthetic_report['path']).write('runtime.json', runtime_info)
        print(json.dumps(synthetic_report['groups'], ensure_ascii=False, indent=2))
    elif RUN_SYNTHETIC:
        print('Synthetic sweep blocked by golden failures (REQUIRE_GOLDEN_PASS=True).')
    extra = [m for m in MODEL_MATRIX if m != ORCHESTRATOR_MODEL]
    if extra:
        orchestrator.backend.unload()   # free the T4 before loading a 9-14B model
        matrix_results = run_model_matrix(extra, selected_packs, retriever, runtime_info)
        print(json.dumps(matrix_results, indent=2))
else:
    print('Golden and synthetic model evaluations NOT RUN (RUN_LIVE=False).')
```

### 11.4 Real results so far (Colab T4, qwen2.5:7b, Ollama)

The tables are generated from committed artifacts when the notebook is built (`results/v37_…`, `results/v38_1_…`, `results/v38_2_…` and `results/v38_3_colab_t4_qwen2.5-7b/`). Nothing is typed by hand. All runs: temperature 0, 8,192-token context.

**Run A — v3.7** (seed 3700):

<!-- generated: v37-results -->

**What it showed.** No case reached a final answer. The model's tool requests were: E01 `ticket_read {"reference":"T-101"}` ×4 (valid but repeated), then `retrieve {}` ×3. E03: `invoice_read {"invoice_id":"INV-301"}` ×3 (the key should be `reference`). E06: `code_read {}` ×3. Each retry was byte-identical because the repair message did not say which key was expected. The cold first call took 119 s against a 120 s timeout.

**Run B — v3.8.1** (seed 3800; constrained schema and progress hint on, warm-up on):

<!-- generated: v381-results -->

**What it showed.** Still 0/3, but for different reasons, and this is progress that the table alone hides:
- **Fixed:** zero tool-argument errors. Every call named the right tool with the right keys (`required_tools` passes in all three golden cases), and the reconcile arithmetic in E03 was correct (`overcharge 5000`, `payable 25000`).
- **New failure 1 — wrong envelope (4 of 18 golden calls).** The flat schema allowed `kind:"tool"` with `tool:null` and findings (E01 call 5, E03 call 9, E06 call 18) and a `final` without `status`/`action` (E03 call 11). The validator refused all of them, correctly. The model knew the answer and mislabelled the step.
- **New failure 2 — repeated reads (E01 calls 3–4, E06 calls 14–17).** The progress hint did not stop `ticket_read` and `code_read` being asked again.
- **New failure 3 — a paraphrased citation.** E03's final quoted a sentence that appears in no observation.
- **New failure 4 — schema text copied into a value (synthetic, E03, 6 calls).** `reference` was set to "The exact case reference identifier quoted in the request." because the argument description sat inside the decoding schema.
- **Weak synthetic data (3 of 6 cases lost before the orchestrator).** The request said "ID 1234567890" while the world id was `ticket_1234567890`; a refund exceeded the schema maximum; and `#calc123` in the request did not match `calc123` in the world.
- **Latency.** Warm-up took 68 s, yet the first timed call took 79.6 s: the warm-up used Ollama's default context and the chat calls use 8,192, which reloads the model. Afterwards, tool turns took a median of about 1.4 s and final answers 4–9 s.

All of these were addressed in v3.8.2 and each fix is pinned by a test that replays the real raw output (`tests/test_v381_artifact.py`).

**Run C — v3.8.2** (seed 3800; state-conditioned schema on):

<!-- generated: v382-results -->

**What it showed.** Still 0/3, and again a new failure:
- **Fixed:** no `output_schema`, `tool_argument_schema` or duplicate-call failure. Every call used the right tool and arguments, and every value was correct.
- **New failure 1 — re-formatted quotes (9 of 9 final answers).** The model cited the right observation with the right value but wrote `"refund_cents": 9000` where the observation holds `"refund_cents":9000`, or paraphrased it, and repeated the mistake after feedback.
- **New failure 2 — wrong `status`.** An E01 synthetic case answered `PROPOSED` where the workflow requires `PENDING_APPROVAL`; all values were right.
- **Weak synthetic data.** All six cases reached the orchestrator (three of six before), but one case used a placeholder reference with contradictory figures and the same-model auditor accepted it.
- **Preflight bug.** `prompt_tokens_estimated` was 2 in every call because the token count measured the keys of the tokenizer output instead of its tokens. The guard that protects the 8K window was therefore inactive.
- **Latency.** The first timed call still took 93.8 s after a 71 s warm-up. The cause is not established; v3.8.3 adds a tiny chat call to the warm-up, which has not been measured yet.

v3.8.3 turned citation quoting from *typing* into *selecting* (`quote_fragments`), fixed `status` and `action` per workflow, repaired the preflight and validated synthetic cases deterministically. Run D measures it.

**Run D — v3.8.3** (seed 3800; grounded quotes on):

<!-- generated: v383-results -->

**What it showed.** The first passing cases: **golden 2/3 and synthetic 4/6.**
- **Confirmed:** no re-formatted quote, a real token estimate (it was 2 before), and a 1.8 s first call (it was 93.8 s). E01 and E03 pass with the right tools, values, citations and approval gate.
- **E06 golden failed on one missing citation.** A finding named `obs-1`, but only `obs-3` was cited. The refusal did not say which id was missing and the model resent the same step three times.
- **Synthetic E03 noisy:** "invoice_id 12345" in the request became the reference `invoice_id_12345` (refused three times).
- **Synthetic E06 noisy:** a model arithmetic limit; the same wrong expression was sent four times until the duplicate guard stopped it.
- **Weak evidence is visible.** E01 passes while citing `"mode":"hybrid"` for `requires_approval`: the rubric checks that a quote is verbatim, not that it supports the finding (backlog V39-02).

v3.8.4 makes the final carry exactly one citation per observation and lets a tool `reference` be chosen only among the identifiers written in the request. Run E measures it.

**Run E — v3.8.4** (seed 3800):

<!-- generated: v384-results -->

**What it showed.** **Golden 3/3 and synthetic 5/6**, with no repair in any golden case and a 1.4 s first call.
- **Confirmed:** the E06 final cites all three observations; E03 noisy uses the reference `12345` written in the request.
- **Still failing:** synthetic E06 noisy. The model re-sent a wrong arithmetic expression and the graph stopped at its step budget. This is a model limit, not a schema fault.
- **Unchanged weakness:** E01 passes while citing `"mode":"hybrid"` for `requires_approval` (backlog V39-02).

### 11.5 Expected results (hypotheses to be measured, not results)

**Scorecard of the v3.8.3 predictions against Run D.**

| Prediction made for v3.8.3 | Run D | Verdict |
|---|---|---|
| `citation_grounding` failures drop to zero | One remains (E06, a missing citation) | **Partly confirmed**: the Run C cause is gone |
| At least 2 of 3 golden cases pass | 2 of 3 | **Confirmed** |
| At least 4 of 6 synthetic cases reach the orchestrator and 2 pass | 6 reached it, 4 passed | **Confirmed** |

**Scorecard of the v3.8.4 predictions against Run E.**

| Prediction made for v3.8.4 | Run E | Verdict |
|---|---|---|
| Golden 3 of 3 pass | 3 of 3 | **Confirmed** |
| At least 5 of 6 synthetic pass and no `wrong_reference` | 5 of 6, no `wrong_reference` | **Confirmed** |

**Run F (the section 11.6 campaign)** measured the ablation, local-model, thinking-model and auditor rows in one session:

<!-- generated: v384-campaign -->

What the campaign showed:

- **The ablation ladder reproduces the five-run history on one model.** The same code with the switches turned on one at a time gives 0/3, 0/3, 2/3 and 3/3. Each switch removes the failure class it was built for, so the improvement is not a lucky seed or a different build.
- **E06 (the arithmetic fix) is the case that separates models, and size alone does not decide it.** `qwen2.5:3b`, `qwen2.5:14b` and `llama3.1:8b` loop on `code_check`; `ministral-3:14b`, `qwen3.5:9b` and `gemma4:12b` pass it.
- **Thinking models keep the grammar constraint.** No `output_schema` failure in any run, with `think` on or off. Thinking costs time (first call 5.6–8.4 s against 2–2.8 s, total 2–3× longer) and bought no accuracy on these cases.
- **`granite4.1:8b` and `llama3.1:8b` fail E01 on `missing_provenance`**: they cite a fragment that does not support `requires_approval`.
- **The different-family auditor study is inconclusive.** 4/6 against 5/6, and the generated cases differ between runs, so a one-case difference is noise. It needs the scale campaign.
- **The main run repeats Run E exactly:** golden 3/3 and synthetic 5/6 with the same single failure (E06 noisy, `step_budget`).

**Scorecard of the Run F predictions.**

| Prediction | Run F | Verdict |
|---|---|---|
| Ablation `GROUNDED_QUOTES=False` reproduces Run C | Rung 2: 2/3, the failure is `citation_grounding` (E01) | **Partly confirmed**: same failure class, fewer cases fail because v3.8.4 fixed others |
| Ablation `STATE_CONDITIONED=False` reproduces Run B | Rung 1: 0/3 with `output_schema` and `duplicate_tool_loop` | **Confirmed** |
| All switches off reproduces v3.7 | Rung 0: 0/3, `duplicate_tool_loop` in all three | **Partly confirmed**: 0/3 as in Run A, but the label is `duplicate_tool_loop`, not `tool_argument_schema` |
| qwen2.5:3b is no better than 7B | 2/3 against 3/3 | **Confirmed** (one case; not significant) |
| qwen2.5:14b is about 2× slower and at least as accurate | First call 2.9 s against 1.4 s; 2/3 against 3/3 | **Falsified on accuracy**: the 14B fails E06 |
| Non-thinking granite/ministral: no envelope failures, differences in citations and arithmetic | granite 2/3 (`missing_provenance`), ministral-3:14b 3/3, no `output_schema` failure | **Confirmed** |
| Thinking models are slower, and may weaken `format` | 3/3 in both modes; first call 4–6× slower with thinking on; no `output_schema` failure | **Partly confirmed**: slower yes, `format` held, no accuracy gain over the 3/3 ceiling |
| Different-family auditor changes the accepted set | 4/6 against 5/6; the generated cases differ, so it is one case of noise | **Not decided** at this sample size |

**Hypotheses that are still open.** The rows scored above are kept for reference; the scale campaign and the API models are the open ones.

| Setting | Prediction | Reasoning | Falsified if |
|---|---|---|---|
| Offline suite (any CPU) | All tests pass in about 1 s | Deterministic | Any failure |
| Ablation `GROUNDED_QUOTES=False` | Reproduces Run C | Same code path as v3.8.2 | It passes 3/3 |
| Ablation `STATE_CONDITIONED=False` | Reproduces the Run B failure mix | Same code path as v3.8.1 | It passes 3/3 |
| Ablation all switches off | Reproduces v3.7 (`tool_argument_schema`) | | It passes |
| T4, qwen2.5:3b | Fewer finals reached; more wrong findings than 7B | Smaller models copy and plan worse | 3B ≥ 7B |
| T4, qwen2.5:14b (Q4, ~9 GB) | About 2× slower per token than 7B and at least as accurate | Decoding on a T4 is memory-bandwidth bound | Not slower, or less accurate |
| T4, non-thinking instruct: granite4.1:8b, ministral-3:8b/14b | Similar to qwen2.5:7b on the envelope (now enforced by the decoder). Differences appear in citations and arithmetic | The grammar makes the envelope model-independent | They fail the envelope |
| T4, hybrid thinking: gemma4:12b, qwen3.5:9b, qwen3:14b | Higher accuracy but slower; risk of `format` being weakened or reasoning tokens exhausting `num_predict` | Thinking spends output tokens before the JSON; some builds are reported to ignore `format` when thinking is disabled | `output_schema` failures, or median step above 20 s |
| T4, gpt-oss:20b, deepseek-r1 (not registered) | Expected to be unusable here | 20B at MXFP4 is near the whole 15 GB; R1 reasoning traces collide with `format` | Not run on purpose; add a profile to test |
| API (`openai:<model>`, `anthropic:<model>`) | Near the ceiling on the three golden cases. Differences are latency and cost | Strong instruction following and exact copying | Repeated grounding failures (would point to the rubric or prompts) |

**Caveats that apply to every row.** (1) Three golden cases cannot separate models: a 95% interval on 0/3 is 0–56%, on 3/3 it is 44–100% (`wilson_95`). Use the scale campaign for statements about rates. (2) The generator, auditor and orchestrator are the same model by default, so their errors are correlated; set `TAGO_AUDITOR_MODEL` to a different family for independence. (3) The model tags listed for the matrix were checked on ollama.com, but their Hugging Face tokenizer ids were not opened here; a wrong id fails the preflight with a clear `ContextError` and is recorded as that model's `error`.

The comparison is only valid when every model runs under the same `PROMPT_VERSION`, schema, budgets and rubric. The matrix (11.3) enforces this.

### 11.6 Experiment campaign: every extra measurement in the same session

**What this section does.** The main run (11.1 to 11.3) measures the shipped configuration. This cell then reuses everything already loaded (the offline-checked code, the retriever, the warm model) and repeats the golden cases under different settings, so a single "Run all" produces all of the data:

| Item | What changes | Question it answers |
|---|---|---|
| Ablation ladder (4 runs) | Switches turned on one at a time: v3.7 behaviour, then constrained schema with progress hints, then state-conditioned steps, then grounded quotes | Which switch removes which failure? |
| Different-family auditor | The synthetic sweep is audited by `CAMPAIGN_AUDITOR` instead of the same model | How much of the synthetic pass rate came from the generator and auditor sharing one model? |
| Other local models | One golden run per model in `CAMPAIGN_MODELS` | Does the result depend on `qwen2.5:7b`? |
| Thinking models | Each model in `CAMPAIGN_THINKING_MODELS` runs with `think` left at its default and with `think=false` | Does a thinking model keep the grammar constraint, and what does switching thinking off change? |

**How it stays safe on a T4.**
- Models are downloaded one at a time and deleted again after use, so the disk never holds more than one extra model.
- A model that fails to download, load or run is recorded as an error and the campaign continues.
- `CAMPAIGN_MAX_MINUTES` stops new items from starting; the rest are recorded as skipped. Results are saved after each item, so a disconnected session still leaves the finished items.
- Every golden run is saved in the same format as the main run (`python -m tago analyze <folder>` reads it) and records the switches it really used.
- Reading the results: three golden cases per item cannot rank models. A difference of one case is not evidence; use the scale campaign for rates.

<!-- code: workflows/experiments.py#campaign -->

**What the next cell does.** It runs the campaign when `RUN_CAMPAIGN` is on and prints one table. Expect roughly two to three hours on a T4 with the default lists (model downloads dominate); shorten the lists or set `CAMPAIGN_MAX_MINUTES` to bound it.

```python
campaign_results = {'path': None, 'items': []}
if RUN_LIVE and RUN_CAMPAIGN:
    from IPython.display import Markdown, display
    campaign_results = run_campaign(selected_packs, retriever, runtime_info, ORCHESTRATOR_MODEL, ablation=CAMPAIGN_ABLATION,
        models=[m for m in CAMPAIGN_MODELS if m != ORCHESTRATOR_MODEL], thinking_models=CAMPAIGN_THINKING_MODELS,
        auditor=CAMPAIGN_AUDITOR or None, n_clean=N_CLEAN, n_noisy=N_NOISY, max_minutes=CAMPAIGN_MAX_MINUTES)
    display(Markdown(campaign_markdown(campaign_results['items'])))
    print('Campaign artifacts:', campaign_results['path'])
else:
    print('Campaign NOT RUN (RUN_LIVE and RUN_CAMPAIGN must both be on).')
```

<a id="s12"></a>
## 12. Analysis, frozen datasets and export

### 12.1 Analyse runs

**What this cell does.** It loads saved run folders (v3.7 or v3.8 format) and builds a comparison table: pass count, failure category, model calls, duplicate calls, repairs, first-call and median warm latency.

<!-- code: workflows/analysis.py#analysis -->

**What the next cell does.** It compares this run with the real v3.7 baseline, if the repository's `results/` folder is present, and prints every failed case in full.

```python
from pathlib import Path
from IPython.display import Markdown, display
runs = [p for p in [Path('results/v37_colab_t4_qwen2.5-7b'), Path(golden_path) if golden_path else None,
                    *[Path(v['path']) for v in matrix_results.values() if 'path' in v],
                    *[Path(i['path']) for i in campaign_results['items'] if i.get('kind') != 'auditor' and i.get('path')]] if p and p.exists()]
if runs:
    display(Markdown(markdown_report([run_summary(p) for p in runs])))
else:
    print('No run folders to analyse yet.')
for row in golden_results + synthetic_rows:
    if not row.get('evaluation', {}).get('ok'):
        print(row['workflow_id'], row.get('evaluation', {}).get('failure_category'), (row.get('state') or {}).get('repair_log'))
```

### 12.2 Freeze, replay and export

**What this cell does.** `freeze_dataset` saves the accepted synthetic inputs (never the model's answers) with a SHA-256 manifest. `replay_frozen` re-runs a model on exactly those inputs, which is how prompt or model changes should be compared. `export_bundle` zips a run folder.

<!-- code: workflows/experiments.py#freeze-replay-export -->

**What the next cell does.** It zips every run folder produced in this session and shows download links. Colab storage is wiped when the session ends, so download the ZIPs and commit them under `results/` to publish the measurement.

```python
artifact_bundles = [export_bundle(p) for p in [golden_path, synthetic_report and synthetic_report['path'],
                    *[v['path'] for v in matrix_results.values() if 'path' in v], campaign_results['path']] if p]
if artifact_bundles:
    from IPython.display import HTML
    import html
    for bundle in artifact_bundles:
        absolute = Path(bundle).resolve()
        if str(absolute).startswith('/content/'):
            href = '/files/' + absolute.relative_to('/content').as_posix()
            display(HTML('<a href="' + html.escape(href, quote=True) + '" download>Download ' + html.escape(absolute.name) + '</a>'))
        else:
            print('Saved:', absolute)
    print('If a link does not open, use the Colab Files pane. Commit the unzipped folder under results/<name>/.')
else:
    print('No live artifacts generated in this session.')
```
