# Changelog

Measured results for each version are in [`results/`](results/); the analysis is in [`docs/v38_review.md`](docs/v38_review.md).

## Unreleased
- Run F measured with the campaign: ablation ladder 0/3, 0/3, 2/3, 3/3; other models and thinking models in `results/v38_4_campaign_colab_t4/`.
- Notebook section 11.6 and `run_campaign()`: ablation ladder, different-family auditor, local and thinking models in one Colab run (`TAGO_RUN_CAMPAIGN`, `TAGO_CAMPAIGN_*`). Run folders now record `grounded_quotes` and the per-backend `think` setting.

## 3.8.4 (prompt `v38.4`) — Run E: golden 3/3, synthetic 5/6 (first fully passing golden run)
- The final answer carries exactly one citation per observation, so a finding can no longer name an uncited observation. The refusal names the missing ids.
- Tool `reference` arguments are chosen among the identifiers written in the request (`reference_candidates`).
- Argument text for `retrieve` rewritten as short guidance (the previous sentence was copied into a query).

## 3.8.3 (prompt `v38.3`) — Run D: golden 2/3, synthetic 4/6 (first passing run)
- Public Hugging Face downloads (embedding model, tokenizers) retry without a token when Colab injects an expired `HF_TOKEN` secret (previously `401 Unauthorized` stopped the run before the first model call).
- Grounded quotes: a citation quote can only be an exact fragment of the cited observation; `status` and `action` are fixed per workflow (`GROUNDED_QUOTES`).
- Token preflight counted the keys of the tokenizer output; fixed, with a plausibility floor.
- Synthetic cases are validated deterministically (placeholders, field names, stated numbers) before the auditor runs.
- Warm-up sends a tiny chat call with the chat context.
- The arithmetic-rule error lists the allowed syntax.
- Model profiles for granite4.1, ministral-3, gemma4, qwen3.5, phi4 and qwen3:14b (status `candidate`, tokenizer ids unverified).

## 3.8.2 (prompt `v38.2`) — Run C: 0/3 golden, all `citation_grounding`
- State-conditioned step schema: only the legal step can be decoded.
- No prose descriptions in the decoding schema; identifier arguments have a no-space pattern.
- The synthetic generator redrafts with the exact validation error.
- Citation errors show the observation text; whitespace is ignored when comparing.

## 3.8.1 (prompt `v38.1`) — Run B: 0/3 golden, all `output_schema`
- Constrained tool schema, progress hint, duplicate-call guard, informative repair messages, warm-up, API backends, failure taxonomy, installable package, CI and a generated notebook.

## 3.7 — Run A: 0/3 golden, all `tool_argument_schema`
- Ten enterprise workflow packs, tokenizer preflight, offline contract suite, first Colab T4 run.
