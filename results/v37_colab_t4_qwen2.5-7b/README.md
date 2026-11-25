# v3.7 golden run — Google Colab T4, Ollama, qwen2.5:7b

Raw artifacts exported by `tago_v37_agent_graph_lab.ipynb` (section 12.4), copied unchanged from
`.legacy/golden-1509ee80638d413d92756669d69bac71/`. Settings: E01, E03, E06; temperature 0; seed 3700;
context 8192; tokenizer `Qwen/Qwen2.5-7B-Instruct`. `runtime.json` records the Colab plugin and the
Ollama version string exactly as the run reported it.

Regenerate the summary with:

    python -m tago analyze results/v37_colab_t4_qwen2.5-7b

Result: 0/3 golden cases passed, all `QUARANTINED` with category `tool_argument_schema`. The synthetic
sweep did not run, because v3.7 blocked it after golden failures. See `docs/v38_review.md` for the analysis.
