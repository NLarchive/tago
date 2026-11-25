# Run C — prompt v38.2, Colab T4, Ollama 0.40.18, qwen2.5:7b

Raw artifacts exported by the notebook (section 12), unchanged. Settings: workflows E01, E03, E06; temperature 0; seed 3800;
context 8,192; constrained schema, progress hint and state-conditioned schema on; one clean and one noisy synthetic case per
workflow; generator, auditor and orchestrator all `qwen2.5:7b`.

    python -m tago analyze results/v38_2_colab_t4_qwen2.5-7b

Result: **0/3 golden, 0/6 synthetic** — but all 6 synthetic cases reached the orchestrator (3 of 6 in run B).
Golden failures are all `citation_grounding`: every call used the right tool and arguments, the findings were correct, and the model
re-formatted the JSON it quoted. No `output_schema`, `tool_argument_schema` or duplicate-call failure remains.
See `docs/v38_review.md`, section "Run C", and `tests/test_v382_artifact.py`.
