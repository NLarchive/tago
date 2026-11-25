# v3.8.1 golden + synthetic run — Google Colab T4, Ollama 0.40.17, qwen2.5:7b

Raw artifacts exported by `tago_v38_t4-agent_graph_orchestration.ipynb` (section 12) from the code at prompt version
`v38.1` and copied unchanged from `results/v38-colab/golden-afd1a63183d6/` and
`results/v38-colab/synthetic-c85fe3ab057b/`. Settings: E01, E03, E06; temperature 0; seed 3800; context 8192;
constrained schema and progress hint on; one clean and one noisy synthetic case per workflow; all three roles `qwen2.5:7b`.

    python -m tago analyze results/v38_1_colab_t4_qwen2.5-7b

Result: **0/3 golden, 0/3 synthetic cases that reached the orchestrator, 3/6 synthetic cases lost at generation.**
Golden failure category `output_schema` ×3; synthetic `wrong_reference` ×3 (the model copied a schema description into
`reference`, or sent `#calc123`). No `tool_argument_schema` failures: the v3.7 problem is fixed. See `docs/v38_review.md`
(section "v3.8.1 run") and `tests/test_v381_artifact.py`, which replays the raw outputs against the v3.8.2 schemas.
