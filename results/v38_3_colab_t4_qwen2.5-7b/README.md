# Run D — prompt v38.3, Colab T4, Ollama 0.40.18, qwen2.5:7b

Raw artifacts exported by the notebook (section 12), unchanged. Settings: workflows E01, E03, E06; temperature 0; seed 3800;
context 8,192; constrained, guided, state-conditioned and grounded-quote decoding on; one clean and one noisy synthetic case per
workflow; generator, auditor and orchestrator all `qwen2.5:7b`; retrieval `hybrid`.

    python -m tago analyze results/v38_3_colab_t4_qwen2.5-7b

Result: **golden 2/3 (E01, E03 pass), synthetic 4/6.** The first run with passing cases. The warm-up took 168 s (75.6 s model load
plus the first chat request) and the first timed call took 1.8 s.
Failures: E06 golden (a finding named `obs-1` but only `obs-3` was cited), synthetic E03 noisy (`invoice_id_12345` used as the
reference), synthetic E06 noisy (the same refused `code_check` four times).
See `docs/v38_review.md`, section 8, and `tests/test_v383_artifact.py`.
