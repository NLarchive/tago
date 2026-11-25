# Run E — prompt v38.4, Colab T4, Ollama 0.40.18, qwen2.5:7b

Raw artifacts exported by the notebook (section 12), unchanged. Settings: workflows E01, E03, E06; temperature 0; seed 3800;
context 8,192; constrained, guided, state-conditioned and grounded-quote decoding on (one citation per observation, reference
candidates); one clean and one noisy synthetic case per workflow; generator, auditor and orchestrator all `qwen2.5:7b`.

    python -m tago analyze results/v38_4_colab_t4_qwen2.5-7b

Result: **golden 3/3, synthetic 5/6.** The warm-up took 179 s (78.5 s model load plus the first chat request); the first timed call
took 1.4 s. No golden case needed a repair.
The only failure is synthetic E06 noisy: the model re-sent a wrong arithmetic expression (`n * (n - 1) // 2 + 100 - 100`), four
duplicate calls were refused and the graph hit its step budget. That is a limit of the 7B model's arithmetic, not a schema fault.
See `docs/v38_review.md`, section 9, and `tests/test_v384_artifact.py`.
