| Item | Kind | Model | Think | Result | Failure categories | First call s | Minutes |
|---|---|---|---|---|---|---|---|
| rung0-v37-behaviour | ablation | qwen2.5:7b | - | 0/3 | {"duplicate_tool_loop": 3} | 2.31 | 0.5 |
| rung1-constrained-guided | ablation | qwen2.5:7b | - | 0/3 | {"duplicate_tool_loop": 2, "output_schema": 1} | 2.51 | 0.4 |
| rung2-state-conditioned | ablation | qwen2.5:7b | - | 2/3 | {"citation_grounding": 1} | 1.46 | 1.0 |
| rung3-grounded-quotes | ablation | qwen2.5:7b | - | 3/3 | {} | 1.36 | 0.7 |
| auditor-granite4.1_8b | auditor | generator/orchestrator qwen2.5:7b, auditor granite4.1:8b | - | 4/6 | {"tool_refused": 1, "step_budget": 1} | - | 4.5 |
| qwen2.5_3b | model | qwen2.5:3b | default | 2/3 | {"duplicate_tool_loop": 1} | 0.71 | 0.3 |
| qwen2.5_14b | model | qwen2.5:14b | default | 2/3 | {"duplicate_tool_loop": 1} | 2.86 | 1.2 |
| llama3.1_8b | model | llama3.1:8b | default | 1/3 | {"missing_provenance": 1, "duplicate_tool_loop": 1} | 2.21 | 0.7 |
| granite4.1_8b | model | granite4.1:8b | default | 2/3 | {"missing_provenance": 1} | 2.46 | 0.9 |
| ministral-3_14b | model | ministral-3:14b | default | 3/3 | {} | 3.86 | 1.4 |
| qwen3.5_9b-think-default | thinking | qwen3.5:9b | default | 3/3 | {} | 8.44 | 2.4 |
| qwen3.5_9b-think-off | thinking | qwen3.5:9b | off | 3/3 | {} | 1.95 | 0.8 |
| gemma4_12b-think-default | thinking | gemma4:12b | default | 3/3 | {} | 5.56 | 3.2 |
| gemma4_12b-think-off | thinking | gemma4:12b | off | 3/3 | {} | 2.77 | 0.9 |
