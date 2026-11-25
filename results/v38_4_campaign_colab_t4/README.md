# Run F — experiment campaign (notebook section 11.6), Colab T4, Ollama 0.40.18

One "Run all" of the v3.8.4 notebook (prompt v38.4, seed 3800, three golden cases per item, one clean and one noisy synthetic case per workflow).
Generator and orchestrator are `qwen2.5:7b` unless the row says otherwise.

| Folder | What it is |
|---|---|
| `golden-rung0…rung3` | Ablation ladder on `qwen2.5:7b`: v3.7 behaviour, + constrained schema and progress hints, + state-conditioned steps, + grounded quotes |
| `golden-<model>` | The same golden cases on another local model (`qwen2.5:3b`, `qwen2.5:14b`, `llama3.1:8b`, `granite4.1:8b`, `ministral-3:14b`) |
| `golden-<model>-think-default` / `-think-off` | A thinking model run with `think` left at its default and with `think=false` (`qwen3.5:9b`, `gemma4:12b`) |
| `synthetic-auditor-granite4.1_8b` | Synthetic sweep with a different-family auditor (generator and orchestrator `qwen2.5:7b`) |
| `campaign_summary.json`, `campaign_report.md` | One row per item, as written by the notebook |

Folder names have the random run id removed; `campaign_summary.json` is unchanged and still shows the Colab paths.
The repeat of the main run is in `../v38_4_repeat_colab_t4_qwen2.5-7b/`. Interpretation: `docs/v38_review.md`, section 10.
