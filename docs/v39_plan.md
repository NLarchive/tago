# TAGO v3.9 plan

The plan is ordered by what the measurements in `docs/v38_review.md` (sections 8–10) show, not by what is easiest.

## What v3.8 established

| Measured fact | Evidence |
|---|---|
| With every decoder constraint on, the 7B model passes 3/3 golden cases | Runs E and F, `results/v38_4_*` |
| Each switch removes one failure class (0/3, 0/3, 2/3, 3/3) | Ablation ladder, Run F |
| The remaining errors are semantic: arithmetic (E06) and the relevance of cited evidence | Run F, N-25, N-27, N-29 |
| Thinking models keep the grammar constraint; thinking costs 2–3× the time | Run F, N-28 |
| Three golden cases cannot rank models (3/3 has a 44–100% interval) | `wilson_95` |

## Priorities

1. **Scale campaign (results task 5).** Before any claim about rates: at least 20 clean and 20 noisy synthetic cases per workflow on the best model, then `freeze_dataset()` so every later model replays identical inputs. This also decides the auditor question (N-30), which one sweep cannot.
2. **E06 arithmetic loops (N-27, V39-10).** Three of five other models loop on `code_check` after a near miss. Test a repair message that shows the failing check's result and the expected values; keep it only if the scale campaign shows a gain.
3. **Evidence relevance (V39-02, V39-08, N-29).** Require a cited quote to contain the finding's value. `granite4.1:8b` and `llama3.1:8b` already fail this check on E01 while `qwen2.5:7b` passes with an unrelated fragment, so a stricter rule would re-baseline every earlier result. Do it once, and re-run Runs E and F.
4. **Different-family auditor at scale (V39-03).** Compare the accepted and rejected sets against the same-model auditor on the frozen dataset.
5. **Workflows with long tool histories (V39-07).** E05 and E10 have at least eight tool calls; measure whether failures grow with the prompt length.

## Later or optional

- Human-approval resume path (V39-05) and an MCP tool-host adapter (V39-06): product features, not measurements.
- OpenAI strict structured outputs (V39-04) and the API matrix: optional, outside the T4 focus.
- E09 `alert_read` should not reveal the sensitive asset (V39-01): changes task difficulty, so schedule it with the re-baseline in item 3.
- Slow first request (V39-09): mitigated by the warm-up; the first call is 1.4 s in practice.

## Not planned

Models that do not fit a 15 GB T4 (for example `gpt-oss:20b`) and models whose reasoning traces collide with the JSON format.
