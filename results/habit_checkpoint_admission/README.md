# HABIT checkpoint-robust admission

This package applies one fixed target-identity plus event-proximity admission
rule to 720 frozen outputs from each of Qwen3-VL-8B, Qwen3-VL-32B, and
InternVL3-8B. The 60 HABIT episodes, queried targets, temporal windows,
event-proximity model, threshold, and reference outcomes are identical across
checkpoints. No admission threshold is refitted per checkpoint.

## Main result

| Checkpoint | Candidate: correct / wrong | Combined rule: correct / wrong |
|---|---:|---:|
| Qwen3-VL-8B | 53 / 91 | 51 / 1 |
| Qwen3-VL-32B | 57 / 86 | 54 / 1 |
| InternVL3-8B | 27 / 71 | 24 / 1 |

The task-balanced combined-minus-candidate change in the all-case wrong
admission rate is -12.50 percentage points for Qwen3-VL-8B (95% CI
[-13.75,-11.25]), -11.81 points for Qwen3-VL-32B
([-13.19,-10.42]), and -9.72 points for InternVL3-8B
([-11.67,-7.78]). Intervals use 10,000 episode-within-task bootstrap draws.

## Files

- `scored_rows_three_checkpoint.csv`: 2,160 frozen foundation-model rows.
- `event_proximity_predictions.csv`: frozen conventional-encoder outputs.
- `checkpoint_admission_decisions.csv`: row-level outputs of the fixed rule.
- `checkpoint_admission_summary.csv`: checkpoint-by-evidence counts and rates.
- `checkpoint_admission_by_task.csv`: task-stratified outcomes.
- `checkpoint_admission_bootstrap.csv`: paired task-balanced contrasts.
- `foundation_model_source_gate.json` and `event_proximity_source_gate.json`:
  upstream source-completion and development/test-separation evidence.
- `gate.json`: fixed-input hashes, denominators, expected counts, and scope.

Run `python analysis/habit_checkpoint_admission.py` from the repository root to
reproduce the derived tables.

## Scope

This is a frozen-output robustness analysis of one admission rule. It shows
that the error-removal pattern is not peculiar to the selected 8B checkpoint.
It does not establish checkpoint independence, closed-loop task success,
physical robot safety, or participant outcomes.
