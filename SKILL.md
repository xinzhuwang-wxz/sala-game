---
name: soar-autoresearch
description: >-
  For autonomous SOAR inference optimization loops. Trigger when user asks for
  iterative tuning, overnight experiments, benchmark-driven optimization, or
  automated keep/discard decisions.
---

# SOAR Strategy Skill

## Trigger Conditions

- 用户提到: 自动调优 / 自动实验 / overnight 跑实验 / 提速竞赛 / SOAR。
- 有明确评估指标: S1/S8/Smax + accuracy。

## Core Rules

1. Accuracy gate first (`normalized_acc >= 97`), otherwise fail.
2. Keep only if estimated final score improves.
3. One primary hypothesis per round.
4. Always append `results.tsv`.

## Strategy Ladder

1. `args_tuning_low_risk`
2. `quantization_medium_risk`
3. `kernel_backend_medium_high_risk`
4. `speculative_high_risk`
5. `stacked_combo_when_stable`

## Suggested Experiment Templates

- args tuning:
  - switch backend
  - tune `chunked-prefill-size`
  - toggle cuda graph/radix cache/warmup
- quantization:
  - GPTQ W4A16 baseline
  - group_size search
  - selective high-precision fallback
- kernels:
  - trim redundant cast/copy
  - sparse topk path micro-optimizations
- speculative:
  - minimal viable MiniCPM compatibility path

## Stop / Pivot Rules

- 3 consecutive crashes in one lane -> downgrade risk lane.
- no gain in 8 rounds -> pivot to next strategy phase.
