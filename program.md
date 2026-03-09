# SOAR AutoResearch Program Manual (中英混合)

面向通用 LLM Agent 的自动实验手册。目标是在 OpenBMB SOAR 竞赛中，持续优化 `MiniCPM-SALA` 推理性能，同时保证正确性门槛。

## 0) Mission / Objective

- Primary objective: maximize final score proxy under competition rule.
- Official score formula:
  - `Final = (S1 * 0.4 + S8 * 0.3 + Smax * 0.3) * C`
  - **IMPORTANT**: S1/S8/Smax are **relative latency scores** output by the platform (higher = better).
    Internally the platform converts raw benchmark durations (seconds, lower=better) to these normalized scores.
    Our local `estimated_score` proxy inverts raw durations: `(0.4/t1 + 0.3/t8 + 0.3/tmax) * C * 10000`.
    **Lower raw benchmark duration → higher S score → higher Final.**
  - `C` depends on normalized accuracy (`overall_accuracy` in `summary.json`):
    - `< 97%` -> `C = 0` (hard fail — total score zeroed)
    - `[97%, 99%)` -> `C = 0.92`
    - `>= 99%` -> `C = 1.0`
- Core rule: **Accuracy first, then speed**. 任何速度提升如果导致 `C=0`，都视为失败。
- **Submissions**: 3 per day maximum. Each attempt is precious — pre-validate locally.
- **Private eval set**: Platform uses hidden `perf_private_set.jsonl` to validate top scorers.
  `ori_accuracy >= 73.6` (i.e. `overall_accuracy >= 92%` based on 80pt baseline) = "valid".
  Only the top-5 valid results are eligible for prizes.

## 0.1) Hardware & Submission Constraints

| Constraint | Value |
|---|---|
| GPU | Single NVIDIA RTX PRO (high-end) |
| CPU | 20 cores |
| Memory | 128 GiB |
| Max eval time | 5 hours (excluding queue wait) |
| Max submission size | 2 GB (.tar.gz) |
| Package manager | **`uv`** (not pip) — use `uv pip install` inside container |
| SGLANG_SERVER_ARGS | **Must use hyphenated names** (e.g. `--dense-as-sparse`, NOT `--dense_as_sparse`). The container does NOT auto-convert underscores. |
| Official default SGLANG_SERVER_ARGS | `--disable-radix-cache --attention-backend flashinfer --chunked-prefill-size 32768` (applied when SGLANG_SERVER_ARGS env is empty/unset) |

## 1) Repo Scope and Allowed Changes

工作目录：`/Users/bamboo/Githubs/sala-game/`

### 可修改 (Allowed)

- `prepare_env.sh`
- `prepare_model.sh`
- `preprocess_model.py`
- `sglang/python/**`
- 自动化脚本与记录文件（如 `run_experiment.sh`, `results.tsv`, `parse_results.py` 等）

### 不建议修改 (Read-only for fair comparison)

- `eval-official/eval_model.py`
- `eval-official/bench_serving.sh`
- `eval-official/perf_public_set.jsonl`

## 2) Experiment Principles

1. 单次实验只改一个主变量（one primary hypothesis）。
2. 每轮必须记录：`S1/S8/Smax/accuracy/C/estimated_score/status/description`。
3. Crash、OOM、timeout 也必须记录。
4. 保留原则：
   - `C=0` -> 必丢弃
   - `C>0` 且估计分数提升 -> keep
   - 无提升或倒退 -> discard
5. 简洁优先：同等收益时，选更简单、更稳定、更易维护的改动。

## 3) Standard Workflow (Infinite Loop)

### Phase A: Initialize Once

1. 创建实验分支：
   - `git checkout -b autoresearch/soar-$(date +%m%d)-gpu0`
2. 确认 `results.tsv` 存在且有表头。
3. 跑一次 baseline，写入第一条 `keep` 记录。

### Phase B: Autonomous Iteration

Repeat until user stops:

1. Read current best row in `results.tsv`.
2. Choose one hypothesis from strategy library.
3. Modify code.
4. Commit with short experiment message.
5. Run:

```bash
bash run_experiment.sh \
  --model-input /path/to/model \
  --description "phaseX: short hypothesis"
```

6. Check `results.tsv` new row.
7. Decision:
   - `status=keep` -> continue from current HEAD
   - `status=discard` or `crash` -> revert experiment commit (`git reset --hard HEAD~1`) and continue

## 4) Evaluation Pipeline (Docker-first)

`run_experiment.sh` 默认走 Docker（适配租用 GPU 环境）：

1. `docker run --gpus all --network host ...`
2. `source prepare_env.sh`
3. `bash prepare_model.sh --input ... --output ...`
4. 启动 SGLang server (`python3 -m sglang.launch_server`)
5. 等待 `/v1/models` ready
6. 跑速度评测：`eval-official/bench_serving.sh`
7. 跑正确性评测：`eval-official/eval_model.py`
8. `parse_results.py` 解析并追加一行到 `results.tsv`

## 5) Decision Rules

### Hard Gates

- Gate-1: server must be healthy.
- Gate-2: accuracy normalized >= 97; else `status=discard`.

### Keep / Discard

- keep:
  - `C > 0` and `estimated_score` improves over best keep row.
- discard:
  - `C = 0`
  - no speed gain
  - unstable behavior (frequent timeout / OOM)

### Crash Handling

- Crash types: launch fail, runtime exception, OOM, benchmark timeout.
- Write row with `status=crash`, metrics set to `0` where unavailable.

## 6) Strategy Library (Expected Impact x Risk)

按优先级从高到低执行：

1. **Phase 1: Server Arg Tuning (高收益, 低风险)**
   - Focus: `SGLANG_SERVER_ARGS`
   - Candidates:
     - `--chunked-prefill-size` (`2048/4096/8192/16384`) — default is 32768, try smaller values
     - `--disable-radix-cache` toggle
     - `--skip-server-warmup` toggle
     - `--cuda-graph-max-bs` and CUDA graph related knobs
   - **KNOWN NON-CANDIDATE**: `--attention-backend minicpm_flashinfer` vs `minicpm_flashattn`
     are **identical** — both map to `MiniCPMSparseBackend(runner)` with no parameter difference
     (see `attention_registry.py:180-197`). Do NOT waste a submission on this switch.

2. **Phase 2: Quantization (高收益, 中风险)**
   - GPTQ W4A16 baseline from `demo-quant/`
   - Explore `group_size`, selective layer precision fallback, dtype compatibility
   - 必须盯住 `C`，防止速度上去但精度掉到 97% 以下

3. **Phase 3: Kernel / Backend Optimization (中高收益, 中高风险)**
   - Targets:
     - `sglang/python/sglang/srt/layers/attention/minicpm_backend.py`
     - `.../minicpm_fuse_kernel.py`
     - `.../minicpm_sparse_kernels.py`
   - Ideas:
     - reduce Python overhead in hot path
     - improve kernel launch config
     - avoid redundant dtype conversion

4. **Phase 4: Speculative Decoding (高潜力, 高风险)**
   - Current blocker: MiniCPM sparse backend lacks speculative support
   - Direction:
     - implement/bridge verify path compatibility for hybrid mixer
     - start from minimal draft length and strict fallback

5. **Combination Stage (只在单项稳定后)**
   - Example: `best args + GPTQ + kernel tweak`
   - 每次只叠加一个新因素，避免不可解释波动

## 7) File-Level Playbook

- `prepare_env.sh`
  - install local sglang
  - set/override runtime args
  - optional environment patching
- `prepare_model.sh` / `preprocess_model.py`
  - quantization, conversion, model artifact rewrite
- `sglang/python/sglang/srt/**`
  - backend logic, kernels, quantization path, speculative path

## 8) Logging and Artifacts

每次实验自动生成：

- `experiments/<timestamp>/server.log`
- `experiments/<timestamp>/bench.log`
- `experiments/<timestamp>/eval.log`
- `outputs/<timestamp>/summary.json` (from eval_model.py — written relative to REPO_DIR, NOT inside experiments/)
- `experiments/<timestamp>/result.json` (from parse_results.py)

长期记录：

- `results.tsv` (single source of truth)

## 9) Night-Run Policy (无人值守)

- Never ask user for continuation once loop starts.
- On failure, auto-record and move on.
- If 3 consecutive crashes in same strategy, switch to lower-risk strategy.
- 每 10 轮回顾一次 `results.tsv`，防止卡局部最优。

## 10) Submission Readiness Checklist

Before platform submission:

1. `prepare_env.sh` runs cleanly by `source`.
2. `prepare_model.sh --input --output` idempotent.
3. model artifact size <= 2GB tar.gz requirement (if applicable).
4. local docker benchmark completed with stable metrics.
5. correctness gate satisfied (`C != 0`).

## 11) Example Commands

Generate synthetic local speed sets:

```bash
python3 gen_test_data.py --out-dir synthetic-data --samples-per-setting 32
```

Run one experiment with official public set:

```bash
bash run_experiment.sh \
  --model-input /data/models/MiniCPM-SALA \
  --description "phase1: flashinfer + chunked_prefill_8192"
```

Run with custom speed sets:

```bash
bash run_experiment.sh \
  --model-input /data/models/MiniCPM-SALA \
  --speed-s1 /workspace/sala-game/synthetic-data/speed_s1_0p1k.jsonl \
  --speed-s8 /workspace/sala-game/synthetic-data/speed_s8_4k.jsonl \
  --speed-smax /workspace/sala-game/synthetic-data/speed_smax_60k_0p5k.jsonl \
  --description "phase2: gptq rtn w4a16"
```

---

核心理念不变：**有效就留，无效就扔；accuracy gate 永远优先于 speed。**
