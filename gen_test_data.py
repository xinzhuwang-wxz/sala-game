#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def mk_text(target_chars: int, seed: int) -> str:
    rnd = random.Random(seed)
    words = [
        "sparse",
        "attention",
        "latency",
        "throughput",
        "kernel",
        "cuda",
        "prefill",
        "decode",
        "quantization",
        "benchmark",
        "optimization",
        "memory",
        "cache",
        "token",
        "prompt",
        "reasoning",
        "system",
    ]
    out = []
    cur = 0
    while cur < target_chars:
        w = rnd.choice(words)
        out.append(w)
        cur += len(w) + 1
    return " ".join(out)[:target_chars]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_rows(
    num: int, prompt_chars: int, completion_tokens: int, tag: str
) -> list[dict]:
    rows = []
    for i in range(num):
        prompt = (
            "Please answer briefly and end with one line 'ANSWER: X'.\n\n"
            + mk_text(prompt_chars, seed=prompt_chars * 1000 + i)
        )
        rows.append(
            {
                "index": i + 1,
                "question": prompt,
                "prompt_tokens": max(1, prompt_chars // 4),
                "completion_tokens": completion_tokens,
                "task": "mcq",
                "gold": "A",
                "tag": tag,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic JSONL test sets for local SOAR benchmarking"
    )
    p.add_argument("--out-dir", default="synthetic-data")
    p.add_argument("--samples-per-setting", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.samples_per_setting
    # winner tips style: 0.1k-4k, 60k-0.5k etc. Here we emulate by char length proxy.
    settings = [
        ("speed_s1_0p1k_4k", 400, 4000),
        ("speed_s8_1k_2k", 4000, 2000),
        ("speed_smax_60k_0p5k", 240000, 500),
    ]

    for name, pchars, ctoks in settings:
        rows = build_rows(n, prompt_chars=pchars, completion_tokens=ctoks, tag=name)
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        print(f"wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
