#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FIELDS = [
    "commit",
    "S1_time",
    "S8_time",
    "Smax_time",
    "accuracy",
    "C_value",
    "estimated_score",
    "status",
    "description",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse SOAR benchmark/eval outputs into results.tsv"
    )
    p.add_argument("--results-tsv", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--bench-log", default=None)
    p.add_argument("--summary-json", default=None)
    p.add_argument("--s1", type=float, default=None)
    p.add_argument("--s8", type=float, default=None)
    p.add_argument("--smax", type=float, default=None)
    p.add_argument(
        "--accuracy", type=float, default=None, help="Normalized accuracy in [0,100]"
    )
    p.add_argument("--force-status", choices=["keep", "discard", "crash"], default=None)
    p.add_argument("--result-json", default=None)
    return p.parse_args()


def parse_bench_log(path: Path) -> tuple[float, float, float]:
    if not path.exists():
        return 0.0, 0.0, 0.0
    text = path.read_text(encoding="utf-8", errors="ignore")

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if {"S1", "S8", "Smax"}.issubset(obj.keys()):
            return (
                float(obj.get("S1", 0.0)),
                float(obj.get("S8", 0.0)),
                float(obj.get("Smax", 0.0)),
            )

    s1 = regex_float(text, r"\bS1\b[^\n]*?([0-9]+(?:\.[0-9]+)?)")
    s8 = regex_float(text, r"\bS8\b[^\n]*?([0-9]+(?:\.[0-9]+)?)")
    smax = regex_float(text, r"\bSmax\b[^\n]*?([0-9]+(?:\.[0-9]+)?)")
    return s1, s8, smax


def parse_accuracy(summary_json: Path) -> float:
    if not summary_json.exists():
        return 0.0
    try:
        obj = json.loads(summary_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0
    if "overall_accuracy" in obj:
        return float(obj["overall_accuracy"])
    if "acc" in obj:
        return float(obj["acc"])
    return 0.0


def regex_float(text: str, pattern: str) -> float:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return 0.0
    return float(m.group(1))


def c_value(acc: float) -> float:
    if acc < 97.0:
        return 0.0
    if acc < 99.0:
        return 0.92
    return 1.0


def estimate_score(s1: float, s8: float, smax: float, c: float) -> float:
    if s1 <= 0 or s8 <= 0 or smax <= 0 or c <= 0:
        return 0.0
    proxy = (0.4 / s1) + (0.3 / s8) + (0.3 / smax)
    return proxy * c * 10000.0


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def best_keep_score(rows: list[dict[str, str]]) -> float:
    best = 0.0
    for r in rows:
        if r.get("status") != "keep":
            continue
        try:
            best = max(best, float(r.get("estimated_score", "0") or 0))
        except ValueError:
            continue
    return best


def append_row(path: Path, row: dict[str, str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    results_tsv = Path(args.results_tsv)

    s1 = args.s1
    s8 = args.s8
    smax = args.smax
    if s1 is None or s8 is None or smax is None:
        if args.bench_log:
            ps1, ps8, psmax = parse_bench_log(Path(args.bench_log))
            if s1 is None:
                s1 = ps1
            if s8 is None:
                s8 = ps8
            if smax is None:
                smax = psmax
        else:
            s1 = s1 or 0.0
            s8 = s8 or 0.0
            smax = smax or 0.0

    acc = args.accuracy
    if acc is None:
        if args.summary_json:
            acc = parse_accuracy(Path(args.summary_json))
        else:
            acc = 0.0

    c = c_value(acc)
    est = estimate_score(float(s1), float(s8), float(smax), c)

    rows = read_existing_rows(results_tsv)
    if args.force_status:
        status = args.force_status
    else:
        if c == 0.0:
            status = "discard"
        else:
            status = "keep" if est > (best_keep_score(rows) + 1e-9) else "discard"

    row = {
        "commit": args.commit,
        "S1_time": f"{float(s1):.6f}",
        "S8_time": f"{float(s8):.6f}",
        "Smax_time": f"{float(smax):.6f}",
        "accuracy": f"{float(acc):.4f}",
        "C_value": f"{c:.2f}",
        "estimated_score": f"{est:.6f}",
        "status": status,
        "description": args.description,
    }
    append_row(results_tsv, row)

    if args.result_json:
        out = {
            "row": row,
            "best_keep_before": best_keep_score(rows),
        }
        Path(args.result_json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
