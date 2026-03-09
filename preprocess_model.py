"""
Model preprocessing for MiniCPM-SALA.

Behavior is controlled by the ENABLE_GPTQ environment variable:

  ENABLE_GPTQ=0 (default)
    Plain file copy — no quantization. Use for baseline / Phase 1 experiments.

  ENABLE_GPTQ=1
    RTN W4A16 GPTQ quantization (no calibration data required).
    Controlled by:
      GPTQ_GROUP_SIZE  (default: 128)
      GPTQ_BITS        (default: 4)

Usage:
    python preprocess_model.py --input /path/to/model --output /path/to/output
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def copy_model(src: Path, dst: Path) -> None:
    """Plain copy — skip hidden files, skip already-existing targets."""
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src.iterdir()):
        if f.name.startswith("."):
            continue
        target = dst / f.name
        if not target.exists():
            if f.is_dir():
                shutil.copytree(f, target)
            else:
                shutil.copy2(f, target)
            count += 1
    print(f"[preprocess] copied {count} new files from {src} to {dst}")


# ---------------------------------------------------------------------------
# RTN W4A16 GPTQ — inlined from demo-quant/quantize_gptq.py
# ---------------------------------------------------------------------------


def quantize_weight_rtn(weight, bits: int = 4, group_size: int = 128):
    """
    Round-To-Nearest W4A16 quantize a single linear layer weight.

    weight shape : [out_features, in_features]
    Returns      : qweight, scales, qzeros, g_idx  (all torch.Tensor)
    """
    import torch

    out_features, in_features = weight.shape
    pack_factor = 32 // bits
    maxq = 2**bits - 1

    if group_size <= 0:
        group_size = in_features
    num_groups = (in_features + group_size - 1) // group_size

    if in_features % group_size != 0:
        pad = group_size * num_groups - in_features
        weight = torch.nn.functional.pad(weight, (0, pad))
        in_features = weight.shape[1]

    w = weight.float()
    w_grouped = w.reshape(out_features, num_groups, group_size)
    w_min = w_grouped.min(dim=2, keepdim=True).values
    w_max = w_grouped.max(dim=2, keepdim=True).values

    scales = (w_max - w_min).clamp(min=1e-10) / maxq
    zeros = -w_min / scales

    qw = torch.clamp(torch.round(w_grouped / scales + zeros), 0, maxq).to(torch.int32)
    qw = qw.reshape(out_features, in_features)

    # qweight: [in_features // pack_factor, out_features]
    qw_t = qw.t().contiguous()
    qweight = torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32)
    for j in range(pack_factor):
        qweight |= qw_t[j::pack_factor, :] << (bits * j)

    # scales: [num_groups, out_features]
    scales_out = scales.squeeze(2).t().contiguous().half()

    # qzeros: [num_groups, out_features // pack_factor]
    zeros_int = torch.clamp(torch.round(zeros.squeeze(2)), 0, maxq).to(torch.int32)
    zeros_t = zeros_int.t().contiguous()
    qzeros = torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32)
    for j in range(pack_factor):
        qzeros |= zeros_t[:, j::pack_factor] << (bits * j)

    g_idx = torch.tensor(
        [i // group_size for i in range(in_features)], dtype=torch.int32
    )

    return qweight, scales_out, qzeros, g_idx


LINEAR_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".o_gate.weight",
    ".z_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


def quantize_gptq(src: Path, dst: Path, bits: int = 4, group_size: int = 128) -> None:
    """RTN W4A16 GPTQ quantization — no calibration data required."""
    import torch
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files
    for f in src.iterdir():
        if f.suffix in (".json", ".py", ".model", ".txt") or f.name in (
            "tokenizer.json",
        ):
            shutil.copy2(f, dst / f.name)

    # Patch config.json: add quantization_config + switch dtype to float16
    with open(src / "config.json") as fh:
        config = json.load(fh)
    config["quantization_config"] = {
        "bits": bits,
        "group_size": group_size,
        "quant_method": "gptq",
        "desc_act": False,
        "sym": False,
    }
    config["torch_dtype"] = "float16"
    with open(dst / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    # Write quantize_config.json (SGLang reads this separately)
    with open(dst / "quantize_config.json", "w") as fh:
        json.dump(
            {
                "bits": bits,
                "group_size": group_size,
                "desc_act": False,
                "sym": False,
                "lm_head": False,
                "dynamic": {},
            },
            fh,
            indent=2,
        )

    # Quantize weight shards
    with open(src / "model.safetensors.index.json") as fh:
        index = json.load(fh)

    shard_files = sorted(set(index["weight_map"].values()))
    new_weight_map: dict = {}
    total_quantized = 0

    for shard_name in shard_files:
        print(f"[preprocess] quantizing {shard_name} ...")
        shard = load_file(str(src / shard_name))
        new_tensors: dict = {}

        for name, tensor in shard.items():
            if name.endswith(LINEAR_SUFFIXES):
                w = tensor.to(torch.float16)
                qweight, scales, qzeros, g_idx = quantize_weight_rtn(
                    w, bits, group_size
                )

                base = name.removesuffix(".weight")
                for suffix, t in [
                    (".qweight", qweight),
                    (".scales", scales),
                    (".qzeros", qzeros),
                    (".g_idx", g_idx),
                ]:
                    new_tensors[base + suffix] = t
                    new_weight_map[base + suffix] = shard_name
                total_quantized += 1
                print(
                    f"  {name}: {list(w.shape)} -> W{bits}A16 group_size={group_size}"
                )
            else:
                new_tensors[name] = tensor.to(torch.float16)
                new_weight_map[name] = shard_name

        save_file(new_tensors, str(dst / shard_name))

    with open(dst / "model.safetensors.index.json", "w") as fh:
        json.dump(
            {"metadata": {"total_size": 0}, "weight_map": new_weight_map}, fh, indent=2
        )

    print(
        f"[preprocess] GPTQ done — {total_quantized} layers quantized to W{bits}A16 (group_size={group_size})"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original model directory")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument(
        "--bits", type=int, default=int(os.environ.get("GPTQ_BITS", "4"))
    )
    parser.add_argument(
        "--group-size", type=int, default=int(os.environ.get("GPTQ_GROUP_SIZE", "128"))
    )
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Input model dir not found: {src}")

    enable_gptq = os.environ.get("ENABLE_GPTQ", "0").strip() == "1"

    if enable_gptq:
        print(f"[preprocess] mode=GPTQ W{args.bits}A16 group_size={args.group_size}")
        quantize_gptq(src, dst, bits=args.bits, group_size=args.group_size)
    else:
        print("[preprocess] mode=copy (set ENABLE_GPTQ=1 to quantize)")
        copy_model(src, dst)


if __name__ == "__main__":
    main()
