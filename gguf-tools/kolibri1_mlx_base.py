#!/usr/bin/env python3
"""Give an MLX Kolibri-1 the GGUF's Q8_0 attention and shared expert, for LoRA training.

Every ds4 recipe (Q8, Q4, Q2) stores attention and the shared expert as the
same Q8_0 tensors, and kolibri1_apply_lora.py fuses adapters into exactly
those. A 4-bit MLX conversion would train the adapter against coarser
weights than it is applied to, so this replaces them, shard by shard and in
place, with the decoded Q8_0 values: dense bf16 by default, closest to the
GGUF, or MLX 8-bit (group 32) where the 1.9 GB more of bf16 does not fit.
Exact affine scales are out of reach either way: bf16 scales and biases
cancel badly, and float32 ones turn every activation float32. The routed
experts keep the MLX recipe; they are frozen in training.

  kolibri1_mlx_base.py --gguf Kolibri-1-Q4.gguf --mlx Kolibri-1-MLX-4bit [--format bf16|q8]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from glm53_quantize import read_exact
from kolibri1_apply_lora import MODULES, QTYPE_Q8_0, decode_q8_0, read_tensors

GROUP_SIZE = 32
BITS = 8


def replace_shard(path, layers, gguf, tensors, mx, dense):
    weights = mx.load(str(path))
    changed = 0
    with open(gguf, "rb") as fp:
        for layer in layers:
            for module, stem in MODULES.items():
                prefix = f"model.layers.{layer}.{module}"
                if f"{prefix}.weight" not in weights:
                    continue
                (cols, rows), kind, offset = tensors[f"blk.{layer}.{stem}.weight"]
                if kind != QTYPE_Q8_0:
                    raise ValueError(f"{gguf}: blk.{layer}.{stem} is type {kind}, expected Q8_0")
                fp.seek(offset)
                values = mx.array(decode_q8_0(read_exact(fp, rows * cols // 32 * 34, prefix), rows, cols))
                if dense:
                    weights[f"{prefix}.weight"] = values.astype(mx.bfloat16)
                    weights.pop(f"{prefix}.scales", None)
                    weights.pop(f"{prefix}.biases", None)
                else:
                    weight, scales, biases = mx.quantize(values.astype(mx.bfloat16), group_size=GROUP_SIZE, bits=BITS)
                    weights[f"{prefix}.weight"] = weight
                    weights[f"{prefix}.scales"] = scales
                    weights[f"{prefix}.biases"] = biases
                changed += 1
    if changed:
        temporary = path.with_suffix(".tmp.safetensors")
        mx.save_safetensors(str(temporary), weights, metadata={"format": "mlx"})
        temporary.replace(path)
    return changed


def precision_report(gguf, tensors, mx):
    """How far the training weights sit from the Q8_0 values, against the 4-bit alternative."""
    (cols, rows), _, offset = tensors["blk.0.attn_q.weight"]
    with open(gguf, "rb") as fp:
        fp.seek(offset)
        values = decode_q8_0(read_exact(fp, rows * cols // 32 * 34, "probe"), rows, cols)
    scale = float(np.abs(values).mean())
    dense = np.array(mx.array(values).astype(mx.bfloat16).astype(mx.float32))
    print(f"blk.0.attn_q bf16 dense: mean |error| / mean |weight| = {float(np.abs(dense - values).mean()) / scale:.4f}")
    for label, group, bits in (("mlx 8-bit/32", GROUP_SIZE, BITS), ("mlx 4-bit/64", 64, 4)):
        packed = mx.quantize(mx.array(values).astype(mx.bfloat16), group_size=group, bits=bits)
        back = np.array(mx.dequantize(*packed, group_size=group, bits=bits).astype(mx.float32))
        print(f"blk.0.attn_q {label}: mean |error| / mean |weight| = "
              f"{float(np.abs(back - values).mean()) / scale:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", type=Path, required=True, help="any ds4 Kolibri-1 GGUF (all share the Q8_0 tensors)")
    parser.add_argument("--mlx", type=Path, required=True, help="MLX Kolibri-1 directory, rewritten in place")
    parser.add_argument("--format", choices=("bf16", "q8"), default="bf16")
    args = parser.parse_args()
    dense = args.format == "bf16"
    import mlx.core as mx
    tensors = read_tensors(args.gguf)
    config_path = args.mlx / "config.json"
    config = json.loads(config_path.read_text())
    layers = range(config["num_hidden_layers"])
    precision_report(args.gguf, tensors, mx)
    index = json.loads((args.mlx / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({index[k] for k in index if any(f".{m}." in k for m in MODULES)})
    total = 0
    for shard in shards:
        total += replace_shard(args.mlx / shard, layers, args.gguf, tensors, mx, dense)
        print(f"{shard}: {total} tensors so far", flush=True)
    if total != len(layers) * len(MODULES):
        raise ValueError(f"replaced {total} tensors, expected {len(layers) * len(MODULES)}")
    # mlx-lm takes a per-module entry over the global recipe; False leaves the layer unquantized.
    entry = False if dense else {"group_size": GROUP_SIZE, "bits": BITS}
    for key in ("quantization", "quantization_config"):
        for layer in layers:
            for module in MODULES:
                if key in config:
                    config[key][f"model.layers.{layer}.{module}"] = entry
    config_path.write_text(json.dumps(config, indent=4) + "\n")
    print(f"{args.mlx}: {total} attention/shared tensors now carry the GGUF's Q8_0 values as {args.format}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"kolibri1-mlx-base: {error}")
