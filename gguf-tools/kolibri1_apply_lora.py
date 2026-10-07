#!/usr/bin/env python3
"""Fuse an mlx-lm LoRA adapter into a Kolibri-1 GGUF.

The adapter may cover attention (q/k/v/o) and the shared expert. Every
recipe stores those as Q8_0 (the F32 fixture as F32), so each touched tensor
is decoded, gets scale * (A @ B)^T added and is re-encoded in place, in a
copy of the base. On APFS the copy is a clone: the output costs only the
rewritten blocks and keeps the base layout byte for byte elsewhere. The
routed experts are never touched, so the recipe's expert quantization stays.

Runs with any Python that has numpy and mlx (mlx reads the adapter's
bfloat16 tensors):
  kolibri1_apply_lora.py --gguf Kolibri-1-Q4.gguf --adapter adapters/ --out Kolibri-1-Q4-tuned.gguf
"""
import argparse
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import numpy as np

from glm53_quantize import align, read_exact, read_u32, read_u64, read_gguf_string, skip_gguf_value

QTYPE_F32 = 0
QTYPE_Q8_0 = 8
GGUF_UINT32 = 4

# mlx-lm module path -> GGUF tensor stem, per kolibri1_quantize.py.
MODULES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.shared_experts.gate_proj": "ffn_gate_shexp",
    "mlp.shared_experts.up_proj": "ffn_up_shexp",
    "mlp.shared_experts.down_proj": "ffn_down_shexp",
}
ADAPTER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)\.lora_([ab])$")


def read_tensors(path):
    """Tensor name -> (shape, type, absolute offset)."""
    tensors = {}
    with open(path, "rb") as fp:
        if read_exact(fp, 4, "magic") != b"GGUF" or read_u32(fp, "version") != 3:
            raise ValueError(f"{path}: expected GGUF v3")
        count = read_u64(fp, "tensor count")
        alignment = 32
        for _ in range(read_u64(fp, "metadata count")):
            key = read_gguf_string(fp, "metadata key")
            kind = read_u32(fp, "metadata type")
            if key == "general.alignment" and kind == GGUF_UINT32:
                alignment = read_u32(fp, key)
            else:
                skip_gguf_value(fp, kind)
        entries = []
        for _ in range(count):
            name = read_gguf_string(fp, "tensor name")
            shape = tuple(read_u64(fp, "dimension") for _ in range(read_u32(fp, "tensor rank")))
            entries.append((name, shape, read_u32(fp, "tensor type"), read_u64(fp, "tensor offset")))
        start = align(fp.tell(), alignment)
    for name, shape, kind, offset in entries:
        tensors[name] = (shape, kind, start + offset)
    return tensors


def read_adapter(adapter_dir):
    """GGUF tensor name -> delta [out, in] as float32."""
    import mlx.core as mx
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if config.get("fine_tune_type", "lora") != "lora":
        raise ValueError(f"{adapter_dir}: only plain LoRA adapters can be fused, not "
                         f"{config['fine_tune_type']}")
    scale = float(config["lora_parameters"]["scale"])
    pairs = {}
    for key, value in mx.load(str(adapter_dir / "adapters.safetensors")).items():
        if (match := ADAPTER_KEY.match(key)) is None or match.group(2) not in MODULES:
            raise ValueError(f"{adapter_dir}: adapter tensor {key} is outside attention and the "
                             f"shared expert; only {', '.join(MODULES)} can be fused")
        name = f"blk.{match.group(1)}.{MODULES[match.group(2)]}.weight"
        pairs.setdefault(name, {})[match.group(3)] = np.array(value.astype(mx.float32))
    deltas = {}
    for name, pair in pairs.items():
        if set(pair) != {"a", "b"}:
            raise ValueError(f"{adapter_dir}: {name} lacks lora_{'b' if 'a' in pair else 'a'}")
        deltas[name] = scale * (pair["a"] @ pair["b"]).T
    return deltas


def decode_q8_0(raw, rows, cols):
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    quants = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (quants * scales).reshape(rows, cols)


def encode_q8_0(values):
    """ggml's reference Q8_0: the scale rounds to f16, quants round half away from zero."""
    blocks = values.astype(np.float32).reshape(-1, 32)
    scale = np.abs(blocks).max(axis=1) / np.float32(127.0)
    inverse = np.divide(np.float32(1.0), scale, out=np.zeros_like(scale), where=scale != 0)
    scaled = blocks * inverse[:, None]
    quants = np.clip(np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5)), -127, 127)
    out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    out[:, :2] = scale.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = quants.astype(np.int8).view(np.uint8)
    return out.tobytes()


def clone(src, dst):
    if platform.system() == "Darwin":
        subprocess.run(["cp", "-c", str(src), str(dst)], check=True)
    else:
        shutil.copyfile(src, dst)


def apply(args):
    deltas = read_adapter(args.adapter)
    tensors = read_tensors(args.gguf)
    for name, delta in deltas.items():
        if name not in tensors:
            raise ValueError(f"{args.gguf}: no tensor {name} for the adapter")
        shape, kind, _ = tensors[name]
        if kind not in (QTYPE_F32, QTYPE_Q8_0) or (shape[1], shape[0]) != delta.shape:
            raise ValueError(f"{name}: expected an F32 or Q8_0 {delta.shape[1]}x{delta.shape[0]} "
                             f"tensor, found type {kind} with shape {shape}")
    if args.out.exists():
        raise ValueError(f"{args.out} exists; refusing to overwrite it")
    clone(args.gguf, args.out)
    largest = 0.0
    with open(args.out, "r+b") as fp:
        for index, (name, delta) in enumerate(sorted(deltas.items())):
            (cols, rows), kind, offset = tensors[name]
            nbytes = rows * cols * 4 if kind == QTYPE_F32 else rows * cols // 32 * 34
            fp.seek(offset)
            raw = read_exact(fp, nbytes, name)
            base = (np.frombuffer(raw, dtype=np.float32).reshape(rows, cols) if kind == QTYPE_F32
                    else decode_q8_0(raw, rows, cols))
            fused = base + delta
            encoded = fused.astype(np.float32).tobytes() if kind == QTYPE_F32 else encode_q8_0(fused)
            fp.seek(offset)
            fp.write(encoded)
            largest = max(largest, float(np.abs(delta).max()))
            if (index + 1) % 50 == 0 or index + 1 == len(deltas):
                print(f"fused {index + 1}/{len(deltas)} tensors", flush=True)
    print(f"{args.out}: {len(deltas)} tensors fused, largest |delta| {largest:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", type=Path, required=True, help="base Kolibri-1 GGUF")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="mlx-lm adapter directory (adapters.safetensors, adapter_config.json)")
    parser.add_argument("--out", type=Path, required=True)
    apply(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"kolibri1-apply-lora: {error}")
