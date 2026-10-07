#!/usr/bin/env python3
"""Checks gguf-tools/kolibri1_apply_lora.py on the F32 mini fixture: a random
adapter over attention and the shared expert must land as exactly base +
scale * (A @ B)^T, every other byte must stay, and Q8_0 decode/encode must
round-trip the base bytes of a real Q8_0 GGUF when one is given.

Usage: test_kolibri1_apply_lora.py --gguf gguf/Kolibri-1-mini.gguf [--q8-gguf Kolibri-1-Q4.gguf]
Needs numpy and mlx."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from kolibri1_apply_lora import QTYPE_Q8_0, decode_q8_0, encode_q8_0, read_tensors

MODULES = {"self_attn.q_proj": "attn_q", "self_attn.o_proj": "attn_output",
           "mlp.shared_experts.down_proj": "ffn_down_shexp"}
SCALE = 2.0
RANK = 4


def write_adapter(directory, tensors):
    rng = np.random.default_rng(20261008)
    arrays, expected = {}, {}
    for layer in (0, 4):
        for module, stem in MODULES.items():
            name = f"blk.{layer}.{stem}.weight"
            cols, rows = tensors[name][0]
            a = rng.normal(0, 0.1, (cols, RANK)).astype(np.float32)
            b = rng.normal(0, 0.1, (RANK, rows)).astype(np.float32)
            arrays[f"model.layers.{layer}.{module}.lora_a"] = mx.array(a)
            arrays[f"model.layers.{layer}.{module}.lora_b"] = mx.array(b)
            expected[name] = SCALE * (a @ b).T
    mx.save_safetensors(str(directory / "adapters.safetensors"), arrays)
    (directory / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "lora_parameters": {"rank": RANK, "scale": SCALE, "dropout": 0.0}}))
    return expected


def check_fixture(gguf):
    tensors = read_tensors(gguf)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        expected = write_adapter(tmp, tensors)
        out = tmp / "tuned.gguf"
        subprocess.run([sys.executable, str(ROOT / "gguf-tools/kolibri1_apply_lora.py"),
                        "--gguf", str(gguf), "--adapter", str(tmp), "--out", str(out)], check=True)
        base, tuned = gguf.read_bytes(), out.read_bytes()
        changed = np.zeros(len(base), dtype=bool)
        worst = 0.0
        for name, delta in expected.items():
            (cols, rows), _, offset = tensors[name]
            span = slice(offset, offset + rows * cols * 4)
            before = np.frombuffer(base[span], dtype=np.float32).reshape(rows, cols)
            after = np.frombuffer(tuned[span], dtype=np.float32).reshape(rows, cols)
            worst = max(worst, float(np.abs(after - (before + delta)).max()))
            changed[span] = True
        differs = np.frombuffer(base, dtype=np.uint8) != np.frombuffer(tuned, dtype=np.uint8)
        outside = int(np.count_nonzero(differs & ~changed))
        ok = worst < 1e-6 and len(base) == len(tuned) and outside == 0
        print(f"apply-lora fixture: {len(expected)} tensors, max|fused - (base + delta)| {worst:.2e}, "
              f"{outside} bytes changed elsewhere {'OK' if ok else 'FAIL'}")
        return ok


def check_q8_round_trip(gguf):
    tensors = read_tensors(gguf)
    names = [n for n, (_, kind, _) in tensors.items() if kind == QTYPE_Q8_0 and ".attn_" in n][:8]
    with open(gguf, "rb") as fp:
        for name in names:
            (cols, rows), _, offset = tensors[name]
            fp.seek(offset)
            raw = fp.read(rows * cols // 32 * 34)
            if encode_q8_0(decode_q8_0(raw, rows, cols)) != raw:
                print(f"apply-lora q8_0 round trip: {name} FAIL")
                return False
    print(f"apply-lora q8_0 round trip: {len(names)} tensors byte-identical OK")
    return bool(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--q8-gguf", type=Path)
    args = parser.parse_args()
    ok = check_fixture(args.gguf)
    if args.q8_gguf:
        ok = check_q8_round_trip(args.q8_gguf) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
