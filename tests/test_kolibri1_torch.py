#!/usr/bin/env python3
"""Parity gate for the torch Kolibri-1 reference.

Extends tests/test_kolibri1_parity.py with a third leg: the torch port
(tests/kolibri1_torch_reference.py, tensor-bearing port of Aleph Alpha's
kolibri1.py) must agree with the NumPy anchor and with ds4 on the mini
fixture.

- anchor (float64)  vs torch (float64): semantic equality of the two
  independent implementations, gate 1e-8 — rounding-order noise is ~1e-10,
  so any real misreading (wrong norm placement, wrong routing, wrong rope)
  shows up orders of magnitude above the gate.
- ds4 (float32 C)   vs torch (float32): the numerics gap that matters for the
  real-weight golden vectors, gate 1e-4 with identical argmax, same tolerance
  as the NumPy parity tool.

Also verifies the Q8_0 dequant against hand-computed bytes, because the mini
fixture is all-F32 and the real artifact's Q8 path would otherwise first run
on the CUDA box with no local check.
"""
import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS4 = ROOT / "ds4"
sys.path.insert(0, str(ROOT / "tests"))

import kolibri1_torch_reference as torchref  # noqa: E402

SEMANTIC_TOLERANCE = 1.0e-8
NUMERIC_TOLERANCE = 1.0e-4

PROMPTS = (
    "Hello",
    "Hello,",
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog again and again. "
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog again and again.",
)


def prompt_tokens(gguf, text):
    proc = subprocess.run([DS4, "-m", gguf, "--raw", "-p", text, "--dump-tokens"],
                          check=True, capture_output=True, text=True)
    return [int(line.split()[0]) for line in proc.stdout.splitlines()
            if line.strip() and not line.startswith("[")]


def argmax_id(doc):
    token = doc["argmax_token"]
    return token["id"] if isinstance(token, dict) else token


def reference_json(gguf, tokens, out, dtype):
    script = ROOT / "tests" / "kolibri1_reference.py"
    subprocess.run([sys.executable, script, "--gguf", gguf,
                    "--tokens", ",".join(map(str, tokens)), "--out", out],
                   check=True, capture_output=True)
    return json.load(open(out))


def torch_json(gguf, tokens, out, dtype):
    script = ROOT / "tests" / "kolibri1_torch_reference.py"
    subprocess.run([sys.executable, script, "--gguf", gguf,
                    "--tokens", ",".join(map(str, tokens)), "--out", out,
                    "--dtype", dtype], check=True, capture_output=True)
    return json.load(open(out))


def ds4_json(gguf, text, out):
    subprocess.run([DS4, "--cpu", "-m", gguf, "--raw", "-p", text,
                    "--dump-logits", out], check=True, capture_output=True)
    return json.load(open(out))


def drift(x, y):
    return max(abs(a - b) for a, b in zip(x, y))


def check_q8_block():
    """One block: fp16 scale, 32 int8 quants; value = scale * quant."""
    scale = np.float32(1.375)          # exactly representable in fp16
    quants = np.array(range(-16, 16), dtype=np.int8)
    expected = np.float32(scale) * quants.astype(np.int32)  # exact int math
    block = struct.pack("<e", np.float16(scale)) + quants.tobytes()
    got = torchref.dequant_q8_0(block, np.float32)
    if not np.array_equal(got.astype(np.float64), expected.astype(np.float64)):
        return False
    # sign + subnormal scales round-trip: tiny scale must not flush to zero
    tiny = struct.pack("<e", np.float16(2.0 ** -15)) + b"\x01" + bytes(31)
    values = torchref.dequant_q8_0(tiny, np.float32)
    return bool(values[0] == np.float32(np.float16(2.0 ** -15)) and
                float(values[0]) > 0.0)


def check(gguf, text):
    tokens = prompt_tokens(gguf, text)
    anchor = reference_json(gguf, tokens, "/tmp/kolibri1_torch_anchor.json", "f64")
    torch64 = torch_json(gguf, tokens, "/tmp/kolibri1_torch_t64.json", "float64")
    torch32 = torch_json(gguf, tokens, "/tmp/kolibri1_torch_t32.json", "float32")
    ds4 = ds4_json(gguf, text, "/tmp/kolibri1_torch_c.json")
    ds4_logits = ds4["logits"] if isinstance(ds4["logits"][0], float) else ds4["logits"]

    semantics = drift(anchor["logits"], torch64["logits"])
    semantics_ok = semantics < SEMANTIC_TOLERANCE
    argument = ds4_logits
    numerics = drift(argument, torch32["logits"])
    numerics_ok = numerics < NUMERIC_TOLERANCE and \
        argmax_id(ds4) == torch32["argmax_token"]
    ok = semantics_ok and numerics_ok
    print(f"kolibri1-torch: len {len(tokens):3d} "
          f"anchor|torch64 {semantics:.2e} ds4|torch32 {numerics:.2e} "
          f"argmax A={anchor['argmax_token']} C={argmax_id(ds4)} "
          f"T={torch32['argmax_token']} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default="gguf/Kolibri-1-mini.gguf")
    args = parser.parse_args()

    if not check_q8_block():
        sys.exit("kolibri1-torch: Q8_0 dequant self-check failed")

    failures = sum(0 if check(args.gguf, text) else 1 for text in PROMPTS)
    if failures:
        sys.exit(f"kolibri1-torch: {failures} prompt(s) failed")
    print("kolibri1 torch parity tests passed")


if __name__ == "__main__":
    main()
