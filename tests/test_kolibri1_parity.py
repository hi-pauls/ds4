#!/usr/bin/env python3
"""Parity gate: ds4 CPU reference logits against the NumPy Kolibri-1 anchor.

Runs text prompts through ds4's pre-tokenizer, feeds the same ids to
tests/kolibri1_reference.py, and requires next-token logits to agree to 1e-4
with the same argmax. Prompt lengths cover a single token, a few tokens and a
prompt longer than the sliding window, so the KV ring wrap is exercised.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS4 = ROOT / "ds4"
REFERENCE = ROOT / "tests" / "kolibri1_reference.py"

PROMPTS = (
    "Hello",
    "Hello,",
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog again and again. "
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog again and again.",
)

TOLERANCE = 1.0e-4


def prompt_tokens(gguf, text):
    proc = subprocess.run([DS4, "-m", gguf, "--raw", "-p", text, "--dump-tokens"],
                          check=True, capture_output=True, text=True)
    return [int(line.split()[0]) for line in proc.stdout.splitlines()
            if line.strip() and not line.startswith("[")]


def argmax_id(doc):
    token = doc["argmax_token"]
    return token["id"] if isinstance(token, dict) else token


def check(gguf, text):
    tokens = prompt_tokens(gguf, text)
    c_out = f"/tmp/kolibri1_parity_c.json"
    r_out = f"/tmp/kolibri1_parity_r.json"
    subprocess.run([DS4, "--cpu", "-m", gguf, "--raw", "-p", text,
                    "--dump-logits", c_out], check=True, capture_output=True)
    subprocess.run([sys.executable, REFERENCE, "--gguf", gguf,
                    "--tokens", ",".join(map(str, tokens)), "--out", r_out],
                   check=True, capture_output=True)
    c = json.load(open(c_out))
    r = json.load(open(r_out))
    drift = max(abs(a - b) for a, b in zip(c["logits"], r["logits"]))
    ca, ra = argmax_id(c), argmax_id(r)
    ok = drift < TOLERANCE and ca == ra
    print(f"kolibri1-parity: len {len(tokens):3d} argmax C={ca} R={ra} "
          f"max|d| {drift:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default="gguf/Kolibri-1-mini.gguf")
    args = parser.parse_args()

    failures = sum(0 if check(args.gguf, text) else 1 for text in PROMPTS)
    if failures:
        sys.exit(f"kolibri1-parity: {failures} prompt(s) failed")
    print("kolibri1 parity tests passed")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"kolibri1-parity: {error}")
