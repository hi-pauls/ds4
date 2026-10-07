#!/usr/bin/env python3
"""Build a tiny Kolibri-1 GGUF for loader, tokenizer and kernel tests.

The dimensions match DS4_SHAPE_KOLIBRI1_MINI in ds4.c: five layers of
4:1 sliding-window/full attention, 64-wide hidden state, 4/2 GQA heads of 16,
eight routed experts with two selected, a 33-token window and 32-wide experts.
The released tokenizer is embedded, so pre-tokenization runs against the real
128k vocabulary and merge table. Weights are seeded and small, and norms are
one, which keeps the fixture logits well behaved.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from glm53_quantize import (
    GGUF_ALIGNMENT, QTYPE_F32, align, kv_bool, kv_f32, kv_string, kv_u32,
    kv_u32_array, qtype_nbytes,
)
from kolibri1_quantize import tokenize_records

LAYERS = 5
N_EMBD = 64
N_VOCAB = 128000
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 16
N_EXPERT = 8
N_EXPERT_USED = 2
N_FF = 32
SLIDING_WINDOW = 33
CONTEXT = 4096
Q_SIZE = N_HEAD * HEAD_DIM
KV_SIZE = N_HEAD_KV * HEAD_DIM


def tensor_header_item(name, dims, offset):
    return (struct.pack("<Q", len(name)) + name.encode() +
            struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) +
            struct.pack("<IQ", QTYPE_F32, offset))


def config():
    return {
        "model_type": "kolibri1",
        "architectures": ["Kolibri1ForCausalLM"],
        "hidden_size": N_EMBD,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": N_HEAD,
        "num_key_value_heads": N_HEAD_KV,
        "head_dim": HEAD_DIM,
        "vocab_size": N_VOCAB,
        "num_experts": N_EXPERT,
        "num_experts_per_tok": N_EXPERT_USED,
        "moe_intermediate_size": N_FF,
        "shared_expert_intermediate_size": N_FF,
        "norm_topk_prob": False,
        "sliding_window": SLIDING_WINDOW,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": True,
        "max_position_embeddings": CONTEXT,
        "layer_types": ["full_attention" if (il + 1) % 5 == 0 else "sliding_attention"
                        for il in range(LAYERS)],
    }


def metadata(revision, cfg):
    layer_types = [1 if (il + 1) % 5 == 0 else 0 for il in range(LAYERS)]
    return [
        kv_string("general.architecture", "kolibri1"),
        kv_string("general.name", "Kolibri-1 mini"),
        kv_string("general.source.revision", revision),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("kolibri1.fixture", "tests/make_kolibri1_mini.py"),
        kv_u32("kolibri1.block_count", LAYERS),
        kv_u32("kolibri1.context_length", CONTEXT),
        kv_u32("kolibri1.embedding_length", N_EMBD),
        kv_u32("kolibri1.feed_forward_length", N_FF),
        kv_u32("kolibri1.expert_count", N_EXPERT),
        kv_u32("kolibri1.expert_used_count", N_EXPERT_USED),
        kv_u32("kolibri1.expert_shared_count", 1),
        kv_bool("kolibri1.expert_weights_norm", False),
        kv_f32("kolibri1.expert_weights_scale", 1.0),
        kv_bool("kolibri1.router.expert_bias", True),
        kv_string("kolibri1.router.selection", "logit_add_bias_sigmoid"),
        kv_u32("kolibri1.attention.head_count", N_HEAD),
        kv_u32("kolibri1.attention.head_count_kv", N_HEAD_KV),
        kv_u32("kolibri1.attention.key_length", HEAD_DIM),
        kv_u32("kolibri1.attention.value_length", HEAD_DIM),
        kv_f32("kolibri1.attention.layer_norm_rms_epsilon", 1.0e-6),
        kv_u32("kolibri1.attention.sliding_window", SLIDING_WINDOW),
        kv_u32_array("kolibri1.attention.layer_types", layer_types),
        kv_bool("kolibri1.attention.full_attention_rope", False),
        kv_f32("kolibri1.rope.freq_base", 10000.0),
        kv_u32("kolibri1.rope.dimension_count", HEAD_DIM),
        kv_string("kolibri1.quantization", "F32 fixture"),
        kv_string("kolibri1.calibration", "fixture"),
        kv_string("kolibri1.config",
                  json.dumps(cfg, sort_keys=True, separators=(",", ":"))),
    ]


def plan():
    """Fixture tensors in file order: (name, GGUF dims, kind)."""
    items = [
        ("token_embd.weight", (N_EMBD, N_VOCAB), "matrix"),
        ("output_norm.weight", (N_EMBD,), "ones"),
        ("output.weight", (N_EMBD, N_VOCAB), "matrix"),
    ]
    for il in range(LAYERS):
        items += [
            (f"blk.{il}.attn_norm.weight", (N_EMBD,), "ones"),
            (f"blk.{il}.post_attn_norm.weight", (N_EMBD,), "ones"),
            (f"blk.{il}.ffn_norm.weight", (N_EMBD,), "ones"),
            (f"blk.{il}.post_ffn_norm.weight", (N_EMBD,), "ones"),
            (f"blk.{il}.attn_q.weight", (N_EMBD, Q_SIZE), "matrix"),
            (f"blk.{il}.attn_k.weight", (N_EMBD, KV_SIZE), "matrix"),
            (f"blk.{il}.attn_v.weight", (N_EMBD, KV_SIZE), "matrix"),
            (f"blk.{il}.attn_output.weight", (Q_SIZE, N_EMBD), "matrix"),
            (f"blk.{il}.attn_q_norm.weight", (HEAD_DIM,), "ones"),
            (f"blk.{il}.attn_k_norm.weight", (HEAD_DIM,), "ones"),
            (f"blk.{il}.ffn_gate_inp.weight", (N_EMBD, N_EXPERT), "matrix"),
            (f"blk.{il}.exp_probs_b.bias", (N_EXPERT,), "bias"),
            (f"blk.{il}.ffn_gate_shexp.weight", (N_EMBD, N_FF), "matrix"),
            (f"blk.{il}.ffn_up_shexp.weight", (N_EMBD, N_FF), "matrix"),
            (f"blk.{il}.ffn_down_shexp.weight", (N_FF, N_EMBD), "matrix"),
            (f"blk.{il}.ffn_gate_exps.weight", (N_EMBD, N_FF, N_EXPERT), "matrix"),
            (f"blk.{il}.ffn_up_exps.weight", (N_EMBD, N_FF, N_EXPERT), "matrix"),
            (f"blk.{il}.ffn_down_exps.weight", (N_FF, N_EMBD, N_EXPERT), "matrix"),
        ]
    return items


def weights(items, seed):
    rng = np.random.default_rng(seed)
    for name, dims, kind in items:
        count = int(np.prod(dims))
        if qtype_nbytes(QTYPE_F32, dims) != count * 4:
            raise ValueError(f"{name}: unexpected fixture size")
        if kind == "ones":
            values = np.ones(count, dtype=np.float32)
        elif kind == "bias":
            values = rng.normal(0.0, 0.5, count).astype(np.float32)
        else:
            # Small weights keep the fixture logits in a sane range and make
            # routing differences visible.
            values = (rng.normal(0.0, 0.05, count) / np.sqrt(dims[0])).astype(np.float32)
        yield values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True,
                        help="snapshot directory with tokenizer.json and tokenizer_config.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--source-revision", default="0" * 40)
    args = parser.parse_args()

    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not os.path.isfile(os.path.join(args.tokenizer, name)):
            raise ValueError(f"{args.tokenizer}: missing {name}")

    cfg = config()
    records = metadata(args.source_revision, cfg) + tokenize_records(args.tokenizer)

    items = plan()
    offset, entries = 0, []
    for name, dims, _ in items:
        nbytes = qtype_nbytes(QTYPE_F32, dims)
        entries.append((name, dims, nbytes, offset))
        offset += align(nbytes, GGUF_ALIGNMENT)
    header_len = 4 + 4 + 8 + 8 + sum(map(len, records))
    header_len += sum(len(tensor_header_item(name, dims, off))
                      for name, dims, _, off in entries)
    data_start = align(header_len, GGUF_ALIGNMENT)

    with open(args.out, "wb") as fp:
        fp.write(b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(records)))
        for record in records:
            fp.write(record)
        for name, dims, _, off in entries:
            fp.write(tensor_header_item(name, dims, off))
        fp.write(bytes(data_start - fp.tell()))
        fp.seek(data_start)
        for (name, _, nbytes, off), values in zip(entries, weights(items, args.seed)):
            if fp.tell() != data_start + off:
                raise ValueError(f"{name}: offset mismatch")
            fp.write(values.tobytes())
            fp.write(bytes(align(nbytes, GGUF_ALIGNMENT) - nbytes))
    print(f"{args.out}: {data_start + offset} bytes, {len(entries)} tensors, vocab {N_VOCAB}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"make-kolibri1-mini: {error}")