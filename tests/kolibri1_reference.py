#!/usr/bin/env python3
"""Independent NumPy forward pass for the Kolibri-1 mini fixture.

This is the correctness anchor for the C implementation: it is written from the
released architecture (model card plus aleph-alpha-inference's kolibri1.py) and
not from ds4.c, so a shared misreading is less likely. It reads a GGUF written by
tests/make_kolibri1_mini.py (F32 or F16 weights) and prints next-token logits for
a list of token ids, in the JSON shape `ds4 --dump-logits` emits.

Block semantics under test. Each pre-norm is a fused add-RMSNorm, so the
residual stream advances at those points and a sublayer's normalized output
joins it there:

  residual = residual + hidden; x = rmsnorm(residual) * input_layernorm
  attn     = o_proj(attend(qk_norm(q), k_norm(k)))
  x_attn   = post_attn_norm(attn)
  residual = residual + x_attn; x = rmsnorm(residual) * post_attention_layernorm
  y        = shared_experts(x) + sum_e sigmoid(logit_e) * expert_e(x)
  y        = post_ffn_norm(y)
  residual = residual + y
  logits   = output(norm(residual))

Routing selects the top-k experts on logit + expert_bias and weights them with
the unbiased sigmoid. Sliding layers apply RoPE over their window; full layers
apply no positional encoding at all.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

GGUF_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 10: "Q2_K",
              12: "Q4_K", 16: "IQ2_XXS", 30: "BF16"}


def read_exact(fp, length):
    data = fp.read(length)
    if len(data) != length:
        raise ValueError("short read")
    return data


def read_u32(fp):
    return struct.unpack("<I", read_exact(fp, 4))[0]


def read_u64(fp):
    return struct.unpack("<Q", read_exact(fp, 8))[0]


def read_string(fp):
    return read_exact(fp, read_u64(fp)).decode("utf-8")


def skip_value(fp, kind):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if kind == 8:
        read_string(fp)
        return
    if kind == 9:
        element = read_u32(fp)
        count = read_u64(fp)
        for _ in range(count):
            skip_value(fp, element)
        return
    if kind not in sizes:
        raise ValueError(f"unsupported GGUF value type {kind}")
    fp.seek(sizes[kind], 1)


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def load_gguf(path):
    with open(path, "rb") as fp:
        if read_exact(fp, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        if read_u32(fp) != 3:
            raise ValueError("expected GGUF v3")
        n_tensors, n_meta = read_u64(fp), read_u64(fp)
        meta = {}
        for _ in range(n_meta):
            key = read_string(fp)
            kind = read_u32(fp)
            if kind == 8:
                meta[key] = read_string(fp)
            elif kind == 4:
                meta[key] = read_u32(fp)
            elif kind == 6:
                meta[key] = struct.unpack("<f", read_exact(fp, 4))[0]
            elif kind == 7:
                meta[key] = bool(read_exact(fp, 1)[0])
            elif kind == 9:
                element = read_u32(fp)
                count = read_u64(fp)
                if element == 4:
                    meta[key] = [read_u32(fp) for _ in range(count)]
                elif element == 8:
                    values = [read_string(fp) for _ in range(count)]
                    meta[key] = values if len(values) <= 64 else f"<{count} strings>"
                else:
                    for _ in range(count):
                        skip_value(fp, element)
            else:
                skip_value(fp, kind)
        tensors = {}
        for _ in range(n_tensors):
            name = read_string(fp)
            dims = [read_u64(fp) for _ in range(read_u32(fp))]
            tensors[name] = {"dims": dims, "type": read_u32(fp), "offset": read_u64(fp)}
        data_start = align_up(fp.tell(), meta.get("general.alignment", 32))
        fp.seek(0)
        blob = fp.read()
    return meta, tensors, blob, data_start


def tensor_array(info, blob, data_start):
    """F64 view of a tensor in linear-algebra order.

    GGUF stores dense tensors as [in, out] with dim0 contiguous and routed
    experts as expert-major [in, out, n] planes. NumPy reshape puts the LAST
    axis fastest, so restoring the mathematical view means reshaping to the
    reversed dims: dense matrices come out as [out, in] rows and expert
    stacks as [n, out, in] planes, ready for `w @ x`.
    """
    dtype = {0: "<f4", 1: "<f2"}.get(info["type"])
    if dtype is None:
        raise ValueError(f"unsupported tensor type {GGUF_TYPES.get(info['type'], info['type'])}")
    count = int(np.prod(info["dims"]))
    values = np.frombuffer(blob, dtype=dtype, count=count,
                           offset=data_start + info["offset"]).astype(np.float64)
    if len(info["dims"]) == 1:
        return values
    return values.reshape(tuple(reversed(info["dims"])))


def dense(x, w):
    """y = W x for a [out, in] matrix (GGUF [in, out] read row-major)."""
    return w @ x


def expert(x, w, index):
    """y = W_e x for a [n, out, in] routed-expert stack."""
    return w[index] @ x


def embed_row(w, token, width):
    """Row `token` of a GGUF [width, vocab] embedding table."""
    flat = w.reshape(-1)
    return flat[token * width:(token + 1) * width]


def rms_norm(x, weight, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * weight


def rope(vec, head_dim, pos, base):
    half = head_dim // 2
    for i in range(half):
        angle = pos * (base ** (-2.0 * i / head_dim))
        c, s = np.cos(angle), np.sin(angle)
        x1, x2 = vec[i], vec[i + half]
        vec[i] = x1 * c - x2 * s
        vec[i + half] = x1 * s + x2 * c


def swiglu(x, gate_w, up_w, down_w, index=None):
    """SwiGLU with GGUF [in, out] matrices; index selects a routed expert."""
    project = dense if index is None else (lambda v, w: expert(v, w, index))
    gate = project(x, gate_w)
    act = gate / (1.0 + np.exp(-gate))
    return project(act * project(x, up_w), down_w)


def forward(meta, tensors, blob, data_start, tokens):
    get = lambda name: tensor_array(tensors[name], blob, data_start)

    n_embd = meta["kolibri1.embedding_length"]
    n_head = meta["kolibri1.attention.head_count"]
    n_kv = meta["kolibri1.attention.head_count_kv"]
    head_dim = meta["kolibri1.attention.key_length"]
    window = meta["kolibri1.attention.sliding_window"]
    eps = meta["kolibri1.attention.layer_norm_rms_epsilon"]
    rope_base = meta["kolibri1.rope.freq_base"]
    layer_types = meta["kolibri1.attention.layer_types"]
    top_k = meta["kolibri1.expert_used_count"]
    embed = get("token_embd.weight")
    out_head = get("output.weight")
    out_norm = get("output_norm.weight")

    history = [{"k": [], "v": []} for _ in range(len(layer_types))]

    for pos, token in enumerate(tokens):
        # vLLM runs one forward per step with residual=None, so the stream is
        # rebuilt from this token's embedding; only the KV cache spans steps.
        residual = embed_row(embed, token, n_embd).copy()
        for il, layer_type in enumerate(layer_types):
            prefix = f"blk.{il}"
            q_w, k_w, v_w, o_w = (get(f"{prefix}.attn_{p}.weight")
                                  for p in ("q", "k", "v", "output"))
            qn_w, kn_w = get(f"{prefix}.attn_q_norm.weight"), get(f"{prefix}.attn_k_norm.weight")

            x = rms_norm(residual, get(f"{prefix}.attn_norm.weight"), eps)
            q = dense(x, q_w)
            k = dense(x, k_w)
            v = dense(x, v_w)
            for h in range(n_head):
                lo, hi = h * head_dim, (h + 1) * head_dim
                q[lo:hi] = rms_norm(q[lo:hi], qn_w, eps)
            for h in range(n_kv):
                lo, hi = h * head_dim, (h + 1) * head_dim
                k[lo:hi] = rms_norm(k[lo:hi], kn_w, eps)

            if layer_type != 1:
                for h in range(n_head):
                    rope(q[h * head_dim:(h + 1) * head_dim], head_dim, pos, rope_base)
                for h in range(n_kv):
                    rope(k[h * head_dim:(h + 1) * head_dim], head_dim, pos, rope_base)

            history[il]["k"].append(k.copy())
            history[il]["v"].append(v.copy())
            start = 0 if layer_type == 1 else max(0, pos + 1 - window)
            scale = head_dim ** -0.5
            group = n_head // n_kv
            context = np.zeros(n_head * head_dim)
            keys_all = np.stack(history[il]["k"][start:pos + 1])
            values_all = np.stack(history[il]["v"][start:pos + 1])
            for h in range(n_head):
                lo, hi = h * head_dim, (h + 1) * head_dim
                kv = (h // group) * head_dim
                scores = (keys_all[:, kv:kv + head_dim] @ q[lo:hi]) * scale
                scores -= scores.max()
                weights = np.exp(scores)
                context[lo:hi] = (weights / weights.sum()) @ values_all[:, kv:kv + head_dim]
            attention = dense(context, o_w)
            attention = rms_norm(attention, get(f"{prefix}.post_attn_norm.weight"), eps)
            residual = residual + attention

            x = rms_norm(residual, get(f"{prefix}.ffn_norm.weight"), eps)
            router = dense(x, get(f"{prefix}.ffn_gate_inp.weight"))
            bias = get(f"{prefix}.exp_probs_b.bias")
            selected = np.argsort(-(router + bias))[:top_k]
            routed = np.zeros(n_embd)
            for expert in selected:
                routed += (1.0 / (1.0 + np.exp(-router[expert]))) * swiglu(
                    x,
                    get(f"{prefix}.ffn_gate_exps.weight"),
                    get(f"{prefix}.ffn_up_exps.weight"),
                    get(f"{prefix}.ffn_down_exps.weight"), expert)
            shared = swiglu(x,
                            get(f"{prefix}.ffn_gate_shexp.weight"),
                            get(f"{prefix}.ffn_up_shexp.weight"),
                            get(f"{prefix}.ffn_down_shexp.weight"))
            hidden = rms_norm(shared + routed, get(f"{prefix}.post_ffn_norm.weight"), eps)
            # The MoE output joins the residual at the next pre-norm; nothing
            # reads the stream in between, so adding it here is equivalent.
            residual = residual + hidden

        logits = dense(rms_norm(residual, out_norm, eps), out_head)

    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tokens", required=True,
                        help="comma-separated prompt ids, or @FILE with one sequence per line")
    parser.add_argument("--out", help="write logits JSON here instead of stdout")
    args = parser.parse_args()

    meta, tensors, blob, data_start = load_gguf(args.gguf)
    if args.tokens.startswith("@"):
        with open(args.tokens[1:], "r", encoding="utf-8") as fp:
            sequences = [[int(t) for t in line.split(",") if t.strip()]
                         for line in fp if line.strip()]
    else:
        sequences = [[int(t) for t in args.tokens.split(",") if t.strip()]]

    results = []
    for tokens in sequences:
        if not tokens:
            raise ValueError("empty token sequence")
        logits = forward(meta, tensors, blob, data_start, tokens)
        results.append({
            "source": "numpy-reference",
            "prompt_tokens": tokens,
            "vocab": int(logits.shape[0]),
            "argmax_token": int(np.argmax(logits)),
            "argmax_logit": float(logits.max()),
            "logits": [float(v) for v in logits],
        })
        print(f"prompt_tokens={len(tokens)} argmax={results[-1]['argmax_token']} "
              f"logit={results[-1]['argmax_logit']:.6f}", file=sys.stderr)

    document = json.dumps(results[0] if len(results) == 1 else results, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write(document)
    else:
        sys.stdout.write(document)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"kolibri1-reference: {error}")