#!/usr/bin/env python3
"""Convert Aleph Alpha Kolibri-1 to DwarfStar's GGUF.

Accepts the FP8 release (F8_E4M3 128x128 block weights with F32 scales) or the
BF16 release; both share tensor names. The routed experts carry 96.7% of the
parameters, so --quant selects only their format while attention, shared
experts and the output head stay Q8_0 and norms/router stay F32.

Uses the project's C quantizers through libds4quants. The full tensor layout is
validated against the source before a single byte is written, and only the
exact source snapshot matching the plan is accepted.
"""

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time

from glm53_manifest import validate_fp8_scales
from glm53_quantize import (
    GGUF_ARRAY, GGUF_STRING, GGUF_UINT32, SourceDB, TensorPlan, Quantizer,
    Imatrix, QTYPE_F32, QTYPE_F16, QTYPE_Q8_0, QTYPE_Q2_K, QTYPE_Q4_K,
    QTYPE_IQ2_XXS, align, conversion_signature, kv_bool, kv_f32, kv_string,
    kv_u32, kv_u32_array, load_resume_state, pack_string, print_plan,
    qtype_nbytes, save_resume_state, tensor_header, write_experts,
    write_regular,
)

GGUF_ALIGNMENT = 32

QUANTIZATION = {
    "f16": "F16 weights; F32 norms/router (exactness reference)",
    "q8": "Q8_0 experts/attention/shared/embedding/output; F32 norms/router",
    "q4": "Q4_K experts; Q8_0 attention/shared/output; F16 embedding",
    "q2": "IQ2_XXS gate/up experts; Q2_K down experts; Q8_0 attention/shared/output; F16 embedding",
}

ARCH = "kolibri1"
SOURCE_URL = "https://huggingface.co/Aleph-Alpha/Kolibri-1"
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

N_LAYER = 50
N_EXPERT = 384
N_HEAD = 48
N_HEAD_KV = 4
HEAD_DIM = 128
HIDDEN = 2560
EXPERT_WIDTH = 512
SLIDING_WINDOW = 513
VOCAB = 128000
RMS_EPS = 1e-6
ROPE_BASE = 10000.0
FULL_ATTN_INTERVAL = 5

WEIGHT_DTYPES = ("F8_E4M3", "BF16", "F16")
FLOAT_DTYPES = ("BF16", "F32", "F16")

LAYER_TAILS = (
    "input_layernorm.weight",
    "post_attn_norm.weight",
    "post_attention_layernorm.weight",
    "post_ffn_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate.weight",
    "moe.router.expert_bias",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
)

TOP_LEVEL = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")


def fail(message):
    raise ValueError(message)


def load_config(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), "rb") as fp:
        config = json.load(fp)
    if config.get("model_type") != "kolibri1" or config.get("architectures") != ["Kolibri1ForCausalLM"]:
        fail("not a Kolibri-1 source checkpoint")
    expected = {
        "hidden_size": HIDDEN,
        "num_hidden_layers": N_LAYER,
        "num_attention_heads": N_HEAD,
        "num_key_value_heads": N_HEAD_KV,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "num_experts": N_EXPERT,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": EXPERT_WIDTH,
        "shared_expert_intermediate_size": EXPERT_WIDTH,
        "norm_topk_prob": False,
        "sliding_window": SLIDING_WINDOW,
        "rms_norm_eps": RMS_EPS,
        "rope_theta": ROPE_BASE,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            fail(f"config.{key} is {config.get(key)!r}, expected {value!r}")
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != N_LAYER:
        fail("config.layer_types must list every layer")
    for layer, kind in enumerate(layer_types):
        want = "full_attention" if (layer + 1) % FULL_ATTN_INTERVAL == 0 else "sliding_attention"
        if kind != want:
            fail(f"config.layer_types[{layer}] is {kind!r}, expected {want!r}")
    return config


def validate_index(weight_map):
    """Name-level structure of the released index; dtype-aware checks follow."""
    layers = set()
    tails = set()
    experts = {}
    for name in weight_map:
        if name.endswith("_scale_inv"):
            if name.removesuffix("_scale_inv") not in weight_map:
                fail(f"{name}: scale tensor without a source weight")
            continue
        if name in TOP_LEVEL:
            continue
        match = LAYER_RE.match(name)
        if not match:
            fail(f"unknown source tensor: {name}")
        layer, tail = int(match.group(1)), match.group(2)
        if layer >= N_LAYER:
            fail(f"{name}: layer out of range")
        layers.add(layer)
        expert = EXPERT_RE.match(tail)
        if expert:
            experts.setdefault(layer, {}).setdefault(int(expert.group(1)), set()).add(expert.group(2))
        elif tail not in LAYER_TAILS or (layer, tail) in tails:
            fail(f"unknown or duplicate layer tensor: {name}")
        else:
            tails.add((layer, tail))
    if layers != set(range(N_LAYER)):
        fail("source index does not cover every layer")
    for layer in range(N_LAYER):
        for tail in LAYER_TAILS:
            if (layer, tail) not in tails:
                fail(f"missing model.layers.{layer}.{tail}")
        if set(experts.get(layer, {})) != set(range(N_EXPERT)):
            fail(f"layer {layer} does not cover every expert")
        for expert, parts in experts[layer].items():
            if parts != {"gate", "up", "down"}:
                fail(f"layer {layer} expert {expert} projections are {sorted(parts)}")


def expert_qtypes(artifact):
    if artifact == "q2":
        return {"gate": QTYPE_IQ2_XXS, "up": QTYPE_IQ2_XXS, "down": QTYPE_Q2_K}
    if artifact == "q4":
        return {"gate": QTYPE_Q4_K, "up": QTYPE_Q4_K, "down": QTYPE_Q4_K}
    if artifact == "q8":
        return {"gate": QTYPE_Q8_0, "up": QTYPE_Q8_0, "down": QTYPE_Q8_0}
    if artifact == "f16":
        return {"gate": QTYPE_F16, "up": QTYPE_F16, "down": QTYPE_F16}
    fail(f"unknown quantization recipe: {artifact}")


def dense_qtypes(artifact):
    if artifact == "f16":
        return {"weight": QTYPE_F16, "embedding": QTYPE_F16, "output": QTYPE_F16}
    if artifact == "q8":
        return {"weight": QTYPE_Q8_0, "embedding": QTYPE_Q8_0, "output": QTYPE_Q8_0}
    return {"weight": QTYPE_Q8_0, "embedding": QTYPE_F16, "output": QTYPE_Q8_0}


def layer_types(config):
    return [1 if kind == "full_attention" else 0 for kind in config["layer_types"]]


def build_plan(db, config, artifact):
    dense = dense_qtypes(artifact)
    routed = expert_qtypes(artifact)
    plan, consumed = [], set()

    def claim(name, shape, dtypes=WEIGHT_DTYPES):
        info = db.info(name)
        if info["shape"] != list(shape) or info["dtype"] not in dtypes:
            fail(f"{name}: unexpected {info['dtype']} {info['shape']}, expected {dtypes} {shape}")
        consumed.add(name)
        if info["dtype"] == "F8_E4M3":
            scale = name + "_scale_inv"
            db.info(scale)
            consumed.add(scale)

    def regular(name, source, shape, qtype, role, dtypes=WEIGHT_DTYPES):
        claim(source, shape, dtypes)
        plan.append(TensorPlan(name, tuple(reversed(shape)), qtype, role, source=source))

    regular("token_embd.weight", "model.embed_tokens.weight", (VOCAB, HIDDEN),
            dense["embedding"], "embedding")
    regular("output_norm.weight", "model.norm.weight", (HIDDEN,), QTYPE_F32, "norm", FLOAT_DTYPES)
    regular("output.weight", "lm_head.weight", (VOCAB, HIDDEN), dense["output"], "output")

    q_size, kv_size = N_HEAD * HEAD_DIM, N_HEAD_KV * HEAD_DIM
    for layer in range(N_LAYER):
        src, dst = f"model.layers.{layer}", f"blk.{layer}"
        for target, source, shape, qtype, role, dtypes in (
            ("attn_norm.weight", "input_layernorm.weight", (HIDDEN,), QTYPE_F32, "norm", FLOAT_DTYPES),
            ("post_attn_norm.weight", "post_attn_norm.weight", (HIDDEN,), QTYPE_F32, "norm", FLOAT_DTYPES),
            ("ffn_norm.weight", "post_attention_layernorm.weight", (HIDDEN,), QTYPE_F32, "norm", FLOAT_DTYPES),
            ("post_ffn_norm.weight", "post_ffn_norm.weight", (HIDDEN,), QTYPE_F32, "norm", FLOAT_DTYPES),
            ("attn_q.weight", "self_attn.q_proj.weight", (q_size, HIDDEN), dense["weight"], "attention", WEIGHT_DTYPES),
            ("attn_k.weight", "self_attn.k_proj.weight", (kv_size, HIDDEN), dense["weight"], "attention", WEIGHT_DTYPES),
            ("attn_v.weight", "self_attn.v_proj.weight", (kv_size, HIDDEN), dense["weight"], "attention", WEIGHT_DTYPES),
            ("attn_output.weight", "self_attn.o_proj.weight", (HIDDEN, q_size), dense["weight"], "attention", WEIGHT_DTYPES),
            ("attn_q_norm.weight", "self_attn.q_norm.weight", (HEAD_DIM,), QTYPE_F32, "norm", FLOAT_DTYPES),
            ("attn_k_norm.weight", "self_attn.k_norm.weight", (HEAD_DIM,), QTYPE_F32, "norm", FLOAT_DTYPES),
            ("ffn_gate_inp.weight", "mlp.gate.weight", (N_EXPERT, HIDDEN), QTYPE_F32, "router", FLOAT_DTYPES),
            ("exp_probs_b.bias", "moe.router.expert_bias", (N_EXPERT,), QTYPE_F32, "router", FLOAT_DTYPES),
        ):
            regular(f"{dst}.{target}", f"{src}.{source}", shape, qtype, role, dtypes)
        for part in ("gate", "up", "down"):
            source = f"{src}.mlp.shared_experts.{part}_proj.weight"
            shape = (EXPERT_WIDTH, HIDDEN) if part != "down" else (HIDDEN, EXPERT_WIDTH)
            regular(f"{dst}.ffn_{part}_shexp.weight", source, shape, dense["weight"], "shared_expert")
        for part, source in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            pattern = f"{src}.mlp.experts.{{expert}}.{source}.weight"
            shape = (EXPERT_WIDTH, HIDDEN) if part != "down" else (HIDDEN, EXPERT_WIDTH)
            for expert in range(N_EXPERT):
                claim(pattern.format(expert=expert), shape)
            item = TensorPlan(
                f"{dst}.ffn_{part}_exps.weight",
                (*reversed(shape), N_EXPERT),
                routed[part],
                f"routed_{part}",
                source=pattern,
                expert_layer=layer,
                expert_part=part,
                expert_count=N_EXPERT,
            )
            item.nbytes = qtype_nbytes(item.qtype, item.shape)
            plan.append(item)

    if consumed != set(db.tensors):
        missing = sorted(set(db.tensors) - consumed)
        fail(f"unclaimed source tensors: {missing[:10]}")
    names = [item.name for item in plan]
    if len(names) != len(set(names)):
        fail("duplicate GGUF tensor names in conversion plan")
    offset = 0
    for item in plan:
        item.offset = offset
        if not item.nbytes:
            item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan


def array_string_record(key, values):
    header = pack_string(key) + struct.pack("<IIQ", GGUF_ARRAY, GGUF_STRING, len(values))
    return header + b"".join(pack_string(value) for value in values)


def array_i32_record(key, values):
    header = pack_string(key) + struct.pack("<IIQ", GGUF_ARRAY, 5, len(values))
    return header + struct.pack(f"<{len(values)}i", *values)


def tokenize_records(hf_dir):
    path = os.path.join(hf_dir, "tokenizer.json")
    with open(path, "rb") as fp:
        document = json.load(fp)
    model = document.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE" or model.get("ignore_merges"):
        fail("unsupported tokenizer; expected a BPE tokenizer with merges enabled")
    vocab, added = model.get("vocab"), document.get("added_tokens")
    if not isinstance(vocab, dict) or not isinstance(added, list):
        fail(f"{path}: unsupported tokenizer structure")
    tokens = [None] * VOCAB
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int) or not 0 <= token_id < VOCAB or tokens[token_id] is not None:
            fail(f"{path}: invalid base vocabulary entry")
        tokens[token_id] = token
    special = set()
    for entry in added:
        token, token_id = entry.get("content"), entry.get("id")
        if not isinstance(token, str) or not isinstance(token_id, int) or not 0 <= token_id < VOCAB or tokens[token_id] is not None:
            fail(f"{path}: invalid added token entry")
        tokens[token_id] = token
        if entry.get("special"):
            special.add(token_id)
    # The released tokenizer has 127,998 ids; the embedding has 128,000 rows.
    for token_id, token in enumerate(tokens):
        if token is None:
            tokens[token_id] = f"[PAD{token_id}]"
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in model.get("merges", [])]
    if not merges:
        fail(f"{path}: tokenizer has no merges")
    with open(os.path.join(hf_dir, "tokenizer_config.json"), "rb") as fp:
        tokenizer_config = json.load(fp)
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        fail("tokenizer_config.json has no chat_template")
    return [
        kv_string("tokenizer.ggml.model", "gpt2"),
        kv_string("tokenizer.ggml.pre", "kolibri1"),
        kv_string("tokenizer.chat_template", chat_template),
        array_string_record("tokenizer.ggml.tokens", tokens),
        array_i32_record("tokenizer.ggml.token_type", [3 if i in special else 1 for i in range(len(tokens))]),
        array_string_record("tokenizer.ggml.merges", merges),
        kv_u32("tokenizer.ggml.eos_token_id", 127906),
        kv_u32("tokenizer.ggml.padding_token_id", 127901),
        kv_bool("tokenizer.ggml.add_bos_token", False),
        kv_bool("tokenizer.ggml.add_eos_token", False),
    ]


def model_metadata(config, source_revision, artifact, fp8):
    records = [
        kv_string("general.architecture", ARCH),
        kv_string("general.name", "Kolibri-1"),
        kv_string("general.source.url", SOURCE_URL + ("" if fp8 else "-BF16")),
        kv_string("general.source.revision", source_revision),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string(f"{ARCH}.config", json.dumps(config, sort_keys=True, separators=(",", ":"))),
        kv_u32(f"{ARCH}.block_count", N_LAYER),
        kv_u32(f"{ARCH}.context_length", config["max_position_embeddings"]),
        kv_u32(f"{ARCH}.embedding_length", HIDDEN),
        kv_u32(f"{ARCH}.feed_forward_length", EXPERT_WIDTH),
        kv_u32(f"{ARCH}.expert_count", N_EXPERT),
        kv_u32(f"{ARCH}.expert_used_count", config["num_experts_per_tok"]),
        kv_u32(f"{ARCH}.expert_shared_count", 1),
        kv_bool(f"{ARCH}.expert_weights_norm", config["norm_topk_prob"]),
        kv_f32(f"{ARCH}.expert_weights_scale", 1.0),
        kv_bool(f"{ARCH}.router.expert_bias", True),
        kv_string(f"{ARCH}.router.selection", "logit_add_bias_sigmoid"),
        kv_u32(f"{ARCH}.attention.head_count", N_HEAD),
        kv_u32(f"{ARCH}.attention.head_count_kv", N_HEAD_KV),
        kv_u32(f"{ARCH}.attention.key_length", HEAD_DIM),
        kv_u32(f"{ARCH}.attention.value_length", HEAD_DIM),
        kv_f32(f"{ARCH}.attention.layer_norm_rms_epsilon", RMS_EPS),
        kv_u32(f"{ARCH}.attention.sliding_window", SLIDING_WINDOW),
        kv_u32_array(f"{ARCH}.attention.layer_types", layer_types(config)),
        kv_bool(f"{ARCH}.attention.full_attention_rope", False),
        kv_f32(f"{ARCH}.rope.freq_base", ROPE_BASE),
        kv_u32(f"{ARCH}.rope.dimension_count", HEAD_DIM),
        kv_string(f"{ARCH}.quantization", QUANTIZATION[artifact]),
        kv_string(f"{ARCH}.calibration",
                  "lossless conversion" if artifact == "f16" else "weight-energy bootstrap"),
    ]
    return records


class KolibriQuantizer(Quantizer):
    def encode(self, array, qtype, imatrix=None):
        if qtype == QTYPE_F16 and self.np.any(self.np.abs(array) > 65504):
            raise ValueError("F16 tensor would overflow; convert from the BF16 release or use q8")
        return super().encode(array, qtype, imatrix)


def write_gguf(args, plan, records, db):
    quantizer = KolibriQuantizer(args.quants_library)
    imatrix = Imatrix(args.imatrix, quantizer.np)
    if args.imatrix:
        for item in plan:
            if item.is_expert and item.name not in imatrix.entries:
                fail(f"missing imatrix tensor {item.name}")
    data_start, data_bytes = print_plan(plan, records, [], GGUF_ALIGNMENT)
    partial, journal = args.out + ".partial", args.out + ".partial.json"
    signature = conversion_signature(plan, records, [], args.imatrix)
    source_identity = [(name, db.info(name)) for name in sorted(db.tensors)]
    signature = hashlib.sha256((signature + json.dumps(source_identity, sort_keys=True)).encode()).hexdigest()
    if os.path.exists(args.out):
        fail(f"refusing to overwrite {args.out}")
    completed = 0
    if os.path.exists(partial) or os.path.exists(journal):
        if not args.resume or not (os.path.exists(partial) and os.path.exists(journal)):
            fail("partial file and journal require --resume")
        completed = load_resume_state(journal, signature, plan)
    end = data_start + (plan[completed - 1].offset +
                        align(plan[completed - 1].nbytes, GGUF_ALIGNMENT) if completed else 0)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free
    if free < data_start + data_bytes - end + (32 << 30):
        fail("insufficient disk space for remaining output plus 32 GiB reserve")
    header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
    header += b"".join(records) + b"".join(tensor_header(item) for item in plan)
    header += bytes(data_start - len(header))
    if os.path.exists(partial):
        with open(partial, "rb") as fp:
            if fp.read(data_start) != header or os.fstat(fp.fileno()).st_size < end:
                fail("partial GGUF is truncated or has a different header")
    else:
        with open(partial, "xb") as fp:
            fp.write(header)
            fp.flush()
            os.fsync(fp.fileno())
        save_resume_state(journal, signature, 0)
    with open(partial, "r+b") as fp:
        fp.truncate(end)
        fp.seek(end)
        for index in range(completed, len(plan)):
            item = plan[index]
            started = time.monotonic()
            if fp.tell() != data_start + item.offset:
                fail(f"incorrect offset for {item.name}")
            if item.is_expert:
                write_experts(fp, item, db, quantizer, imatrix, args.threads)
            else:
                write_regular(fp, item, db, quantizer)
            if fp.tell() != data_start + item.offset + item.nbytes:
                fail(f"incorrect payload size for {item.name}")
            fp.write(bytes(align(item.nbytes, GGUF_ALIGNMENT) - item.nbytes))
            fp.flush()
            os.fsync(fp.fileno())
            save_resume_state(journal, signature, index + 1)
            print(f"[{index + 1}/{len(plan)}] {item.name}: {item.nbytes / (1 << 30):.3f} GiB, "
                  f"{time.monotonic() - started:.1f}s", flush=True)
    os.rename(partial, args.out)
    os.unlink(journal)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="Kolibri-1 or Kolibri-1-BF16 snapshot directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--quant", choices=QUANTIZATION, default="q4")
    parser.add_argument("--imatrix")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=os.path.join(os.path.dirname(__file__), f"libds4quants.{suffix}"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    if not 1 <= args.threads <= 32:
        parser.error("threads must be between 1 and 32")
    if args.quant == "f16" and args.imatrix:
        parser.error("--imatrix does not apply to the lossless F16 recipe")
    return args


def main():
    args = parse_args()
    config = load_config(args.hf)
    db = SourceDB(args.hf, index_validator=validate_index, scale_validator=validate_fp8_scales)
    try:
        fp8 = any(info["dtype"] == "F8_E4M3" for info in db.tensors.values())
        plan = build_plan(db, config, args.quant)
        records = model_metadata(config, args.source_revision, args.quant, fp8)
        records += tokenize_records(args.hf)
        if args.imatrix:
            records.append(kv_string("quantize.imatrix.file", os.path.basename(args.imatrix)))
        if args.dry_run:
            print(f"source: {'FP8' if fp8 else 'BF16'} {len(db.tensors)} tensors")
            print_plan(plan, records, [], GGUF_ALIGNMENT)
            for item in plan:
                print(json.dumps(dataclasses.asdict(item), sort_keys=True))
        else:
            if args.overwrite:
                for path in (args.out + ".partial", args.out + ".partial.json"):
                    if os.path.exists(path):
                        os.unlink(path)
            write_gguf(args, plan, records, db)
            print(f"kolibri1-quantize: wrote {args.out}", file=sys.stderr)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"kolibri1-quantize: {error}")
