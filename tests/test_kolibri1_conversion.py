#!/usr/bin/env python3
"""Kolibri-1 conversion fixtures: index, plan, tokenizer, quantizer, writer.

Runs without the released checkpoint: the full tensor catalogue is synthesized
from the pinned shape constants, and payload tests use small in-memory tensors.
"""

import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from glm53_quantize import (
    QTYPE_F32, QTYPE_F16, QTYPE_Q8_0, QTYPE_Q4_K, QTYPE_IQ2_XXS, QTYPE_Q2_K,
    TensorPlan, align, kv_string, load_tokenizer_records, qtype_nbytes,
)
from kolibri1_quantize import (
    EXPERT_WIDTH, FULL_ATTN_INTERVAL, HIDDEN, N_EXPERT, N_HEAD, N_HEAD_KV,
    N_LAYER, QUANTIZATION, RMS_EPS, ROPE_BASE, SLIDING_WINDOW, VOCAB,
    HEAD_DIM, KolibriQuantizer, build_plan, load_config, tokenize_records,
    validate_index, write_gguf,
)
import kolibri1_validate_gguf as artifact_audit

SUFFIX = "dylib" if sys.platform == "darwin" else "so"


def source_config():
    return {
        "model_type": "kolibri1",
        "architectures": ["Kolibri1ForCausalLM"],
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
        "max_position_embeddings": 262144,
        "layer_types": ["full_attention" if (i + 1) % FULL_ATTN_INTERVAL == 0
                        else "sliding_attention" for i in range(N_LAYER)],
    }


def catalogue(fp8=True):
    """Every released tensor name with its real shape and dtype."""
    tensors = {}

    def put(name, shape, dtype):
        tensors[name] = {"shape": list(shape), "dtype": dtype}

    def put_weight(name, shape):
        put(name, shape, "F8_E4M3" if fp8 else "BF16")
        if fp8:
            put(name + "_scale_inv", [(dim + 127) // 128 for dim in shape], "F32")

    put("model.embed_tokens.weight", (VOCAB, HIDDEN), "BF16")
    put("model.norm.weight", (HIDDEN,), "BF16")
    put("lm_head.weight", (VOCAB, HIDDEN), "BF16")
    for layer in range(N_LAYER):
        prefix = f"model.layers.{layer}"
        put(f"{prefix}.input_layernorm.weight", (HIDDEN,), "BF16")
        put(f"{prefix}.post_attn_norm.weight", (HIDDEN,), "BF16")
        put(f"{prefix}.post_attention_layernorm.weight", (HIDDEN,), "BF16")
        put(f"{prefix}.post_ffn_norm.weight", (HIDDEN,), "BF16")
        put_weight(f"{prefix}.self_attn.q_proj.weight", (N_HEAD * HEAD_DIM, HIDDEN))
        put_weight(f"{prefix}.self_attn.k_proj.weight", (N_HEAD_KV * HEAD_DIM, HIDDEN))
        put_weight(f"{prefix}.self_attn.v_proj.weight", (N_HEAD_KV * HEAD_DIM, HIDDEN))
        put_weight(f"{prefix}.self_attn.o_proj.weight", (HIDDEN, N_HEAD * HEAD_DIM))
        put(f"{prefix}.self_attn.q_norm.weight", (HEAD_DIM,), "BF16")
        put(f"{prefix}.self_attn.k_norm.weight", (HEAD_DIM,), "BF16")
        put(f"{prefix}.mlp.gate.weight", (N_EXPERT, HIDDEN), "BF16")
        put(f"{prefix}.moe.router.expert_bias", (N_EXPERT,), "BF16")
        for part in ("gate", "up", "down"):
            shape = (EXPERT_WIDTH, HIDDEN) if part != "down" else (HIDDEN, EXPERT_WIDTH)
            put_weight(f"{prefix}.mlp.shared_experts.{part}_proj.weight", shape)
            for expert in range(N_EXPERT):
                put_weight(f"{prefix}.mlp.experts.{expert}.{part}_proj.weight", shape)
    return tensors


class CatalogueDB:
    def __init__(self, tensors):
        self.tensors = tensors

    def info(self, name):
        return self.tensors[name]

    def close(self):
        pass


class MemoryDB:
    def __init__(self, name, weights, scales, dtype="F8_E4M3"):
        self.name = name
        self.arrays = {name: weights, name + "_scale_inv": scales}
        self.tensors = {
            name: {"shape": list(weights.shape), "dtype": dtype},
            name + "_scale_inv": {"shape": list(scales.shape), "dtype": "F32"},
        }

    def info(self, name):
        return self.tensors[name]

    def read(self, name):
        return self.arrays[name].tobytes()

    def iter_read(self, name, byte_start=0, byte_count=None):
        data = self.read(name)
        yield data[byte_start:None if byte_count is None else byte_start + byte_count]

    def close(self):
        pass


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = KolibriQuantizer(str(ROOT / "gguf-tools" / f"libds4quants.{SUFFIX}"))

    def test_index(self):
        names = catalogue()
        self.assertEqual(len(names), 116303)
        validate_index(dict.fromkeys(names, "shard.safetensors"))
        mutators = (
            "model.layers.7.mlp.experts.300.up_proj.weight",
            "model.layers.2.post_ffn_norm.weight",
            "model.layers.49.mlp.shared_experts.gate_proj.weight",
        )
        for victim in mutators:
            reduced = {name: "shard.safetensors" for name in names if name != victim}
            with self.assertRaises(ValueError):
                validate_index(reduced)
        extra = dict.fromkeys(names, "shard.safetensors")
        extra["model.layers.0.self_attn.rotary.weight"] = "shard.safetensors"
        with self.assertRaisesRegex(ValueError, "unknown|duplicate"):
            validate_index(extra)

    def test_config(self):
        config = source_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(tmp)["num_experts"], N_EXPERT)
            for key, value in (("num_experts_per_tok", 8), ("norm_topk_prob", True),
                               ("sliding_window", 256), ("rope_theta", 1000.0)):
                broken = dict(config, **{key: value})
                path.write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError, key):
                    load_config(tmp)

    def test_plan(self):
        for fp8, expected_expert_dtype in ((True, "F8_E4M3"), (False, "BF16")):
            tensors = catalogue(fp8=fp8)
            db = CatalogueDB(tensors)
            plan = build_plan(db, source_config(), "q4")
            self.assertEqual(len(plan), 3 + N_LAYER * 18)
            by_name = {item.name: item for item in plan}
            embedding = by_name["token_embd.weight"]
            self.assertEqual((embedding.shape, embedding.qtype),
                             ((HIDDEN, VOCAB), QTYPE_F16))
            self.assertEqual(embedding.nbytes, HIDDEN * VOCAB * 2)
            expert = by_name["blk.7.ffn_gate_exps.weight"]
            self.assertEqual((expert.shape, expert.qtype, expert.expert_count),
                             ((HIDDEN, EXPERT_WIDTH, N_EXPERT), QTYPE_Q4_K, N_EXPERT))
            self.assertEqual(expert.nbytes,
                             qtype_nbytes(QTYPE_Q4_K, (HIDDEN, EXPERT_WIDTH, N_EXPERT)))
            down = by_name["blk.7.ffn_down_exps.weight"]
            self.assertEqual(down.shape, (EXPERT_WIDTH, HIDDEN, N_EXPERT))
            attention = by_name["blk.49.attn_q.weight"]
            self.assertEqual(attention.shape, (HIDDEN, N_HEAD * HEAD_DIM))
            self.assertEqual(by_name["blk.49.post_ffn_norm.weight"].qtype, QTYPE_F32)
            self.assertEqual(by_name["blk.49.exp_probs_b.bias"].source,
                             "model.layers.49.moe.router.expert_bias")
            for previous, item in zip(plan, plan[1:]):
                self.assertEqual(item.offset, previous.offset + align(previous.nbytes))
            if fp8:
                extra = dict(tensors)
                extra["model.layers.0.self_attn.rotary.weight"] = {"shape": [8], "dtype": "F32"}
                with self.assertRaisesRegex(ValueError, "unclaimed"):
                    build_plan(CatalogueDB(extra), source_config(), "q4")
            else:
                self.assertEqual(tensors["model.layers.0.mlp.gate.weight"]["dtype"], "BF16")

    def test_recipes(self):
        db = CatalogueDB(catalogue())
        for artifact in QUANTIZATION:
            plan = build_plan(db, source_config(), artifact)
            expert_types = {item.qtype for item in plan if item.is_expert}
            if artifact == "f16":
                self.assertEqual(expert_types, {QTYPE_F16})
            elif artifact == "q8":
                self.assertEqual(expert_types, {QTYPE_Q8_0})
            elif artifact == "q4":
                self.assertEqual(expert_types, {QTYPE_Q4_K})
            else:
                self.assertEqual(expert_types, {QTYPE_IQ2_XXS, QTYPE_Q2_K})

    def test_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = {
                "model": {"type": "BPE", "ignore_merges": False,
                          "vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]},
                "added_tokens": [
                    {"id": 127906, "content": "<|im_end|>", "special": True},
                    {"id": 127907, "content": "<think>", "special": False},
                ],
            }
            (Path(tmp) / "tokenizer.json").write_text(json.dumps(tokenizer))
            (Path(tmp) / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "{{ messages }}"}))
            records = tokenize_records(tmp)
            self.assertIn(b"kolibri1", b"".join(records))
            with tempfile.NamedTemporaryFile(suffix=".gguf") as fp:
                fp.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(records)) + b"".join(records))
                fp.flush()
                _, tokens = load_tokenizer_records(fp.name)
            self.assertEqual(len(tokens), VOCAB)
            self.assertEqual((tokens[0], tokens[1], tokens[127906], tokens[127907]),
                             ("a", "b", "<|im_end|>", "<think>"))
            self.assertEqual(tokens[127998], "[PAD127998]")
            broken = dict(tokenizer, model=dict(tokenizer["model"], ignore_merges=True))
            (Path(tmp) / "tokenizer.json").write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "unsupported tokenizer"):
                tokenize_records(tmp)

    def test_fp8_blocks(self):
        codes = (np.arange(128 * 256, dtype=np.uint32) % 200).astype(np.uint8).reshape(128, 256)
        codes[(codes & 0x7F) == 0x7F] = 0
        scales = np.array([[2.0 ** -3, 2.0 ** -2]], dtype=np.float32)
        db = MemoryDB("model.layers.1.self_attn.q_proj.weight", codes, scales)
        actual = self.q.to_f32(db, db.name)
        expected = self.q.fp8_lut[codes] * np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)
        np.testing.assert_array_equal(actual, expected)
        encoded = self.q.encode(actual, QTYPE_Q8_0)
        self.assertEqual(len(encoded), qtype_nbytes(QTYPE_Q8_0, (256, 128)))
        with self.assertRaisesRegex(ValueError, "overflow"):
            self.q.encode(np.array([[70000.0]], np.float32), QTYPE_F16)

    def test_validator_payload(self):
        codes = (np.arange(64 * 128, dtype=np.uint32) % 180).astype(np.uint8).reshape(64, 128)
        codes[(codes & 0x7F) == 0x7F] = 0
        scales = np.ones((1, 1), dtype=np.float32)
        db = MemoryDB("model.layers.1.self_attn.v_proj.weight", codes, scales)
        item = TensorPlan("blk.1.attn_v.weight", (128, 64), QTYPE_Q8_0, "attention",
                          source=db.name)
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        payload = self.q.encode(self.q.to_f32(db, db.name), item.qtype)
        fp = io.BytesIO(payload)
        artifact_audit.check_payload(fp, 0, item, db, self.q, None)
        fp.seek(len(payload) // 2)
        fp.write(bytes([payload[len(payload) // 2] ^ 1]))
        with self.assertRaisesRegex(ValueError, "differs from source"):
            artifact_audit.check_payload(fp, 0, item, db, self.q, None)

    def test_resume(self):
        class DB:
            tensors = {f"test.{i}": {"shape": [2, 32], "dtype": "F32"} for i in range(3)}

            def info(self, name):
                return self.tensors[name]

            def read(self, name):
                return (np.arange(64, dtype=np.float32) + int(name[-1])).tobytes()

            def close(self):
                pass

        db = DB()
        plan = []
        for name in db.tensors:
            item = TensorPlan(name, (32, 2), QTYPE_F32, "test", source=name)
            item.nbytes = qtype_nbytes(item.qtype, item.shape)
            item.offset = sum(align(t.nbytes) for t in plan)
            plan.append(item)
        records = [kv_string("general.architecture", "kolibri1"),
                   kv_string("general.source.revision", "0" * 40)]
        original = KolibriQuantizer.to_f32

        def interrupted(quantizer, db, name, *args):
            if name == "test.1":
                raise OSError("interrupted conversion")
            return original(quantizer, db, name, *args)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = types.SimpleNamespace(out=str(Path(tmp) / "model.gguf"), imatrix=None,
                                         quants_library=self.q.lib._name, threads=2, resume=False)
            partial = Path(args.out + ".partial")
            journal = Path(args.out + ".partial.json")
            with mock.patch.object(KolibriQuantizer, "to_f32", interrupted):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    write_gguf(args, plan, records, db)
            self.assertEqual(json.loads(journal.read_text())["completed"], 1)
            with self.assertRaisesRegex(ValueError, "require --resume"):
                write_gguf(args, plan, records, db)
            args.resume = True
            with self.assertRaisesRegex(ValueError, "does not match"):
                write_gguf(args, plan, records + [kv_string("changed", "recipe")], db)
            with partial.open("ab") as fp:
                fp.write(b"unfinished tensor payload")
            write_gguf(args, plan, records, db)
            self.assertFalse(partial.exists())
            self.assertFalse(journal.exists())
            result = Path(args.out).read_bytes()
            expected = b"".join(db.read(t.source) + bytes(align(t.nbytes) - t.nbytes)
                                for t in plan)
            self.assertEqual(result[-len(expected):], expected)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                write_gguf(args, plan, records, db)


if __name__ == "__main__":
    unittest.main()
