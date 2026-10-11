"""Split qwen4exp sources: conversion rules, exact dense conversions and the assembled layout."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen4_gguf_q2 as q2
from qwen4_pack_to_qwen4exp import Arr, Reader, kv_bytes, w_str


def write_gguf(path, metadata, tensors):
    """tensors: name -> (type, dims, payload), laid out in order at 32-byte alignment."""
    prefix = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
    offset, layout = 0, []
    for name, (kind, dims, payload) in tensors.items():
        prefix += w_str(name) + struct.pack('<I', len(dims)) + struct.pack('<' + 'Q' * len(dims), *dims)
        prefix += struct.pack('<IQ', kind, offset)
        layout.append((offset, payload))
        offset = (offset + len(payload) + 31) // 32 * 32
    data_start = (len(prefix) + 31) // 32 * 32
    blob = bytearray(prefix + bytes(data_start - len(prefix) + offset))
    for pos, payload in layout:
        blob[data_start + pos:data_start + pos + len(payload)] = payload
    path.write_bytes(bytes(blob))


def q4k_rows(rows, rng):
    blocks = bytearray()
    for _ in range(rows):
        blocks += np.array([0.01, 0.005], dtype='<f2').tobytes()
        blocks += rng.integers(0, 256, 12 + 128, dtype=np.uint8).tobytes()
    return bytes(blocks)


class SplitSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        self.bf16 = (rng.standard_normal(4 * 8).astype('<f4').view('<u4') >> 16).astype('<u2').tobytes()
        self.copied = rng.integers(0, 1 << 15, 64, dtype=np.uint16).astype('<u2').tobytes()
        self.q4k = q4k_rows(2, rng)
        self.table = rng.integers(0, 1 << 14, 160 * 3, dtype=np.uint16).astype('<u2').tobytes()
        split = {'split.count': (2, 2), 'split.tensors.count': (4, 4)}
        write_gguf(self.root / 'm-00001-of-00002.gguf', {
            'general.architecture': (8, 'qwen4exp'), 'qwen4exp.expert_count': (4, 288),
            'qwen4exp.embedding_length_per_layer_input': (4, 160),
            'qwen4exp.ple.head_offsets': (9, Arr(10, [0, 1])),
            'qwen4exp.ple.head_vocab_sizes': (9, Arr(4, [1, 2])), 'split.no': (2, 0), **split}, {
            'blk.0.attn_q.weight': (q2.T_Q4_K, [256, 2], self.q4k),
            'blk.1.ple_conv1d.weight': (q2.T_BF16, [4, 8], self.bf16),
            'blk.0.ssm_out.weight': (q2.T_BF16, [32, 2], self.copied)})
        write_gguf(self.root / 'm-00002-of-00002.gguf', {'split.no': (2, 1), **split}, {
            q2.TABLE: (q2.T_BF16, [160, 3], self.table)})
        self.first = self.root / 'm-00001-of-00002.gguf'

    def test_conversion_rules(self):
        self.assertEqual(q2.conversion('blk.3.ffn_gate_exps.weight', q2.T_Q4_K, [2560, 640, 288]), q2.T_IQ2_XXS)
        self.assertEqual(q2.conversion('blk.47.ffn_down_exps.weight', 6, [640, 2560, 288]), q2.T_Q2_K)
        self.assertEqual(q2.target_dims('blk.0.ffn_down_exps.weight', 6, [640, 2560, 288]), [768, 2560, 288])
        self.assertEqual(q2.conversion('blk.48.ffn_up_exps.weight', q2.T_BF16, [2560, 640, 288]), q2.T_Q8_0)
        self.assertIsNone(q2.conversion('blk.48.ffn_up_exps.weight', q2.T_Q4_K, [2560, 640, 288]))
        self.assertEqual(q2.conversion('blk.4.ffn_down_shexp.weight', 6, [640, 2560]), q2.T_F16)
        self.assertIsNone(q2.conversion('blk.4.attn_qkv.weight', q2.T_BF16, [2560, 10240]))
        self.assertIsNone(q2.conversion('blk.4.ple_conv1d.weight', q2.T_F16, [4, 10240]))
        with self.assertRaises(SystemExit):
            q2.conversion(q2.TABLE, q2.T_Q8_0, [160, 3])

    def test_dense_conversions_keep_the_values(self):
        source = q2.Source(self.first)
        self.assertEqual(len(source.tensors), 4)
        half, _ = q2.convert_tensor(source, 'blk.0.attn_q.weight', q2.T_F16, None, None, None, 1)
        grid = dequantize(np.frombuffer(self.q4k, dtype=np.uint8).reshape(2, 144), GGMLQuantizationType.Q4_K)
        error = np.abs(np.frombuffer(half, dtype='<f2').astype(np.float32).reshape(2, 256) - grid)
        self.assertLess(error.max(), 1e-3 * np.abs(grid).max())
        single, _ = q2.convert_tensor(source, 'blk.1.ple_conv1d.weight', q2.T_F32, None, None, None, 1)
        widened = (np.frombuffer(self.bf16, dtype='<u2').astype('<u4') << 16).view('<f4')
        self.assertEqual(single, widened.tobytes())

    def test_table_from_another_gguf_replaces_a_missing_shard(self):
        donor = self.root / 'donor.gguf'
        write_gguf(donor, {'general.architecture': (8, 'qwen4exp')}, {
            'blk.0.other.weight': (q2.T_F32, [8], bytes(32)), q2.TABLE: (q2.T_BF16, [160, 3], self.table)})
        (self.root / 'm-00002-of-00002.gguf').unlink()
        with self.assertRaises(SystemExit):
            q2.Source(self.first)
        source = q2.Source(self.first, donor)
        self.assertEqual(len(source.tensors), 4)
        self.assertEqual(bytes(source.raw(q2.TABLE)), self.table)
        write_gguf(donor, {'general.architecture': (8, 'qwen4exp')}, {
            q2.TABLE: (q2.T_BF16, [160, 2], self.table[:640])})
        with self.assertRaises(SystemExit):
            q2.Source(self.first, donor)

    def test_assembled_file_has_the_table_last_and_aligned(self):
        source = q2.Source(self.first)
        work = self.root / 'work'
        work.mkdir()
        tensors = {}
        for name in ('blk.0.attn_q.weight', 'blk.1.ple_conv1d.weight'):
            _, kind, dims, _ = source.tensors[name]
            target = q2.conversion(name, kind, dims)
            raw, _ = q2.convert_tensor(source, name, target, None, None, None, 1)
            (work / name).write_bytes(raw)
            tensors[name] = {'type': target, 'dims': dims, 'bytes': len(raw),
                             'sha256': hashlib.sha256(raw).hexdigest()}
        (work / 'tensors.json').write_text(json.dumps({
            'source': str(self.first.resolve()), 'imatrix': {'sha256': 'test'}, 'map_sha256': 'test',
            'tensors': tensors}))
        out = self.root / 'out.gguf'
        q2.assemble(type('Args', (), {'source': self.first, 'table': None, 'work_dir': work, 'out': out})())
        result = Reader(str(out))
        self.assertEqual(list(result.tensors)[-1], q2.TABLE)
        self.assertEqual((result.data_start + result.tensors[q2.TABLE][2]) % q2.TABLE_ALIGN, 0)
        self.assertNotIn('split.count', result.kv)
        self.assertEqual(result.kv['qwen4exp.expert_count'], 288)
        self.assertEqual(result.tensors['blk.0.attn_q.weight'][0], q2.T_F16)
        result.f.seek(result.data_start + result.tensors[q2.TABLE][2])
        self.assertEqual(result.f.read(len(self.table)), self.table)
        result.f.seek(result.data_start + result.tensors['blk.0.ssm_out.weight'][2])
        self.assertEqual(result.f.read(len(self.copied)), self.copied)
        result.f.close()


if __name__ == '__main__':
    unittest.main()
