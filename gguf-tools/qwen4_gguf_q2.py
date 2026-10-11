#!/usr/bin/env python3
"""Requantize a llama.cpp qwen4exp GGUF's routed experts to the DS4 Q2 recipe.

The source may be split and may be an expert-pruned derivative, such as one
keeping 288 of the 512 experts. Trunk gate/up experts become IQ2_XXS and down
experts Q2_K with their 640 inputs padded to 768, both weighted by the pinned
base-model imatrix. The `map` stage matches every pruned expert to the base
expert it was kept from by its router row, so each takes its own statistics.

Tensors DS4 cannot execute are converted. Dense projections in other block
formats become F16, which keeps a Q4_K grid to about 2e-4 where Q8_0 would add
6e-3. MTP experts in other formats become Q8_0, the closest type with a tiled
prefill path; F16 experts would fall back to per-token rows. A BF16 n-gram
convolution becomes F32. Every other tensor, the BF16 n-gram table included,
is copied unchanged. The table goes last, on a page boundary, where DS4 reads
rows without mapping it.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time

import numpy as np
from gguf import GGMLQuantizationType
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize

from qwen4_pack import QwenGGUFImatrix, GGMLQuantizer, Q2_IMATRIX_SHA256
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str

T_F32, T_F16, T_Q4_0, T_Q8_0, T_Q2_K, T_Q4_K, T_IQ2_XXS, T_BF16, T_MXFP4 = 0, 1, 2, 8, 10, 12, 16, 30, 39
DENSE_TYPES = {T_F32, T_F16, T_BF16, T_Q8_0, T_Q4_0}
EXPERT_TYPES = {T_F32, T_F16, T_Q4_0, T_Q8_0, T_Q2_K, T_Q4_K, T_IQ2_XXS, T_MXFP4}
TRUNK_LAYERS = 48
TABLE = 'per_layer_token_embd.weight'
TABLE_ALIGN = 1 << 16
EXPERT_RE = re.compile(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight')
MIN_COSINE = 0.999
MIN_MARGIN = 0.005


def tensor_bytes(kind, dims):
    block, size = GGML_QUANT_SIZES[GGMLQuantizationType(kind)]
    n = math.prod(dims)
    if n % block:
        raise SystemExit(f'{n} values do not fill whole {GGMLQuantizationType(kind).name} blocks')
    return n // block * size


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


class Source:
    """A possibly split GGUF: metadata from the first shard, tensors from all of them. With `table`, the
    n-gram table comes from that GGUF instead and a shard holding only the table may be absent; the
    caller must have established that the two tables are the same bytes."""

    def __init__(self, first, table=None):
        match = re.fullmatch(r'(.*)-00001-of-(\d{5})\.gguf', first.name)
        count = int(match[2]) if match else 1
        paths = [first.with_name(f'{match[1]}-{i:05d}-of-{match[2]}.gguf') for i in range(1, count + 1)] \
            if match else [first]
        missing = [p for p in paths if not p.is_file()]
        if missing and table is None:
            raise SystemExit(f'Missing shard {missing[0]}')
        self.readers = [Reader(str(p)) for p in paths if p.is_file()]
        self.kv, self.kv_types = self.readers[0].kv, self.readers[0].kv_types
        self.maps = {}
        self.tensors = {}
        for reader in self.readers:
            for name, (kind, dims, offset) in reader.tensors.items():
                self.tensors[name] = (reader, kind, dims, offset)
        if table is not None:
            self.tensors.pop(TABLE, None)
            donor = Reader(str(table))
            kind, dims, offset = donor.tensors[TABLE]
            # DS4's rule: the table may be padded, but must cover every head's rows
            offsets = self.kv['qwen4exp.ple.head_offsets'].values
            rows = max(o + s for o, s in zip(offsets, self.kv['qwen4exp.ple.head_vocab_sizes'].values))
            if (kind != T_BF16 or len(dims) != 2 or dims[1] < rows or
                    dims[0] != self.kv['qwen4exp.embedding_length_per_layer_input']):
                raise SystemExit(f'{table}: table {dims} type {kind} does not fit this model\'s n-grams')
            self.readers.append(donor)
            self.tensors[TABLE] = (donor, kind, dims, offset)
        if len(self.tensors) != self.kv.get('split.tensors.count', len(self.tensors)):
            raise SystemExit(f'{first}: shards hold {len(self.tensors)} tensors, '
                             f'expected {self.kv["split.tensors.count"]}')
        if self.kv.get('general.architecture') != 'qwen4exp':
            raise SystemExit(f'{first}: not a qwen4exp GGUF')

    def raw(self, name, start=0, length=None):
        reader, kind, dims, offset = self.tensors[name]
        if reader not in self.maps:
            self.maps[reader] = np.memmap(reader.f.name, dtype=np.uint8, mode='r')
        size = tensor_bytes(kind, dims)
        length = size - start if length is None else length
        if start + length > size:
            raise SystemExit(f'{name}: read past the tensor end')
        begin = reader.data_start + offset + start
        return self.maps[reader][begin:begin + length]

    def rows(self, name, first_row, count):
        """Rows of a tensor as float32, each dims[0] values wide."""
        _, kind, dims, _ = self.tensors[name]
        row = tensor_bytes(kind, [dims[0]])
        raw = np.asarray(self.raw(name, first_row * row, count * row)).reshape(count, row)
        return np.ascontiguousarray(dequantize(raw, GGMLQuantizationType(kind)), dtype=np.float32)


def router(source, layer, experts):
    values = source.rows(f'blk.{layer}.ffn_gate_inp.weight', 0, experts)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def expert_map(args):
    source, base = Source(args.source, args.table), Source(args.base)
    count, base_count = source.kv['qwen4exp.expert_count'], base.kv['qwen4exp.expert_count']
    if count > base_count:
        raise SystemExit(f'Source has {count} experts, more than the base model\'s {base_count}')
    layers = {}
    for layer in range(TRUNK_LAYERS):
        similarity = router(source, layer, count) @ router(base, layer, base_count).T
        order = np.argsort(-similarity, axis=1)
        best = similarity[np.arange(count), order[:, 0]]
        margin = best - similarity[np.arange(count), order[:, 1]]
        if len(set(order[:, 0].tolist())) != count or best.min() < MIN_COSINE or margin.min() < MIN_MARGIN:
            raise SystemExit(f'Layer {layer}: ambiguous match (min cosine {best.min():.6f}, '
                             f'min margin {margin.min():.6f}, {len(set(order[:, 0].tolist()))} distinct)')
        layers[str(layer)] = {'base_expert': order[:, 0].tolist(), 'min_cosine': float(best.min()),
                              'min_margin': float(margin.min())}
        print(f'Layer {layer}: min cosine {best.min():.6f}, min margin {margin.min():.4f}', flush=True)
    save(args.map, {'source': str(args.source.resolve()), 'base': str(args.base.resolve()),
                    'experts': count, 'base_experts': base_count, 'layers': layers})


def conversion(name, kind, dims):
    """The target type of a tensor, or None to copy it unchanged."""
    match = EXPERT_RE.fullmatch(name)
    target = None
    if match and int(match[1]) < TRUNK_LAYERS:
        target = T_Q2_K if match[2] == 'down' else T_IQ2_XXS
    elif match:
        target = None if kind in EXPERT_TYPES else T_Q8_0
    elif name == TABLE:
        if kind != T_BF16:
            raise SystemExit(f'{TABLE} must be the original BF16 table')
    elif name.endswith('.ple_conv1d.weight') and kind not in (T_F32, T_F16):
        target = T_F32
    elif kind not in DENSE_TYPES:
        target = T_F16
    return target


def target_dims(name, kind, dims):
    return [768, *dims[1:]] if conversion(name, kind, dims) == T_Q2_K else dims


def convert_tensor(source, name, target, layer_map, imatrix, quant, threads):
    _, kind, dims, _ = source.tensors[name]
    experts = dims[2] if len(dims) == 3 else 1
    rows = math.prod(dims[1:2]) if len(dims) == 3 else math.prod(dims[1:])
    fallbacks = []

    def convert(expert):
        values = source.rows(name, expert * rows, rows)
        result = None
        if target == T_F32:
            result = values.astype('<f4').tobytes()
        elif target == T_F16:
            half = values.astype('<f2')
            if not np.all(np.isfinite(half)):
                raise SystemExit(f'{name}: values overflow F16')
            result = half.tobytes()
        elif target == T_Q8_0:
            result = quant.encode(values, 'Q8_0')
        else:
            sums, counts = imatrix.entries[name]
            base = layer_map[expert]
            if counts[base] > 0:
                weights = sums[base] / counts[base]
            else:
                weights = np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float32)
                fallbacks.append(expert)
            if target == T_Q2_K:
                values = np.pad(values, ((0, 0), (0, 768 - values.shape[1])))
                weights = np.pad(weights, (0, 768 - weights.shape[0]))
            result = quant.encode(values, 'Q2_K' if target == T_Q2_K else 'IQ2_XXS',
                                  np.ascontiguousarray(weights, dtype='<f4'))
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        raw = b''.join(pool.map(convert, range(experts)))
    return raw, sorted(fallbacks)


def quantize(args):
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = Source(args.source, args.table)
    mapping = json.loads(args.map.read_text())
    if mapping['experts'] != source.kv['qwen4exp.expert_count']:
        raise SystemExit('Expert map does not belong to this source')
    imatrix = QwenGGUFImatrix(args.imatrix, Q2_IMATRIX_SHA256)
    quant = GGMLQuantizer(args.library)
    quant.lib.ds4q_quantize_init(T_IQ2_XXS)  # shared lookup tables, before the worker threads
    manifest = args.work_dir / 'tensors.json'
    identity = {'source': str(args.source.resolve()), 'imatrix': imatrix.provenance(),
                'library_sha256': digest(args.library), 'map_sha256': digest(args.map),
                'padding': {'logical_input': 640, 'physical_input': 768}}
    report = json.loads(manifest.read_text()) if manifest.exists() else {**identity, 'tensors': {}}
    for key, value in identity.items():
        if report[key] != value:
            raise SystemExit(f'Resume provenance mismatch: {key}')
    for name, (_, kind, dims, _) in source.tensors.items():
        target = conversion(name, kind, dims)
        if target is None:
            continue
        path = args.work_dir / name
        done = report['tensors'].get(name)
        if done and path.is_file() and digest(path) == done['sha256']:
            print(f'{name}: verified existing', flush=True)
            continue
        start = time.monotonic()
        match = EXPERT_RE.fullmatch(name)
        layer_map = mapping['layers'][match[1]]['base_expert'] if target in (T_Q2_K, T_IQ2_XXS) else None
        raw, fallbacks = convert_tensor(source, name, target, layer_map, imatrix, quant, args.threads)
        if len(raw) != tensor_bytes(target, target_dims(name, kind, dims)):
            raise SystemExit(f'Invalid output size for {name}')
        tmp = path.with_suffix('.incomplete')
        tmp.write_bytes(raw)
        tmp.replace(path)
        report['tensors'][name] = {'type': target, 'dims': target_dims(name, kind, dims), 'bytes': len(raw),
                                   'sha256': hashlib.sha256(raw).hexdigest(), 'source_type': kind,
                                   'source_sha256': hashlib.sha256(source.raw(name)).hexdigest(),
                                   'weight_energy_fallback': fallbacks, 'seconds': time.monotonic() - start}
        save(manifest, report)
        print(f'{name}: {GGMLQuantizationType(kind).name} -> {GGMLQuantizationType(target).name}, '
              f'{len(raw)} bytes, {time.monotonic() - start:.1f}s', flush=True)
    imatrix.close()


def assemble(args):
    if args.out.exists() or Path(str(args.out) + '.incomplete').exists():
        raise SystemExit('Output already exists; choose a new path')
    source = Source(args.source, args.table)
    report = json.loads((args.work_dir / 'tensors.json').read_text())
    if report['source'] != str(args.source.resolve()):
        raise SystemExit('Converted tensors belong to another source')
    names = [n for n in source.tensors if n != TABLE] + [TABLE]
    missing = [n for n in names if conversion(n, *source.tensors[n][1:3]) is not None and n not in report['tensors']]
    if missing:
        raise SystemExit(f'{len(missing)} tensors are not converted yet, first {missing[0]}')
    metadata = {k: (source.kv_types[k], v) for k, v in source.kv.items() if not k.startswith('split.')}
    metadata['general.alignment'] = (4, 32)
    metadata['general.name'] = (8, f'{source.kv.get("general.name", "qwen4exp")}, IQ2_XXS imatrix trunk, '
                                   'padded Q2_K down, MTP')
    metadata['general.file_type'] = (4, 19)  # LLAMA_FTYPE_MOSTLY_IQ2_XXS
    metadata['ds4.iq2.imatrix_sha256'] = (8, report['imatrix']['sha256'])
    metadata['ds4.qwen4.down.logical_input'] = (4, 640)
    metadata['ds4.qwen4.down.physical_input'] = (4, 768)
    metadata['ds4.qwen4.expert_map_sha256'] = (8, report['map_sha256'])
    plan = []
    for name in names:
        _, kind, dims, _ = source.tensors[name]
        done = report['tensors'].get(name)
        kind, dims = (done['type'], done['dims']) if done else (kind, dims)
        plan.append([name, kind, dims, 0, tensor_bytes(kind, dims)])
    prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(metadata))
    prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
    header = len(prefix) + sum(len(w_str(n)) + 4 + 8 * len(d) + 12 for n, _, d, _, _ in plan)
    data_start = (header + 31) // 32 * 32
    offset = 0
    for entry in plan:
        if entry[0] == TABLE:
            offset = (data_start + offset + TABLE_ALIGN - 1) // TABLE_ALIGN * TABLE_ALIGN - data_start
        entry[3] = offset
        offset = (offset + entry[4] + 31) // 32 * 32
    for name, kind, dims, pos, _ in plan:
        prefix += w_str(name) + struct.pack('<I', len(dims))
        prefix += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, pos)
    if len(prefix) != header:
        raise SystemExit('Internal error: header size changed')
    tmp = Path(str(args.out) + '.incomplete')
    records = {}
    with tmp.open('wb') as out:
        out.write(prefix + bytes(data_start - len(prefix)))
        for name, kind, dims, pos, size in plan:
            out.seek(data_start + pos)
            converted = name in report['tensors']
            payload = (args.work_dir / name).open('rb') if converted else None
            h = hashlib.sha256()
            for start in range(0, size, 256 << 20):
                length = min(256 << 20, size - start)
                chunk = payload.read(length) if converted else bytes(source.raw(name, start, length))
                if len(chunk) != length:
                    raise SystemExit(f'Short read: {name}')
                out.write(chunk)
                h.update(chunk)
            if payload:
                payload.close()
            if converted and h.hexdigest() != report['tensors'][name]['sha256']:
                raise SystemExit(f'Converted payload changed since quantization: {name}')
            records[name] = {'type': kind, 'bytes': size, 'sha256': h.hexdigest(), 'converted': converted}
        out.truncate(data_start + offset)
    check = Reader(str(tmp))
    for name, record in records.items():
        check.f.seek(check.data_start + check.tensors[name][2])
        h = hashlib.sha256()
        left = record['bytes']
        while left:
            chunk = check.f.read(min(left, 8 << 20))
            if not chunk:
                raise SystemExit(f'Short verification read: {name}')
            h.update(chunk)
            left -= len(chunk)
        if h.hexdigest() != record['sha256']:
            raise SystemExit(f'Written tensor mismatch: {name}')
    if (check.data_start + check.tensors[TABLE][2]) % TABLE_ALIGN:
        raise SystemExit('n-gram table is not page aligned')
    check.f.close()
    tmp.replace(args.out)
    save(Path(str(args.out) + '.json'), {'source': report['source'], 'conversion': {
        k: v for k, v in report.items() if k != 'tensors'}, 'tensors': records,
        'bytes': args.out.stat().st_size})
    print(f'Verified {args.out}: {args.out.stat().st_size} bytes', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    m = sub.add_parser('map', help='match the source experts to the base experts')
    m.add_argument('--base', type=Path, required=True, help='a 512-expert qwen4exp GGUF of the base model')
    m.add_argument('--map', type=Path, required=True)
    q = sub.add_parser('quantize', help='convert the tensors DS4 needs in another format')
    q.add_argument('--map', type=Path, required=True)
    q.add_argument('--imatrix', type=Path, required=True)
    q.add_argument('--library', type=Path, required=True)
    q.add_argument('--threads', type=int, default=8)
    a = sub.add_parser('assemble', help='write the single DS4 GGUF')
    a.add_argument('--out', type=Path, required=True)
    for p in (m, q, a):
        p.add_argument('--source', type=Path, required=True, help='the GGUF, or its first shard')
        p.add_argument('--table', type=Path,
                       help='take the n-gram table from this GGUF; only once it is shown to be the same bytes')
    for p in (q, a):
        p.add_argument('--work-dir', type=Path, required=True)
    args = ap.parse_args()
    {'map': expert_map, 'quantize': quantize, 'assemble': assemble}[args.stage](args)


if __name__ == '__main__':
    main()
