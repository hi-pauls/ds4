/* Kolibri-1 kernels for CUDA, ported from metal/kolibri1.metal: per-head QK
 * RMSNorm + NeoX rope into f32 or q8_0 K/V caches (sliding layers address a
 * ring of cache_rows, full layers absolute rows; ds4_gpu_kolibri_set_kv_q8
 * picks the row format), GQA span attention with optional key splits, the
 * sigmoid router and the routed/shared MoE sum.  Included by
 * ds4_cuda.cu after the Qwen kernels, whose MoE mid/down, dense and norm
 * wrappers the Kolibri graph also uses.  DS4_KOLIBRI1_CUDA_KERNELS_ONLY keeps
 * the wrappers out, so tests/test_kolibri1_cuda.cu can check the kernels
 * against a host reference without the rest of the backend. */

#include <cuda_fp16.h>
#include <stdint.h>

namespace kolibri1_cuda {

constexpr unsigned ATTN_NSG = 4;          /* warps per attention block, each owning part of a q-head group */
constexpr unsigned ATTN_MAX_SPLITS = 1024;
constexpr unsigned ATTN_TILE = 32;        /* keys attn_decode_tiles scores at once, one per lane */
constexpr unsigned ATTN_TILE_MAX_SPLITS = 256;
constexpr unsigned ATTN_Q8_KEY_WORDS = 36; /* shared words per staged q8 key: 34 of a D=128 row, padded so rows stay 16-byte aligned and lane-wide 16-byte reads hit distinct banks */
constexpr unsigned ATTN_Q8_ROW_WORDS = 34;     /* words of a D=128 q8 row as attn_decode_tiles_q8_pipe stages it: dense, since its lanes read 8-byte words */
constexpr unsigned ATTN_Q8_WARP_COPIES = ATTN_TILE / ATTN_NSG * ATTN_Q8_ROW_WORDS / 2;  /* 8-byte words a warp moves per tile: 8 rows of 17 */
constexpr unsigned ATTN_Q8_LANE_COPIES = (ATTN_Q8_WARP_COPIES + 31u) / 32u;
constexpr unsigned ATTN_MERGE_THREADS = 512;   /* attn_merge_wide block */
constexpr unsigned ATTN_Q8_PIECE_KEYS = 8192; /* keys attn_batch_products_q8 dequantizes at a time */
constexpr unsigned ATTN_GROUP_MAX = 12;   /* query heads per kv head, three per warp */
constexpr unsigned ROUTER_MAX_USED = 8;
constexpr unsigned ROUTER_NSG = 4;
constexpr unsigned ROUTER_LANE_MAX = 4;   /* 32 lanes * 4 warps * 4 = 512 experts */

struct attn_prep_args {
    unsigned n_tokens, n_head, n_head_kv, head_dim, pos0, cache_rows, use_rope, q8;
    float eps;
    float rope_freq[64];
};

struct attn_args {
    unsigned n_tokens, n_head, n_head_kv, head_dim, pos0, window, cache_rows, n_splits, keys_per_split;
    float scale;
};

__device__ __forceinline__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d));
    return x;
}

__device__ __forceinline__ float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

/* Pair p touches {p, p + head/2} only, so lanes rotate disjoint pairs. */
__device__ __forceinline__ void rope_stage(float *staged, unsigned D, unsigned pos,
                                           const float *freq, unsigned lane) {
    const unsigned nh = D / 2u;
    for (unsigned p = lane; p < nh; p += 32u) {
        const float theta = (float)pos * freq[p];
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = staged[p], x1 = staged[p + nh];
        staged[p] = x0 * c - x1 * s;
        staged[p + nh] = x0 * s + x1 * c;
    }
}

/* q8_0 K/V rows, the layout of the Qwen caches: a (position, kv head) row is
 * D int8 values then D/32 half scales, one per 32-value block, quantized
 * against the scale as stored so dequantization inverts it.  The caches are
 * then Hkv * kv_row_bytes_q8(D) bytes per position (row), still indexed as
 * the f32 ones; D % 64 == 0 keeps every row on a 4-byte boundary for the word
 * loads below.  A warp step i = lane + 32 * b covers block b, so its max is a
 * warp max. */
__host__ __device__ __forceinline__ unsigned kv_row_bytes_q8(unsigned D) { return D + D / 16u; }

__device__ __forceinline__ void kv_store_q8(uint8_t *row, unsigned D, unsigned i, float v) {
    const float amax = warp_max(fabsf(v));
    const __half dh = __float2half_rn(amax / 127.0f);
    const float d = __half2float(dh), id = d > 0.0f ? 1.0f / d : 0.0f;
    row[i] = (uint8_t)(int8_t)fminf(fmaxf(roundf(v * id), -127.0f), 127.0f);
    if (!(i & 31u)) ((__half *)(row + D))[i / 32u] = dh;
}

/* values lane * 4 .. + 3 of a q8 row, the four a lane owns in attn_decode:
 * the word and its block's scale bits, then the floats */
__device__ __forceinline__ void kv_fetch4_q8(uint32_t &w, unsigned short &s, const uint8_t *row, unsigned D, unsigned lane) {
    w = *(const uint32_t *)(row + lane * 4u);
    s = ((const unsigned short *)(row + D))[lane / 8u];
}

__device__ __forceinline__ void kv_unpack4_q8(float *out, uint32_t w, unsigned short s) {
    const float d = __half2float(__ushort_as_half(s));
    out[0] = d * (float)(int8_t)(w & 0xffu);
    out[1] = d * (float)(int8_t)((w >> 8) & 0xffu);
    out[2] = d * (float)(int8_t)((w >> 16) & 0xffu);
    out[3] = d * (float)(int8_t)(w >> 24);
}

/* The first n values (a multiple of 4) of a q8 cache as floats, the layout of
 * the f32 caches; a thread converts one word. */
__global__ void kv_dequant_q8(float *out, const uint8_t *cache, unsigned n, unsigned D) {
    const unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i >= n) return;
    const unsigned r = i / D, d = i - r * D;
    const uint8_t *row = cache + (uint64_t)r * kv_row_bytes_q8(D);
    const float s = __half2float(((const __half *)(row + D))[d / 32u]);
    const uint32_t w = *(const uint32_t *)(row + d);
    *(float4 *)(out + i) = make_float4(s * (float)(int8_t)(w & 0xffu), s * (float)(int8_t)((w >> 8) & 0xffu),
                                       s * (float)(int8_t)((w >> 16) & 0xffu), s * (float)(int8_t)(w >> 24));
}

/* One warp per (head slot, token): slots 0..H-1 norm (and rope) a query
 * head, H..H+Hkv-1 a key head plus its value row, the last copies V (the
 * q8 rows are complete after the key slots, so it has nothing to do). */
__global__ void attn_prep(attn_prep_args a, const float *qproj, const float *kproj, const float *vproj,
                          const float *gq, const float *gk, float *qout, float *kc, float *vc) {
    const unsigned slot = blockIdx.x, tok = blockIdx.y, lane = threadIdx.x;
    const unsigned H = a.n_head, Hkv = a.n_head_kv, D = a.head_dim;
    if (tok >= a.n_tokens || slot > H + Hkv || (a.q8 && slot == H + Hkv)) return;
    const unsigned pos = a.pos0 + tok;
    const uint64_t row = a.cache_rows ? pos % a.cache_rows : pos;
    __shared__ float staged[128];
    if (slot < H + Hkv) {
        const bool isq = slot < H;
        const unsigned h = isq ? slot : slot - H;
        const float *src = isq ? qproj + ((uint64_t)tok * H + h) * D : kproj + ((uint64_t)tok * Hkv + h) * D;
        const float *gamma = isq ? gq : gk;
        float ss = 0.0f;
        for (unsigned i = lane * 4u; i < D; i += 128u)
            ss += src[i] * src[i] + src[i + 1] * src[i + 1] + src[i + 2] * src[i + 2] + src[i + 3] * src[i + 3];
        const float r = 1.0f / sqrtf(warp_sum(ss) / (float)D + a.eps);
        for (unsigned i = lane * 4u; i < D; i += 128u)
            for (unsigned j = 0; j < 4u; j++) staged[i + j] = src[i + j] * r * gamma[i + j];
        __syncwarp();
        if (a.use_rope) {
            rope_stage(staged, D, pos, a.rope_freq, lane);
            __syncwarp();
        }
        if (isq) {
            float *dst = qout + ((uint64_t)tok * H + h) * D;
            for (unsigned i = lane; i < D; i += 32u) dst[i] = staged[i];
        } else if (a.q8) {
            uint8_t *dk = (uint8_t *)kc + (row * Hkv + h) * kv_row_bytes_q8(D);
            uint8_t *dv = (uint8_t *)vc + (row * Hkv + h) * kv_row_bytes_q8(D);
            const float *vs = vproj + ((uint64_t)tok * Hkv + h) * D;
            for (unsigned i = lane; i < D; i += 32u) {
                kv_store_q8(dk, D, i, staged[i]);
                kv_store_q8(dv, D, i, vs[i]);
            }
        } else {
            float *dk = kc + (row * Hkv + h) * D;
            float *dv = vc + (row * Hkv + h) * D;
            const float *vs = vproj + ((uint64_t)tok * Hkv + h) * D;
            for (unsigned i = lane; i < D; i += 32u) {
                dk[i] = staged[i];
                dv[i] = vs[i];
            }
        }
    } else {
        const float *src = vproj + (uint64_t)tok * Hkv * D;
        float *dst = vc + row * Hkv * D;
        for (unsigned i = lane; i < Hkv * D; i += 32u) dst[i] = src[i];
    }
}

/* Online-softmax GQA over the span a token attends: each warp owns up to
 * three query heads of the kv head's group, a lane four head dims.  One
 * split writes out; otherwise each head leaves (m, l, acc) partials.  Q8
 * reads q8_0 rows (kv_store_q8) instead of f32. */
template<bool Q8>
__global__ void attn_decode(attn_args a, const float *q, const float *kc, const float *vc,
                            float *out, float *part) {
    const unsigned split = blockIdx.x, kvh = blockIdx.y, tok = blockIdx.z;
    const unsigned warp = threadIdx.x / 32u, lane = threadIdx.x & 31u;
    if (split >= a.n_splits || kvh >= a.n_head_kv || tok >= a.n_tokens) return;
    const unsigned H = a.n_head, Hkv = a.n_head_kv, D = a.head_dim, group = H / Hkv;
    const unsigned pos = a.pos0 + tok;
    const unsigned n = a.window ? min(pos + 1u, a.window) : pos + 1u;
    const unsigned start = a.window ? pos + 1u - n : 0u;
    const unsigned hps = (group + ATTN_NSG - 1u) / ATTN_NSG, g0 = warp * hps;
    if (g0 >= group) return;
    const unsigned ng = min(hps, group - g0);
    const unsigned k0 = split * a.keys_per_split, k1 = min(n, k0 + a.keys_per_split);
    float qv[3][4], m[3], l[3], acc[3][4];
    for (unsigned g = 0; g < 3u; g++) {
        if (g < ng) {
            const float *qh = q + ((uint64_t)tok * H + kvh * group + g0 + g) * D;
            for (unsigned j = 0; j < 4u; j++) {
                const unsigned d = lane * 4u + j;
                qv[g][j] = d < D ? qh[d] * a.scale : 0.0f;
            }
        }
        m[g] = -3.0e38f;
        l[g] = 0.0f;
        for (unsigned j = 0; j < 4u; j++) acc[g][j] = 0.0f;
    }
    /* q8 rows: the next row's words are fetched before the current one goes
     * through the reductions, so the loads overlap them */
    uint32_t kwn = 0u, vwn = 0u;
    unsigned short ksn = 0u, vsn = 0u;
    if (Q8 && k0 < k1 && lane * 4u < D) {
        const unsigned p = start + k0;
        const uint64_t row = a.cache_rows ? p % a.cache_rows : p;
        kv_fetch4_q8(kwn, ksn, (const uint8_t *)kc + (row * Hkv + kvh) * kv_row_bytes_q8(D), D, lane);
        kv_fetch4_q8(vwn, vsn, (const uint8_t *)vc + (row * Hkv + kvh) * kv_row_bytes_q8(D), D, lane);
    }
    for (unsigned idx = k0; idx < k1; idx++) {
        const unsigned p = start + idx;
        const uint64_t row = a.cache_rows ? p % a.cache_rows : p;
        float kv[4], vv[4];
        if (Q8) {
            const uint32_t kwc = kwn, vwc = vwn;
            const unsigned short ksc = ksn, vsc = vsn;
            if (idx + 1u < k1 && lane * 4u < D) {
                const unsigned pn = start + idx + 1u;
                const uint64_t rown = a.cache_rows ? pn % a.cache_rows : pn;
                kv_fetch4_q8(kwn, ksn, (const uint8_t *)kc + (rown * Hkv + kvh) * kv_row_bytes_q8(D), D, lane);
                kv_fetch4_q8(vwn, vsn, (const uint8_t *)vc + (rown * Hkv + kvh) * kv_row_bytes_q8(D), D, lane);
            }
            kv_unpack4_q8(kv, kwc, ksc);
            kv_unpack4_q8(vv, vwc, vsc);
        } else {
            const float *kr = kc + (row * Hkv + kvh) * D, *vr = vc + (row * Hkv + kvh) * D;
            for (unsigned j = 0; j < 4u; j++) {
                const unsigned d = lane * 4u + j;
                kv[j] = d < D ? kr[d] : 0.0f;
                vv[j] = d < D ? vr[d] : 0.0f;
            }
        }
        for (unsigned g = 0; g < ng; g++) {
            float s = 0.0f;
            for (unsigned j = 0; j < 4u; j++) s += qv[g][j] * kv[j];
            s = warp_sum(s);
            const float m_new = fmaxf(m[g], s), corr = expf(m[g] - m_new), w = expf(s - m_new);
            l[g] = l[g] * corr + w;
            for (unsigned j = 0; j < 4u; j++) acc[g][j] = acc[g][j] * corr + w * vv[j];
            m[g] = m_new;
        }
    }
    for (unsigned g = 0; g < ng; g++) {
        if (a.n_splits == 1) {
            float *dst = out + ((uint64_t)tok * H + kvh * group + g0 + g) * D;
            const float inv = l[g] > 0.0f ? 1.0f / l[g] : 0.0f;
            for (unsigned j = 0; j < 4u; j++) {
                const unsigned d = lane * 4u + j;
                if (d < D) dst[d] = acc[g][j] * inv;
            }
        } else {
            float *dst = part + ((((uint64_t)tok * Hkv + kvh) * a.n_splits + split) * group + g0 + g) * (2u + D);
            if (!lane) { dst[0] = m[g]; dst[1] = l[g]; }
            for (unsigned j = 0; j < 4u; j++) {
                const unsigned d = lane * 4u + j;
                if (d < D) dst[2u + d] = acc[g][j];
            }
        }
    }
}

/* attn_decode for full layers deep into a context, where its warp reduction
 * and two expf per (key, head) cost far more than reading the caches.  A
 * block takes its split a tile of keys at a time: the K tile goes through
 * shared memory transposed, so a lane holds one key and dots it with three
 * query heads without a reduction; the softmax state advances once per tile;
 * then a thread owns one head dim of V for all heads of the group.  Each K and
 * V row is read once per kv head, coalesced.  ks rotates the keys of a dim by
 * dim / 4, which keeps both the float4 stores and the per-key reads off
 * shared bank conflicts.  Leaves the partials attn_merge expects. */
__global__ void attn_decode_tiles(attn_args a, const float *q, const float *kc, const float *vc,
                                  float *out, float *part) {
    const unsigned split = blockIdx.x, kvh = blockIdx.y, tok = blockIdx.z;
    const unsigned tid = threadIdx.x, warp = tid / 32u, lane = tid & 31u;
    if (split >= a.n_splits || kvh >= a.n_head_kv || tok >= a.n_tokens) return;
    const unsigned H = a.n_head, Hkv = a.n_head_kv, D = a.head_dim, group = H / Hkv;
    const unsigned hps = ATTN_GROUP_MAX / ATTN_NSG, g0 = warp * hps;
    const unsigned ng = g0 < group ? min(hps, group - g0) : 0u;
    const unsigned n = a.pos0 + tok + 1u;
    const unsigned k0 = split * a.keys_per_split, k1 = min(n, k0 + a.keys_per_split);
    __shared__ float ks[128 * ATTN_TILE], qs[ATTN_GROUP_MAX * 128], ws[ATTN_GROUP_MAX * ATTN_TILE];
    __shared__ float ms[ATTN_GROUP_MAX], ls[ATTN_GROUP_MAX], cs[ATTN_GROUP_MAX];
    float acc[ATTN_GROUP_MAX];
    for (unsigned g = 0; g < ATTN_GROUP_MAX; g++) {
        if (tid < D)
            qs[g * 128u + tid] = g < group ? q[((uint64_t)tok * H + kvh * group + g) * D + tid] * a.scale : 0.0f;
        acc[g] = 0.0f;
    }
    for (unsigned i = tid; i < ATTN_GROUP_MAX * ATTN_TILE; i += blockDim.x) ws[i] = 0.0f;
    if (tid < ATTN_GROUP_MAX) { ms[tid] = -3.0e38f; ls[tid] = 0.0f; cs[tid] = 0.0f; }
    __syncthreads();
    for (unsigned kt = k0; kt < k1; kt += ATTN_TILE) {
        const unsigned nt = min(ATTN_TILE, k1 - kt);
        for (unsigned j = warp; j < nt; j += ATTN_NSG) {
            if (lane * 4u >= D) continue;
            const float4 k4 = *(const float4 *)(kc + ((uint64_t)(kt + j) * Hkv + kvh) * D + lane * 4u);
            float *dst = ks + lane * 4u * ATTN_TILE + ((j + lane) & 31u);
            dst[0] = k4.x;
            dst[ATTN_TILE] = k4.y;
            dst[2u * ATTN_TILE] = k4.z;
            dst[3u * ATTN_TILE] = k4.w;
        }
        __syncthreads();
        float s[ATTN_GROUP_MAX / ATTN_NSG] = { 0.0f, 0.0f, 0.0f };
        if (lane < nt)
            for (unsigned i = 0; i < D; i++) {
                const float kv = ks[i * ATTN_TILE + ((lane + (i >> 2)) & 31u)];
                for (unsigned g = 0; g < hps; g++) s[g] += qs[(g0 + g) * 128u + i] * kv;
            }
        for (unsigned g = 0; g < ng; g++) {
            const unsigned gh = g0 + g;
            const float sc = lane < nt ? s[g] : -3.0e38f, m_old = ms[gh];
            const float m_new = fmaxf(m_old, warp_max(sc));
            const float w = lane < nt ? expf(sc - m_new) : 0.0f;
            const float l_tile = warp_sum(w);
            ws[gh * ATTN_TILE + lane] = w;
            if (!lane) {
                const float corr = expf(m_old - m_new);
                cs[gh] = corr;
                ls[gh] = ls[gh] * corr + l_tile;
                ms[gh] = m_new;
            }
        }
        __syncthreads();
        if (tid < D) {
            for (unsigned g = 0; g < ATTN_GROUP_MAX; g++) acc[g] *= cs[g];
            for (unsigned j = 0; j < nt; j++) {
                const float vv = vc[((uint64_t)(kt + j) * Hkv + kvh) * D + tid];
                for (unsigned g = 0; g < ATTN_GROUP_MAX; g++) acc[g] += ws[g * ATTN_TILE + j] * vv;
            }
        }
        __syncthreads();
    }
    for (unsigned g = 0; g < group; g++) {
        if (a.n_splits == 1) {
            if (tid < D) out[((uint64_t)tok * H + kvh * group + g) * D + tid] = acc[g] * (ls[g] > 0.0f ? 1.0f / ls[g] : 0.0f);
        } else {
            float *dst = part + ((((uint64_t)tok * Hkv + kvh) * a.n_splits + split) * group + g) * (2u + D);
            if (tid == g) { dst[0] = ms[g]; dst[1] = ls[g]; }
            if (tid < D) dst[2u + tid] = acc[g];
        }
    }
}

/* attn_decode_tiles for q8 caches, built to spend fewer instructions than the
 * f32 kernel rather than the same on a quarter of the bytes.  The query is
 * quantized once per 32-value block into an int8 hi part and a lo part that
 * carries the residual at 1/127 of the hi step (about 15 bits of the query
 * survive), so a key block is one dp4a chain per part and one multiply by
 * the two block scales.  The K tile is staged as raw rows at a stride of
 * ATTN_Q8_KEY_WORDS and read back as 16-byte words, conflict-free across the
 * lanes.  In the V phase a warp takes every fourth key and a lane four dims
 * as one word, fetched before the K phase so the loads overlap it, a key's
 * twelve weights being three 16-byte broadcast reads, and the four warps'
 * partial sums meet in shared memory at the end.  Same grid, split layout
 * and partials as attn_decode_tiles. */
__global__ void attn_decode_tiles_q8(attn_args a, const float *q, const float *kc, const float *vc,
                                     float *out, float *part) {
    const unsigned split = blockIdx.x, kvh = blockIdx.y, tok = blockIdx.z;
    const unsigned tid = threadIdx.x, warp = tid / 32u, lane = tid & 31u;
    if (split >= a.n_splits || kvh >= a.n_head_kv || tok >= a.n_tokens) return;
    const unsigned H = a.n_head, Hkv = a.n_head_kv, D = a.head_dim, group = H / Hkv;
    const unsigned hps = ATTN_GROUP_MAX / ATTN_NSG, g0 = warp * hps;
    const unsigned ng = g0 < group ? min(hps, group - g0) : 0u;
    const unsigned n = a.pos0 + tok + 1u;
    const unsigned k0 = split * a.keys_per_split, k1 = min(n, k0 + a.keys_per_split);
    const unsigned rb = kv_row_bytes_q8(D), kw = D / 4u, nb = D / 32u;
    __shared__ __align__(16) uint32_t kq[ATTN_TILE * ATTN_Q8_KEY_WORDS];
    __shared__ __align__(16) int8_t qq[ATTN_GROUP_MAX * 2u * 128u];
    __shared__ __align__(16) float ws[ATTN_TILE * ATTN_GROUP_MAX];
    __shared__ float qd[ATTN_GROUP_MAX * 4u], ms[ATTN_GROUP_MAX], ls[ATTN_GROUP_MAX], cs[ATTN_GROUP_MAX];
    float acc[ATTN_GROUP_MAX][4];
    for (unsigned g = 0; g < ATTN_GROUP_MAX; g++)
        for (unsigned c = 0; c < 4u; c++) acc[g][c] = 0.0f;
    for (unsigned p = warp; p < ATTN_GROUP_MAX * 4u; p += ATTN_NSG) {
        const unsigned g = p / 4u, b = p % 4u;
        const float v = g < group && b < nb ? q[((uint64_t)tok * H + kvh * group + g) * D + b * 32u + lane] * a.scale : 0.0f;
        const float amax = warp_max(fabsf(v)), dq = amax / 127.0f, id = dq > 0.0f ? 1.0f / dq : 0.0f;
        const float hi = rintf(v * id);
        const float lo = fminf(fmaxf(rintf((v - hi * dq) * id * 127.0f), -127.0f), 127.0f);
        qq[(g * 2u) * 128u + b * 32u + lane] = (int8_t)hi;
        qq[(g * 2u + 1u) * 128u + b * 32u + lane] = (int8_t)lo;
        if (!lane) qd[p] = dq;
    }
    for (unsigned i = tid; i < ATTN_TILE * ATTN_GROUP_MAX; i += blockDim.x) ws[i] = 0.0f;
    if (tid < ATTN_GROUP_MAX) { ms[tid] = -3.0e38f; ls[tid] = 0.0f; cs[tid] = 0.0f; }
    __syncthreads();
    for (unsigned kt = k0; kt < k1; kt += ATTN_TILE) {
        const unsigned nt = min(ATTN_TILE, k1 - kt);
        for (unsigned j = warp; j < nt; j += ATTN_NSG) {
            const uint32_t *kr = (const uint32_t *)((const uint8_t *)kc + ((uint64_t)(kt + j) * Hkv + kvh) * rb);
            if (lane < kw) kq[j * ATTN_Q8_KEY_WORDS + lane] = kr[lane];
            if (lane < rb / 4u - kw) kq[j * ATTN_Q8_KEY_WORDS + kw + lane] = kr[kw + lane];
        }
        /* this warp's V words and scales of the tile, in flight through the K phase */
        uint32_t v4[ATTN_TILE / ATTN_NSG];
        unsigned short vsb[ATTN_TILE / ATTN_NSG];
        for (unsigned k = 0; k < ATTN_TILE / ATTN_NSG; k++) {
            const unsigned j = warp + k * ATTN_NSG;
            v4[k] = 0u;
            vsb[k] = 0u;
            if (j < nt && lane < kw) {
                const uint8_t *vr = (const uint8_t *)vc + ((uint64_t)(kt + j) * Hkv + kvh) * rb;
                v4[k] = ((const uint32_t *)vr)[lane];
                vsb[k] = ((const unsigned short *)(vr + D))[lane / 8u];
            }
        }
        __syncthreads();
        float s[ATTN_GROUP_MAX / ATTN_NSG] = { 0.0f, 0.0f, 0.0f };
        if (lane < nt) {
            const uint32_t *kr = kq + lane * ATTN_Q8_KEY_WORDS;
            const __half *kd = (const __half *)(kr + kw);
            for (unsigned b = 0; b < nb; b++) {
                const int4 x0 = *(const int4 *)(kr + b * 8u), x1 = *(const int4 *)(kr + b * 8u + 4u);
                const float dk = __half2float(kd[b]);
                for (unsigned g = 0; g < hps; g++) {
                    const int4 *qh = (const int4 *)(qq + ((g0 + g) * 2u) * 128u + b * 32u);
                    const int4 *ql = (const int4 *)(qq + ((g0 + g) * 2u + 1u) * 128u + b * 32u);
                    const int4 h0 = qh[0], h1 = qh[1], l0 = ql[0], l1 = ql[1];
                    int ih = __dp4a(x0.x, h0.x, 0), il = __dp4a(x0.x, l0.x, 0);
                    ih = __dp4a(x0.y, h0.y, ih); il = __dp4a(x0.y, l0.y, il);
                    ih = __dp4a(x0.z, h0.z, ih); il = __dp4a(x0.z, l0.z, il);
                    ih = __dp4a(x0.w, h0.w, ih); il = __dp4a(x0.w, l0.w, il);
                    ih = __dp4a(x1.x, h1.x, ih); il = __dp4a(x1.x, l1.x, il);
                    ih = __dp4a(x1.y, h1.y, ih); il = __dp4a(x1.y, l1.y, il);
                    ih = __dp4a(x1.z, h1.z, ih); il = __dp4a(x1.z, l1.z, il);
                    ih = __dp4a(x1.w, h1.w, ih); il = __dp4a(x1.w, l1.w, il);
                    s[g] += dk * qd[(g0 + g) * 4u + b] * ((float)ih + (float)il * (1.0f / 127.0f));
                }
            }
        }
        for (unsigned g = 0; g < ng; g++) {
            const unsigned gh = g0 + g;
            const float sc = lane < nt ? s[g] : -3.0e38f, m_old = ms[gh];
            const float m_new = fmaxf(m_old, warp_max(sc));
            const float w = lane < nt ? expf(sc - m_new) : 0.0f;
            const float l_tile = warp_sum(w);
            ws[lane * ATTN_GROUP_MAX + gh] = w;
            if (!lane) {
                const float corr = expf(m_old - m_new);
                cs[gh] = corr;
                ls[gh] = ls[gh] * corr + l_tile;
                ms[gh] = m_new;
            }
        }
        __syncthreads();
        for (unsigned g = 0; g < ATTN_GROUP_MAX; g++)
            for (unsigned c = 0; c < 4u; c++) acc[g][c] *= cs[g];
        for (unsigned k = 0; k < ATTN_TILE / ATTN_NSG; k++) {
            const unsigned j = warp + k * ATTN_NSG;
            if (j < nt && lane < kw) {
                const float dv = __half2float(__ushort_as_half(vsb[k]));
                const float v0 = dv * (float)(int8_t)(v4[k] & 0xffu), v1 = dv * (float)(int8_t)((v4[k] >> 8) & 0xffu);
                const float v2 = dv * (float)(int8_t)((v4[k] >> 16) & 0xffu), v3 = dv * (float)(int8_t)(v4[k] >> 24);
                const float4 *wj = (const float4 *)(ws + j * ATTN_GROUP_MAX);
                for (unsigned g4 = 0; g4 < ATTN_GROUP_MAX / 4u; g4++) {
                    const float4 w = wj[g4];
                    acc[g4 * 4u][0] += w.x * v0; acc[g4 * 4u][1] += w.x * v1; acc[g4 * 4u][2] += w.x * v2; acc[g4 * 4u][3] += w.x * v3;
                    acc[g4 * 4u + 1u][0] += w.y * v0; acc[g4 * 4u + 1u][1] += w.y * v1; acc[g4 * 4u + 1u][2] += w.y * v2; acc[g4 * 4u + 1u][3] += w.y * v3;
                    acc[g4 * 4u + 2u][0] += w.z * v0; acc[g4 * 4u + 2u][1] += w.z * v1; acc[g4 * 4u + 2u][2] += w.z * v2; acc[g4 * 4u + 2u][3] += w.z * v3;
                    acc[g4 * 4u + 3u][0] += w.w * v0; acc[g4 * 4u + 3u][1] += w.w * v1; acc[g4 * 4u + 3u][2] += w.w * v2; acc[g4 * 4u + 3u][3] += w.w * v3;
                }
            }
        }
        __syncthreads();
    }
    float *red = (float *)kq;   /* 4 warps * 2 heads * 128 dims fit the staged tile */
    for (unsigned r = 0; r < ATTN_GROUP_MAX / 2u; r++) {
        if (lane < kw)
            for (unsigned i = 0; i < 2u; i++)
                *(float4 *)(red + (warp * 2u + i) * 128u + lane * 4u) =
                    make_float4(acc[r * 2u + i][0], acc[r * 2u + i][1], acc[r * 2u + i][2], acc[r * 2u + i][3]);
        __syncthreads();
        for (unsigned i = 0; i < 2u; i++) {
            const unsigned g = r * 2u + i;
            if (g >= group) continue;
            float o = 0.0f;
            if (tid < D)
                for (unsigned w = 0; w < ATTN_NSG; w++) o += red[(w * 2u + i) * 128u + tid];
            if (a.n_splits == 1) {
                if (tid < D) out[((uint64_t)tok * H + kvh * group + g) * D + tid] = o * (ls[g] > 0.0f ? 1.0f / ls[g] : 0.0f);
            } else {
                float *dst = part + ((((uint64_t)tok * Hkv + kvh) * a.n_splits + split) * group + g) * (2u + D);
                if (tid == g) { dst[0] = ms[g]; dst[1] = ls[g]; }
                if (tid < D) dst[2u + tid] = o;
            }
        }
        __syncthreads();
    }
}

/* attn_decode_tiles_q8 for D = 128 with the next tile's K and V rows fetched
 * into registers while the current one is computed (a warp moves 8 rows as
 * 8-byte words, lane-contiguous in both memories, so the staged rows are
 * dense at 34 words: for 8-byte lane reads that stride is conflict-free),
 * and a thread owning one head dim of V for all heads of the group, as
 * attn_decode_tiles does, which drops the per-warp accumulator reduction and
 * a third of the registers.  Scores are the ones attn_decode_tiles_q8 forms;
 * the V sums run over the keys in order instead of across four warps.  Same
 * grid, split layout and partials. */
__global__ void __launch_bounds__(128) attn_decode_tiles_q8_pipe(attn_args a, const float *q, const float *kc,
                                                                 const float *vc, float *out, float *part) {
    constexpr unsigned D = 128, RW = ATTN_Q8_ROW_WORDS, NB = D / 32u;
    const unsigned split = blockIdx.x, kvh = blockIdx.y, tok = blockIdx.z;
    const unsigned tid = threadIdx.x, warp = tid / 32u, lane = tid & 31u;
    if (split >= a.n_splits || kvh >= a.n_head_kv || tok >= a.n_tokens) return;
    const unsigned H = a.n_head, Hkv = a.n_head_kv, group = H / Hkv;
    const unsigned hps = ATTN_GROUP_MAX / ATTN_NSG, g0 = warp * hps;
    const unsigned ng = g0 < group ? min(hps, group - g0) : 0u;
    const unsigned n = a.pos0 + tok + 1u;
    const unsigned k0 = split * a.keys_per_split, k1 = min(n, k0 + a.keys_per_split);
    const uint64_t rb = kv_row_bytes_q8(D);
    __shared__ __align__(16) uint32_t kq[ATTN_TILE * RW], vq[ATTN_TILE * RW];
    __shared__ __align__(16) int8_t qq[ATTN_GROUP_MAX * 2u * 128u];
    __shared__ __align__(16) float ws[ATTN_TILE * ATTN_GROUP_MAX];
    __shared__ float qd[ATTN_GROUP_MAX * 4u], ms[ATTN_GROUP_MAX], ls[ATTN_GROUP_MAX], cs[ATTN_GROUP_MAX];
    float acc[ATTN_GROUP_MAX];
    #pragma unroll
    for (unsigned g = 0; g < ATTN_GROUP_MAX; g++) acc[g] = 0.0f;
    for (unsigned p = warp; p < ATTN_GROUP_MAX * 4u; p += ATTN_NSG) {
        const unsigned g = p / 4u, b = p % 4u;
        const float v = g < group && b < NB ? q[((uint64_t)tok * H + kvh * group + g) * D + b * 32u + lane] * a.scale : 0.0f;
        const float amax = warp_max(fabsf(v)), dq = amax / 127.0f, id = dq > 0.0f ? 1.0f / dq : 0.0f;
        const float hi = rintf(v * id);
        const float lo = fminf(fmaxf(rintf((v - hi * dq) * id * 127.0f), -127.0f), 127.0f);
        qq[(g * 2u) * 128u + b * 32u + lane] = (int8_t)hi;
        qq[(g * 2u + 1u) * 128u + b * 32u + lane] = (int8_t)lo;
        if (!lane) qd[p] = dq;
    }
    for (unsigned i = tid; i < ATTN_TILE * ATTN_GROUP_MAX; i += blockDim.x) ws[i] = 0.0f;
    if (tid < ATTN_GROUP_MAX) { ms[tid] = -3.0e38f; ls[tid] = 0.0f; cs[tid] = 0.0f; }
    uint2 kp[ATTN_Q8_LANE_COPIES], vp[ATTN_Q8_LANE_COPIES];
    const uint8_t *kc8 = (const uint8_t *)kc, *vc8 = (const uint8_t *)vc;
    auto fetch = [&](unsigned kt, unsigned nt) {
        #pragma unroll
        for (unsigned k = 0; k < ATTN_Q8_LANE_COPIES; k++) {
            const unsigned i = lane + 32u * k, j = warp * (ATTN_TILE / ATTN_NSG) + i / (RW / 2u), c = i % (RW / 2u);
            if (i < ATTN_Q8_WARP_COPIES && j < nt) {
                const uint64_t at = ((uint64_t)(kt + j) * Hkv + kvh) * rb + c * 8u;
                kp[k] = *(const uint2 *)(kc8 + at);
                vp[k] = *(const uint2 *)(vc8 + at);
            }
        }
    };
    fetch(k0, min(ATTN_TILE, k1 - k0));
    const unsigned vw = tid >> 2, vsh = (tid & 3u) * 8u, vb = tid >> 5;
    for (unsigned kt = k0; kt < k1; kt += ATTN_TILE) {
        const unsigned nt = min(ATTN_TILE, k1 - kt);
        #pragma unroll
        for (unsigned k = 0; k < ATTN_Q8_LANE_COPIES; k++) {
            const unsigned i = lane + 32u * k;
            if (i < ATTN_Q8_WARP_COPIES) {
                ((uint2 *)kq)[warp * ATTN_Q8_WARP_COPIES + i] = kp[k];
                ((uint2 *)vq)[warp * ATTN_Q8_WARP_COPIES + i] = vp[k];
            }
        }
        __syncthreads();
        if (kt + ATTN_TILE < k1) fetch(kt + ATTN_TILE, min(ATTN_TILE, k1 - kt - ATTN_TILE));
        float s[ATTN_GROUP_MAX / ATTN_NSG] = { 0.0f, 0.0f, 0.0f };
        if (lane < nt) {
            const int2 *kr = (const int2 *)kq + lane * (RW / 2u);
            const uint2 kdw = *(const uint2 *)(kr + D / 8u);
            #pragma unroll
            for (unsigned b = 0; b < NB; b++) {
                const int2 x0 = kr[b * 4u], x1 = kr[b * 4u + 1u], x2 = kr[b * 4u + 2u], x3 = kr[b * 4u + 3u];
                const float dk = __half2float(__ushort_as_half((unsigned short)((b < 2u ? kdw.x : kdw.y) >> (16u * (b & 1u)))));
                #pragma unroll
                for (unsigned g = 0; g < hps; g++) {
                    const int4 *qh = (const int4 *)(qq + ((g0 + g) * 2u) * 128u + b * 32u);
                    const int4 *ql = (const int4 *)(qq + ((g0 + g) * 2u + 1u) * 128u + b * 32u);
                    const int4 h0 = qh[0], h1 = qh[1], l0 = ql[0], l1 = ql[1];
                    int ih = __dp4a(x0.x, h0.x, 0), il = __dp4a(x0.x, l0.x, 0);
                    ih = __dp4a(x0.y, h0.y, ih); il = __dp4a(x0.y, l0.y, il);
                    ih = __dp4a(x1.x, h0.z, ih); il = __dp4a(x1.x, l0.z, il);
                    ih = __dp4a(x1.y, h0.w, ih); il = __dp4a(x1.y, l0.w, il);
                    ih = __dp4a(x2.x, h1.x, ih); il = __dp4a(x2.x, l1.x, il);
                    ih = __dp4a(x2.y, h1.y, ih); il = __dp4a(x2.y, l1.y, il);
                    ih = __dp4a(x3.x, h1.z, ih); il = __dp4a(x3.x, l1.z, il);
                    ih = __dp4a(x3.y, h1.w, ih); il = __dp4a(x3.y, l1.w, il);
                    s[g] += dk * qd[(g0 + g) * 4u + b] * ((float)ih + (float)il * (1.0f / 127.0f));
                }
            }
        }
        for (unsigned g = 0; g < ng; g++) {
            const unsigned gh = g0 + g;
            const float sc = lane < nt ? s[g] : -3.0e38f, m_old = ms[gh];
            const float m_new = fmaxf(m_old, warp_max(sc));
            const float w = lane < nt ? expf(sc - m_new) : 0.0f;
            const float l_tile = warp_sum(w);
            ws[lane * ATTN_GROUP_MAX + gh] = w;
            if (!lane) {
                const float corr = expf(m_old - m_new);
                cs[gh] = corr;
                ls[gh] = ls[gh] * corr + l_tile;
                ms[gh] = m_new;
            }
        }
        __syncthreads();
        #pragma unroll
        for (unsigned g = 0; g < ATTN_GROUP_MAX; g++) acc[g] *= cs[g];
        for (unsigned j = 0; j < nt; j++) {
            const uint32_t *vr = vq + j * RW;
            const float dv = __half2float(((const __half *)(vr + D / 4u))[vb]);
            const float v = dv * (float)(int8_t)(vr[vw] >> vsh);
            const float4 *wj = (const float4 *)(ws + j * ATTN_GROUP_MAX);
            const float4 w0 = wj[0], w1 = wj[1], w2 = wj[2];
            acc[0] += w0.x * v; acc[1] += w0.y * v; acc[2] += w0.z * v; acc[3] += w0.w * v;
            acc[4] += w1.x * v; acc[5] += w1.y * v; acc[6] += w1.z * v; acc[7] += w1.w * v;
            acc[8] += w2.x * v; acc[9] += w2.y * v; acc[10] += w2.z * v; acc[11] += w2.w * v;
        }
        __syncthreads();
    }
    for (unsigned g = 0; g < group; g++) {
        if (a.n_splits == 1) {
            out[((uint64_t)tok * H + kvh * group + g) * D + tid] = acc[g] * (ls[g] > 0.0f ? 1.0f / ls[g] : 0.0f);
        } else {
            float *dst = part + ((((uint64_t)tok * Hkv + kvh) * a.n_splits + split) * group + g) * (2u + D);
            if (tid == g) { dst[0] = ms[g]; dst[1] = ls[g]; }
            dst[2u + tid] = acc[g];
        }
    }
}

/* Combines the split partials of one (head, token). */
__global__ void attn_merge(attn_args a, const float *part, float *out) {
    const unsigned h = blockIdx.x, tok = blockIdx.y, lane = threadIdx.x;
    if (h >= a.n_head || tok >= a.n_tokens) return;
    const unsigned Hkv = a.n_head_kv, D = a.head_dim, group = a.n_head / Hkv;
    const unsigned kvh = h / group, g = h % group;
    const uint64_t stride = (uint64_t)group * (2u + D);
    const float *base = part + (((uint64_t)tok * Hkv + kvh) * a.n_splits * group + g) * (2u + D);
    float mm = -3.0e38f;
    for (unsigned s = 0; s < a.n_splits; s++) mm = fmaxf(mm, base[s * stride]);
    float ll = 0.0f, o[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (unsigned s = 0; s < a.n_splits; s++) {
        const float *p = base + s * stride;
        const float c = p[1] > 0.0f ? expf(p[0] - mm) : 0.0f;
        ll += p[1] * c;
        for (unsigned j = 0; j < 4u; j++) {
            const unsigned d = lane * 4u + j;
            if (d < D) o[j] += p[2u + d] * c;
        }
    }
    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    float *dst = out + ((uint64_t)tok * a.n_head + h) * D;
    for (unsigned j = 0; j < 4u; j++) {
        const unsigned d = lane * 4u + j;
        if (d < D) dst[d] = o[j] * inv;
    }
}

/* attn_merge with the splits spread over the warps of a block instead of
 * walked by one warp, whose serial loads were most of its time at 256
 * splits; the sums run per warp and then across the warps. */
__global__ void __launch_bounds__(ATTN_MERGE_THREADS) attn_merge_wide(attn_args a, const float *part, float *out) {
    constexpr unsigned NW = ATTN_MERGE_THREADS / 32u;
    const unsigned h = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, warp = tid / 32u, lane = tid & 31u;
    if (h >= a.n_head || tok >= a.n_tokens) return;
    const unsigned Hkv = a.n_head_kv, D = a.head_dim, group = a.n_head / Hkv;
    const unsigned kvh = h / group, g = h % group;
    const uint64_t stride = (uint64_t)group * (2u + D);
    const float *base = part + (((uint64_t)tok * Hkv + kvh) * a.n_splits * group + g) * (2u + D);
    __shared__ float red[NW], redl[NW];
    __shared__ __align__(16) float redo[NW * 128];
    float mm = -3.0e38f;
    for (unsigned s = tid; s < a.n_splits; s += ATTN_MERGE_THREADS) mm = fmaxf(mm, base[s * stride]);
    mm = warp_max(mm);
    if (!lane) red[warp] = mm;
    __syncthreads();
    for (unsigned w = 0; w < NW; w++) mm = fmaxf(mm, red[w]);
    float ll = 0.0f, o[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma unroll 4
    for (unsigned s = warp; s < a.n_splits; s += NW) {
        const float *p = base + s * stride;
        const float2 ml = *(const float2 *)p;
        const float c = ml.y > 0.0f ? expf(ml.x - mm) : 0.0f;
        ll += ml.y * c;
        if (lane * 4u < D) {
            const float2 v0 = *(const float2 *)(p + 2u + lane * 4u), v1 = *(const float2 *)(p + 4u + lane * 4u);
            o[0] += v0.x * c; o[1] += v0.y * c; o[2] += v1.x * c; o[3] += v1.y * c;
        }
    }
    *(float4 *)(redo + warp * 128u + lane * 4u) = make_float4(o[0], o[1], o[2], o[3]);
    if (!lane) redl[warp] = ll;
    __syncthreads();
    if (warp == 0) {
        ll = 0.0f;
        o[0] = o[1] = o[2] = o[3] = 0.0f;
        for (unsigned w = 0; w < NW; w++) {
            const float4 v = *(const float4 *)(redo + w * 128u + lane * 4u);
            ll += redl[w];
            o[0] += v.x; o[1] += v.y; o[2] += v.z; o[3] += v.w;
        }
        const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
        float *dst = out + ((uint64_t)tok * a.n_head + h) * D;
        for (unsigned j = 0; j < 4u; j++) {
            const unsigned d = lane * 4u + j;
            if (d < D) dst[d] = o[j] * inv;
        }
    }
}

/* DS4_KOLIBRI1_OLD_ATTN=1 keeps attn_decode_tiles_q8 and attn_merge for A/B. */
static bool attn_old_kernels(void) {
    static const bool old = getenv("DS4_KOLIBRI1_OLD_ATTN") != NULL;
    return old;
}

__device__ __forceinline__ bool router_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

/* Top-k on logit + expert bias, weights = sigmoid(logit) * scale left
 * unnormalized; ties keep the lower expert index like kolibri1_route. Each
 * warp ranks its lanes' experts, then warp 0 ranks the candidates. */
__global__ void router_topk(const float *logits, const float *bias, int *selected, float *weights,
                            unsigned T, unsigned NE, unsigned NS, float scale) {
    const unsigned tok = blockIdx.x, warp = threadIdx.x / 32u, lane = threadIdx.x & 31u;
    const unsigned nsg = blockDim.x / 32u;
    if (tok >= T) return;
    __shared__ float cand_v[ROUTER_NSG * ROUTER_MAX_USED];
    __shared__ int cand_i[ROUTER_NSG * ROUTER_MAX_USED];
    const float *lg = logits + (uint64_t)tok * NE;
    const float unset = -3.402823466e+38f;
    float mine[ROUTER_LANE_MAX];
    for (unsigned k = 0; k < ROUTER_LANE_MAX; k++) {
        const unsigned e = warp * 32u + lane + 32u * nsg * k;
        mine[k] = e < NE ? lg[e] + bias[e] : unset;
    }
    for (unsigned r = 0; r < NS; r++) {
        float bv = unset;
        int bi = 0x7fffffff;
        for (unsigned k = 0; k < ROUTER_LANE_MAX; k++)
            if (mine[k] > bv) { bv = mine[k]; bi = (int)(warp * 32u + lane + 32u * nsg * k); }
        for (int off = 16; off; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (router_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (!lane) { cand_v[warp * NS + r] = bv; cand_i[warp * NS + r] = bi; }
        for (unsigned k = 0; k < ROUTER_LANE_MAX; k++)
            if ((int)(warp * 32u + lane + 32u * nsg * k) == bi) mine[k] = unset;
    }
    __syncthreads();
    if (warp) return;
    const unsigned n_cand = nsg * NS;
    float cv[ROUTER_NSG];
    int ci[ROUTER_NSG];
    for (unsigned j = 0; j < ROUTER_NSG; j++) {
        const unsigned c = lane + 32u * j;
        cv[j] = c < n_cand ? cand_v[c] : unset;
        ci[j] = c < n_cand ? cand_i[c] : 0x7fffffff;
    }
    for (unsigned r = 0; r < NS; r++) {
        float bv = unset;
        int bi = 0x7fffffff;
        for (unsigned j = 0; j < ROUTER_NSG; j++)
            if (router_better(cv[j], ci[j], bv, bi)) { bv = cv[j]; bi = ci[j]; }
        for (int off = 16; off; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (router_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        for (unsigned j = 0; j < ROUTER_NSG; j++) if (ci[j] == bi) cv[j] = unset;
        if (!lane) {
            selected[(uint64_t)tok * NS + r] = bi;
            weights[(uint64_t)tok * NS + r] = sigmoid(lg[bi]) * scale;
        }
    }
}

/* part[t][NS] is the shared expert: it joins unweighted, routed slots through
 * their router weights. */
__global__ void moe_sum(const float *part, const float *weights, float *out, unsigned T, unsigned NS, unsigned dim) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    if (tok >= T || d >= dim) return;
    const uint64_t stride = NS + 1u;
    float acc = part[((uint64_t)tok * stride + NS) * dim + d];
    for (unsigned s = 0; s < NS; s++)
        acc += weights[(uint64_t)tok * NS + s] * part[((uint64_t)tok * stride + s) * dim + d];
    out[(uint64_t)tok * dim + d] = acc;
}

/* As moe_sum, with the shared expert's output already in out instead of in
 * the part's last slot. */
__global__ void moe_sum_onto(const float *part, const float *weights, float *out, unsigned T, unsigned NS, unsigned dim) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    if (tok >= T || d >= dim) return;
    const uint64_t stride = NS + 1u;
    float acc = out[(uint64_t)tok * dim + d];
    for (unsigned s = 0; s < NS; s++)
        acc += weights[(uint64_t)tok * NS + s] * part[((uint64_t)tok * stride + s) * dim + d];
    out[(uint64_t)tok * dim + d] = acc;
}

static attn_prep_args make_attn_prep_args(unsigned T, unsigned H, unsigned Hkv, unsigned D, unsigned pos0,
                                          unsigned cache_rows, unsigned use_rope, float rope_base, float eps,
                                          bool q8) {
    attn_prep_args a = { T, H, Hkv, D, pos0, cache_rows, use_rope, q8 ? 1u : 0u, eps, { 0 } };
    for (unsigned i = 0; i < D / 2u; i++) a.rope_freq[i] = powf(rope_base, -2.0f * (float)i / (float)D);
    return a;
}

/* DS4_KOLIBRI1_ATTN_SPLITS=N lowers either kernel's limit for A/B. */
static unsigned attn_split_limit(unsigned most) {
    static const unsigned asked = [] {
        const char *set = getenv("DS4_KOLIBRI1_ATTN_SPLITS");
        return set ? (unsigned)strtoul(set, NULL, 10) : 0u;
    }();
    return asked && asked < most ? asked : most;
}

static attn_args make_attn_args(unsigned T, unsigned H, unsigned Hkv, unsigned D, unsigned pos0,
                                unsigned window, unsigned cache_rows, bool split, float scale) {
    const unsigned n_keys = window ? std::min(pos0 + T, window) : pos0 + T;   /* the last token's span */
    unsigned n_splits = split ? (n_keys + 31u) / 32u : 1u;
    if (n_splits < 1u) n_splits = 1u;
    const unsigned limit = attn_split_limit(ATTN_MAX_SPLITS);
    if (n_splits > limit) n_splits = limit;
    const attn_args a = { T, H, Hkv, D, pos0, window, cache_rows, n_splits, (n_keys + n_splits - 1u) / n_splits, scale };
    return a;
}

/* For attn_decode_tiles: splits of whole tiles, so only a split's last tile
 * can be short. */
static attn_args make_attn_tile_args(unsigned T, unsigned H, unsigned Hkv, unsigned D, unsigned pos0, float scale) {
    const unsigned n_keys = pos0 + T;
    const unsigned most = std::min((n_keys + ATTN_TILE - 1u) / ATTN_TILE, attn_split_limit(ATTN_TILE_MAX_SPLITS));
    const unsigned keys = ((n_keys + most - 1u) / most + ATTN_TILE - 1u) / ATTN_TILE * ATTN_TILE;
    const attn_args a = { T, H, Hkv, D, pos0, 0u, 0u, (n_keys + keys - 1u) / keys, keys, scale };
    return a;
}

/* Scores of one head, one column per token: softmax over the keys the token
 * may attend, rows 0 .. pos0 + token, and zero for the later ones. */
__global__ void attn_softmax_columns(float *scores, unsigned n_keys, unsigned pos0) {
    const unsigned tok = blockIdx.x, tid = threadIdx.x, n = min(pos0 + tok + 1u, n_keys);
    float *col = scores + (uint64_t)tok * n_keys;
    __shared__ float red[256];
    float m = -3.0e38f;
    for (unsigned i = tid; i < n; i += blockDim.x) m = fmaxf(m, col[i]);
    red[tid] = m;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float l = 0.0f;
    for (unsigned i = tid; i < n; i += blockDim.x) {
        const float w = expf(col[i] - m);
        col[i] = w;
        l += w;
    }
    red[tid] = l;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    for (unsigned i = tid; i < n_keys; i += blockDim.x) col[i] = i < n ? col[i] * inv : 0.0f;
}

/* attn_softmax_columns over the keys k0 .. k0 + keys of one head, carrying
 * the running (max, denominator) of each (token, head) in ml across the
 * pieces of a context: the piece's scores become exp(s - m_new), the
 * token's output row (out, stride H * D) is rescaled by exp(m_old - m_new),
 * and on the piece holding the token's last key both are also divided by the
 * final denominator, so the V product that follows (beta 1) completes the
 * row.  k0 == 0 starts the state and zeroes the row. */
__global__ void attn_softmax_piece(float *scores, float *ml, float *out, unsigned keys, unsigned k0,
                                   unsigned pos0, unsigned H, unsigned h, unsigned D) {
    const unsigned tok = blockIdx.x, tid = threadIdx.x, pos = pos0 + tok;
    const unsigned valid = pos + 1u > k0 ? min(keys, pos + 1u - k0) : 0u;
    const bool last = pos + 1u > k0 && pos + 1u <= k0 + keys;
    float *col = scores + (uint64_t)tok * keys, *state = ml + ((uint64_t)tok * H + h) * 2u;
    __shared__ float red[256];
    float m = -3.0e38f;
    for (unsigned i = tid; i < valid; i += blockDim.x) m = fmaxf(m, col[i]);
    red[tid] = m;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float m_old = k0 ? state[0] : -3.0e38f, l_old = k0 ? state[1] : 0.0f;
    const float m_new = fmaxf(m_old, red[0]), c = expf(m_old - m_new);
    __syncthreads();
    float l = 0.0f;
    for (unsigned i = tid; i < valid; i += blockDim.x) {
        const float w = expf(col[i] - m_new);
        col[i] = w;
        l += w;
    }
    red[tid] = l;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float l_new = l_old * c + red[0];
    const float norm = last ? (l_new > 0.0f ? 1.0f / l_new : 0.0f) : 1.0f;
    for (unsigned i = tid; i < keys; i += blockDim.x) col[i] = i < valid ? col[i] * norm : 0.0f;
    if (valid) {
        float *o = out + ((uint64_t)tok * H + h) * D;
        for (unsigned d = tid; d < D; d += blockDim.x) o[d] = k0 ? o[d] * c * norm : 0.0f;
        if (!tid) { state[0] = m_new; state[1] = l_new; }
    }
}

#ifndef DS4_KOLIBRI1_CUDA_KERNELS_ONLY

static std::unordered_map<int, std::pair<float *, uint64_t>> g_attn_scores;   /* by CUDA device */
static std::unordered_map<int, std::pair<float *, uint64_t>> g_attn_q8_prompt; /* attn_batch_products_q8's pieces, by CUDA device */

/* One per-device scratch grown to need bytes; NULL when the device is out. */
static float *attn_scratch(std::pair<float *, uint64_t> &s, uint64_t need) {
    if (s.second < need) {
        (void)cudaFree(s.first);
        void *grown = NULL;
        s = { NULL, 0 };
        if (cudaMalloc(&grown, need) == cudaSuccess) s = { (float *)grown, need };
        else (void)cudaGetLastError();
    }
    return s.first;
}

/* Full-context attention for a batch as two matrix products per head: the
 * per-token kernel walks every key serially for every row, which for a prompt
 * chunk deep into a long context is the whole cost of the layer.  The caches
 * interleave the heads, which a leading dimension of all heads' width skips
 * over.  Rows are taken in groups whose score matrix stays within a fixed
 * size.  Returns 0 when it cannot run and the caller falls back. */
static int attn_batch_products(float *out, const float *q, const float *kc, const float *vc,
                               unsigned T, unsigned H, unsigned Hkv, unsigned D, unsigned pos0, float scale) {
    const unsigned n_keys = pos0 + T, group = H / Hkv;
    const uint64_t budget = 256ull << 20;
    const unsigned rows = (unsigned)std::max<uint64_t>(1, std::min<uint64_t>(T, budget / ((uint64_t)n_keys * 4)));
    int device = -1, done = 0;
    cublasHandle_t blas = cuda_cublas_for_tier(g_current_logical_tier < 0 ? 0 : g_current_logical_tier);
    if (blas && cudaGetDevice(&device) == cudaSuccess) {
        float *scores = attn_scratch(g_attn_scores[device], (uint64_t)rows * n_keys * 4);
        bool ok = scores != NULL;
        const float one = 1.0f, zero = 0.0f;
        for (unsigned t0 = 0; t0 < T && ok; t0 += rows) {
            const unsigned n = std::min(rows, T - t0), keys = pos0 + t0 + n;
            for (unsigned h = 0; h < H && ok; h++) {
                const unsigned kvh = h / group;
                const float *qh = q + ((uint64_t)t0 * H + h) * D;
                ok = cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)keys, (int)n, (int)D, &scale,
                                 kc + (uint64_t)kvh * D, (int)(Hkv * D), qh, (int)(H * D), &zero,
                                 scores, (int)keys) == CUBLAS_STATUS_SUCCESS;
                if (ok) {
                    attn_softmax_columns<<<n, 256, 0, cuda_decode_stream()>>>(scores, keys, pos0 + t0);
                    ok = qwen4_cuda::launched() &&
                         cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, (int)D, (int)n, (int)keys, &one,
                                     vc + (uint64_t)kvh * D, (int)(Hkv * D), scores, (int)keys, &zero,
                                     out + ((uint64_t)t0 * H + h) * D, (int)(H * D)) == CUBLAS_STATUS_SUCCESS;
                }
            }
        }
        done = ok;
        static bool told;
        if (ok && !told) {
            told = true;
            fprintf(stderr, "ds4: Kolibri-1 full-context attention of prompt chunks runs as matrix products\n");
        }
    }
    return done;
}

/* attn_batch_products for q8 caches.  Floats of the whole context would want
 * hundreds of MB after the expert caches were sized, so the keys go in
 * pieces of ATTN_Q8_PIECE_KEYS: a piece is dequantized once per layer into a
 * float K and V of all kv heads (the f32 layout), then every row group and
 * head runs the score product, attn_softmax_piece carries the softmax across
 * the pieces, and the V product accumulates into out.  One allocation per
 * device holds the pieces, the scores and the (max, denominator) state:
 * about 70 MB at the released shape with an 8192-token chunk, a quarter of
 * what the f32 path's score scratch may grow to. */
static int attn_batch_products_q8(float *out, const float *q, const float *kc, const float *vc,
                                  unsigned T, unsigned H, unsigned Hkv, unsigned D, unsigned pos0, float scale) {
    const unsigned n_keys = pos0 + T, group = H / Hkv, piece = std::min(n_keys, ATTN_Q8_PIECE_KEYS);
    const uint64_t budget = 32ull << 20;
    const unsigned rows = (unsigned)std::max<uint64_t>(1, std::min<uint64_t>(T, budget / ((uint64_t)piece * 4)));
    const uint64_t kv_floats = (uint64_t)piece * Hkv * D, score_floats = (uint64_t)rows * piece;
    const uint64_t need = (2 * kv_floats + score_floats + (uint64_t)T * H * 2) * 4;
    int device = -1, done = 0;
    cublasHandle_t blas = cuda_cublas_for_tier(g_current_logical_tier < 0 ? 0 : g_current_logical_tier);
    if (blas && cudaGetDevice(&device) == cudaSuccess) {
        float *mem = attn_scratch(g_attn_q8_prompt[device], need);
        static bool told, told_short;
        if (!mem) {
            if (!told_short) {
                told_short = true;
                fprintf(stderr, "ds4: Kolibri-1 q8 prompt attention: device %d has no %llu MB of scratch left\n",
                        device, (unsigned long long)(need >> 20));
            }
        } else {
            float *kf = mem, *vf = kf + kv_floats, *scores = vf + kv_floats, *ml = scores + score_floats;
            const float one = 1.0f, zero = 0.0f;
            bool ok = true;
            for (unsigned p0 = 0; p0 < n_keys && ok; p0 += piece) {
                const unsigned pk = std::min(piece, n_keys - p0), values = pk * Hkv * D;
                const uint64_t at = (uint64_t)p0 * Hkv * kv_row_bytes_q8(D);
                kv_dequant_q8<<<(values / 4 + 255) / 256, 256, 0, cuda_decode_stream()>>>(kf, (const uint8_t *)kc + at, values, D);
                kv_dequant_q8<<<(values / 4 + 255) / 256, 256, 0, cuda_decode_stream()>>>(vf, (const uint8_t *)vc + at, values, D);
                ok = qwen4_cuda::launched();
                for (unsigned t0 = 0; t0 < T && ok; t0 += rows) {
                    const unsigned n = std::min(rows, T - t0), end = pos0 + t0 + n;   /* keys the group's last token sees */
                    if (p0 >= end) continue;
                    const unsigned keys = std::min(pk, end - p0);
                    for (unsigned h = 0; h < H && ok; h++) {
                        const unsigned kvh = h / group;
                        const float *qh = q + ((uint64_t)t0 * H + h) * D;
                        ok = cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)keys, (int)n, (int)D, &scale,
                                         kf + (uint64_t)kvh * D, (int)(Hkv * D), qh, (int)(H * D), &zero,
                                         scores, (int)keys) == CUBLAS_STATUS_SUCCESS;
                        if (ok) {
                            attn_softmax_piece<<<n, 256, 0, cuda_decode_stream()>>>(
                                scores, ml + (uint64_t)t0 * H * 2, out + (uint64_t)t0 * H * D, keys, p0, pos0 + t0, H, h, D);
                            ok = qwen4_cuda::launched() &&
                                 cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, (int)D, (int)n, (int)keys, &one,
                                             vf + (uint64_t)kvh * D, (int)(Hkv * D), scores, (int)keys, &one,
                                             out + ((uint64_t)t0 * H + h) * D, (int)(H * D)) == CUBLAS_STATUS_SUCCESS;
                        }
                    }
                }
            }
            done = ok;
            if (ok && !told) {
                told = true;
                fprintf(stderr, "ds4: Kolibri-1 q8 full-context attention of prompt chunks runs as matrix products "
                                "in pieces of %u keys\n", piece);
            }
        }
    }
    return done;
}

#endif

} // namespace kolibri1_cuda

#ifndef DS4_KOLIBRI1_CUDA_KERNELS_ONLY

extern "C" int ds4_gpu_kolibri_attn_prep_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *qproj, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const void *map, uint64_t size, uint64_t g_q_offset, uint64_t g_k_offset,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D,
        uint32_t pos0, uint32_t cache_rows, uint32_t use_rope, float rope_base, float eps) {
    using namespace kolibri1_cuda;
    const bool q8 = g_kolibri1_kv_q8;
    const uint64_t qb = (uint64_t)T * H * D * 4, kb = (uint64_t)T * Hkv * D * 4;
    const uint64_t cb = (uint64_t)(cache_rows ? cache_rows : pos0 + T) * Hkv * (q8 ? kv_row_bytes_q8(D) : D * 4);
    if (!T || !D || D > 128 || D % 4 || (q8 && D % 64) || !Hkv || H % Hkv || (cache_rows && T > cache_rows) ||
        !qwen4_cuda::tensor(qproj, qb) || !qwen4_cuda::tensor(kproj, kb) || !qwen4_cuda::tensor(vproj, kb) ||
        !qwen4_cuda::tensor(q_out, qb) || !qwen4_cuda::tensor(k_cache, cb) || !qwen4_cuda::tensor(v_cache, cb)) return 0;
    const char *gq = qwen4_cuda::weight(map, size, g_q_offset, (uint64_t)D * 4);
    const char *gk = qwen4_cuda::weight(map, size, g_k_offset, (uint64_t)D * 4);
    if (!gq || !gk) return 0;
    attn_prep<<<dim3(H + Hkv + 1, T), 32, 0, cuda_decode_stream()>>>(
        make_attn_prep_args(T, H, Hkv, D, pos0, cache_rows, use_rope, rope_base, eps, q8),
        (const float *)qproj->ptr, (const float *)kproj->ptr, (const float *)vproj->ptr,
        (const float *)gq, (const float *)gk, (float *)q_out->ptr, (float *)k_cache->ptr, (float *)v_cache->ptr);
    return qwen4_cuda::launched();
}

extern "C" uint64_t ds4_gpu_kolibri_attn_part_floats(uint32_t T, uint32_t H, uint32_t D) {
    return (uint64_t)T * H * kolibri1_cuda::ATTN_MAX_SPLITS * (2u + D);
}

extern "C" int ds4_gpu_kolibri_attn_decode_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache, ds4_gpu_tensor *part,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D,
        uint32_t pos0, uint32_t window, uint32_t cache_rows, float scale) {
    using namespace kolibri1_cuda;
    const bool q8 = g_kolibri1_kv_q8;
    const uint64_t qb = (uint64_t)T * H * D * 4;
    const uint64_t cb = (uint64_t)(cache_rows ? cache_rows : pos0 + T) * Hkv * (q8 ? kv_row_bytes_q8(D) : D * 4);
    if (!T || !Hkv || H % Hkv || H / Hkv > ATTN_GROUP_MAX || !D || D > 128 || D % 4 || (q8 && D % 64) ||
        (cache_rows && T > cache_rows) ||
        !qwen4_cuda::tensor(q, qb) || !qwen4_cuda::tensor(k_cache, cb) || !qwen4_cuda::tensor(v_cache, cb) ||
        !qwen4_cuda::tensor(out, qb) ||
        (part && !qwen4_cuda::tensor(part, ds4_gpu_kolibri_attn_part_floats(T, H, D) * 4))) return 0;
    /* DS4_KOLIBRI1_NO_ATTN_TILES=1 keeps attn_decode on full layers for A/B.
     * The tiled kernel reads K as float4. */
    static const bool tiled = getenv("DS4_KOLIBRI1_NO_ATTN_TILES") == NULL;
    const bool tiles = tiled && !window && !cache_rows && part && !((uintptr_t)k_cache->ptr & 15u);
    const attn_args a = tiles ? make_attn_tile_args(T, H, Hkv, D, pos0, scale)
                              : make_attn_args(T, H, Hkv, D, pos0, window, cache_rows, part != NULL, scale);
    /* DS4_KOLIBRI1_NO_ATTN_PRODUCTS=1 keeps the per-token kernel for A/B. */
    static const bool products = getenv("DS4_KOLIBRI1_NO_ATTN_PRODUCTS") == NULL;
    if (products && !window && !cache_rows && !part && T >= 64 && !g_decode_graph_capturing) {
        if ((q8 ? attn_batch_products_q8 : attn_batch_products)(
                (float *)out->ptr, (const float *)q->ptr, (const float *)k_cache->ptr,
                (const float *)v_cache->ptr, T, H, Hkv, D, pos0, scale)) return 1;
        static bool told;
        if (!told) {
            told = true;
            fprintf(stderr, "ds4: Kolibri-1 prompt attention falls back to the per-token kernel (%u tokens, %u keys); "
                            "expect it to be slow\n", T, pos0 + T);
        }
    }
    const bool old = attn_old_kernels();
    const auto kernel = tiles ? (q8 ? (D == 128 && !old ? attn_decode_tiles_q8_pipe : attn_decode_tiles_q8) : attn_decode_tiles)
                              : (q8 ? attn_decode<true> : attn_decode<false>);
    kernel<<<dim3(a.n_splits, Hkv, T), 32 * ATTN_NSG, 0, cuda_decode_stream()>>>(
        a, (const float *)q->ptr, (const float *)k_cache->ptr, (const float *)v_cache->ptr,
        (float *)out->ptr, a.n_splits > 1 ? (float *)part->ptr : NULL);
    if (!qwen4_cuda::launched()) return 0;
    if (a.n_splits == 1) return 1;
    if (old) attn_merge<<<dim3(H, T), 32, 0, cuda_decode_stream()>>>(a, (const float *)part->ptr, (float *)out->ptr);
    else attn_merge_wide<<<dim3(H, T), ATTN_MERGE_THREADS, 0, cuda_decode_stream()>>>(a, (const float *)part->ptr, (float *)out->ptr);
    return qwen4_cuda::launched();
}

extern "C" int ds4_gpu_kolibri_router_topk_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, uint64_t bias_offset, const void *map, uint64_t size,
        uint32_t T, uint32_t NE, uint32_t NS, float scale) {
    using namespace kolibri1_cuda;
    if (!T || !NE || NE > 512 || !NS || NS > ROUTER_MAX_USED || NS > NE ||
        !qwen4_cuda::tensor(logits, (uint64_t)T * NE * 4) ||
        !qwen4_cuda::tensor(selected, (uint64_t)T * NS * 4) || !qwen4_cuda::tensor(weights, (uint64_t)T * NS * 4)) return 0;
    const char *bias = qwen4_cuda::weight(map, size, bias_offset, (uint64_t)NE * 4);
    if (!bias) return 0;
    router_topk<<<T, 32 * ROUTER_NSG, 0, cuda_decode_stream()>>>((const float *)logits->ptr, (const float *)bias,
        (int *)selected->ptr, (float *)weights->ptr, T, NE, NS, scale);
    return qwen4_cuda::launched();
}

extern "C" int ds4_gpu_kolibri_moe_sum_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *part, const ds4_gpu_tensor *weights,
        uint32_t T, uint32_t NS, uint32_t dim) {
    using namespace kolibri1_cuda;
    if (!T || !NS || !dim ||
        !qwen4_cuda::tensor(part, (uint64_t)T * (NS + 1) * dim * 4) ||
        !qwen4_cuda::tensor(weights, (uint64_t)T * NS * 4) || !qwen4_cuda::tensor(out, (uint64_t)T * dim * 4)) return 0;
    moe_sum<<<dim3((dim + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (const float *)part->ptr, (const float *)weights->ptr, (float *)out->ptr, T, NS, dim);
    return qwen4_cuda::launched();
}

extern "C" int ds4_gpu_kolibri_moe_sum_onto_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *part, const ds4_gpu_tensor *weights,
        uint32_t T, uint32_t NS, uint32_t dim) {
    using namespace kolibri1_cuda;
    if (!T || !NS || !dim ||
        !qwen4_cuda::tensor(part, (uint64_t)T * (NS + 1) * dim * 4) ||
        !qwen4_cuda::tensor(weights, (uint64_t)T * NS * 4) || !qwen4_cuda::tensor(out, (uint64_t)T * dim * 4)) return 0;
    moe_sum_onto<<<dim3((dim + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (const float *)part->ptr, (const float *)weights->ptr, (float *)out->ptr, T, NS, dim);
    return qwen4_cuda::launched();
}

#endif
