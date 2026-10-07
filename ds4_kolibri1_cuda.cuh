/* Kolibri-1 kernels for CUDA, ported from metal/kolibri1.metal: per-head QK
 * RMSNorm + NeoX rope into f32 K/V caches (sliding layers address a ring of
 * cache_rows, full layers absolute rows), GQA span attention with optional
 * key splits, the sigmoid router and the routed/shared MoE sum.  Included by
 * ds4_cuda.cu after the Qwen kernels, whose MoE mid/down, dense and norm
 * wrappers the Kolibri graph also uses.  DS4_KOLIBRI1_CUDA_KERNELS_ONLY keeps
 * the wrappers out, so tests/test_kolibri1_cuda.cu can check the kernels
 * against a host reference without the rest of the backend. */

#include <stdint.h>

namespace kolibri1_cuda {

constexpr unsigned ATTN_NSG = 4;          /* warps per attention block, each owning part of a q-head group */
constexpr unsigned ATTN_MAX_SPLITS = 64;
constexpr unsigned ROUTER_MAX_USED = 8;
constexpr unsigned ROUTER_NSG = 4;
constexpr unsigned ROUTER_LANE_MAX = 4;   /* 32 lanes * 4 warps * 4 = 512 experts */

struct attn_prep_args {
    unsigned n_tokens, n_head, n_head_kv, head_dim, pos0, cache_rows, use_rope;
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

/* One warp per (head slot, token): slots 0..H-1 norm (and rope) a query
 * head, H..H+Hkv-1 a key head plus its value row, the last copies V. */
__global__ void attn_prep(attn_prep_args a, const float *qproj, const float *kproj, const float *vproj,
                          const float *gq, const float *gk, float *qout, float *kc, float *vc) {
    const unsigned slot = blockIdx.x, tok = blockIdx.y, lane = threadIdx.x;
    const unsigned H = a.n_head, Hkv = a.n_head_kv, D = a.head_dim;
    if (tok >= a.n_tokens || slot > H + Hkv) return;
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
 * split writes out; otherwise each head leaves (m, l, acc) partials. */
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
    for (unsigned idx = k0; idx < k1; idx++) {
        const unsigned p = start + idx;
        const uint64_t row = a.cache_rows ? p % a.cache_rows : p;
        const float *kr = kc + (row * Hkv + kvh) * D, *vr = vc + (row * Hkv + kvh) * D;
        float kv[4], vv[4];
        for (unsigned j = 0; j < 4u; j++) {
            const unsigned d = lane * 4u + j;
            kv[j] = d < D ? kr[d] : 0.0f;
            vv[j] = d < D ? vr[d] : 0.0f;
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

static attn_prep_args make_attn_prep_args(unsigned T, unsigned H, unsigned Hkv, unsigned D, unsigned pos0,
                                          unsigned cache_rows, unsigned use_rope, float rope_base, float eps) {
    attn_prep_args a = { T, H, Hkv, D, pos0, cache_rows, use_rope, eps, { 0 } };
    for (unsigned i = 0; i < D / 2u; i++) a.rope_freq[i] = powf(rope_base, -2.0f * (float)i / (float)D);
    return a;
}

static attn_args make_attn_args(unsigned T, unsigned H, unsigned Hkv, unsigned D, unsigned pos0,
                                unsigned window, unsigned cache_rows, bool split, float scale) {
    const unsigned n_keys = pos0 + T;
    unsigned n_splits = split ? (n_keys + 31u) / 32u : 1u;
    if (n_splits < 1u) n_splits = 1u;
    if (n_splits > ATTN_MAX_SPLITS) n_splits = ATTN_MAX_SPLITS;
    const attn_args a = { T, H, Hkv, D, pos0, window, cache_rows, n_splits, (n_keys + n_splits - 1u) / n_splits, scale };
    return a;
}

} // namespace kolibri1_cuda

#ifndef DS4_KOLIBRI1_CUDA_KERNELS_ONLY

extern "C" int ds4_gpu_kolibri_attn_prep_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *qproj, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const void *map, uint64_t size, uint64_t g_q_offset, uint64_t g_k_offset,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D,
        uint32_t pos0, uint32_t cache_rows, uint32_t use_rope, float rope_base, float eps) {
    using namespace kolibri1_cuda;
    const uint64_t qb = (uint64_t)T * H * D * 4, kb = (uint64_t)T * Hkv * D * 4;
    const uint64_t cb = (uint64_t)(cache_rows ? cache_rows : pos0 + T) * Hkv * D * 4;
    if (!T || !D || D > 128 || D % 4 || !Hkv || H % Hkv || (cache_rows && T > cache_rows) ||
        !qwen4_cuda::tensor(qproj, qb) || !qwen4_cuda::tensor(kproj, kb) || !qwen4_cuda::tensor(vproj, kb) ||
        !qwen4_cuda::tensor(q_out, qb) || !qwen4_cuda::tensor(k_cache, cb) || !qwen4_cuda::tensor(v_cache, cb)) return 0;
    const char *gq = qwen4_cuda::weight(map, size, g_q_offset, (uint64_t)D * 4);
    const char *gk = qwen4_cuda::weight(map, size, g_k_offset, (uint64_t)D * 4);
    if (!gq || !gk) return 0;
    attn_prep<<<dim3(H + Hkv + 1, T), 32, 0, cuda_decode_stream()>>>(
        make_attn_prep_args(T, H, Hkv, D, pos0, cache_rows, use_rope, rope_base, eps),
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
    const attn_args a = make_attn_args(T, H, Hkv, D, pos0, window, cache_rows, part != NULL, scale);
    const uint64_t qb = (uint64_t)T * H * D * 4;
    const uint64_t cb = (uint64_t)(cache_rows ? cache_rows : pos0 + T) * Hkv * D * 4;
    if (!T || !Hkv || H % Hkv || H / Hkv > 12 || !D || D > 128 || D % 4 || (cache_rows && T > cache_rows) ||
        !qwen4_cuda::tensor(q, qb) || !qwen4_cuda::tensor(k_cache, cb) || !qwen4_cuda::tensor(v_cache, cb) ||
        !qwen4_cuda::tensor(out, qb) ||
        (a.n_splits > 1 && !qwen4_cuda::tensor(part, ds4_gpu_kolibri_attn_part_floats(T, H, D) * 4))) return 0;
    attn_decode<<<dim3(a.n_splits, Hkv, T), 32 * ATTN_NSG, 0, cuda_decode_stream()>>>(
        a, (const float *)q->ptr, (const float *)k_cache->ptr, (const float *)v_cache->ptr,
        (float *)out->ptr, a.n_splits > 1 ? (float *)part->ptr : NULL);
    if (!qwen4_cuda::launched()) return 0;
    if (a.n_splits == 1) return 1;
    attn_merge<<<dim3(H, T), 32, 0, cuda_decode_stream()>>>(a, (const float *)part->ptr, (float *)out->ptr);
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

#endif
