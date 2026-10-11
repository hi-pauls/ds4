/* Qwen3.8 Flash Next. Included by ds4_cuda.cu so model residency, streams
 * and temporary allocations have the same lifetime as the other CUDA paths. */

#include "ds4_qwen4_vision.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace qwen4_cuda {

__device__ __forceinline__ float sum(float x);
__device__ __forceinline__ float sigmoid(float x);
template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid = NULL, const uint8_t *signs = NULL);
__device__ float injection(const float *inj, unsigned hc, unsigned stream);

struct rope_args { float freq[32], scale; unsigned nrot; };
static float rope_freq[32];
static float rope_scale = 1;
static bool rope_set;

static rope_args rope(unsigned nrot, float base) {
    rope_args r = {};
    r.nrot = nrot;
    r.scale = rope_set ? rope_scale : 1;
    for (unsigned i = 0; i < nrot / 2; i++)
        r.freq[i] = rope_set ? rope_freq[i] : powf(base, -2.0f * i / nrot);
    return r;
}

__device__ void apply_rope(float *row, const uint32_t *pos, rope_args r) {
    const unsigned i = threadIdx.x;
    if (i < r.nrot / 2) {
        const float theta = (float)pos[i % 3] * r.freq[i];
        const float c = cosf(theta) * r.scale, s = sinf(theta) * r.scale;
        const float a = row[i], b = row[i + r.nrot / 2];
        row[i] = a * c - b * s;
        row[i + r.nrot / 2] = a * s + b * c;
    }
    __syncwarp();
}

/* q8_0 K/V rows, the Metal layout: the D int8 values of a (position, kv head)
 * row, then its D/32 half scales. Lane i of a warp step owns value lane+32*b,
 * so a step is one block and its max is a warp max. */
__device__ __forceinline__ unsigned kv_row_bytes_q8(unsigned D) { return D + D / 16; }

__device__ __forceinline__ void kv_store_q8(int8_t *row, unsigned D, unsigned i, float v) {
    float amax = fabsf(v);
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
    /* quantize against the scale as stored, so dequantization inverts it */
    const __half dh = __float2half_rn(amax / 127.0f);
    const float d = __half2float(dh), id = d > 0 ? 1.0f / d : 0;
    row[i] = (int8_t)fminf(fmaxf(roundf(v * id), -127.0f), 127.0f);
    if (!(i & 31)) ((__half *)(row + D))[i / 32] = dh;
}

/* value lane+32*b of row r, from either cache format */
template<bool Q8>
__device__ __forceinline__ float kv_at(const __half *cache, uint64_t r, unsigned D, unsigned b, unsigned lane) {
    if (Q8) {
        const int8_t *row = (const int8_t *)cache + r * kv_row_bytes_q8(D);
        return __half2float(((const __half *)(row + D))[b]) * (float)row[lane + 32 * b];
    }
    return __half2float(cache[r * D + lane + 32 * b]);
}

__global__ void attn_prep(float *qout, float *gate, __half *kc, __half *vc,
        float *iqout, float *ikc, const float *qg, const float *kp, const float *vp,
        const float *iq, const float *ik, const uint32_t *pos3,
        const float *gq, const float *gk, const float *giq,
        unsigned H, unsigned Hkv, unsigned D, unsigned Hi, unsigned Di,
        unsigned pos0, unsigned ik_ring, float eps, rope_args rp, bool q8) {
    const unsigned slot = blockIdx.x, t = blockIdx.y, pos = pos0 + t, lane = threadIdx.x;
    __shared__ float row[256];
    if (slot == H + Hkv + Hi) {
        /* raw indexer keys live at pos % ik_ring; 0 keeps one row per position */
        const uint64_t key_row = ik_ring ? pos % ik_ring : pos;
        for (unsigned i = lane; i < Di; i += 32) ikc[key_row * Di + i] = ik[(uint64_t)t * Di + i];
        return;
    }
    const bool isq = slot < H, isk = !isq && slot < H + Hkv;
    const unsigned h = isq ? slot : isk ? slot - H : slot - H - Hkv;
    const unsigned dim = isq || isk ? D : Di;
    const float *src = isq ? qg + ((uint64_t)t * H + h) * 2 * D :
                      isk ? kp + ((uint64_t)t * Hkv + h) * D : iq + ((uint64_t)t * Hi + h) * Di;
    const float *gamma = isq ? gq : isk ? gk : giq;
    const unsigned npt = dim / 32;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) { const float v = src[lane * npt + i]; ss += v * v; }
    const float inv = rsqrtf(sum(ss) / dim + eps);
    for (unsigned i = lane; i < dim; i += 32) row[i] = src[i] * inv * gamma[i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)pos * 4, rp);
    for (unsigned i = lane; i < dim; i += 32) {
        if (isq) {
            qout[((uint64_t)t * H + h) * D + i] = row[i];
            gate[((uint64_t)t * H + h) * D + i] = src[D + i];
        } else if (isk) {
            const uint64_t r = (uint64_t)pos * Hkv + h;
            const float v = vp[((uint64_t)t * Hkv + h) * D + i];
            if (q8) {
                kv_store_q8((int8_t *)kc + r * kv_row_bytes_q8(D), D, i, row[i]);
                kv_store_q8((int8_t *)vc + r * kv_row_bytes_q8(D), D, i, v);
            } else {
                kc[r * D + i] = __float2half_rn(row[i]);
                vc[r * D + i] = __float2half_rn(v);
            }
        } else iqout[((uint64_t)t * Hi + h) * Di + i] = row[i];
    }
}

__global__ void block_key(__half *out, const float *ik, const uint32_t *pos3,
        const float *gamma, unsigned block0, unsigned ratio, unsigned D, unsigned ik_ring, float eps, rope_args rp) {
    const unsigned b = block0 + blockIdx.x, lane = threadIdx.x, npt = D / 32;
    __shared__ float row[128];
    float ss = 0, v[4];
    for (unsigned i = 0; i < npt; i++) {
        float a = 0;
        for (unsigned t = 0; t < ratio; t++) {
            const uint64_t pos = (uint64_t)b * ratio + t;
            a += ik[(ik_ring ? pos % ik_ring : pos) * D + lane * npt + i];
        }
        v[i] = a / ratio;
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) row[lane * npt + i] = v[i] * inv * gamma[lane * npt + i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)b * ratio * 4, rp);
    for (unsigned i = lane; i < D; i += 32) out[(uint64_t)b * D + i] = __float2half_rn(row[i]);
}

__global__ void idx_score(float *out, const float *q, const __half *key,
        unsigned N, unsigned H, unsigned D, unsigned pos0, unsigned ratio) {
    const unsigned b = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, lane = threadIdx.x & 31;
    if (b >= N) return;
    float score = 0;
    if (b >= (pos0 + t + 1) / ratio) score = -3e38f;
    else for (unsigned h = 0; h < H; h++) {
        float a = 0;
        for (unsigned i = lane; i < D; i += 32)
            a += q[((uint64_t)t * H + h) * D + i] * __half2float(key[(uint64_t)b * D + i]);
        score += fmaxf(sum(a), 0);
    }
    if (!lane) out[(uint64_t)t * N + b] = score;
}

__global__ void tile_max(unsigned *out, const float *scores, unsigned N, unsigned tiles) {
    const unsigned tile = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (tile >= tiles) return;
    unsigned v = 0;
    for (unsigned i = tile * 8; i < min(N, tile * 8 + 8); i++)
        v = max(v, __float_as_uint(fmaxf(scores[(uint64_t)t * N + i], 0)));
    out[(uint64_t)t * tiles + tile] = v;
}

/* Exact radix threshold and stable gather, with the same greater-than then
 * equal-score ordering as Metal. No context-dependent candidate truncation. */
__global__ void idx_select(int *out, const float *score, unsigned N, unsigned K) {
    const unsigned tid = threadIdx.x, t = blockIdx.x;
    __shared__ unsigned hist[256], gt[256], eq[256], threshold, need;
    const float *row = score + (uint64_t)t * N;
    if (!tid) { threshold = 0; need = K; }
    __syncthreads();
    for (unsigned pass = 0; pass < 4; pass++) {
        const unsigned shift = 24 - 8 * pass, mask = pass ? (0xffffffffu << (shift + 8)) : 0;
        hist[tid] = 0;
        __syncthreads();
        for (unsigned i = tid; i < N; i += 256) {
            const unsigned key = __float_as_uint(fmaxf(row[i], 0));
            if ((key & mask) == threshold) atomicAdd(hist + ((key >> shift) & 255), 1u);
        }
        __syncthreads();
        if (!tid) for (int d = 255; d >= 0; d--) {
            if (hist[d] >= need) { threshold |= (unsigned)d << shift; break; }
            need -= hist[d];
        }
        __syncthreads();
    }
    const unsigned chunk = (N + 255) / 256, begin = min(N, tid * chunk), end = min(N, begin + chunk);
    unsigned ng = 0, ne = 0;
    for (unsigned i = begin; i < end; i++) {
        const unsigned key = __float_as_uint(fmaxf(row[i], 0));
        ng += key > threshold; ne += key == threshold;
    }
    gt[tid] = ng; eq[tid] = ne;
    __syncthreads();
    if (!tid) {
        unsigned pg = 0, pe = 0;
        for (unsigned i = 0; i < 256; i++) {
            const unsigned g = gt[i], e = eq[i];
            gt[i] = pg; eq[i] = pe; pg += g; pe += e;
        }
    }
    __syncthreads();
    unsigned g = gt[tid], e = eq[tid];
    for (unsigned i = begin; i < end; i++) {
        const unsigned key = __float_as_uint(fmaxf(row[i], 0));
        if (key > threshold) out[(uint64_t)t * K + g++] = i;
        else if (key == threshold) { if (e < need) out[(uint64_t)t * K + K - need + e] = i; e++; }
    }
}

__global__ void idx_expand(int *out, unsigned *count, const int *blocks,
        unsigned K, unsigned ratio, unsigned pos0, unsigned stride) {
    const unsigned t = blockIdx.x, pos = pos0 + t, tail = (pos + 1) / ratio * ratio;
    for (unsigned i = threadIdx.x; i < K * ratio; i += blockDim.x)
        out[(uint64_t)t * stride + i] = blocks[(uint64_t)t * K + i / ratio] * ratio + i % ratio;
    for (unsigned i = tail + threadIdx.x; i <= pos; i += blockDim.x)
        out[(uint64_t)t * stride + K * ratio + i - tail] = i;
    if (!threadIdx.x) count[t] = K * ratio + pos + 1 - tail;
}

template<unsigned D, bool Q8>
__global__ void attention(float *out, float *partial, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse,
        unsigned splits, unsigned per, float scale) {
    const unsigned h = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, split = blockIdx.z;
    if (h >= H) return;
    const unsigned lane = threadIdx.x & 31, kh = h / (H / Hkv), n = sparse ? counts[t] : pos0 + t + 1;
    float qv[D / 32], acc[D / 32] = {}, m = -3e38f, denom = 0;
    for (unsigned i = 0; i < D / 32; i++) qv[i] = q[((uint64_t)t * H + h) * D + lane + 32 * i] * scale;
    for (unsigned j = split * per; j < min(n, (split + 1) * per); j++) {
        const unsigned p = sparse ? (unsigned)sel[(uint64_t)t * stride + j] : j;
        if (p > pos0 + t) continue;
        const uint64_t r = (uint64_t)p * Hkv + kh;
        float score = 0;
        for (unsigned i = 0; i < D / 32; i++) score += qv[i] * kv_at<Q8>(kc, r, D, i, lane);
        score = sum(score);
        const float nm = fmaxf(m, score), correction = expf(m - nm), w = expf(score - nm);
        denom = denom * correction + w;
        for (unsigned i = 0; i < D / 32; i++) acc[i] = acc[i] * correction + w * kv_at<Q8>(vc, r, D, i, lane);
        m = nm;
    }
    if (splits == 1) {
        for (unsigned i = 0; i < D / 32; i++) {
            const uint64_t p = ((uint64_t)t * H + h) * D + lane + 32 * i;
            out[p] = (denom > 0 ? acc[i] / denom : 0) * sigmoid(gate[p]);
        }
    } else {
        float *dst = partial + (((uint64_t)t * H + h) * splits + split) * (D + 2);
        if (!lane) { dst[0] = m; dst[1] = denom; }
        for (unsigned i = 0; i < D / 32; i++) dst[2 + lane + 32 * i] = acc[i];
    }
}

__global__ void attn_merge(float *out, const float *partial, const float *gate,
        unsigned H, unsigned D, unsigned splits) {
    const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    const float *p = partial + ((uint64_t)t * H + h) * splits * (D + 2);
    float m = -3e38f, denom = 0, acc = 0;
    for (unsigned s = 0; s < splits; s++) m = fmaxf(m, p[s * (D + 2)]);
    for (unsigned s = 0; s < splits; s++) {
        const float *row = p + s * (D + 2);
        const float w = row[1] > 0 ? expf(row[0] - m) : 0;
        denom += row[1] * w; acc += row[2 + d] * w;
    }
    const uint64_t i = ((uint64_t)t * H + h) * D + d;
    out[i] = (denom > 0 ? acc / denom : 0) * sigmoid(gate[i]);
}

__device__ __forceinline__ unsigned half2_bits(__half2 h) { return *reinterpret_cast<unsigned *>(&h); }

/* 32 q8_0 rows of D=256 into the kv tile of attention_group: one 32-value block
 * per thread, dequantized to half as cp.async cannot. Rows at UINT_MAX are zero. */
__device__ __forceinline__ void kv_tile_q8(__half (*kv)[264], const __half *cache,
        const unsigned *positions, unsigned Hkv, unsigned kh, unsigned tid) {
    const unsigned r = tid/8, blk = tid%8, p = positions[r];
    int4 *dst = reinterpret_cast<int4 *>(&kv[r][blk*32]);
    if (p == UINT_MAX) {
        #pragma unroll
        for (unsigned u = 0; u < 4; u++) dst[u] = make_int4(0,0,0,0);
        return;
    }
    const int8_t *row = (const int8_t *)cache+((uint64_t)p*Hkv+kh)*kv_row_bytes_q8(256);
    const float d = __half2float(((const __half *)(row+256))[blk]);
    const int4 *src = reinterpret_cast<const int4 *>(row+blk*32);
    #pragma unroll
    for (unsigned u = 0; u < 4; u++) {
        const int4 raw = src[u/2];
        const int w0 = u&1 ? raw.z : raw.x, w1 = u&1 ? raw.w : raw.y;
        dst[u] = make_int4(
            half2_bits(__floats2half2_rn(d*(float)(int8_t)w0,d*(float)(int8_t)(w0>>8))),
            half2_bits(__floats2half2_rn(d*(float)(int8_t)(w0>>16),d*(float)(int8_t)(w0>>24))),
            half2_bits(__floats2half2_rn(d*(float)(int8_t)w1,d*(float)(int8_t)(w1>>8))),
            half2_bits(__floats2half2_rn(d*(float)(int8_t)(w1>>16),d*(float)(int8_t)(w1>>24))));
    }
}

/* The heads in a KV group share the same selected keys. Keep their output
 * accumulators in registers and use tensor cores for both products. Scaled
 * residual components retain the fine part of the FP32 queries/probabilities;
 * K and V are already half, so neither requires further rounding. */
template<bool Q8>
__global__ void attention_group(float *out, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse, float scale) {
#if __CUDA_ARCH__ >= 800
    const unsigned D = 256, tid = threadIdx.x, lane = tid&31, warp = tid/32;
    const unsigned t = blockIdx.y, kh = blockIdx.x, group = H/Hkv;
    const unsigned qr = tid/16, col = tid%16, h = kh*group+qr;
    const unsigned n = sparse ? counts[t] : pos0+t+1;
    __shared__ __align__(32) __half qh[16][264], ql[16][264], kv[32][264];
    union Scores { float part[2][16][32]; __half prob[2][16][40]; };
    __shared__ __align__(32) Scores scores;
    __shared__ float qs[16], max_score[16], denom[16], correction[16];
    __shared__ unsigned positions[32];
    float mx = 0;
    for (unsigned d = col; d < D; d += 16)
        if (qr < group) mx = fmaxf(mx,fabsf(q[((uint64_t)t*H+h)*D+d]*scale));
    for (unsigned off = 8; off; off /= 2) mx = fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,off,16));
    const int exp = mx > 0 ? max(-120,min(120,(int)((__float_as_uint(mx)>>23)&255)-127)) : 0;
    const float inv = ldexpf(1,-exp);
    if (!col) { qs[qr] = ldexpf(1,exp); max_score[qr] = -3e38f; denom[qr] = 0; }
    for (unsigned d = col; d < D; d += 16) {
        const float v = qr < group ? q[((uint64_t)t*H+h)*D+d]*scale*inv : 0;
        qh[qr][d] = __float2half_rn(v);
        ql[qr][d] = __float2half_rn((v-__half2float(qh[qr][d]))*4096);
    }
    float result[4][4] = {};
    __syncthreads();
    for (unsigned j0 = 0; j0 < n; j0 += 32) {
        if (tid < 32) {
            const unsigned j = j0+tid;
            const unsigned p = j < n ? (sparse ? (unsigned)sel[(uint64_t)t*stride+j] : j) : UINT_MAX;
            positions[tid] = p <= pos0+t ? p : UINT_MAX;
        }
        __syncthreads();
        if (Q8) kv_tile_q8(kv,kc,positions,Hkv,kh,tid);
        else {
            for (unsigned i = tid*8; i < 32*D; i += 256*8) {
                const unsigned r = i/D, d = i%D, p = positions[r];
                tt_cp_async_16B(&kv[r][d],kc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
            }
            tt_cp_async_commit();
            tt_cp_async_wait_group<0>();
        }
        __syncthreads();
        float hi[4] = {}, lo[4] = {};
        const unsigned split = warp/4, key0 = (warp%4)*8;
        for (unsigned k = split*128; k < (split+1)*128; k += 16) {
            uint32_t ah[4], al[4], b[2];
            tt_ldmatrix_x4(ah,&qh[lane%16][k+(lane/16)*8]);
            tt_ldmatrix_x4(al,&ql[lane%16][k+(lane/16)*8]);
            tt_ldmatrix_x2(b,&kv[key0+lane%8][k+((lane%16)/8)*8]);
            tt_mma_m16n8k16_f16_f32(hi,ah,b);
            tt_mma_m16n8k16_f16_f32(lo,al,b);
        }
        #pragma unroll
        for (unsigned i = 0; i < 4; i++)
            scores.part[split][tt_mma_c_i(lane,i)][key0+tt_mma_c_j(lane,i)] = hi[i]+lo[i]*0x1p-12f;
        __syncthreads();
        float prob[2], peak = max_score[qr];
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const unsigned key = col+i*16;
            prob[i] = positions[key] != UINT_MAX ?
                (scores.part[0][qr][key]+scores.part[1][qr][key])*qs[qr] : -3e38f;
            peak = fmaxf(peak,prob[i]);
        }
        for (unsigned off = 8; off; off /= 2) peak = fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,off,16));
        const float old = expf(max_score[qr]-peak);
        float total = 0;
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            prob[i] = positions[col+i*16] != UINT_MAX ? expf(prob[i]-peak) : 0;
            total += prob[i];
        }
        for (unsigned off = 8; off; off /= 2) total += __shfl_xor_sync(0xffffffff,total,off,16);
        __syncthreads();
        if (!col) {
            correction[qr] = old;
            max_score[qr] = peak;
            denom[qr] = denom[qr]*old+total;
        }
        #pragma unroll
        for (unsigned i = 0; i < 2; i++) {
            const __half p = __float2half_rn(prob[i]);
            scores.prob[0][qr][col+i*16] = p;
            scores.prob[1][qr][col+i*16] = __float2half_rn((prob[i]-__half2float(p))*4096);
        }
        if (Q8) kv_tile_q8(kv,vc,positions,Hkv,kh,tid);
        else {
            for (unsigned i = tid*8; i < 32*D; i += 256*8) {
                const unsigned r = i/D, d = i%D, p = positions[r];
                tt_cp_async_16B(&kv[r][d],vc+((uint64_t)(p == UINT_MAX ? 0 : p)*Hkv+kh)*D+d,p != UINT_MAX);
            }
            tt_cp_async_commit();
            tt_cp_async_wait_group<0>();
        }
        __syncthreads();
        #pragma unroll
        for (unsigned tile = 0; tile < 4; tile++) {
            float hi[4] = {}, lo[4] = {};
            #pragma unroll
            for (unsigned k = 0; k < 32; k += 16) {
                uint32_t ah[4], al[4], b[2];
                tt_ldmatrix_x4(ah,&scores.prob[0][lane%16][k+(lane/16)*8]);
                tt_ldmatrix_x4(al,&scores.prob[1][lane%16][k+(lane/16)*8]);
                tt_ldmatrix_x2_trans(b,&kv[k+lane%16][warp*32+tile*8]);
                tt_mma_m16n8k16_f16_f32(hi,ah,b);
                tt_mma_m16n8k16_f16_f32(lo,al,b);
            }
            #pragma unroll
            for (unsigned i = 0; i < 4; i++) result[tile][i] =
                result[tile][i]*correction[tt_mma_c_i(lane,i)]+(hi[i]+lo[i]*0x1p-12f);
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned tile = 0; tile < 4; tile++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = tt_mma_c_i(lane,i), d = warp*32+tile*8+tt_mma_c_j(lane,i);
            if (r < group) {
                const uint64_t dst = ((uint64_t)t*H+kh*group+r)*D+d;
                out[dst] = (denom[r] > 0 ? result[tile][i]/denom[r] : 0)*sigmoid(gate[dst]);
            }
        }
    }
#endif
}

static bool tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static uint64_t row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case 0: return n * 4;
    case 1: case 30: return n * 2;
    case 2: return n % 32 ? 0 : n / 32 * 18;
    case 8: return n % 32 ? 0 : n / 32 * 34;
    case 39: return n % 32 ? 0 : n / 32 * 17;
    case 10: return n % 256 ? 0 : n / 256 * 84;
    case 12: return n % 256 ? 0 : n / 256 * 144;
    case 16: return n % 256 ? 0 : n / 256 * 66;
    default: return 0;
    }
}

static const char *weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes) {
    if (!map || !bytes || off > size || bytes > size - off) return NULL;
    return cuda_resolve_weight_ptr(map, off, bytes, 0, "Qwen weights");
}

/* Temporary device memory on the current device: with the layers of a graph
 * spread over several devices the shared slab would belong to one of them. */
static void *tmp_alloc(uint64_t bytes, const char *what) {
    int tier = 0, device = -1;
    if (g_n_gpus > 1 && cudaGetDevice(&device) == cudaSuccess)
        for (int t = 0; t < g_n_gpus; t++)
            if (g_gpu[t].device_id == device) tier = t;
    return cuda_tmp_alloc_on(tier, bytes, what);
}

static int launched(void) { return cuda_ok(cudaGetLastError(), "Qwen kernel"); }

__device__ __forceinline__ float sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}

__device__ __forceinline__ float block_sum(float x, float *shared) {
    x = sum(x);
    if (!(threadIdx.x & 31)) shared[threadIdx.x / 32] = x;
    __syncthreads();
    float result = 0;
    for (unsigned i = 0; i < blockDim.x / 32; i++) result += shared[i];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }
__device__ __forceinline__ float softplus(float x) {
    return x > 20 ? x : x < -20 ? expf(x) : log1pf(expf(x));
}

/* These readers preserve the GGUF values, including padded Q2_K down rows.
 * Templates remove unused formats from each matrix kernel. */
template<unsigned TYPE>
__device__ __forceinline__ float value(const char *row, unsigned i,
        const uint64_t *grid_table = NULL, const uint8_t *sign_table = NULL) {
    if (TYPE == 0) return ((const float *)row)[i];
    if (TYPE == 1) return __half2float(((const __half *)row)[i]);
    if (TYPE == 30) return __bfloat162float(((const __nv_bfloat16 *)row)[i]);
    if (TYPE == 8) {
        const char *b = row + (i / 32) * 34;
        return __half2float(*(const __half *)b) * (float)((const int8_t *)b)[2 + i % 32];
    }
    if (TYPE == 2) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 18;
        return __half2float(*(const __half *)b) *
            (float)((int)((b[2 + i % 16] >> (4 * (i % 32 / 16))) & 15) - 8);
    }
    if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 17;
        const unsigned q = (b[1 + i % 16] >> (4 * (i % 32 / 16))) & 15;
        const float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0] << 23);
        return (q & 8 ? -levels[q & 7] : levels[q & 7]) * scale;
    }
    if (TYPE == 10) {
        const cuda_block_q2_K *b = (const cuda_block_q2_K *)row + i / 256;
        const unsigned j = i % 256, sc = b->scales[j / 16];
        const unsigned q = (b->qs[j / 128 * 32 + j % 32] >> (2 * (j % 128 / 32))) & 3;
        return dev_f16_to_f32(b->d) * (sc & 15) * q - dev_f16_to_f32(b->dmin) * (sc >> 4);
    }
    if (TYPE == 12) {
        const cuda_block_q4_K *b = (const cuda_block_q4_K *)row + i / 256;
        const unsigned j = i % 256, group = j / 32;
        unsigned sc, mn;
        if (group < 4) { sc = b->scales[group] & 63; mn = b->scales[group + 4] & 63; }
        else {
            sc = (b->scales[group + 4] & 15) | ((b->scales[group - 4] >> 6) << 4);
            mn = (b->scales[group + 4] >> 4) | ((b->scales[group] >> 6) << 4);
        }
        const unsigned q = (b->qs[j / 64 * 32 + j % 32] >> (4 * (group & 1))) & 15;
        return dev_f16_to_f32(b->d) * sc * q - dev_f16_to_f32(b->dmin) * mn;
    }
    if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row + i / 256;
        const unsigned j = i % 256, group = j / 32, sub = j % 32 / 8;
        const uint16_t *p = b->qs + group * 4;
        const unsigned grid_ids = (unsigned)p[0] | ((unsigned)p[1] << 16);
        const unsigned signs_scale = (unsigned)p[2] | ((unsigned)p[3] << 16);
        const unsigned gi = (grid_ids >> (8 * sub)) & 255, si = (signs_scale >> (7 * sub)) & 127;
        const uint64_t grid = grid_table ? grid_table[gi] : cuda_iq2xxs_grid[gi];
        const unsigned signs = sign_table ? sign_table[si] : cuda_ksigns_iq2xs[si];
        float v = (float)((grid >> (8 * (j & 7))) & 255);
        if (signs & (1u << (j & 7))) v = -v;
        return dev_f16_to_f32(b->d) * (.5f + (signs_scale >> 28)) * .25f * v;
    }
    return 0;
}

__device__ __forceinline__ float scalar(const char *row, unsigned i, unsigned type) {
    switch (type) {
    case 0: return value<0>(row, i);
    case 1: return value<1>(row, i);
    case 2: return value<2>(row, i);
    case 8: return value<8>(row, i);
    case 30: return value<30>(row, i);
    case 39: return value<39>(row, i);
    default: return 0;
    }
}

template<unsigned TYPE>
__device__ __forceinline__ float4 value4(const char *row, unsigned i,
        const uint64_t *grid_table, const uint8_t *sign_table) {
    float4 out;
    float *v = (float *)&out;
    if (TYPE == 1) {
        const uint2 bits = *(const uint2 *)(row+i*2);
        const float2 a = __half22float2(*(__half2 *)&bits.x), b = __half22float2(*(__half2 *)&bits.y);
        out = make_float4(a.x,a.y,b.x,b.y);
    } else if (TYPE == 0) {
        out = *(const float4 *)(row+i*4);
    } else if (TYPE == 8) {
        const char *b = row+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        out = make_float4((float)(int8_t)q[0]*scale,(float)(int8_t)(q[0]>>8)*scale,
                         (float)(int8_t)q[1]*scale,(float)(int8_t)(q[1]>>8)*scale);
    } else if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row+(i/32)*17;
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0]<<23);
        unsigned packed;
        memcpy(&packed,b+1+i%16,4);
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const unsigned q = (packed>>(k*8+(i%32/16)*4))&15, mag = q&7;
            const float level = mag < 2 ? .5f*mag : __uint_as_float(((mag/2+126)<<23)|((mag&1)<<22));
            v[k] = (q&8 ? -level : level)*scale;
        }
    } else if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row+i/256;
        const unsigned j = i%256, sub = (j%32)/8;
        const uint16_t *p = b->qs+(j/32)*4;
        const unsigned ids = (unsigned)p[0]|((unsigned)p[1]<<16);
        const unsigned ss = (unsigned)p[2]|((unsigned)p[3]<<16);
        const uint64_t grid = grid_table[(ids>>(8*sub))&255];
        const unsigned signs = sign_table[(ss>>(7*sub))&127];
        const float scale = dev_f16_to_f32(b->d)*(.5f+(ss>>28))*.25f;
        /* Apply four signs before conversion. IQ2's nonzero magnitudes keep
         * the packed negations from carrying into neighboring bytes. */
        const unsigned offset = j&7;
        const unsigned bits = (((signs>>offset)&15)*0x00204081u)&0x01010101u;
        const unsigned packed = (((unsigned)(grid>>(8*offset)))^(bits*255u))+bits;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            v[k] = scale*(float)(int8_t)(packed>>(8*k));
        }
    } else if (TYPE == 10 || TYPE == 12) {
        const unsigned j = i%256;
        unsigned sc, mn, qs, shift;
        float d, dm;
        if (TYPE == 10) {
            const cuda_block_q2_K *b = (const cuda_block_q2_K *)row+i/256;
            const unsigned s = b->scales[j/16];
            sc = s&15; mn = s>>4;
            d = dev_f16_to_f32(b->d); dm = dev_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/128)*32+j%32);
            shift = 2*((j%128)/32);
        } else {
            const cuda_block_q4_K *b = (const cuda_block_q4_K *)row+i/256;
            const unsigned group = j/32;
            if (group < 4) { sc = b->scales[group]&63; mn = b->scales[group+4]&63; }
            else {
                sc = (b->scales[group+4]&15)|((b->scales[group-4]>>6)<<4);
                mn = (b->scales[group+4]>>4)|((b->scales[group]>>6)<<4);
            }
            d = dev_f16_to_f32(b->d); dm = dev_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/64)*32+j%32);
            shift = 4*(group&1);
        }
        const float scale = d*sc, offset = dm*mn;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = scale*((qs>>(8*k+shift))&(TYPE == 10 ? 3 : 15))-offset;
    } else {
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = value<TYPE>(row,i+k,grid_table,sign_table);
    }
    return out;
}

static uint64_t expert_row_bytes(unsigned type, unsigned K) {
    if (type == 10 && K % 32) return 0;
    return row_bytes(type, type == 10 ? ((uint64_t)K + 255) / 256 * 256 : K);
}

__global__ void router(int *selected, float *weights, const float *logits,
        const float *x, const char *gate, float *shared_gate,
        unsigned NE, unsigned NS, unsigned K, unsigned type) {
    const unsigned t = blockIdx.x, tid = threadIdx.x;
    __shared__ float p[512], maxima[256], red[32];
    __shared__ unsigned ids[256];
    float mx = -FLT_MAX;
    for (unsigned e = tid; e < NE; e += 256) mx = fmaxf(mx, logits[(uint64_t)t * NE + e]);
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (tid < stride) maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
        __syncthreads();
    }
    mx = maxima[0];
    float ps = 0;
    for (unsigned e = tid; e < NE; e += 256) { p[e] = expf(logits[(uint64_t)t * NE + e] - mx); ps += p[e]; }
    const float total = block_sum(ps, red);
    for (unsigned e = tid; e < NE; e += 256) p[e] /= total;
    if (K) {
        float v = 0;
        for (unsigned i = tid; i < K; i += 256) v += scalar(gate, i, type) * x[(uint64_t)t * K + i];
        v = block_sum(v, red);
        if (!tid) shared_gate[t] = v;
    }
    __syncthreads();
    for (unsigned s = 0; s < NS; s++) {
        float best = -1;
        unsigned id = UINT_MAX;
        for (unsigned e = tid; e < NE; e += 256) if (p[e] > best) { best = p[e]; id = e; }
        maxima[tid] = best; ids[tid] = id;
        __syncthreads();
        for (unsigned stride = 128; stride; stride /= 2) {
            if (tid < stride && (maxima[tid + stride] > maxima[tid] ||
                (maxima[tid + stride] == maxima[tid] && ids[tid + stride] < ids[tid]))) {
                maxima[tid] = maxima[tid + stride]; ids[tid] = ids[tid + stride];
            }
            __syncthreads();
        }
        if (!tid) {
            selected[(uint64_t)t * NS + s] = ids[0];
            weights[(uint64_t)t * NS + s] = maxima[0];
            p[ids[0]] = -1;
        }
        __syncthreads();
    }
    if (tid < NS) {
        float denom = 0;
        for (unsigned s = 0; s < NS; s++) denom += weights[(uint64_t)t * NS + s];
        red[tid] = denom;
    }
    __syncthreads();
    if (tid < NS) weights[(uint64_t)t * NS + tid] /= red[tid];
}

/* The id a kernel computes a pair with: the router's expert through the
 * device's slot table when there is one (see moe_stream_table_row), -1 for
 * an expert this device does not hold.  One thread per pair leaves the
 * slot in mapped, which is what moe_merge reads. */
__device__ __forceinline__ int moe_pair_id(const int *selected, const int *table, int *mapped,
        uint64_t pair, unsigned NT, bool report) {
    int e = selected[pair];
    if (table) {
        e = (unsigned)e < NT ? table[e] : -1;
        /* Only the mid launch is given mapped; down maps the same ids. */
        if (report && mapped) mapped[pair] = e;
    }
    return e;
}

/* Four 2-bit values of a Q2_K word against x, each scale * q - min as
 * value4 forms it; q becomes a float by exponent bias, which is exact and
 * spares the conversion instruction. */
__device__ __forceinline__ float q2k_dot4(float acc, unsigned q, float scale, float offset, float4 xv) {
    const float *xs = (const float *)&xv;
    #pragma unroll
    for (unsigned k = 0; k < 4; k++) {
        const float qf = __uint_as_float(((q >> (8 * k)) & 3) | 0x4B000000u) - 8388608.0f;
        acc += (scale * qf - offset) * xs[k];
    }
    return acc;
}

/* A warp over one or two Q2_K rows: a lane takes one 32-bit word of a
 * block's qs, which holds 16 values under four scales, so a block header is
 * decoded once per 16 values rather than once per 4.  Sixteen lanes to a
 * block, two blocks to a step. */
template<bool TWO>
__device__ __forceinline__ void q2k_rows(const char *r0, const char *r1, const float *xt, unsigned K,
        float &a, float &b) {
    const unsigned lane = threadIdx.x & 31, w = lane & 15, h = w >> 3, m = w & 7;
    for (unsigned blk = lane >> 4; blk * 256 < K; blk += 2) {
        const char *b0 = r0 + blk * 84, *b1 = TWO ? r1 + blk * 84 : NULL;
        const unsigned dd0 = *(const unsigned *)(b0 + 80), q0 = *(const unsigned *)(b0 + 16 + 4 * w);
        const unsigned sa0 = *(const unsigned *)(b0 + 8 * h), sb0 = *(const unsigned *)(b0 + 8 * h + 4);
        const float d0 = dev_f16_to_f32((uint16_t)dd0), dm0 = dev_f16_to_f32((uint16_t)(dd0 >> 16));
        unsigned dd1 = 0, q1 = 0, sa1 = 0, sb1 = 0;
        float d1 = 0, dm1 = 0;
        if (TWO) {
            dd1 = *(const unsigned *)(b1 + 80); q1 = *(const unsigned *)(b1 + 16 + 4 * w);
            sa1 = *(const unsigned *)(b1 + 8 * h); sb1 = *(const unsigned *)(b1 + 8 * h + 4);
            d1 = dev_f16_to_f32((uint16_t)dd1); dm1 = dev_f16_to_f32((uint16_t)(dd1 >> 16));
        }
        #pragma unroll
        for (unsigned s = 0; s < 4; s++) {
            const unsigned j = blk * 256 + h * 128 + 32 * s + 4 * m;
            if (j < K) {
                const float4 xv = *(const float4 *)(xt + j);
                const unsigned pick = 0x4440u | (2 * (s & 1) + (m >> 2));
                const unsigned sc0 = __byte_perm(s < 2 ? sa0 : sb0, 0, pick);
                a = q2k_dot4(a, q0 >> (2 * s), d0 * (float)(sc0 & 15), dm0 * (float)(sc0 >> 4), xv);
                if (TWO) {
                    const unsigned sc1 = __byte_perm(s < 2 ? sa1 : sb1, 0, pick);
                    b = q2k_dot4(b, q1 >> (2 * s), d1 * (float)(sc1 & 15), dm1 * (float)(sc1 >> 4), xv);
                }
            }
        }
    }
}

/* Eight IQ2_XXS values, one grid entry, against x: the signed bytes value4
 * forms, each scale * byte, the byte a float by exponent bias. */
__device__ __forceinline__ float iq2_dot8(float acc, uint64_t grid, unsigned signs, float scale, const float *x) {
    #pragma unroll
    for (unsigned q = 0; q < 2; q++) {
        const unsigned bits = (((signs >> (4 * q)) & 15) * 0x00204081u) & 0x01010101u;
        const unsigned packed = (((unsigned)(grid >> (32 * q))) ^ (bits * 255u)) + bits;
        const unsigned biased = packed ^ 0x80808080u;
        const float4 xv = *(const float4 *)(x + 4 * q);
        const float *xs = (const float *)&xv;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const float v = __uint_as_float(__byte_perm(biased, 0x4B000000u, 0x7440u | k)) - 8388736.0f;
            acc += (scale * v) * xs[k];
        }
    }
    return acc;
}

/* A warp over one or two IQ2_XXS rows, 16 values a lane: the block scale
 * and a 32-value group's ids and signs are read once per 16 values rather
 * than once per 4. */
template<bool TWO>
__device__ __forceinline__ void iq2_rows(const char *r0, const char *r1, const float *xt, unsigned K,
        const uint64_t *grid_table, const uint8_t *sign_table, float &a, float &b) {
    const unsigned lane = threadIdx.x & 31, w = lane & 15, g = w >> 1, half = w & 1;
    for (unsigned blk = lane >> 4; blk * 256 < K; blk += 2) {
        const cuda_block_iq2_xxs *b0 = (const cuda_block_iq2_xxs *)(r0 + blk * 66);
        const cuda_block_iq2_xxs *b1 = TWO ? (const cuda_block_iq2_xxs *)(r1 + blk * 66) : NULL;
        const uint16_t *p0 = b0->qs + g * 4, *p1 = TWO ? b1->qs + g * 4 : NULL;
        const unsigned ids0 = p0[0] | (unsigned)p0[1] << 16, ss0 = p0[2] | (unsigned)p0[3] << 16;
        const float scale0 = dev_f16_to_f32(b0->d) * (.5f + (ss0 >> 28)) * .25f;
        unsigned ids1 = 0, ss1 = 0;
        float scale1 = 0;
        if (TWO) {
            ids1 = p1[0] | (unsigned)p1[1] << 16; ss1 = p1[2] | (unsigned)p1[3] << 16;
            scale1 = dev_f16_to_f32(b1->d) * (.5f + (ss1 >> 28)) * .25f;
        }
        #pragma unroll
        for (unsigned e = 0; e < 2; e++) {
            const unsigned sub = 2 * half + e;
            const float *xs = xt + blk * 256 + g * 32 + sub * 8;
            a = iq2_dot8(a, grid_table[(ids0 >> (8 * sub)) & 255], sign_table[(ss0 >> (7 * sub)) & 127], scale0, xs);
            if (TWO) b = iq2_dot8(b, grid_table[(ids1 >> (8 * sub)) & 255], sign_table[(ss1 >> (7 * sub)) & 127],
                                  scale1, xs);
        }
    }
}

/* A warp over one or two rows of any format through value4, four values a
 * lane a step, as moe_mv reads them. */
template<unsigned TYPE, bool TWO>
__device__ __forceinline__ void value4_rows(const char *r0, const char *r1, const float *xt, unsigned K,
        const uint64_t *grid_table, const uint8_t *sign_table, float &a, float &b) {
    for (unsigned i = (threadIdx.x & 31) * 4; i < K; i += 128) {
        const float4 xv = *(const float4 *)(xt + i), av = value4<TYPE>(r0, i, grid_table, sign_table);
        a += av.x * xv.x; a += av.y * xv.y; a += av.z * xv.z; a += av.w * xv.w;
        if (TWO) {
            const float4 bv = value4<TYPE>(r1, i, grid_table, sign_table);
            b += bv.x * xv.x; b += bv.y * xv.y; b += bv.z * xv.z; b += bv.w * xv.w;
        }
    }
}

/* The shared expert's rows in a format fixed per kernel row rather than
 * switched on per value: four values a lane where the format and the
 * alignment allow, one otherwise. */
template<unsigned ST, bool TWO>
__device__ __forceinline__ void plain_rows(const char *r0, const char *r1, const float *xt, unsigned K,
        float &a, float &b) {
    const unsigned lane = threadIdx.x & 31;
    const uintptr_t align = ST == 0 ? 15 : ST == 1 || ST == 30 ? 7 : 1;
    if ((ST == 0 || ST == 1 || ST == 8 || ST == 30 || ST == 39) && !(K & 3) && !((uintptr_t)xt & 15) &&
        !((uintptr_t)r0 & align) && (!TWO || !((uintptr_t)r1 & align))) {
        for (unsigned i = lane * 4; i < K; i += 128) {
            const float4 xv = *(const float4 *)(xt + i);
            float4 av, bv = make_float4(0, 0, 0, 0);
            if (ST == 30) {
                const uint2 w0 = *(const uint2 *)(r0 + i * 2);
                av = make_float4(__uint_as_float(w0.x << 16), __uint_as_float(w0.x & 0xffff0000u),
                                 __uint_as_float(w0.y << 16), __uint_as_float(w0.y & 0xffff0000u));
                if (TWO) {
                    const uint2 w1 = *(const uint2 *)(r1 + i * 2);
                    bv = make_float4(__uint_as_float(w1.x << 16), __uint_as_float(w1.x & 0xffff0000u),
                                     __uint_as_float(w1.y << 16), __uint_as_float(w1.y & 0xffff0000u));
                }
            } else {
                av = value4<ST>(r0, i, NULL, NULL);
                if (TWO) bv = value4<ST>(r1, i, NULL, NULL);
            }
            a += av.x * xv.x; a += av.y * xv.y; a += av.z * xv.z; a += av.w * xv.w;
            if (TWO) { b += bv.x * xv.x; b += bv.y * xv.y; b += bv.z * xv.z; b += bv.w * xv.w; }
        }
    } else {
        for (unsigned i = lane; i < K; i += 32) {
            const float xv = xt[i];
            a += value<ST>(r0, i) * xv;
            if (TWO) b += value<ST>(r1, i) * xv;
        }
    }
}

/* The routed experts' rows against x: eight rows to a block, the shared
 * expert through a path chosen once per row, Q2_K and IQ2_XXS decoded a
 * block header per 16 values. */
template<unsigned TYPE, bool DOWN>
__global__ void moe_mv_rows(float *out, const float *x, const int *selected, const int *table, int *mapped,
        const char *w0, const char *w1, const char *sh0, const char *sh1,
        unsigned shared_type, unsigned NE, unsigned NT, unsigned NS, unsigned K, unsigned M,
        uint64_t rb, uint64_t srb, uint64_t estride) {
    const unsigned row = blockIdx.x * 8 + threadIdx.x / 32, slot = blockIdx.y, t = blockIdx.z;
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    if (row >= M) return;
    const bool shared = slot == NS;
    const unsigned stride = NS + (shared_type != UINT_MAX);
    const uint64_t pair = (uint64_t)t * stride + slot;
    const float *xt = x + (DOWN ? pair : t) * K;
    float a = 0, b = 0;
    bool summed = false;
    if (shared) {
        const char *r0 = sh0 + row * srb, *r1 = DOWN ? NULL : sh1 + row * srb;
        switch (shared_type) {
        case 0: plain_rows<0, !DOWN>(r0, r1, xt, K, a, b); break;
        case 1: plain_rows<1, !DOWN>(r0, r1, xt, K, a, b); break;
        case 2: plain_rows<2, !DOWN>(r0, r1, xt, K, a, b); break;
        case 8: plain_rows<8, !DOWN>(r0, r1, xt, K, a, b); break;
        case 30: plain_rows<30, !DOWN>(r0, r1, xt, K, a, b); break;
        case 39: plain_rows<39, !DOWN>(r0, r1, xt, K, a, b); break;
        default: break;
        }
    } else {
        const int e = moe_pair_id(selected, table, mapped, (uint64_t)t * NS + slot, NT, !blockIdx.x && !threadIdx.x);
        if (e >= 0 && (unsigned)e < NE) {
            const uint64_t off = (uint64_t)e * estride + row * rb;
            const char *r0 = w0 + off, *r1 = DOWN ? NULL : w1 + off;
            const bool aligned = !((uintptr_t)xt & 15);
            if (TYPE == 10 && aligned) q2k_rows<!DOWN>(r0, r1, xt, K, a, b);
            else if (TYPE == 16 && aligned) iq2_rows<!DOWN>(r0, r1, xt, K, grid_table, sign_table, a, b);
            else if ((TYPE == 12 || TYPE == 39) && aligned)
                value4_rows<TYPE, !DOWN>(r0, r1, xt, K, grid_table, sign_table, a, b);
            else {
                a = dot<TYPE>(r0, xt, K, grid_table, sign_table);
                if (!DOWN) b = dot<TYPE>(r1, xt, K, grid_table, sign_table);
                summed = true;
            }
        }
    }
    if (!summed) { a = sum(a); b = sum(b); }
    if (!(threadIdx.x & 31)) out[pair * M + row] = DOWN ? a : silu(a) * b;
}

/* The rows another device or the CPU computed, into the slots this device's
 * own kernel left at zero because it does not hold their experts. */
__global__ void moe_merge(float *part, const float *ext, const int *selected, unsigned NS, unsigned stride,
        unsigned M) {
    const unsigned slot = blockIdx.y, t = blockIdx.z, d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t pair = (uint64_t)t * NS + slot;
    if (d < M && selected[pair] < 0) part[((uint64_t)t * stride + slot) * M + d] = ext[pair * M + d];
}

/* Arithmetic for its own sake: see moe_stream_keep_awake. */
__global__ void keep_awake(unsigned iterations, unsigned *sink) {
    unsigned v = threadIdx.x;
    for (unsigned i = 0; i < iterations; i++) v = v * 1664525u + 1013904223u;
    if (!threadIdx.x) *sink = v;
}

static int moe_mv_dispatch(float *out, const float *x, const int *sel,
        const char *w0, const char *w1, const char *s0, const char *s1,
        unsigned type, unsigned st, unsigned NE, unsigned T, unsigned NS, unsigned K, unsigned M, bool down,
        uint64_t expert_stride = 0, const int *table = NULL, int *mapped = NULL, unsigned NT = 0) {
    /* expert_stride is the distance between two experts' first rows where it
     * is not one expert's rows: the streaming arena keeps gate, up and down
     * of an expert together.  table, when given, maps sel through
     * moe_pair_id and the kernel leaves the slots in mapped. */
    const uint64_t rb = expert_row_bytes(type, K), srb = row_bytes(st, K);
    const uint64_t es = expert_stride ? expert_stride : (uint64_t)M * rb;
    const dim3 rows((M + 7) / 8, NS + (st != UINT_MAX), T);
#define QWEN_MOE_ARGS out,x,sel,table,mapped,w0,w1,s0,s1,st,NE,NT,NS,K,M,rb,srb,es
#define QWEN_MOE(TYPE) case TYPE: \
    if (down) moe_mv_rows<TYPE, true><<<rows,256,0,cuda_decode_stream()>>>(QWEN_MOE_ARGS); \
    else moe_mv_rows<TYPE, false><<<rows,256,0,cuda_decode_stream()>>>(QWEN_MOE_ARGS); \
    break
    switch (type) {
    QWEN_MOE(0); QWEN_MOE(1); QWEN_MOE(2); QWEN_MOE(8); QWEN_MOE(10);
    QWEN_MOE(12); QWEN_MOE(16); QWEN_MOE(30); QWEN_MOE(39);
    default: return 0;
    }
#undef QWEN_MOE
#undef QWEN_MOE_ARGS
    return launched();
}

__global__ void moe_reduce(float *out, float *R, const float *inj, const float *part,
        const float *weights, const float *gate, const float *shared,
        unsigned NS, unsigned stride, unsigned D, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    __shared__ float inject[4];
    if (threadIdx.x < hc) inject[threadIdx.x] = injection(inj + (uint64_t)t * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= D) return;
    float v = 0;
    for (unsigned s = 0; s < NS; s++) v += weights[(uint64_t)t * NS + s] * part[((uint64_t)t * stride + s) * D + d];
    if (gate) v += sigmoid(gate[t]) * (shared ? shared[(uint64_t)t * D + d] : part[((uint64_t)t * stride + NS) * D + d]);
    out[(uint64_t)t * D + d] = v;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)t * hc + s) * D + d] += inject[s] * v;
}

__global__ void expert_lists(int *lists, int *counts, const int *selected,
        unsigned pairs, unsigned NE, unsigned cap) {
    __shared__ unsigned counters[512];
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counters[e] = 0;
    __syncthreads();
    for (unsigned p = threadIdx.x; p < pairs; p += blockDim.x) {
        const unsigned e = (unsigned)selected[p];
        if (e < NE) {
            const unsigned i = atomicAdd(counters + e, 1u);
            if (i < cap) lists[(uint64_t)e * cap + i] = p;
        }
    }
    __syncthreads();
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counts[e] = min(counters[e], cap);
}

/* Tiles share dequantized weights across 16 tokens without requantizing the
 * activations. Expert counts determine the work; no expert list is truncated. */
template<unsigned TYPE, bool DOWN, bool EXPERT>
__global__ void matrix(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned T, unsigned NS, unsigned NO,
        unsigned K, unsigned M, unsigned cap, uint64_t rb) {
    const unsigned r = threadIdx.x % 16, c = threadIdx.x / 16, e = blockIdx.y;
    const unsigned row = blockIdx.x * 16 + r;
    const unsigned count = EXPERT ? (unsigned)counts[e] : T;
    __shared__ float a[16][33], b[16][33], v[16][33];
    for (unsigned t0 = blockIdx.z * 16; t0 < count; t0 += gridDim.z * 16) {
        const unsigned item = t0 + c;
        const int pair = EXPERT && item < count ? lists[(uint64_t)e * cap + item] : (int)item;
        const unsigned tok = EXPERT ? pair / NS : item, slot = EXPERT ? pair % NS : 0;
        float acc = 0, up = 0;
        for (unsigned k0 = 0; k0 < K; k0 += 32) {
            for (unsigned j = threadIdx.x; j < 16 * 32; j += 256) {
                const unsigned rr = j / 32, kk = j % 32, global_row = blockIdx.x * 16 + rr;
                const unsigned logical = k0 + kk;
                const char *wr = w0 + ((uint64_t)(EXPERT ? e : 0) * M + global_row) * rb;
                a[rr][kk] = global_row < M && logical < K ? value<TYPE>(wr, logical) : 0;
                if (EXPERT && !DOWN) {
                    const char *ur = w1 + ((uint64_t)e * M + global_row) * rb;
                    b[rr][kk] = global_row < M && logical < K ? value<TYPE>(ur, logical) : 0;
                }
                const unsigned ti = t0 + rr;
                const int pp = EXPERT && ti < count ? lists[(uint64_t)e * cap + ti] : (int)ti;
                const unsigned tt = EXPERT ? pp / NS : ti, ss = EXPERT ? pp % NS : 0;
                const uint64_t xx = DOWN && EXPERT ? (uint64_t)tt * NO + ss : tt;
                v[rr][kk] = ti < count && logical < K ? x[xx * K + logical] : 0;
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < 32; k++) {
                acc += a[r][k] * v[c][k];
                if (EXPERT && !DOWN) up += b[r][k] * v[c][k];
            }
            __syncthreads();
        }
        if (row < M && item < count) {
            const uint64_t op = EXPERT ? (uint64_t)tok * NO + slot : item;
            out[op * M + row] = EXPERT && !DOWN ? silu(acc) * up : acc;
        }
    }
}

__global__ void expert_tiles(unsigned *prefix, const int *counts, unsigned NE, unsigned nt) {
    unsigned n = 0;
    prefix[0] = 0;
    for (unsigned e = 0; e < NE; e++) {
        n += ((unsigned)counts[e]+nt-1)/nt;
        prefix[e+1] = n;
    }
}

/* Two TF32 components retain the fine part of each FP32 operand. Keep the
 * correction products in their own accumulator: repeatedly adding them to
 * the much larger leading product loses their precision on tensor cores. */
template<unsigned TYPE>
__global__ void matrix_tc(float *out, const float *x, const char *w0,
        unsigned T, unsigned K, unsigned M, uint64_t rb) {
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32, r0 = blockIdx.x*64;
    /* Padding separates the banks of adjacent rows in the TF32 fragment
     * loads. The leading planes also hold the completed output tiles. */
    __shared__ __align__(32) float ah[64][36], al[64][36];
    __shared__ __align__(32) float bh[16][36], bl[16][36];
    for (unsigned t0 = blockIdx.z*16; t0 < T; t0 += gridDim.z*16) {
        wm::fragment<wm::accumulator,16,16,8,float> acc, corr, total;
        wm::fill_fragment(acc,0);
        wm::fill_fragment(corr,0);
        wm::fill_fragment(total,0);
        for (unsigned k0 = 0; k0 < K; k0 += 32) {
            for (unsigned j = tid; j < 64*32; j += 128) {
                const unsigned r = j/32, kk = j%32, row = r0+r, col = k0+kk;
                const uint64_t off = (uint64_t)row*rb;
                const float a = row < M && col < K ? value<TYPE>(w0+off,col) : 0;
                ah[r][kk] = wm::__float_to_tf32(a);
                al[r][kk] = wm::__float_to_tf32(a-ah[r][kk]);
            }
            for (unsigned j = tid; j < 16*32; j += 128) {
                const unsigned item = t0+j/32, kk = j%32, col = k0+kk;
                const float b = item < T && col < K ? x[(uint64_t)item*K+col] : 0;
                bh[j/32][kk] = wm::__float_to_tf32(b);
                bl[j/32][kk] = wm::__float_to_tf32(b-bh[j/32][kk]);
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < 32; k += 8) {
                wm::fragment<wm::matrix_a,16,16,8,wm::precision::tf32,wm::row_major> a, alo;
                wm::fragment<wm::matrix_b,16,16,8,wm::precision::tf32,wm::col_major> b, blo;
                wm::load_matrix_sync(a,&ah[warp*16][k],36);
                wm::load_matrix_sync(alo,&al[warp*16][k],36);
                wm::load_matrix_sync(b,&bh[0][k],36);
                wm::load_matrix_sync(blo,&bl[0][k],36);
                wm::mma_sync(corr,alo,blo,corr);
                wm::mma_sync(corr,alo,b,corr);
                wm::mma_sync(corr,a,blo,corr);
                wm::mma_sync(acc,a,b,acc);
            }
            __syncthreads();
            /* Bound tensor-core accumulation depth. CUDA FP32 adds the
             * partial sums, avoiding drift on long HC projection rows. */
            if ((k0+32)%256 == 0 || k0+32 >= K) {
                for (unsigned i = 0; i < acc.num_elements; i++) {
                    total.x[i] += acc.x[i] + corr.x[i];
                }
                wm::fill_fragment(acc,0);
                wm::fill_fragment(corr,0);
            }
        }
        wm::store_matrix_sync(&ah[warp*16][0],total,36,wm::mem_row_major);
        __syncthreads();
        for (unsigned j = tid; j < 64*16; j += 128) {
            const unsigned row = r0+j/16, item = t0+j%16;
            if (row < M && item < T) out[(uint64_t)item*M+row] = ah[j/16][j%16];
        }
        __syncthreads();
    }
#endif
}

__device__ __forceinline__ void mma_tf32(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

template<unsigned TYPE, bool DOWN, unsigned NC>
__global__ void matrix_reg(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
#if __CUDA_ARCH__ >= 800
    const unsigned lane = threadIdx.x&31, warp = threadIdx.x/32;
    const unsigned row_tiles = (M+31)/32, job = blockIdx.x/row_tiles;
    if (job >= tiles[NE]) return;
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    unsigned e = 0, end = NE;
    while (e < end) {
        const unsigned mid = (e+end)/2;
        if (tiles[mid+1] <= job) e = mid+1;
        else end = mid;
    }
    const unsigned count = counts[e], t0 = (job-tiles[e])*16*NC;
    const unsigned r0 = (blockIdx.x%row_tiles)*32+(warp%2)*16, c0 = t0+(warp/2)*8*NC;
    if (r0 >= M || c0 >= count) return;
    float acc[NC][4] = {}, corr[NC][4] = {}, total[NC][4] = {};
    float uacc[NC][4] = {}, ucorr[NC][4] = {}, utotal[NC][4] = {};
    const float *xp[NC];
    #pragma unroll
    for (unsigned nc = 0; nc < NC; nc++) {
        const unsigned item = c0+nc*8+lane/4;
        const unsigned pair = item < count ? lists[(uint64_t)e*cap+item] : 0;
        xp[nc] = x+(DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS)*K;
    }
    for (unsigned k0 = 0; k0 < K; k0 += 8) {
        unsigned a[4], al[4], u[4], ul[4], b[2], bl[2];
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned r = r0+lane/4+(i%2)*8, k = k0+lane%4+(i/2)*4;
            const uint64_t off = ((uint64_t)e*M+r)*rb;
            const float v = r < M && k < K ? value<TYPE>(w0+off,k,grid_table,sign_table) : 0;
            const float hi = nvcuda::wmma::__float_to_tf32(v);
            a[i] = __float_as_uint(hi);
            al[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            if (!DOWN) {
                const float v = r < M && k < K ? value<TYPE>(w1+off,k,grid_table,sign_table) : 0;
                const float hi = nvcuda::wmma::__float_to_tf32(v);
                u[i] = __float_as_uint(hi);
                ul[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            }
        }
        #pragma unroll
        for (unsigned nc = 0; nc < NC; nc++) {
            if (c0+nc*8 >= count) continue;
            const unsigned item = c0+nc*8+lane/4;
            #pragma unroll
            for (unsigned i = 0; i < 2; i++) {
                const unsigned k = k0+lane%4+i*4;
                const float v = item < count && k < K ? xp[nc][k] : 0;
                const float hi = nvcuda::wmma::__float_to_tf32(v);
                b[i] = __float_as_uint(hi);
                bl[i] = __float_as_uint(nvcuda::wmma::__float_to_tf32(v-hi));
            }
            mma_tf32(corr[nc],al,bl);
            mma_tf32(corr[nc],al,b);
            mma_tf32(corr[nc],a,bl);
            mma_tf32(acc[nc],a,b);
            if (!DOWN) {
                mma_tf32(ucorr[nc],ul,bl);
                mma_tf32(ucorr[nc],ul,b);
                mma_tf32(ucorr[nc],u,bl);
                mma_tf32(uacc[nc],u,b);
            }
            if ((k0+8)%256 == 0 || k0+8 >= K) {
                #pragma unroll
                for (unsigned i = 0; i < 4; i++) {
                    total[nc][i] += acc[nc][i]+corr[nc][i];
                    acc[nc][i] = corr[nc][i] = 0;
                    if (!DOWN) { utotal[nc][i] += uacc[nc][i]+ucorr[nc][i]; uacc[nc][i] = ucorr[nc][i] = 0; }
                }
            }
        }
    }
    #pragma unroll
    for (unsigned nc = 0; nc < NC; nc++) {
        #pragma unroll
        for (unsigned i = 0; i < 4; i++) {
            const unsigned row = r0+lane/4+(i/2)*8, item = c0+nc*8+(lane%4)*2+(i%2);
            if (row < M && item < count) {
                const unsigned pair = lists[(uint64_t)e*cap+item];
                out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row] = DOWN ? total[nc][i] : silu(total[nc][i])*utotal[nc][i];
            }
        }
    }
#endif
}

/* Match the half operands of the production Metal expert tiles. Products
 * accumulate in FP32; only the matrix operands are rounded to half. */
template<unsigned TYPE, bool DOWN, unsigned NT, unsigned NR = 64>
__global__ void matrix_half_tile(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32, nr = (M+NR-1)/NR, job = blockIdx.x/nr;
    if (job >= tiles[NE]) return;
    unsigned e = 0, end = NE;
    while (e < end) {
        const unsigned mid = (e+end)/2;
        if (tiles[mid+1] <= job) e = mid+1; else end = mid;
    }
    const unsigned count = counts[e], t0 = (job-tiles[e])*NT, r0 = (blockIdx.x%nr)*NR;
    const unsigned wr = warp%(NR/16), wc = warp/(NR/16);
    constexpr unsigned NC = NT/(8/(NR/16));
    union Tile {
        __half ab[(NR+(DOWN ? 0 : NR)+NT)*72];
        float c[NR][NT];
    };
    __shared__ __align__(32) Tile tile;
    __half (*a)[72] = (__half (*)[72])tile.ab;
    __half (*u)[72] = a+NR;
    __half (*b)[72] = a+NR+(DOWN ? 0 : NR);
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = tid; i < 256; i += 256) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = tid; i < 128; i += 256) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    wm::fragment<wm::accumulator,16,16,16,float> acc[NC/16], up[NC/16];
    #pragma unroll
    for (unsigned j = 0; j < NC/16; j++) {
        wm::fill_fragment(acc[j],0);
        if (!DOWN) wm::fill_fragment(up[j],0);
    }
    for (unsigned k0 = 0; k0 < K; k0 += 64) {
        for (unsigned i = tid*4; i < NR*64; i += 256*4) {
            const unsigned row = r0+i/64, k = k0+i%64;
            const uint64_t off = ((uint64_t)e*M+row)*rb;
            const float4 av = row < M && k < K ? value4<TYPE>(w0+off,k,grid_table,sign_table) : make_float4(0,0,0,0);
            *(__half2 *)&a[i/64][i%64] = __floats2half2_rn(av.x,av.y);
            *(__half2 *)&a[i/64][i%64+2] = __floats2half2_rn(av.z,av.w);
            if (!DOWN) {
                const float4 uv = row < M && k < K ? value4<TYPE>(w1+off,k,grid_table,sign_table) : make_float4(0,0,0,0);
                *(__half2 *)&u[i/64][i%64] = __floats2half2_rn(uv.x,uv.y);
                *(__half2 *)&u[i/64][i%64+2] = __floats2half2_rn(uv.z,uv.w);
            }
        }
        for (unsigned i = tid; i < NT*64; i += 256) {
            const unsigned item = t0+i/64, k = k0+i%64;
            const unsigned pair = item < count ? lists[(uint64_t)e*cap+item] : 0;
            const uint64_t row = DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS;
            b[i/64][i%64] = __float2half_rn(item < count && k < K ? x[row*K+k] : 0);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 64; k += 16) {
            wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> af, uf;
            wm::load_matrix_sync(af,&a[wr*16][k],72);
            if (!DOWN) wm::load_matrix_sync(uf,&u[wr*16][k],72);
            #pragma unroll
            for (unsigned j = 0; j < NC/16; j++) {
                wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> bf;
                wm::load_matrix_sync(bf,&b[wc*NC+j*16][k],72);
                wm::mma_sync(acc[j],af,bf,acc[j]);
                if (!DOWN) wm::mma_sync(up[j],uf,bf,up[j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned j = 0; j < NC/16; j++) {
        if (!DOWN) for (unsigned i = 0; i < acc[j].num_elements; i++)
            acc[j].x[i] = silu(acc[j].x[i])*up[j].x[i];
        wm::store_matrix_sync(&tile.c[wr*16][wc*NC+j*16],acc[j],NT,wm::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = tid; i < NR*NT; i += 256) {
        const unsigned row = r0+i/NT, item = t0+i%NT;
        if (row < M && item < count) {
            const unsigned pair = lists[(uint64_t)e*cap+item];
            out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row] = tile.c[i/NT][i%NT];
        }
    }
#endif
}

static int matrix_dispatch(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned type, unsigned NE, unsigned T,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, bool down,
        unsigned *own_tiles = NULL) {
    /* own_tiles: NE+1 words on the current device, for callers that run on a
     * device other than the one the shared temporary belongs to. */
    const dim3 grid((M + 15) / 16, lists ? NE : 1, std::min(8u, (T + 15) / 16));
    const uint64_t rb = expert_row_bytes(type, K);
    if (ds4_cuda_attn_tokentile_arch_ok() && !getenv("DS4_CUDA_NO_TF32")) {
        if (lists && !g_quality_mode && (type == 16 || type == 10 || type == 12 || type == 39)) {
            const unsigned nt = T >= 2048 ? 64 : 32;
            unsigned *tiles = own_tiles ? own_tiles :
                              (unsigned *)tmp_alloc(((uint64_t)NE+1)*4,"Qwen expert tiles");
            if (!tiles) return 0;
            expert_tiles<<<1,1,0,cuda_decode_stream()>>>(tiles,counts,NE,nt);
            if (!launched()) return 0;
            /* Wider rows reuse activations; Q2_K down is faster at 64 rows. */
            const unsigned nr = nt == 64 && type != 10 ? 128 : 64;
            const uint64_t blocks = (((uint64_t)T*NS+nt-1)/nt+NE)*((M+nr-1)/nr);
            if (blocks > INT_MAX) return 0;
#define QWEN_HALF(TYPE, DOWN) \
            if (nt == 64) matrix_half_tile<TYPE,DOWN,64,(TYPE == 10 ? 64 : 128)><<<blocks,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
            else matrix_half_tile<TYPE,DOWN,32><<<blocks,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb)
#define QWEN_HALF_TYPE(TYPE) case TYPE: if (down) { QWEN_HALF(TYPE,true); } else { QWEN_HALF(TYPE,false); } break
            switch (type) { QWEN_HALF_TYPE(16); QWEN_HALF_TYPE(10); QWEN_HALF_TYPE(12); QWEN_HALF_TYPE(39); }
#undef QWEN_HALF_TYPE
#undef QWEN_HALF
            return launched();
        }
        unsigned *tiles = NULL;
        dim3 tcgrid((M+63)/64,1,std::min(8u,(T+15)/16));
        if (lists) {
            tiles = own_tiles ? own_tiles :
                    (unsigned *)tmp_alloc(((uint64_t)NE+1)*4,"Qwen expert tile schedule");
            if (!tiles) return 0;
            expert_tiles<<<1,1,0,cuda_decode_stream()>>>(tiles,counts,NE,32);
            if (!launched()) return 0;
            const uint64_t jobs = ((uint64_t)T*NS+31)/32+NE;
            const uint64_t blocks = jobs*((M+31)/32);
            if (blocks > INT_MAX) return 0;
            tcgrid = dim3((unsigned)blocks,1,1);
        }
#define QWEN_TC(TYPE) case TYPE: \
        if (!lists) matrix_tc<TYPE><<<tcgrid,128,0,cuda_decode_stream()>>>(out,x,w0,T,K,M,rb); \
        else if (down) matrix_reg<TYPE,true,2><<<tcgrid,128,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
        else matrix_reg<TYPE,false,2><<<tcgrid,128,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); break
        switch (type) {
        QWEN_TC(0); QWEN_TC(1); QWEN_TC(2); QWEN_TC(8); QWEN_TC(10);
        QWEN_TC(12); QWEN_TC(16); QWEN_TC(30); QWEN_TC(39);
        default: return 0;
        }
#undef QWEN_TC
        return launched();
    }
#define QWEN_MM(TYPE) case TYPE: \
    if (!lists) matrix<TYPE,false,false><<<grid,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); \
    else if (down) matrix<TYPE,true,true><<<grid,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); \
    else matrix<TYPE,false,true><<<grid,256,0,cuda_decode_stream()>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); break
    switch (type) {
    QWEN_MM(0); QWEN_MM(1); QWEN_MM(2); QWEN_MM(8); QWEN_MM(10);
    QWEN_MM(12); QWEN_MM(16); QWEN_MM(30); QWEN_MM(39);
    default: return 0;
    }
#undef QWEN_MM
    return launched();
}

template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid, const uint8_t *signs) {
    float acc = 0;
    if (TYPE == 1 && !(n%4) && !((uintptr_t)x&15) && !((uintptr_t)row&7)) {
        for (unsigned i = (threadIdx.x&31)*4; i < n; i += 128) {
            const float4 w = value4<TYPE>(row,i,grid,signs), v = *(const float4 *)(x+i);
            acc += w.x*v.x; acc += w.y*v.y; acc += w.z*v.z; acc += w.w*v.w;
        }
    } else {
        for (unsigned i = threadIdx.x & 31; i < n; i += 32) acc += value<TYPE>(row, i, grid, signs) * x[i];
    }
    return sum(acc);
}

template<unsigned TYPE>
__global__ void matvec(float *out, const char *w, const float *x,
                       unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    if (row >= M) return;
    const float v = dot<TYPE>(w + row * stride, x + (uint64_t)tok * K, K);
    if (!(threadIdx.x & 31)) out[(uint64_t)tok * M + row] = v;
}

template<unsigned TYPE, unsigned ROWS>
__global__ void matvec_rows(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    float a[ROWS] = {};
    const char *wr = w+(uint64_t)row*stride;
    if (TYPE == 1 && !(K%4) && !((uintptr_t)x&15) && !((uintptr_t)wr&7)) {
        for (unsigned i = lane*4; i < K; i += 128) {
            const float4 v = value4<TYPE>(wr,i,NULL,NULL);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 xv = *(const float4 *)(x+(uint64_t)t*K+i);
                a[t] += v.x*xv.x; a[t] += v.y*xv.y; a[t] += v.z*xv.z; a[t] += v.w*xv.w;
            }
        }
    } else {
        for (unsigned i = lane; i < K; i += 32) {
            const float v = value<TYPE>(wr,i);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) a[t] += v*x[(uint64_t)t*K+i];
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(a[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

template<unsigned ROWS>
__global__ void matvec_q8(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    const char *wr = w+(uint64_t)row*stride;
    float acc[ROWS] = {};
    for (unsigned i = lane*4; i < K; i += 128) {
        const char *b = wr+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        const unsigned q01 = q[0], q23 = q[1];
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) {
            const float4 v = *(const float4 *)(x+(uint64_t)t*K+i);
            float part = (float)(int8_t)q01*v.x;
            part += (float)(int8_t)(q01>>8)*v.y;
            part += (float)(int8_t)q23*v.z;
            part += (float)(int8_t)(q23>>8)*v.w;
            acc[t] += part*scale;
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(acc[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

static int matvec_dispatch(float *out, const char *w, const float *x,
                           unsigned type, unsigned T, unsigned K, unsigned M) {
    const dim3 grid((M + 3) / 4, T);
    const uint64_t stride = row_bytes(type, K);
    if (!stride) return 0;
    if (type == 8 && T <= 8 && !((uintptr_t)x&15)) {
#define QWEN_Q8_ROWS(N) matvec_q8<N><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride)
        if (T == 1) { QWEN_Q8_ROWS(1); }
        else if (T == 2) { QWEN_Q8_ROWS(2); }
        else if (T <= 4) { QWEN_Q8_ROWS(4); }
        else { QWEN_Q8_ROWS(8); }
#undef QWEN_Q8_ROWS
        return launched();
    }
#define QWEN_MV(TYPE) case TYPE: \
    if (T == 2) matvec_rows<TYPE,2><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
    else if (T > 2 && T <= 4) matvec_rows<TYPE,4><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
    else if (T > 4 && T <= 8) matvec_rows<TYPE,8><<<(M+3)/4,128,0,cuda_decode_stream()>>>(out,w,x,T,K,M,stride); \
    else matvec<TYPE><<<grid,128,0,cuda_decode_stream()>>>(out,w,x,K,M,stride); break
    switch (type) {
    QWEN_MV(0); QWEN_MV(1); QWEN_MV(2); QWEN_MV(8); QWEN_MV(10);
    QWEN_MV(12); QWEN_MV(16); QWEN_MV(30); QWEN_MV(39);
    default: return 0;
    }
#undef QWEN_MV
    return launched();
}

template<unsigned TYPE>
__global__ void unpack(float *out, const char *w, unsigned K, unsigned M, uint64_t rb) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (uint64_t)K*M) out[i] = value<TYPE>(w+(i/K)*rb,i%K);
}

/* Power-of-two row scaling protects half range. Preserve the residual too,
 * so dense projections retain substantially more than half precision. */
template<unsigned TYPE>
__global__ void pack_half_components(float *scales, __half *hi, __half *lo,
        const char *x, unsigned K, uint64_t rb) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    __shared__ float maxima[256], inv;
    float mx = 0;
    for (unsigned k = tid; k < K; k += 256) mx = fmaxf(mx,fabsf(value<TYPE>(x+(uint64_t)row*rb,k)));
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned d = 128; d; d /= 2) {
        if (tid < d) maxima[tid] = fmaxf(maxima[tid],maxima[tid+d]);
        __syncthreads();
    }
    if (!tid) {
        const int e = maxima[0] > 0 ? max(-120,min(120,(int)((__float_as_uint(maxima[0])>>23)&255)-127)) : 0;
        inv = ldexpf(1,-e); scales[row] = ldexpf(1,e);
    }
    __syncthreads();
    for (unsigned k = tid; k < K; k += 256) {
        const float v = value<TYPE>(x+(uint64_t)row*rb,k)*inv;
        const __half h = __float2half_rn(v);
        hi[(uint64_t)row*K+k] = h;
        lo[(uint64_t)row*K+k] = __float2half_rn((v-__half2float(h))*4096);
    }
}

__global__ void dense_rescale(float *out, const float *scales,
        unsigned M, unsigned N) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*N) out[i] *= scales[i/M];
}

/* Short projections reuse the weights for both activation components.
 * Keep separate FP32 sums and the same scaled residual as the cuBLAS path. */
__global__ void dense_f16_components(float *out, const __half *xh, const __half *xl,
        const __half *w, const float *scales, unsigned T, unsigned K, unsigned M) {
#if __CUDA_ARCH__ >= 800
    namespace wm = nvcuda::wmma;
    const unsigned tid = threadIdx.x, warp = tid/32;
    const unsigned m0 = blockIdx.x*64, t0 = blockIdx.y*64;
    union Tile { __half data[3][64][72]; float result[64][64]; };
    __shared__ __align__(32) Tile tile;
    wm::fragment<wm::accumulator,16,16,16,float> hi[2], lo[2];
    for (unsigned j = 0; j < 2; j++) {
        wm::fill_fragment(hi[j],0);
        wm::fill_fragment(lo[j],0);
    }
    for (unsigned k0 = 0; k0 < K; k0 += 64) {
        for (unsigned i = tid*8; i < 4096; i += 2048) {
            const unsigned r = i/64, k = i%64;
            const bool valid_w = m0+r < M, valid_x = t0+r < T;
            tt_cp_async_16B(&tile.data[0][r][k],w+((uint64_t)(valid_w ? m0+r : 0))*K+k0+k,valid_w);
            tt_cp_async_16B(&tile.data[1][r][k],xh+((uint64_t)(valid_x ? t0+r : 0))*K+k0+k,valid_x);
            tt_cp_async_16B(&tile.data[2][r][k],xl+((uint64_t)(valid_x ? t0+r : 0))*K+k0+k,valid_x);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
        __syncthreads();
        for (unsigned k = 0; k < 64; k += 16) {
            wm::fragment<wm::matrix_a,16,16,16,__half,wm::row_major> a;
            wm::load_matrix_sync(a,&tile.data[0][(warp%4)*16][k],72);
            for (unsigned j = 0; j < 2; j++) {
                wm::fragment<wm::matrix_b,16,16,16,__half,wm::col_major> b, c;
                wm::load_matrix_sync(b,&tile.data[1][(warp/4)*32+j*16][k],72);
                wm::load_matrix_sync(c,&tile.data[2][(warp/4)*32+j*16][k],72);
                wm::mma_sync(hi[j],a,b,hi[j]);
                wm::mma_sync(lo[j],a,c,lo[j]);
            }
        }
        __syncthreads();
    }
    for (unsigned j = 0; j < 2; j++) {
        for (unsigned i = 0; i < hi[j].num_elements; i++) hi[j].x[i] += lo[j].x[i]*0x1p-12f;
        wm::store_matrix_sync(&tile.result[(warp%4)*16][(warp/4)*32+j*16],hi[j],64,wm::mem_row_major);
    }
    __syncthreads();
    for (unsigned i = tid; i < 4096; i += 256) {
        const unsigned m = m0+i%64, t = t0+i/64;
        if (m < M && t < T) out[(uint64_t)t*M+m] = tile.result[i%64][i/64]*scales[t];
    }
#endif
}

static int dense_f16_blas(float *out, const float *x, const __half *w,
        unsigned T, unsigned K, unsigned M) {
    const uint64_t xn = (uint64_t)T*K;
    __half *hi = (__half *)tmp_alloc((xn+T)*4, "Qwen F16 projection scratch");
    if (!hi) return 0;
    __half *lo = hi+xn;
    float *scales = (float *)(lo+xn);
    pack_half_components<0><<<T,256,0,cuda_decode_stream()>>>(scales,hi,lo,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    if (K <= 512 && K%64 == 0 && T >= 32 && ds4_cuda_attn_tokentile_arch_ok()) {
        dense_f16_components<<<dim3((M+63)/64,(T+63)/64),256,0,cuda_decode_stream()>>>(out,hi,lo,w,scales,T,K,M);
        return launched();
    }
    const float zero = 0, one = 1, low = 0x1p-12f;
    if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&low,w,CUDA_R_16F,K,lo,CUDA_R_16F,K,&zero,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 residual projection") ||
        !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&one,w,CUDA_R_16F,K,hi,CUDA_R_16F,K,&one,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 leading projection")) return 0;
    dense_rescale<<<((uint64_t)M*T+255)/256,256,0,cuda_decode_stream()>>>(out,scales,M,T);
    return launched();
}

__global__ void dense_rescale2(float *out, const float *xs, const float *ws,
        unsigned M, unsigned T, unsigned stride) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*T) {
        const uint64_t dst = (i/M)*stride+i%M;
        out[dst] = out[dst]*xs[i/M]*ws[i%M];
    }
}

static int dense_q8_blas(float *out, const float *x, const char *w,
        unsigned T, unsigned K, unsigned M) {
    const unsigned bound = (unsigned)std::max(UINT64_C(1),(UINT64_C(32)<<20)/((uint64_t)K*4));
    const unsigned tile = std::min(M,bound >= 64 ? bound/64*64 : bound);
    const uint64_t xn = (uint64_t)T*K, wn = (uint64_t)tile*K;
    __half *xh = (__half *)tmp_alloc((xn+wn+T+tile)*4,"Qwen Q8 projection scratch");
    if (!xh) return 0;
    __half *xl = xh+xn, *wh = xl+xn, *wl = wh+wn;
    float *xs = (float *)(wl+wn), *ws = xs+T;
    pack_half_components<0><<<T,256,0,cuda_decode_stream()>>>(xs,xh,xl,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    const float zero = 0, one = 1, low = 0x1p-12f;
    const uint64_t rb = row_bytes(8,K);
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        pack_half_components<8><<<n,256,0,cuda_decode_stream()>>>(ws,wh,wl,w+(uint64_t)r*rb,K,rb);
        if (!launched()) return 0;
        if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&low,wl,CUDA_R_16F,K,xh,CUDA_R_16F,K,&zero,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 weight residual") ||
            !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&low,wh,CUDA_R_16F,K,xl,CUDA_R_16F,K,&one,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 input residual") ||
            !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&one,wh,CUDA_R_16F,K,xh,CUDA_R_16F,K,&one,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 leading product")) return 0;
        dense_rescale2<<<((uint64_t)n*T+255)/256,256,0,cuda_decode_stream()>>>(out+r,xs,ws,n,T,M);
        if (!launched()) return 0;
    }
    return 1;
}

/* Bound the temporary weight expansion instead of retaining an FP32 copy
 * of each projection. cuBLAS uses FP32 math, including FP32 activations. */
static int dense_blas(float *out, const float *x, const char *w,
        unsigned type, unsigned T, unsigned K, unsigned M) {
    const unsigned tile = std::min(M, (unsigned)std::max(UINT64_C(1), (UINT64_C(64)*1024*1024)/((uint64_t)K*4)));
    float *scratch = type ? (float *)tmp_alloc((uint64_t)tile*K*4, "Qwen dense FP32 tile") : NULL;
    if (type && !scratch) return 0;
    const uint64_t rb = row_bytes(type,K);
    const float alpha = 1, beta = 0;
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        const float *wf = type ? scratch : (const float *)w+(uint64_t)r*K;
        if (type) {
#define QWEN_UNPACK(TYPE) case TYPE: unpack<TYPE><<<((uint64_t)n*K+255)/256,256,0,cuda_decode_stream()>>>(scratch,w+(uint64_t)r*rb,K,n,rb); break
            switch (type) {
            QWEN_UNPACK(1); QWEN_UNPACK(2); QWEN_UNPACK(8); QWEN_UNPACK(10);
            QWEN_UNPACK(12); QWEN_UNPACK(16); QWEN_UNPACK(30); QWEN_UNPACK(39);
            default: return 0;
            }
#undef QWEN_UNPACK
            if (!launched()) return 0;
        }
        if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&alpha,wf,CUDA_R_32F,K,x,CUDA_R_32F,K,&beta,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT), "Qwen FP32 projection")) return 0;
    }
    return 1;
}

__global__ void mtp_stage(float *cat, const float *e, const float *R,
        const float *ge, const float *gh, unsigned E, unsigned hc, float eps) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    const bool emb = row == 0;
    const unsigned n = emb ? E : E * hc;
    const float *rs = emb ? e : R;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < n; i += blockDim.x) ss += rs[i] * rs[i];
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    const float *src = emb ? e : R + (uint64_t)(row - 1) * E;
    const float *g = emb ? ge : gh + (uint64_t)(row - 1) * E;
    float *o = cat + (uint64_t)row * 2 * E;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        o[(emb ? 0 : E) + i] = src[i] * inv * g[i];
        o[(emb ? E : 0) + i] = 0;
    }
}

__global__ void mtp_combine(float *out, const float *proj, unsigned E, unsigned hc) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E * hc) out[i] = proj[i % E] + proj[E + i];
}

struct max_pair { float value; int index; };
template<bool FINISH>
__global__ void argmax(int *out, max_pair *scratch, const float *logits, unsigned n) {
    const unsigned tid = threadIdx.x;
    __shared__ max_pair best[256];
    max_pair v = {-1e30f, 0};
    for (unsigned i = (FINISH ? 0 : blockIdx.x * 4096) + tid;
         i < (FINISH ? n : min(n, (blockIdx.x + 1) * 4096)); i += 256) {
        const max_pair p = FINISH ? scratch[i] : max_pair{logits[i], (int)i};
        if (p.value > v.value || (p.value == v.value && p.index < v.index)) v = p;
    }
    best[tid] = v;
    __syncthreads();
    for (unsigned s = 128; s; s /= 2) {
        if (tid < s) {
            const max_pair p = best[tid + s], q = best[tid];
            if (p.value > q.value || (p.value == q.value && p.index < q.index)) best[tid] = p;
        }
        __syncthreads();
    }
    if (!tid) { if (FINISH) *out = best[0].index; else scratch[blockIdx.x] = best[0]; }
}

__global__ void vis_patch(float *x, const float *a, const float *b, const float *bias,
        const float *pos, unsigned N, unsigned E) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)N * E) x[i] = a[i] + b[i] + bias[i % E] + pos[i];
}

__global__ void vis_norm(float *out, const float *x, const float *w, const float *b, unsigned E, float eps) {
    const unsigned tid = threadIdx.x;
    const uint64_t base = (uint64_t)blockIdx.x * E;
    __shared__ float red[32];
    float v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) v += x[base + i];
    const float mean = block_sum(v, red) / E;
    v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { const float d = x[base + i] - mean; v += d*d; }
    const float inv = rsqrtf(block_sum(v, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];
}

__global__ void vis_qkv(float *q, float *k, float *v, const float *qkv, const float *bias,
        unsigned H, unsigned D, unsigned grid_w) {
    const unsigned row = blockIdx.x, tid = threadIdx.x, E = H*D, hd = D/2, qd = D/4;
    const unsigned blk = row/4, within = row%4, w2 = grid_w/2;
    const float hp = (blk/w2)*2 + within/2, wp = (blk%w2)*2 + within%2;
    const float *src = qkv + (uint64_t)row*3*E;
    const uint64_t off = (uint64_t)row*E;
    for (unsigned idx = tid; idx < 2*H*hd; idx += blockDim.x) {
        const unsigned part = idx/(H*hd), rem = idx%(H*hd), h = rem/hd, i = rem%hd;
        const unsigned base = part*E + h*D;
        const float x0 = src[base+i] + bias[base+i], x1 = src[base+i+hd] + bias[base+i+hd];
        const unsigned j = i < qd ? i : i-qd;
        const float theta = (i < qd ? hp : wp) * powf(10000.0f, -2.0f*j/hd);
        float s, c; sincosf(theta, &s, &c);
        float *dst = part ? k : q;
        dst[off+h*D+i] = x0*c - x1*s;
        dst[off+h*D+i+hd] = x0*s + x1*c;
    }
    for (unsigned i = tid; i < E; i += blockDim.x) v[off+i] = src[2*E+i] + bias[2*E+i];
}

__global__ void vis_attention(float *out, const float *q, const float *k, const float *v,
        unsigned N, unsigned H, unsigned D) {
    const unsigned row = blockIdx.x, h = blockIdx.y, lane = threadIdx.x, E = H*D;
    const uint64_t qb = (uint64_t)row*E + h*D;
    float query[3] = {}, acc[3] = {};
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) query[j] = q[qb+lane+j*32];
    float mx = -INFINITY, denom = 0;
    for (unsigned p = 0; p < N; p++) {
        const uint64_t kb = (uint64_t)p*E + h*D;
        float score = 0;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) score += query[j]*k[kb+lane+j*32];
        score = sum(score) * rsqrtf((float)D);
        const float nm = fmaxf(mx, score), old = expf(mx-nm), prob = expf(score-nm);
        denom = denom*old + prob;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) acc[j] = acc[j]*old + prob*v[kb+lane+j*32];
        mx = nm;
    }
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) out[qb+lane+j*32] = acc[j]/denom;
}

__global__ void vis_add(float *x, const float *add, const float *bias, unsigned N, unsigned E, unsigned mode) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (uint64_t)N*E) return;
    if (add) { x[i] += add[i] + bias[i%E]; return; }
    float t = x[i] + bias[i%E];
    if (!mode) t *= .5f * (1 + tanhf(fminf(30, fmaxf(-30, .7978845608f*(t+.044715f*t*t*t)))));
    else if (mode == 1) t *= .5f * (1 + erff(t*.70710678f));
    x[i] = t;
}

__global__ void hc_norm(float *xn, float *inj, const float *R, const float *gamma,
                        const char *wi, unsigned type, unsigned E, unsigned hc,
                        unsigned ni, float eps) {
    const unsigned stream = blockIdx.x / 8, chunk = blockIdx.x % 8, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, dim = E * hc;
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) ss += R[base + i] * R[base + i];
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    float acc[4] = {};
    const unsigned per = (E + 7) / 8, end = min(E, (chunk + 1) * per);
    for (unsigned i = chunk * per + tid; i < end; i += blockDim.x) {
        const float v = R[base + i] * inv * gamma[stream * E + i];
        xn[base + i] = v;
        for (unsigned j = 0; j < ni; j++) acc[j] += scalar(wi, j * dim + stream * E + i, type) * v;
    }
    for (unsigned j = 0; j < ni; j++) {
        const float v = block_sum(acc[j], red);
        if (!tid) inj[((uint64_t)tok * hc * 8 + stream * 8 + chunk) * ni + j] = v;
    }
}

/* Prefill has enough tokens to share one normalization across all eight
 * injection chunks. Each warp preserves the four partial sums of the
 * decode kernel, including their order, without rereading R eight times. */
template<unsigned TYPE>
__global__ void hc_norm_prefill(float *xn, float *inj, const float *R,
        const float *gamma, const char *wi, unsigned E, unsigned hc,
        unsigned ni, float eps) {
    const unsigned stream = blockIdx.x, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, lane = tid & 31, chunk = tid / 32;
    const unsigned dim = E * hc, per = (E + 7) / 8;
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    __shared__ float red[32];
    float ss = 0;
    if (tid < 128) for (unsigned i = tid; i < E; i += 128)
        ss += R[base+i] * R[base+i];
    const float inv = rsqrtf(block_sum(ss,red) / E + eps);
    float acc[4][4] = {};
    const unsigned end = min(E,(chunk+1)*per);
    #pragma unroll
    for (unsigned part = 0; part < 4; part++) {
        for (unsigned i = chunk*per+lane+part*32; i < end; i += 128) {
            const float v = R[base+i] * inv * gamma[stream*E+i];
            xn[base+i] = v;
            #pragma unroll
            for (unsigned j = 0; j < 4; j++) if (j < ni)
                acc[j][part] += value<TYPE>(wi,j*dim+stream*E+i) * v;
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; j++) if (j < ni) {
        float v = 0;
        #pragma unroll
        for (unsigned part = 0; part < 4; part++) v += sum(acc[j][part]);
        if (!lane) inj[((uint64_t)tok*hc*8+stream*8+chunk)*ni+j] = v;
    }
}

template<unsigned TYPE>
__global__ void hc_mix(float *out, const float *xn, const float *lo,
                       const char *up, unsigned E, unsigned hc, unsigned rank, uint64_t rb) {
    const unsigned d = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    __shared__ __align__(16) float activated[512];
    if (rank <= 512) {
        for (unsigned r = threadIdx.x; r < rank; r += blockDim.x)
            activated[r] = silu(lo[(uint64_t)tok*rank+r]/hc);
        __syncthreads();
    }
    if (d >= E) return;
    const unsigned lane = threadIdx.x & 31, stream = lane / 8, l = lane % 8;
    float a = 0;
    if (stream < hc) {
        const char *row = up+((uint64_t)stream*E+d)*rb;
        if (!(rank%4) && rank <= 512 &&
            !(TYPE == 0 ? (uintptr_t)row&15 : TYPE == 1 ? (uintptr_t)row&7 : 0)) {
            for (unsigned r = l*4; r < rank; r += 32) {
                const float4 w = value4<TYPE>(row,r,NULL,NULL), v = *(const float4 *)(activated+r);
                a += w.x*v.x; a += w.y*v.y; a += w.z*v.z; a += w.w*v.w;
            }
        } else {
            for (unsigned r = l; r < rank; r += 8)
                a += value<TYPE>(row,r) *
                    (rank <= 512 ? activated[r] : silu(lo[(uint64_t)tok * rank + r] / hc));
        }
    }
    for (unsigned off = 1; off <= 4; off *= 2) a += __shfl_xor_sync(0xffffffff, a, off);
    float v = stream < hc ? sigmoid(a) * xn[((uint64_t)tok * hc + stream) * E + d] : 0;
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 16);
    if (!lane) out[(uint64_t)tok * E + d] = v / hc;
}

__device__ float injection(const float *inj, unsigned hc, unsigned stream) {
    float v = 0;
    for (unsigned i = 0; i < hc * 8; i++) v += inj[i * hc + stream];
    return 2 * sigmoid(v / hc);
}

__global__ void hc_combine(float *R, const float *out, const float *inj, unsigned E, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    __shared__ float weights[4];
    if (threadIdx.x < hc) weights[threadIdx.x] = injection(inj + (uint64_t)tok * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= E) return;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)tok * hc + s) * E + d] += weights[s] * out[(uint64_t)tok * E + d];
}

__global__ void hc_lo(float *dst, const float *src, uint64_t n, unsigned hc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = silu(src[i] / hc);
}

__global__ void hc_rows(float *dst, const float *up, const float *xn, unsigned E, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= E) return;
    float v = 0;
    for (unsigned s = 0; s < hc; s++) {
        const uint64_t i = ((uint64_t)t * hc + s) * E + d;
        v += sigmoid(up[i]) * xn[i];
    }
    dst[(uint64_t)t * E + d] = v / hc;
}

/* Decode (T == 1) launches that share one activation between several weights
 * or carry the next elementwise step.  A row keeps the arithmetic of the
 * launch it came from: matvec_q8<1> for q8_0 rows, dot<TYPE> otherwise. */
__device__ __forceinline__ float dot_q8_row(const char *wr, const float *x, unsigned K) {
    float acc = 0;
    for (unsigned i = (threadIdx.x & 31) * 4; i < K; i += 128) {
        const char *b = wr + (i / 32) * 34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b + 2 + i % 32);
        const unsigned q01 = q[0], q23 = q[1];
        const float4 v = *(const float4 *)(x + i);
        float part = (float)(int8_t)q01 * v.x;
        part += (float)(int8_t)(q01 >> 8) * v.y;
        part += (float)(int8_t)q23 * v.z;
        part += (float)(int8_t)(q23 >> 8) * v.w;
        acc += part * scale;
    }
    return sum(acc);
}

__device__ __forceinline__ float row_dot(unsigned type, const char *row, const float *x, unsigned K) {
    switch (type) {
    case 0: return dot<0>(row, x, K);
    case 1: return dot<1>(row, x, K);
    case 2: return dot<2>(row, x, K);
    case 30: return dot<30>(row, x, K);
    default: return (uintptr_t)x & 15 ? dot<8>(row, x, K) : dot_q8_row(row, x, K);
    }
}

struct matvec_set {
    const char *w[5];
    float *out[5];
    uint64_t stride[5];
    unsigned rows[5], type[5], n;
};

__global__ void multi_matvec(matvec_set s, const float *x, unsigned K) {
    unsigned row = blockIdx.x * 4 + threadIdx.x / 32;
    const char *w = NULL;
    float *out = NULL;
    unsigned type = 0;
    uint64_t stride = 0;
    #pragma unroll
    for (unsigned i = 0; i < 5; i++) {
        if (i < s.n && !w) {
            if (row < s.rows[i]) { w = s.w[i]; out = s.out[i]; type = s.type[i]; stride = s.stride[i]; }
            else row -= s.rows[i];
        }
    }
    if (!w) return;
    const float v = row_dot(type, w + row * stride, x, K);
    if (!(threadIdx.x & 31)) out[row] = v;
}

/* The output projection with hc_combine behind it: a warp's row is the
 * element the combine adds to the hc streams. */
__global__ void matvec_combine(float *R, float *blk, const float *inj, const char *w, const float *x,
                               unsigned type, unsigned K, unsigned M, uint64_t stride, unsigned hc) {
    __shared__ float weights[4];
    if (threadIdx.x < hc) weights[threadIdx.x] = injection(inj, hc, threadIdx.x);
    __syncthreads();
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= M) return;
    const float v = row_dot(type, w + row * stride, x, K);
    if (threadIdx.x & 31) return;
    blk[row] = v;
    for (unsigned s = 0; s < hc; s++) R[(uint64_t)s * M + row] += weights[s] * v;
}

/* hc_norm and the low-rank projection of xn in one launch.  The first hc*8
 * blocks are hc_norm as above; each block after them owns four rows of the
 * projection and normalizes R again into shared memory, with the same
 * reductions, rather than wait for the xn those blocks write. */
__global__ void hc_norm_down(float *xn, float *inj, float *lo, const float *R, const float *gamma,
                             const char *wi, unsigned itype, const char *down, unsigned dtype, uint64_t rb,
                             unsigned E, unsigned hc, unsigned ni, unsigned rank, float eps) {
    extern __shared__ __align__(16) float xs[];
    __shared__ float red[32];
    const unsigned tid = threadIdx.x, dim = E * hc;
    if (blockIdx.x < hc * 8) {
        const unsigned stream = blockIdx.x / 8, chunk = blockIdx.x % 8;
        const uint64_t base = (uint64_t)stream * E;
        float ss = 0;
        for (unsigned i = tid; i < E; i += blockDim.x) ss += R[base + i] * R[base + i];
        const float inv = rsqrtf(block_sum(ss, red) / E + eps);
        float acc[4] = {};
        const unsigned per = (E + 7) / 8, end = min(E, (chunk + 1) * per);
        for (unsigned i = chunk * per + tid; i < end; i += blockDim.x) {
            const float v = R[base + i] * inv * gamma[stream * E + i];
            xn[base + i] = v;
            for (unsigned j = 0; j < ni; j++) acc[j] += scalar(wi, j * dim + stream * E + i, itype) * v;
        }
        for (unsigned j = 0; j < ni; j++) {
            const float v = block_sum(acc[j], red);
            if (!tid) inj[(stream * 8 + chunk) * ni + j] = v;
        }
        return;
    }
    for (unsigned s = 0; s < hc; s++) {
        const uint64_t base = (uint64_t)s * E;
        float ss = 0;
        for (unsigned i = tid; i < E; i += blockDim.x) ss += R[base + i] * R[base + i];
        const float inv = rsqrtf(block_sum(ss, red) / E + eps);
        for (unsigned i = tid; i < E; i += blockDim.x) xs[base + i] = R[base + i] * inv * gamma[s * E + i];
    }
    __syncthreads();
    const unsigned row = (blockIdx.x - hc * 8) * 4 + tid / 32;
    if (row >= rank) return;
    const float v = row_dot(dtype, down + row * rb, xs, dim);
    if (!(tid & 31)) lo[row] = v;
}

} // namespace qwen4_cuda

namespace qwen4_cuda {

__global__ void conv(float *x, float *history, const float *w, unsigned T,
                     unsigned C, unsigned K, bool activate,
                     float *snap, unsigned snap_t, float *snap2, unsigned snap2_t) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[3], taps[4];
    for (unsigned i = 0; i < K - 1; i++) win[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = w[c * K + i];
    for (unsigned t = 0; t < T; t++) {
        const uint64_t pos = (uint64_t)t * C + c;
        const float raw = x[pos];
        float v = taps[K - 1] * raw;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * win[i];
        for (unsigned i = 0; i + 2 < K; i++) win[i] = win[i + 1];
        win[K - 2] = raw;
        x[pos] = activate ? silu(v) : v;
        if (snap && t == snap_t) for (unsigned i = 0; i < K - 1; i++) snap[(uint64_t)i * C + c] = win[i];
        if (snap2 && t == snap2_t) for (unsigned i = 0; i < K - 1; i++) snap2[(uint64_t)i * C + c] = win[i];
    }
    for (unsigned i = 0; i < K - 1; i++) history[(uint64_t)i * C + c] = win[i];
}

__global__ void gdn_prep(float *qkv, float *a, float *b, const float *A, const float *bias,
                         unsigned Hk, unsigned Hv, unsigned D) {
    const unsigned h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const unsigned C = (2 * Hk + Hv) * D, npt = D / 32;
    float *q = qkv + (uint64_t)t * C + h * D + lane * npt, *k = q + Hk * D;
    float qs = 0, ks = 0;
    for (unsigned i = 0; i < npt; i++) { qs += q[i] * q[i]; ks += k[i] * k[i]; }
    qs = rsqrtf(sum(qs) + 1e-6f) * rsqrtf((float)D);
    ks = rsqrtf(sum(ks) + 1e-6f);
    for (unsigned i = 0; i < npt; i++) { q[i] *= qs; k[i] *= ks; }
    if (!h) for (unsigned j = lane; j < Hv; j += 32) {
        const uint64_t p = (uint64_t)t * Hv + j;
        a[p] = expf(A[j] * softplus(a[p] + bias[j]));
        b[p] = sigmoid(b[p]);
    }
}

/* One warp owns a state row across the whole chunk. In particular, MTP
 * snapshots contain the state after the requested token, not the final row. */
template<unsigned ROWS, unsigned D>
__global__ void gdn_scan(float *out, float *state, const float *qkv,
                         const float *a, const float *b, unsigned T, unsigned Hk,
                         unsigned Hv, float *snap, unsigned st,
                         float *snap2, unsigned st2) {
    const unsigned dv = (blockIdx.x * 4 + threadIdx.x / 32)*ROWS, h = blockIdx.y;
    if (dv >= D) return;
    const unsigned npt = D / 32, k0 = (threadIdx.x & 31) * npt, kh = h % Hk;
    const unsigned C = (2 * Hk + Hv) * D;
    const uint64_t idx = ((uint64_t)h * D + dv) * D + k0;
    float s[ROWS][4];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) s[r][i] = state[idx+(uint64_t)r*D+i];
    for (unsigned t = 0; t < T; t++) {
        const float *q = qkv + (uint64_t)t * C + kh * D + k0, *k = q + Hk * D;
        const float decay = a[(uint64_t)t * Hv + h], beta = b[(uint64_t)t * Hv + h];
        #pragma unroll
        for (unsigned r = 0; r < ROWS; r++) {
            const float v = qkv[(uint64_t)t*C+2*Hk*D+h*D+dv+r];
            float u = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] *= decay; u += s[r][i]*k[i]; }
            const float delta = (v-sum(u))*beta;
            float o = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] += k[i]*delta; o += s[r][i]*q[i]; }
            o = sum(o);
            if (!(threadIdx.x&31)) out[((uint64_t)t*Hv+h)*D+dv+r] = o;
            if (snap && t == st) for (unsigned i = 0; i < npt; i++) snap[idx+(uint64_t)r*D+i] = s[r][i];
            if (snap2 && t == st2) for (unsigned i = 0; i < npt; i++) snap2[idx+(uint64_t)r*D+i] = s[r][i];
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) state[idx+(uint64_t)r*D+i] = s[r][i];
}

__global__ void gdn_out(float *o, const float *z, const float *w, unsigned H, unsigned D, float eps) {
    const unsigned h = blockIdx.x, t = blockIdx.y, npt = D / 32, k0 = threadIdx.x * npt;
    const uint64_t idx = ((uint64_t)t * H + h) * D + k0;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) ss += o[idx + i] * o[idx + i];
    const float r = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) o[idx + i] = o[idx + i] * r * w[k0 + i] * sigmoid(z[idx + i]);
}

__global__ void ngram_gate(float *gated, float *normed, const float *R, const float *key,
                           const float *val, const float *gk, const float *gq, const float *gc,
                           unsigned E, unsigned hc, float eps) {
    const unsigned s = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const uint64_t base = ((uint64_t)t * hc + s) * E;
    __shared__ float red[32];
    float sk = 0, sq = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { sk += key[base + i] * key[base + i]; sq += R[base + i] * R[base + i]; }
    const float ik = rsqrtf(block_sum(sk, red) / E + eps);
    const float iq = rsqrtf(block_sum(sq, red) / E + eps);
    float dot = 0;
    for (unsigned i = tid; i < E; i += blockDim.x)
        dot += (key[base + i] * ik * gk[s * E + i]) * (R[base + i] * iq * gq[s * E + i]);
    const float a = block_sum(dot, red) * rsqrtf((float)E);
    const float mag = sqrtf(fmaxf(fabsf(a), 1e-6f));
    const float gate = sigmoid(a > 0 ? mag : a < 0 ? -mag : 0);
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        const float v = gate * val[(uint64_t)t * E + i];
        gated[base + i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) normed[base + i] = gated[base + i] * inv * gc[s * E + i];
}

__global__ void ngram_conv(float *R, const float *gated, const float *normed, float *history,
                           const char *w, unsigned type, unsigned T, unsigned C, unsigned K,
                           unsigned dilation, float *snap, unsigned st, float *snap2, unsigned st2) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const unsigned H = (K - 1) * dilation;
    float hist[9], taps[4];
    for (unsigned i = 0; i < H; i++) hist[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = scalar(w, c * K + i, type);
    for (unsigned t = 0; t < T; t++) {
        const uint64_t p = (uint64_t)t * C + c;
        const float cur = normed[p];
        float v = taps[K - 1] * cur;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * hist[i * dilation];
        for (unsigned i = 0; i + 1 < H; i++) hist[i] = hist[i + 1];
        hist[H - 1] = cur;
        R[p] += gated[p] + silu(v);
        if (snap && t == st) for (unsigned i = 0; i < H; i++) snap[(uint64_t)i * C + c] = hist[i];
        if (snap2 && t == st2) for (unsigned i = 0; i < H; i++) snap2[(uint64_t)i * C + c] = hist[i];
    }
    for (unsigned i = 0; i < H; i++) history[(uint64_t)i * C + c] = hist[i];
}

} // namespace qwen4_cuda

extern "C" int ds4_gpu_qwen4_conv_stream_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *history,
        const void *map, uint64_t size, uint64_t offset, uint32_t T, uint32_t C, uint32_t K, bool activate) {
    using namespace qwen4_cuda;
    if (!T || !C || K < 2 || K > 4 || !tensor(x, (uint64_t)T * C * 4) || !tensor(history, (uint64_t)(K - 1) * C * 4)) return 0;
    const char *w = weight(map, size, offset, (uint64_t)C * K * 4);
    if (!w) return 0;
    conv<<<(C + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)x->ptr, (float *)history->ptr,
        (const float *)w, T, C, K, activate, NULL, UINT_MAX, NULL, UINT_MAX);
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_prep_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *a, ds4_gpu_tensor *b,
        const void *map, uint64_t size, uint64_t ao, uint64_t bo, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D) {
    using namespace qwen4_cuda;
    if (!T || !Hk || !Hv || D < 32 || D > 128 || D % 32 ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4)) return 0;
    const char *A = weight(map, size, ao, (uint64_t)Hv * 4), *bias = weight(map, size, bo, (uint64_t)Hv * 4);
    if (!A || !bias) return 0;
    gdn_prep<<<dim3(Hk, T), 32, 0, cuda_decode_stream()>>>((float *)qkv->ptr,
        (float *)a->ptr, (float *)b->ptr, (const float *)A, (const float *)bias, Hk, Hv, D);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_verify_rows_exact(bool on) { (void)on; }

extern "C" int ds4_gpu_qwen4_gdn_scan_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t bytes = (uint64_t)Hv * D * D * 4;
    if (!T || !Hk || !Hv || Hv % Hk || D < 32 || D > 128 || D % 32 ||
        !tensor(out, (uint64_t)T * Hv * D * 4) || !tensor(state, bytes) ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4) ||
        (snap && !tensor(snap, bytes)) || (snap2 && !tensor(snap2, bytes))) return 0;
#define QWEN_GDN(ROWS, DIM) gdn_scan<ROWS,DIM><<<dim3((DIM+4*ROWS-1)/(4*ROWS),Hv),128,0,cuda_decode_stream()>>>((float *)out->ptr, \
        (float *)state->ptr,(const float *)qkv->ptr,(const float *)a->ptr,(const float *)b->ptr, \
        T,Hk,Hv,snap ? (float *)snap->ptr : NULL,st,snap2 ? (float *)snap2->ptr : NULL,st2)
#define QWEN_GDN_DIM(DIM) case DIM: if (T > 8) { QWEN_GDN(4,DIM); } else { QWEN_GDN(1,DIM); } break
    switch (D) { QWEN_GDN_DIM(32); QWEN_GDN_DIM(64); QWEN_GDN_DIM(96); QWEN_GDN_DIM(128); }
#undef QWEN_GDN_DIM
#undef QWEN_GDN
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_out_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *z,
        const void *map, uint64_t size, uint64_t off, uint32_t T, uint32_t H, uint32_t D, float eps) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * H * D;
    if (!n || D < 32 || D > 128 || D % 32 || !tensor(out, n * 4) || !tensor(z, n * 4)) return 0;
    const char *w = weight(map, size, off, (uint64_t)D * 4);
    if (!w) return 0;
    gdn_out<<<dim3(H, T), 32, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)z->ptr, (const float *)w, H, D, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_gate_tensor(ds4_gpu_tensor *gated, ds4_gpu_tensor *normed,
        const ds4_gpu_tensor *R, const ds4_gpu_tensor *key, const ds4_gpu_tensor *val,
        const void *map, uint64_t size, uint64_t ko, uint64_t qo, uint64_t co,
        uint32_t T, uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_cuda;
    const uint64_t bytes = (uint64_t)T * E * hc * 4, wb = (uint64_t)E * hc * 4;
    if (!T || !E || !hc || hc > 4 || !tensor(gated, bytes) || !tensor(normed, bytes) ||
        !tensor(R, bytes) || !tensor(key, bytes) || !tensor(val, (uint64_t)T * E * 4)) return 0;
    const char *gk = weight(map, size, ko, wb), *gq = weight(map, size, qo, wb), *gc = weight(map, size, co, wb);
    if (!gk || !gq || !gc) return 0;
    ngram_gate<<<dim3(hc, T), 128, 0, cuda_decode_stream()>>>((float *)gated->ptr, (float *)normed->ptr,
        (const float *)R->ptr, (const float *)key->ptr, (const float *)val->ptr,
        (const float *)gk, (const float *)gq, (const float *)gc, E, hc, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_conv_tensor(ds4_gpu_tensor *R, const ds4_gpu_tensor *gated,
        const ds4_gpu_tensor *normed, ds4_gpu_tensor *history, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t T, uint32_t C, uint32_t K, uint32_t dilation,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * C * 4, hb = (uint64_t)(K - 1) * dilation * C * 4;
    if (!T || !C || K < 2 || K > 4 || !dilation || dilation > 3 ||
        (type != 0 && type != 1) || !tensor(R, n) || !tensor(gated, n) || !tensor(normed, n) ||
        !tensor(history, hb) || (snap && !tensor(snap, hb)) || (snap2 && !tensor(snap2, hb))) return 0;
    const char *w = weight(map, size, off, row_bytes(type, (uint64_t)C * K));
    if (!w) return 0;
    ngram_conv<<<(C + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)R->ptr,
        (const float *)gated->ptr, (const float *)normed->ptr, (float *)history->ptr, w, type, T, C, K,
        dilation, snap ? (float *)snap->ptr : NULL, st, snap2 ? (float *)snap2->ptr : NULL, st2);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_norm_tensor(ds4_gpu_tensor *xn, ds4_gpu_tensor *inj,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t go, uint64_t io,
        uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * E * hc;
    if (!T || !E || !hc || hc > 8 || ni > 4 ||
        !tensor(xn, n * 4) || !tensor(R, n * 4) ||
        (ni && !tensor(inj, (uint64_t)T * hc * 8 * ni * 4))) return 0;
    const char *gamma = weight(map, size, go, (uint64_t)E * hc * 4);
    const char *wi = ni ? weight(map, size, io, row_bytes(type, (uint64_t)E * hc) * ni) : gamma;
    if (!gamma || !wi || (type != 0 && type != 1 && type != 8)) return 0;
    if (T > 8) {
#define QWEN_HC_NORM(TYPE) hc_norm_prefill<TYPE><<<dim3(hc,T),256,0,cuda_decode_stream()>>>((float *)xn->ptr, \
        ni ? (float *)inj->ptr : NULL,(const float *)R->ptr,(const float *)gamma,wi,E,hc,ni,eps)
        if (type == 0) { QWEN_HC_NORM(0); }
        else if (type == 1) { QWEN_HC_NORM(1); }
        else { QWEN_HC_NORM(8); }
#undef QWEN_HC_NORM
        return launched();
    }
    hc_norm<<<dim3(hc * 8, T), 128, 0, cuda_decode_stream()>>>((float *)xn->ptr,
        ni ? (float *)inj->ptr : NULL, (const float *)R->ptr, (const float *)gamma, wi, type, E, hc, ni, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_gate_mix_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset, uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t rank) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !rank || !tensor(out, (uint64_t)T * E * 4) ||
        !tensor(xn, (uint64_t)T * E * hc * 4) || !tensor(lo, (uint64_t)T * rank * 4)) return 0;
    const char *w = weight(map, size, offset, row_bytes(type, rank) * E * hc);
    if (!w || (type != 0 && type != 1 && type != 8)) return 0;
#define QWEN_HC(TYPE) hc_mix<TYPE><<<dim3((E + 3) / 4, T),128,0,cuda_decode_stream()>>>((float *)out->ptr, \
        (const float *)xn->ptr,(const float *)lo->ptr,w,E,hc,rank,row_bytes(TYPE,rank))
    if (type == 0) { QWEN_HC(0); }
    else if (type == 1) { QWEN_HC(1); }
    else { QWEN_HC(8); }
#undef QWEN_HC
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_combine_tensor(ds4_gpu_tensor *R,
        const ds4_gpu_tensor *out, const ds4_gpu_tensor *inj, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !tensor(R, (uint64_t)T * E * hc * 4) ||
        !tensor(out, (uint64_t)T * E * 4) || !tensor(inj, (uint64_t)T * hc * hc * 8 * 4)) return 0;
    hc_combine<<<dim3((E + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (float *)R->ptr, (const float *)out->ptr, (const float *)inj->ptr, E, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_lo_act_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src, uint32_t T, uint32_t hc, uint32_t rank) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * rank;
    if (!n || !hc || !tensor(dst, n * 4) || !tensor(src, n * 4)) return 0;
    hc_lo<<<(n + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)dst->ptr, (const float *)src->ptr, n, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_mix_rows_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *up, const ds4_gpu_tensor *xn, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !E || !hc || hc > 4 || !tensor(dst, (uint64_t)T * E * 4) ||
        !tensor(up, (uint64_t)T * hc * E * 4) || !tensor(xn, (uint64_t)T * hc * E * 4)) return 0;
    hc_rows<<<dim3((E + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (float *)dst->ptr, (const float *)up->ptr, (const float *)xn->ptr, E, hc);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_rope(const float *freq, uint32_t n, float scale) {
    qwen4_cuda::rope_set = freq != NULL;
    qwen4_cuda::rope_scale = freq ? scale : 1;
    memset(qwen4_cuda::rope_freq, 0, sizeof(qwen4_cuda::rope_freq));
    if (freq) memcpy(qwen4_cuda::rope_freq, freq, std::min(n, 32u) * sizeof(float));
}

extern "C" int ds4_gpu_qwen4_attn_prep_tensor(ds4_gpu_tensor *q, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, ds4_gpu_tensor *iqout, ds4_gpu_tensor *ikc,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kp, const ds4_gpu_tensor *vp,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3,
        const void *map, uint64_t size, uint64_t qo, uint64_t ko, uint64_t io,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t nrot,
        uint32_t Hi, uint32_t Di, uint32_t pos0, uint32_t cap, uint32_t ik_ring, float base, float eps) {
    using namespace qwen4_cuda;
    const uint64_t qb = (uint64_t)T * H * D * 4, kb = (uint64_t)T * Hkv * D * 4, ib = (uint64_t)T * Hi * Di * 4;
    /* A ring must hold the chunk and the block its first row may complete. */
    if (!T || !H || !Hkv || H % Hkv || !Hi || D < 32 || D > 256 || D % 32 ||
        Di < 32 || Di > 128 || Di % 32 || nrot > 64 || nrot > D || nrot > Di || nrot % 2 ||
        (uint64_t)pos0 + T > cap || (ik_ring && T + 3u > ik_ring) ||
        !tensor(q, qb) || !tensor(gate, qb) || !tensor(qg, qb * 2) ||
        !tensor(kp, kb) || !tensor(vp, kb) || !tensor(iq, ib) || !tensor(iqout, ib) ||
        !tensor(ik, (uint64_t)T * Di * 4) || !tensor(ikc, (uint64_t)(ik_ring ? ik_ring : cap) * Di * 4) ||
        !tensor(kc, (uint64_t)cap * Hkv * ds4_gpu_qwen4_kv_row_bytes(D)) ||
        !tensor(vc, (uint64_t)cap * Hkv * ds4_gpu_qwen4_kv_row_bytes(D)) ||
        !tensor(pos3, (uint64_t)cap * 16)) return 0;
    const char *gq = weight(map, size, qo, D * 4), *gk = weight(map, size, ko, D * 4), *giq = weight(map, size, io, Di * 4);
    if (!gq || !gk || !giq) return 0;
    attn_prep<<<dim3(H + Hkv + Hi + 1, T), 32, 0, cuda_decode_stream()>>>((float *)q->ptr,
        (float *)gate->ptr, (__half *)kc->ptr, (__half *)vc->ptr, (float *)iqout->ptr, (float *)ikc->ptr,
        (const float *)qg->ptr, (const float *)kp->ptr, (const float *)vp->ptr, (const float *)iq->ptr,
        (const float *)ik->ptr, (const uint32_t *)pos3->ptr, (const float *)gq, (const float *)gk, (const float *)giq,
        H, Hkv, D, Hi, Di, pos0, ik_ring, eps, rope(nrot, base), g_qwen4_kv_q8);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_block_key_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3, const void *map, uint64_t size,
        uint64_t off, uint32_t b0, uint32_t N, uint32_t ratio, uint32_t D, uint32_t nrot,
        uint32_t ik_ring, float base, float eps) {
    using namespace qwen4_cuda;
    const uint64_t rows = (uint64_t)b0 + N;
    if (!N || !ratio || D < 32 || D > 128 || D % 32 || nrot > D || nrot > 64 || nrot % 2 ||
        !tensor(out, rows * D * 2) || !tensor(ik, (ik_ring ? ik_ring : rows * ratio) * D * 4) ||
        !tensor(pos3, rows * ratio * 16)) return 0;
    const char *w = weight(map, size, off, D * 4);
    if (!w) return 0;
    block_key<<<N, 32, 0, cuda_decode_stream()>>>((__half *)out->ptr, (const float *)ik->ptr,
        (const uint32_t *)pos3->ptr, (const float *)w, b0, ratio, D, ik_ring, eps, rope(nrot, base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_score_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *tiles,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *key, uint32_t T, uint32_t N,
        uint32_t H, uint32_t D, uint32_t pos0, uint32_t ratio) {
    using namespace qwen4_cuda;
    const unsigned nt = (N + 7) / 8;
    if (!T || !N || !H || !D || !ratio || !tensor(out, (uint64_t)T * N * 4) ||
        !tensor(q, (uint64_t)T * H * D * 4) || !tensor(key, (uint64_t)N * D * 2) ||
        (tiles && !tensor(tiles, (uint64_t)T * nt * 4))) return 0;
    idx_score<<<dim3((N + 3) / 4, T), 128, 0, cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)q->ptr, (const __half *)key->ptr, N, H, D, pos0, ratio);
    if (!launched()) return 0;
    if (tiles) tile_max<<<dim3((nt + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(
        (unsigned *)tiles->ptr, (const float *)out->ptr, N, nt);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_select_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *score,
        const ds4_gpu_tensor *tiles, uint32_t N, uint32_t T, uint32_t K) {
    using namespace qwen4_cuda;
    (void)tiles;
    if (!T || !K || K > N || !tensor(out, (uint64_t)T * K * 4) || !tensor(score, (uint64_t)T * N * 4)) return 0;
    idx_select<<<T, 256, 0, cuda_decode_stream()>>>((int *)out->ptr, (const float *)score->ptr, N, K);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_expand_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *count,
        const ds4_gpu_tensor *blocks, uint32_t T, uint32_t K, uint32_t ratio, uint32_t pos0, uint32_t stride) {
    using namespace qwen4_cuda;
    if (!T || !K || !ratio || (uint64_t)K * ratio + ratio - 1 > stride ||
        !tensor(out, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4) || !tensor(blocks, (uint64_t)T * K * 4)) return 0;
    idx_expand<<<T, 256, 0, cuda_decode_stream()>>>((int *)out->ptr, (unsigned *)count->ptr,
        (const int *)blocks->ptr, K, ratio, pos0, stride);
    return launched();
}

extern "C" uint64_t ds4_gpu_qwen4_attn_part_floats(uint32_t T, uint32_t H, uint32_t D) {
    return (uint64_t)T * H * 64 * (D + 2);
}

extern "C" int ds4_gpu_qwen4_dense_mm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t off, uint32_t type, uint32_t T, uint32_t K, uint32_t M) {
    using namespace qwen4_cuda;
    if (!T || !K || !M || !tensor(x, (uint64_t)T*K*4) || !tensor(out, (uint64_t)T*M*4)) return 0;
    const uint64_t rb = row_bytes(type,K);
    if (!rb) return 0;
    const char *w = weight(map, size, off, rb*M);
    if (!w) return 0;
    if (T <= 8) return matvec_dispatch((float *)out->ptr, w, (const float *)x->ptr, type, T, K, M);
    if (type == 1 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode && !getenv("DS4_CUDA_NO_TF32"))
        return dense_f16_blas((float *)out->ptr,(const float *)x->ptr,(const __half *)w,T,K,M);
    if (type == 8 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode && !getenv("DS4_CUDA_NO_TF32"))
        return dense_q8_blas((float *)out->ptr,(const float *)x->ptr,w,T,K,M);
    if (g_cublas_ready && T >= 32 && K <= INT_MAX && M <= INT_MAX && T <= INT_MAX)
        return dense_blas((float *)out->ptr,(const float *)x->ptr,w,type,T,K,M);
    return matrix_dispatch((float *)out->ptr, (const float *)x->ptr, w, NULL, NULL, NULL,
                           type, 1, T, 1, 1, K, M, 0, false);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *map,
        uint64_t size, uint64_t off, uint64_t K, uint64_t M, const ds4_gpu_tensor *x, uint64_t T) {
    if (K > UINT_MAX || M > UINT_MAX || T > UINT_MAX) return 0;
    return ds4_gpu_qwen4_dense_mm_tensor(out, x, map, size, off, 8, T, K, M);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *w,
        uint32_t K, uint32_t M, const ds4_gpu_tensor *x) {
    using namespace qwen4_cuda;
    const uint64_t rb = row_bytes(8, K);
    if (!K || !M || !rb || !tensor(w, rb*M) || !tensor(x, (uint64_t)K*4) || !tensor(out, (uint64_t)M*4)) return 0;
    return matvec_dispatch((float *)out->ptr, (const char *)w->ptr, (const float *)x->ptr, 8, 1, K, M);
}

extern "C" int ds4_gpu_qwen4_multi_gemv_tensor(const ds4_gpu_tensor *x, uint32_t T,
        uint32_t K, uint32_t N, ds4_gpu_tensor *const *outs, const void *map, uint64_t size,
        const uint64_t *offsets, const uint32_t *types, const uint32_t *rows) {
    if (!N || N > 4 || !outs || !offsets || !types || !rows) return 0;
    for (unsigned i = 0; i < N; i++)
        if (!ds4_gpu_qwen4_dense_mm_tensor(outs[i], x, map, size, offsets[i], types[i], T, K, rows[i])) return 0;
    return 1;
}

extern "C" int ds4_gpu_qwen4_q8_pair_tensor(ds4_gpu_tensor *o0, ds4_gpu_tensor *o1,
        const void *map, uint64_t size, uint64_t w0, uint64_t w1, uint64_t K,
        uint64_t M0, uint64_t M1, const ds4_gpu_tensor *x, uint64_t T) {
    if ((M0 & 1) || (M1 & 1)) return 0;
    return ds4_gpu_qwen4_matmul_q8_0_tensor(o0, map, size, w0, K, M0, x, T) &&
           ds4_gpu_qwen4_matmul_q8_0_tensor(o1, map, size, w1, K, M1, x, T);
}

extern "C" int ds4_gpu_qwen4_router_topk_tensor(ds4_gpu_tensor *sel, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t K, ds4_gpu_tensor *sg, uint32_t T, uint32_t NE, uint32_t NS) {
    using namespace qwen4_cuda;
    if (!T || !NE || NE > 512 || !NS || NS > NE || NS > 32 ||
        !tensor(sel, (uint64_t)T*NS*4) || !tensor(weights, (uint64_t)T*NS*4) || !tensor(logits, (uint64_t)T*NE*4)) return 0;
    const char *wg = K ? weight(map, size, off, row_bytes(type, K)) : NULL;
    if (K && (!wg || !tensor(x, (uint64_t)T*K*4) || !tensor(sg, (uint64_t)T*4))) return 0;
    router<<<T,256,0,cuda_decode_stream()>>>((int *)sel->ptr, (float *)weights->ptr, (const float *)logits->ptr,
        K ? (const float *)x->ptr : NULL, wg, K ? (float *)sg->ptr : NULL, NE, NS, K, type);
    return launched();
}

static bool qwen4_matvec_type(uint32_t type) {
    return type == 0 || type == 1 || type == 2 || type == 8 || type == 30;
}

/* One row of x against up to five weights in one launch (decode).  A weight
 * type without a row_dot case takes the separate launches instead. */
extern "C" int ds4_gpu_qwen4_multi_matvec_tensor(const ds4_gpu_tensor *x, uint32_t K, uint32_t N,
        ds4_gpu_tensor *const *outs, const void *map, uint64_t size,
        const uint64_t *offsets, const uint32_t *types, const uint32_t *rows) {
    using namespace qwen4_cuda;
    if (!N || N > 5 || !outs || !offsets || !types || !rows || !K || !tensor(x, (uint64_t)K * 4)) return 0;
    bool typed = true;
    for (unsigned i = 0; i < N; i++) typed = typed && qwen4_matvec_type(types[i]);
    if (!typed) {
        for (unsigned i = 0; i < N; i++)
            if (!ds4_gpu_qwen4_dense_mm_tensor(outs[i], x, map, size, offsets[i], types[i], 1, K, rows[i])) return 0;
        return 1;
    }
    matvec_set s = {};
    unsigned total = 0;
    for (unsigned i = 0; i < N; i++) {
        const uint64_t rb = row_bytes(types[i], K);
        if (!rb || !rows[i] || !tensor(outs[i], (uint64_t)rows[i] * 4)) return 0;
        s.w[i] = weight(map, size, offsets[i], rb * rows[i]);
        if (!s.w[i]) return 0;
        s.out[i] = (float *)outs[i]->ptr;
        s.stride[i] = rb;
        s.rows[i] = rows[i];
        s.type[i] = types[i];
        total += rows[i];
    }
    s.n = N;
    multi_matvec<<<(total + 3) / 4, 128, 0, cuda_decode_stream()>>>(s, (const float *)x->ptr, K);
    return launched();
}

extern "C" int ds4_gpu_qwen4_matvec_combine_tensor(ds4_gpu_tensor *R, ds4_gpu_tensor *blk,
        const ds4_gpu_tensor *inj, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t K, uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    const uint64_t rb = row_bytes(type, K);
    if (!K || !E || !hc || hc > 4 || !rb || !tensor(x, (uint64_t)K * 4) || !tensor(blk, (uint64_t)E * 4) ||
        !tensor(R, (uint64_t)E * hc * 4) || !tensor(inj, (uint64_t)hc * hc * 8 * 4)) return 0;
    if (!qwen4_matvec_type(type))
        return ds4_gpu_qwen4_dense_mm_tensor(blk, x, map, size, off, type, 1, K, E) &&
               ds4_gpu_qwen4_hc_combine_tensor(R, blk, inj, 1, E, hc);
    const char *w = weight(map, size, off, rb * E);
    if (!w) return 0;
    matvec_combine<<<(E + 3) / 4, 128, 0, cuda_decode_stream()>>>((float *)R->ptr, (float *)blk->ptr,
        (const float *)inj->ptr, w, (const float *)x->ptr, type, K, E, rb, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_norm_down_tensor(ds4_gpu_tensor *xn, ds4_gpu_tensor *inj, ds4_gpu_tensor *lo,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t go, uint64_t io, uint32_t itype,
        uint64_t dof, uint32_t dtype, uint32_t E, uint32_t hc, uint32_t ni, uint32_t rank, float eps) {
    using namespace qwen4_cuda;
    const uint64_t dim = (uint64_t)E * hc, rb = row_bytes(dtype, dim);
    /* the normalized streams must fit the default shared memory beside red[] */
    if (dim * 4 > 48u * 1024 - 256 || !qwen4_matvec_type(dtype))
        return ds4_gpu_qwen4_hc_norm_tensor(xn, inj, R, map, size, go, io, itype, 1, E, hc, ni, eps) &&
               ds4_gpu_qwen4_dense_mm_tensor(lo, xn, map, size, dof, dtype, 1, (uint32_t)dim, rank);
    if (!E || !hc || hc > 8 || ni > 4 || !rank || !rb || !tensor(xn, dim * 4) || !tensor(R, dim * 4) ||
        !tensor(lo, (uint64_t)rank * 4) || (ni && !tensor(inj, (uint64_t)hc * 8 * ni * 4))) return 0;
    const char *gamma = weight(map, size, go, dim * 4);
    const char *wi = ni ? weight(map, size, io, row_bytes(itype, dim) * ni) : gamma;
    const char *down = weight(map, size, dof, rb * rank);
    if (!gamma || !wi || !down || (itype != 0 && itype != 1 && itype != 8)) return 0;
    hc_norm_down<<<hc * 8 + (rank + 3) / 4, 128, dim * 4, cuda_decode_stream()>>>((float *)xn->ptr,
        ni ? (float *)inj->ptr : NULL, (float *)lo->ptr, (const float *)R->ptr, (const float *)gamma, wi, itype,
        down, dtype, rb, (unsigned)E, hc, ni, rank, eps);
    return launched();
}

/* SSD streaming for the per-token MoE kernels (Kolibri-1): the routed experts
 * are not resident.  Each CUDA device keeps an arena of expert slots, one
 * expert's gate, up and down weights per slot, filled from the host mapping
 * on demand and reused least recently used first.  The kernels address an
 * expert as base + id * expert bytes, so between _begin and _end the mid and
 * down wrappers pass the arena as the base and slot numbers as the ids: the
 * kernels themselves are the resident ones. */
namespace qwen4_cuda {

struct moe_stream_cache {
    /* One arena; a slot is an expert's gate, up and down back to back, so a
     * load or a save is one copy.  up and down point into the first slot. */
    char *gate = NULL, *up = NULL, *down = NULL;
    int *selected = NULL;                 /* slot of every (token, slot) in the open batch */
    uint64_t selected_cap = 0;
    uint64_t gate_bytes = 0, down_bytes = 0, slot_bytes = 0;   /* per expert */
    uint32_t slots = 0, filled = 0;
    std::vector<uint32_t> heat;           /* uses per slot; travels with the expert between tiers */    std::unordered_map<uint64_t, uint32_t> slot_of;   /* layer << 32 | expert */
    std::vector<uint64_t> key, stamp;
    bool provisional = false;             /* sized for the first forward only */
    bool resized = false;
    uint32_t last_layer = 0;
    char *stage = NULL;                   /* page-locked ring of experts read from the file */
    uint32_t stage_used = 0;
    uint64_t uses = 0, misses = 0;        /* since the last DS4_CUDA_MOE_STREAM_STATS line */
    uint64_t helped = 0, from_ram = 0;
    double host_s = 0.0, device_s = 0.0;
    /* Only used while this device helps another one (see g_moe_stream.helper):
     * the inputs and outputs of its share of a batch, here and page-locked. */
    float *hx = NULL, *hmid = NULL, *hpart = NULL, *host_part = NULL;
    uint64_t hx_cap = 0, hmid_cap = 0, hpart_cap = 0, host_part_cap = 0;
    /* As the primary: the rows the helper and the CPU computed, a pair to a
     * row, on their way into part in one copy and one kernel. */
    float *ext = NULL, *host_ext = NULL;
    uint64_t ext_cap = 0, host_ext_cap = 0;
    bool ext_used = false;
    /* The last tail_slots slots double as the buffer a whole layer of experts
     * is loaded into for a large batch: while one is being processed
     * (tail_busy) they are out of the cache, between batches they hold experts
     * like the rest, so that memory is never idle. */
    uint32_t tail_slots = 0;
    bool tail_busy = false;
    uint64_t hops = 0;
    double hop_s[4] = {0.0, 0.0, 0.0, 0.0};   /* down launch, wait + read, write back, mid */
    /* The slot table the single-token kernels map the router's ids through,
     * [layer][expert] -> slot or -1 (see moe_stream_table_row).  generation
     * counts changes to key[]; a layer's row is uploaded again when the
     * generation it was built at is behind. */
    int *table = NULL, *table_rows = NULL;   /* device; page-locked ring the rows go out of */
    uint32_t table_layers = 0, table_experts = 0, table_next = 0;
    std::vector<uint64_t> table_gen;
    uint64_t generation = 1;
    cudaEvent_t selected_event = NULL;    /* the selection has reached the host */
};

static const uint64_t moe_stream_no_key = UINT64_MAX;
static const uint32_t moe_stream_stage_slots = 64;
static std::unordered_map<int, moe_stream_cache> g_moe_stream_caches;   /* by CUDA device */
static uint64_t g_moe_stream_clock;       /* one clock, so stamps compare across devices */
static bool g_moe_stream_by_use;          /* the open batch replaces by use count */
/* With DS4_CUDA_MOE_STREAM_HELPER set, a second CUDA device that runs none of
 * the graph holds experts too and evaluates the ones it holds: the layer's
 * input goes over, its expert outputs come back.  That costs two small
 * copies per layer where a miss costs an expert upload. */
static struct {
    const moe_stream_cache *cache;        /* NULL outside _begin .. _end */
    uint64_t gate_off, up_off, down_off;
    moe_stream_cache *helper;             /* NULL when it has no share of this batch */
    int device, helper_device;
    std::vector<uint64_t> helper_pairs;
    /* The CPU's share: pairs whose expert stays in the RAM tier, and where. */
    std::vector<uint64_t> cpu_pairs;
    std::vector<uint32_t> cpu_slots;
    bool cpu_done;
    unsigned mid_type, mid_K, mid_M;      /* of the open batch, for the CPU's gate and up */
    float *host_x;                        /* the layer input, page-locked */
    uint64_t host_x_cap;
    /* Page-locked, so the selection comes back and the slot numbers go out as
     * queued copies: one wait per layer instead of three blocking calls. */
    int *host_ids, *host_helper_ids;
    uint64_t host_ids_cap, host_helper_ids_cap;
    const ds4_gpu_tensor *input;          /* announced for the next _begin, see _stream_input */
    uint32_t input_width;
    bool input_ready;                     /* host_x holds the open batch's input */
    double wait_s;                        /* in _begin's wait, since the last stats line */
    uint64_t waits;
    /* A single token's expert stage through the slot table: _begin queues
     * the selection's read-back and leaves the rest to _down_tensor, which
     * settles it once the kernels are queued (see moe_stream_settle). */
    bool deferred;
    ds4_gpu_stream_expert_table table;    /* the caller's copy may not outlive _begin */
    uint32_t T, NS;
    moe_stream_cache *candidate;          /* the helper, if any, before it has a share */
    uint64_t uploaded;                    /* the cache generation the layer's row went out at */
    struct {                              /* the mid kernel's arguments, for running it again */
        bool queued;
        float *out;
        const float *x;
        const int *sel;
        const char *g, *u, *sg, *su;
        unsigned type, st, K, M;
    } mid;
} g_moe_stream;

typedef void (*moe_cpu_quantize_fn)(const float *x, void *xq, uint32_t n);
typedef void (*moe_cpu_rows_fn)(uint32_t type, const void *rows, uint64_t row_bytes, uint32_t in_dim,
                                uint32_t first_row, uint32_t end_row, const void *xq, float *out);
/* With DS4_CUDA_MOE_STREAM_CPU set, once every device slot is taken an
 * expert that no device holds is evaluated from the RAM tier by the CPU
 * rather than swapped in: a swap moves two experts over the bus, which costs
 * more than the arithmetic. */
static struct {
    moe_cpu_quantize_fn quantize;
    moe_cpu_rows_fn rows;
    uint32_t xq_block_bytes;
    std::vector<char> xq, midq;
    std::vector<float> gate, up, mid;
    float *out;                           /* page-locked, cpu pair major */
    uint64_t out_cap;
    uint64_t pairs, layers;               /* since the last stats line */
    double seconds;
} g_moe_cpu;

/* Runs job(0 .. count-1) on every hardware thread but a few, the caller's
 * included, and returns when all are done. */
struct moe_cpu_pool {
    static constexpr auto round_poll_us = 3000;
    std::vector<std::thread> workers;
    std::mutex lock;
    std::condition_variable wake;
    const std::function<void(uint32_t)> *job = NULL;
    std::atomic<uint32_t> next{0}, running{0};
    std::atomic<uint64_t> round{0};
    std::atomic<uint64_t> busy_ns{0};     /* summed over threads, for the stats line */
    uint32_t count = 0;

    void work() {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = next.fetch_add(1); i < count; i = next.fetch_add(1)) (*job)(i);
        busy_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
    }
    /* A layer is two rounds a few microseconds apart and the next layer's
     * follow within a millisecond or two while a token is generated.  Waking
     * a sleeping thread costs more than a round's work, and run() waits for
     * every worker: a worker polls across that gap before it sleeps, and
     * run() polls for the last worker rather than sleeping itself. */
    bool next_round(uint64_t seen) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(round_poll_us);
        bool there = round.load() != seen;
        while (!there && std::chrono::steady_clock::now() < until) {
            for (int i = 0; i < 64 && !there; i++) there = round.load(std::memory_order_relaxed) != seen;
            std::this_thread::yield();
        }
        return there;
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            if (!next_round(seen)) {
                std::unique_lock<std::mutex> held(lock);
                wake.wait(held, [&] { return round.load() != seen; });
            }
            seen = round.load();
            work();
            running--;
        }
    }
    void run(uint32_t n, const std::function<void(uint32_t)> &fn) {
        if (workers.empty()) {
            unsigned hardware = std::thread::hardware_concurrency();
#ifdef _WIN32
            /* On a hybrid CPU one thread per fast core: a slow core or a
             * second thread on a fast one adds little to this arithmetic and
             * the round waits for whichever thread finishes last. */
            DWORD bytes = 0;
            (void)GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &bytes);
            std::vector<char> cores(bytes);
            if (bytes && GetLogicalProcessorInformationEx(RelationProcessorCore,
                    (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)cores.data(), &bytes)) {
                unsigned fast = 0, best = 0;
                KAFFINITY fast_mask = 0;
                for (DWORD at = 0; at < bytes;) {
                    const auto *core = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(cores.data() + at);
                    if (core->Processor.EfficiencyClass > best) {
                        best = core->Processor.EfficiencyClass;
                        fast = 0;
                        fast_mask = 0;
                    }
                    if (core->Processor.EfficiencyClass == best) {
                        fast++;
                        if (core->Processor.GroupMask[0].Group == 0) fast_mask |= core->Processor.GroupMask[0].Mask;
                    }
                    at += core->Size;
                }
                /* The loop below keeps four back; this leaves one, the caller's. */
                if (best && fast) hardware = fast + 3;
                /* The whole process stays on the fast cores, not only these
                 * workers: a round waits for its last thread, and so does the
                 * device for the thread that feeds it.  With the slow cores
                 * busy elsewhere a thread placed on one waited a scheduler
                 * slice, 15 ms where the work takes 0.2 (700 ms a token
                 * against 47).  With the slow cores idle the process left
                 * to the scheduler still took 0.26 ms an expert against 0.11. */
                if (best && fast_mask && SetProcessAffinityMask(GetCurrentProcess(), fast_mask))
                    fprintf(stderr, "ds4: process kept on the %u fast cores (affinity %llx)\n", fast,
                            (unsigned long long)fast_mask);
            }
#endif
            for (unsigned i = 0; i + 4 < (hardware ? hardware : 8u); i++) {
                workers.emplace_back([this] { loop(); });
                workers.back().detach();
            }
        }
        {
            std::lock_guard<std::mutex> held(lock);
            job = &fn;
            count = n;
            next = 0;
            running = (uint32_t)workers.size();
            round++;
        }
        wake.notify_all();
        work();
        while (running.load() != 0) std::this_thread::yield();
    }
};
static moe_cpu_pool g_moe_cpu_pool;

template<typename TValue>
static bool moe_stream_grow(TValue *&p, uint64_t &cap, uint64_t bytes, bool host, const char *what) {
    if (cap >= bytes) return true;
    if (host) (void)cudaFreeHost(p);
    else (void)cudaFree(p);
    p = NULL;
    cap = 0;
    void *grown = NULL;
    if (!cuda_ok(host ? cudaMallocHost(&grown, bytes) : cudaMalloc(&grown, bytes), what)) return false;
    p = (TValue *)grown;
    cap = bytes;
    return true;
}

static bool moe_stream_wait(cudaStream_t stream, const char *what);
static bool moe_stream_ram_settle(uint32_t ram_slot);
static void moe_stream_promote_drained(void);

/* A blocking wait gives the thread up to the scheduler, which on Windows
 * costs most of a millisecond however short the work; with
 * DS4_CUDA_MOE_STREAM_SPIN the thread polls for a few milliseconds first. */
static bool moe_stream_wait(cudaStream_t stream, const char *what) {
    static const bool spin = getenv("DS4_CUDA_MOE_STREAM_SPIN") != NULL;
    if (spin) {
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
        while (cudaStreamQuery(stream) == cudaErrorNotReady && std::chrono::steady_clock::now() < give_up) {}
        (void)cudaGetLastError();
    }
    return cuda_ok(cudaStreamSynchronize(stream), what);
}

/* The same for an event: the selection has reached the host while kernels
 * queued behind it may still be running. */
static bool moe_stream_event_wait(cudaEvent_t event, const char *what) {
    static const bool spin = getenv("DS4_CUDA_MOE_STREAM_SPIN") != NULL;
    if (spin) {
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
        while (cudaEventQuery(event) == cudaErrorNotReady && std::chrono::steady_clock::now() < give_up) {}
        (void)cudaGetLastError();
    }
    return cuda_ok(cudaEventSynchronize(event), what);
}

/* The open batch's input on the host when _begin did not read it: one round
 * trip, paid only by a layer the CPU route or the helper has a share of.
 * x was complete when the selection was, so nothing new is waited for. */
static bool moe_stream_fetch_input(const float *x, uint64_t bytes) {
    if (!g_moe_stream.input_ready)
        g_moe_stream.input_ready =
            moe_stream_grow(g_moe_stream.host_x, g_moe_stream.host_x_cap, bytes, true, "MoE expert input staging") &&
            cuda_ok(cudaMemcpy(g_moe_stream.host_x, x, bytes, cudaMemcpyDeviceToHost), "MoE expert input read");
    return g_moe_stream.input_ready;
}

/* One layer's row of the device's slot table, [layer][expert] -> slot or -1,
 * built from key[] and uploaded on the stream when the cache changed since
 * it last went out: the single-token kernels map the router's ids through
 * it (moe_pair_id), so they can be queued before the host has seen the
 * selection.  A page-locked row is written again four layers on, by when
 * the wait each layer takes behind its selection has run the copy. */
static const uint32_t moe_stream_table_ring = 4, moe_stream_table_layers = 64;

static bool moe_stream_table_alloc(moe_stream_cache &c, uint32_t layer, uint32_t experts, cudaStream_t stream) {
    bool ok = true;
    if (c.table_experts != experts || c.table_layers <= layer) {
        const uint32_t layers = layer + 1 > moe_stream_table_layers ? layer + 1 : moe_stream_table_layers;
        const uint64_t bytes = (uint64_t)layers * experts * 4;
        void *table = NULL, *rows = NULL;
        /* Kernels queued on the old table may still be reading it. */
        ok = cuda_ok(cudaStreamSynchronize(stream), "MoE slot table wait");
        (void)cudaFree(c.table);
        (void)cudaFreeHost(c.table_rows);
        c.table = c.table_rows = NULL;
        c.table_layers = c.table_experts = 0;
        ok = ok && cuda_ok(cudaMalloc(&table, bytes), "MoE slot table") &&
             cuda_ok(cudaMallocHost(&rows, (uint64_t)moe_stream_table_ring * experts * 4), "MoE slot table staging") &&
             cuda_ok(cudaMemsetAsync(table, 0xff, bytes, stream), "MoE slot table clear");
        if (ok) {
            c.table = (int *)table;
            c.table_rows = (int *)rows;
            c.table_layers = layers;
            c.table_experts = experts;
            c.table_gen.assign(layers, 0);
        } else {
            (void)cudaFree(table);
            (void)cudaFreeHost(rows);
        }
    }
    return ok;
}

static bool moe_stream_table_row(moe_stream_cache &c, uint32_t layer, uint32_t experts, cudaStream_t stream) {
    bool ok = moe_stream_table_alloc(c, layer, experts, stream);
    if (ok && c.table_gen[layer] != c.generation) {
        int *row = c.table_rows + (uint64_t)(c.table_next++ % moe_stream_table_ring) * experts;
        for (uint32_t e = 0; e < experts; e++) row[e] = -1;
        for (uint32_t s = 0; s < c.slots; s++)
            if (c.key[s] >> 32 == layer && (uint32_t)c.key[s] < experts) row[(uint32_t)c.key[s]] = (int)s;
        ok = cuda_ok(cudaMemcpyAsync(c.table + (uint64_t)layer * experts, row, (uint64_t)experts * 4,
                                     cudaMemcpyHostToDevice, stream), "MoE slot table row");
        if (ok) c.table_gen[layer] = c.generation;
    }
    return ok;
}

/* DS4_CUDA_MOE_STREAM_RAM_MIB sizes a host arena for experts that no device
 * holds; shared by all devices.  Without it they are read from the file
 * mapping every time, which is the disk whenever the file does not fit the
 * page cache.  The arena is private memory in chunks, page-locked as far as
 * the driver allows: Windows refuses one large locked block, and whatever it
 * will not lock is still RAM the file cache cannot drop. */
struct moe_stream_ram {
    std::vector<char *> chunks;
    uint64_t slot_bytes = 0;
    uint32_t slots = 0, chunk_slots = 0;
    uint32_t locked_slots = 0;            /* the first this many slots are page-locked */
    bool tried = false;
    std::unordered_map<uint64_t, uint32_t> slot_of;
    std::vector<uint64_t> key, stamp;
    std::vector<uint32_t> uses;
    /* A slot an expert was loaded from is free once that copy has run, which
     * the next batch's wait for the device guarantees: released until then. */
    std::vector<uint32_t> free_slots, released;
    double file_s = 0.0;                  /* experts the CPU needed that only the file had, for the stats line */
    uint64_t file_reads = 0;
    char *at(uint32_t slot) const { return chunks[slot / chunk_slots] + (uint64_t)(slot % chunk_slots) * slot_bytes; }
};
static moe_stream_ram g_moe_stream_ram;
static uint64_t moe_stream_env_mib(const char *name, uint64_t fallback);

/* Where a decoded token's wall time goes, for DS4_CUDA_MOE_STREAM_STATS: each
 * lap charges the time since the previous one to a phase, so the phases add up
 * to the whole token with nothing left out and no extra waits put in. */
enum moe_prof_phase {
    moe_prof_outside,      /* everything between two layers' expert stages: the dense stages' host side */
    moe_prof_wait,         /* the device catching up before the selection can be read: its queued work */
    moe_prof_prepare,      /* sorting the selection, loads, the helper's selection */
    moe_prof_gap,          /* the caller between the expert calls */
    moe_prof_mid,          /* gate and up: launches here, and on the helper */
    moe_prof_down,         /* down: launch, helper results, the CPU's share */
    moe_prof_phases
};
static struct {
    bool running;
    std::chrono::steady_clock::time_point mark;
    double seconds[moe_prof_phases];
    uint64_t layers;
} g_moe_prof;

static void moe_prof_lap(moe_prof_phase phase) {
    static const bool wanted = getenv("DS4_CUDA_MOE_STREAM_STATS") != NULL;
    if (wanted) {
        const auto now = std::chrono::steady_clock::now();
        if (g_moe_prof.running) g_moe_prof.seconds[phase] += std::chrono::duration<double>(now - g_moe_prof.mark).count();
        g_moe_prof.mark = now;
    }
}

static void moe_stream_ram_init(moe_stream_ram &ram, uint64_t slot_bytes) {
    if (ram.tried) return;
    ram.tried = true;
    const uint64_t slots = moe_stream_env_mib("DS4_CUDA_MOE_STREAM_RAM_MIB", 0) / slot_bytes;
    if (!slots || slots > UINT32_MAX) return;
    ram.slot_bytes = slot_bytes;
    ram.chunk_slots = (uint32_t)((256ull << 20) / slot_bytes) + 1u;
    uint64_t locked = 0, lock_limit = 10240ull << 20;
#ifdef _WIN32
    /* What can be locked depends on what is free now, not on what is
     * installed: with 17 GiB free, 9 GiB locked here left the layer reader
     * and the driver's own transfers failing with out of memory, 4 GiB did
     * not.  A third of what is free is a rule of thumb from those two runs;
     * DS4_CUDA_MOE_STREAM_RAM_LOCK_MIB overrides it. */
    MEMORYSTATUSEX memory;
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory) && memory.ullAvailPhys / 3 < lock_limit) lock_limit = memory.ullAvailPhys / 3;
#endif
    lock_limit = moe_stream_env_mib("DS4_CUDA_MOE_STREAM_RAM_LOCK_MIB", lock_limit);
    bool lock = true;
    for (uint64_t have = 0; have < slots; have += ram.chunk_slots) {
        const uint64_t bytes = (uint64_t)ram.chunk_slots * slot_bytes;
        void *chunk = NULL;
        /* Locking up to the driver's limit leaves its own transfers failing
         * with out of memory, even after handing chunks back; stay well under. */
        lock = lock && locked + bytes <= lock_limit;
        if (lock && cudaMallocHost(&chunk, bytes) != cudaSuccess) {
            /* The driver's limit is reached: hand two chunks back, or its own
             * transfers find nothing left to lock and fail. */
            (void)cudaGetLastError();
            lock = false;
            chunk = NULL;
            for (size_t back = 0; back < 2 && back < ram.chunks.size(); back++) {
                char *&locked_chunk = ram.chunks[ram.chunks.size() - 1 - back];
                (void)cudaFreeHost(locked_chunk);
                locked_chunk = (char *)malloc(bytes);
                locked -= bytes;
                if (!locked_chunk) {
                    fprintf(stderr, "ds4: CUDA MoE expert RAM tier ran out of memory\n");
                    exit(1);
                }
            }
        }
        if (chunk) locked += bytes;
        else if (!(chunk = malloc(bytes))) break;
        ram.chunks.push_back((char *)chunk);
    }
    ram.slots = (uint32_t)(ram.chunks.size() * ram.chunk_slots);
    ram.locked_slots = (uint32_t)(locked / slot_bytes);
    ram.key.assign(ram.slots, moe_stream_no_key);
    ram.stamp.assign(ram.slots, 0);
    ram.uses.assign(ram.slots, 0);
    for (uint32_t i = ram.slots; i > 0; i--) ram.free_slots.push_back(i - 1u);
    fprintf(stderr, "ds4: CUDA MoE expert RAM tier: %u experts, %.2f GiB, %.2f GiB of it page-locked\n",
            ram.slots, (double)((uint64_t)ram.slots * slot_bytes) / 1073741824.0, (double)locked / 1073741824.0);
}

static uint64_t moe_stream_oldest(const moe_stream_cache &c) {
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < c.slots; i++)
        if (c.stamp[i] < oldest) oldest = c.stamp[i];
    return oldest;
}

static uint64_t moe_stream_env_mib(const char *name, uint64_t fallback) {
    const char *env = getenv(name);
    return env && env[0] ? strtoull(env, NULL, 10) << 20 : fallback;
}

static void moe_stream_cache_release(moe_stream_cache &c) {
    (void)cudaFree(c.gate);
    (void)cudaFreeHost(c.stage);
    c.stage_used = 0;
    (void)cudaFree(c.hx);
    (void)cudaFree(c.hmid);
    (void)cudaFree(c.hpart);
    (void)cudaFreeHost(c.host_part);
    c.hx = c.hmid = c.hpart = c.host_part = NULL;
    c.hx_cap = c.hmid_cap = c.hpart_cap = c.host_part_cap = 0;
    c.gate = c.up = c.down = c.stage = NULL;
    c.slots = c.filled = 0;
    c.tail_slots = 0;
    c.tail_busy = false;
    c.slot_of.clear();
    (void)cudaFree(c.table);
    (void)cudaFreeHost(c.table_rows);
    c.table = c.table_rows = NULL;
    c.table_layers = c.table_experts = 0;
    c.table_gen.clear();
    c.generation++;
}

static const uint64_t moe_stream_tail_key = UINT64_MAX - 1;

/* Takes the tail slots out of the cache for a layer buffer; the experts they
 * held are back to where the file has them. */
static void moe_stream_tail_reserve(moe_stream_cache &c) {
    for (uint32_t s = c.slots - c.tail_slots; s < c.slots && !c.tail_busy; s++) {
        if (c.key[s] == moe_stream_no_key) c.filled++;
        else c.slot_of.erase(c.key[s]);
        c.key[s] = moe_stream_tail_key;
        /* Never free to a load, never the least used. */
        c.stamp[s] = UINT64_MAX;
        c.heat[s] = UINT32_MAX;
    }
    c.tail_busy = c.tail_slots != 0;
    c.generation++;
}

static void moe_stream_tail_release(moe_stream_cache &c) {
    for (uint32_t s = c.slots - c.tail_slots; s < c.slots && c.tail_busy; s++) {
        c.key[s] = moe_stream_no_key;
        c.stamp[s] = 0;
        c.heat[s] = 0;
        c.filled--;
    }
    c.tail_busy = false;
    c.generation++;
}

/* The arena takes the device memory that is free, less a reserve for cuBLAS
 * and later allocations.  What is free is only known after one forward: a
 * single device caches each weight, and the converted copies its dense
 * kernels use, on first use.  So the first forward runs on a provisional
 * arena of one layer's experts, and the arena is sized for real when the
 * second one starts.  --ssd-streaming-cache-experts sets the slot count per
 * device outright, DS4_CUDA_MOE_STREAM_CACHE_MIB the size and
 * DS4_CUDA_MOE_STREAM_RESERVE_MIB the reserve. */
static bool moe_stream_cache_init(moe_stream_cache &c, int device, uint64_t gate_bytes,
                                  uint64_t down_bytes, uint32_t min_slots, uint32_t layer_experts) {
    size_t free_bytes = 0, total_bytes = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "MoE expert cache memory query")) return false;
    const uint64_t reserve = moe_stream_env_mib("DS4_CUDA_MOE_STREAM_RESERVE_MIB", 512ull << 20);
    const uint64_t budget = moe_stream_env_mib("DS4_CUDA_MOE_STREAM_CACHE_MIB",
                                               free_bytes > reserve ? free_bytes - reserve : 0);
    const uint64_t slot_bytes = 2 * gate_bytes + down_bytes;
    c.provisional = !c.resized && !g_stream_expert_budget && budget / slot_bytes > layer_experts;
    const uint64_t slots = g_stream_expert_budget ? g_stream_expert_budget :
                           c.provisional ? layer_experts : budget / slot_bytes;
    if (slots < min_slots || slots > UINT32_MAX) {
        fprintf(stderr, "ds4: CUDA MoE expert cache on device %d needs %u slots of %.2f MiB but %.2f GiB "
                        "fits %llu (%.2f of %.2f GiB free); free device memory or lower the context\n",
                device, min_slots, (double)slot_bytes / 1048576.0, (double)budget / 1073741824.0,
                (unsigned long long)slots, (double)free_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
        return false;
    }
    void *arena = NULL, *stage = NULL;
    if (!cuda_ok(cudaMalloc(&arena, slots * slot_bytes), "MoE expert cache") ||
        !cuda_ok(cudaMallocHost(&stage, moe_stream_stage_slots * slot_bytes), "MoE expert cache staging")) {
        (void)cudaFree(arena);
        return false;
    }
    c.gate = (char *)arena;
    c.up = c.gate + gate_bytes;
    c.down = c.gate + 2 * gate_bytes;
    c.stage = (char *)stage;
    c.gate_bytes = gate_bytes;
    c.down_bytes = down_bytes;
    c.slot_bytes = slot_bytes;
    c.slots = (uint32_t)slots;
    c.key.assign(slots, moe_stream_no_key);
    c.stamp.assign(slots, 0);
    c.heat.assign(slots, 0);
    c.generation++;
    if (!c.provisional) {
        fprintf(stderr, "ds4: CUDA MoE expert cache on device %d: %u experts, %.2f GiB\n",
                device, c.slots, (double)(slots * slot_bytes) / 1073741824.0);
    }
    return true;
}

#ifdef _WIN32
/* One expert part from the file into dst.  Through the mapping an expert
 * the page cache has dropped arrives a page fault at a time, 5 to 9 ms for
 * 1.4 MiB; one unbuffered read of the sectors around it is the drive's speed. */
static bool moe_layer_reader_open(void);
static bool moe_stream_file_read(char *dst, const char *map, uint64_t offset, uint64_t bytes);
#else
static bool moe_stream_file_read(char *dst, const char *map, uint64_t offset, uint64_t bytes) {
    memcpy(dst, map + offset, bytes);
    return true;
}
#endif

/* The slot holding (layer, expert), loaded first when it is not cached.  The
 * victim is the least recently used slot that the open batch has not touched. */
static int moe_stream_slot(moe_stream_cache &c, const ds4_gpu_stream_expert_table *t, uint32_t expert) {
    const uint64_t key = (uint64_t)t->layer << 32 | expert;
    int slot = -1;
    const auto found = c.slot_of.find(key);
    if (found != c.slot_of.end()) {
        slot = (int)found->second;
    } else {
        /* With the CPU path the device keeps what is used most, so the
         * least used goes; otherwise the least recently used. */
        uint64_t oldest = g_moe_stream_clock;
        uint32_t coldest = UINT32_MAX;
        for (uint32_t i = 0; i < c.slots; i++) {
            const bool free_now = c.stamp[i] < g_moe_stream_clock;
            if (free_now && (g_moe_stream_by_use ? c.heat[i] < coldest : c.stamp[i] < oldest)) {
                oldest = c.stamp[i];
                coldest = c.heat[i];
                slot = (int)i;
            }
        }
        if (slot >= 0) {
            /* An expert is in exactly one place: a device slot, a slot of the
             * RAM tier, or only the file.  The one this load pushes out of the
             * device goes to RAM, so nothing used recently has to come from
             * the disk again.  Every copy is queued on the stream the kernels
             * run on and nothing waits here: the save runs before the load
             * that overwrites it, the load before the kernel that reads it. */
            moe_stream_ram &ram = g_moe_stream_ram;
            moe_stream_ram_init(ram, c.slot_bytes);
            const cudaStream_t stream = cuda_decode_stream();
            char *at = c.gate + (uint64_t)slot * c.slot_bytes;
            const uint64_t victim = c.key[slot];
            c.filled += victim == moe_stream_no_key;
            const auto in_ram = ram.slot_of.find(key);
            const bool ram_hit = in_ram != ram.slot_of.end();
            const uint32_t source_slot = ram_hit ? in_ram->second : UINT32_MAX;
            bool ok = true;
            if (victim != moe_stream_no_key && ram.slots) {
                int keep = -1;
                if (!ram.free_slots.empty()) {
                    keep = (int)ram.free_slots.back();
                    ram.free_slots.pop_back();
                } else {
                    uint64_t oldest = UINT64_MAX;
                    for (uint32_t i = 0; i < ram.slots; i++) {
                        if (ram.key[i] != moe_stream_no_key && i != source_slot && ram.stamp[i] < oldest) {
                            oldest = ram.stamp[i];
                            keep = (int)i;
                        }
                    }
                    if (keep >= 0) ram.slot_of.erase(ram.key[keep]);
                }
                if (keep >= 0) {
                    ok = cuda_ok(cudaMemcpyAsync(ram.at((uint32_t)keep), at, c.slot_bytes, cudaMemcpyDeviceToHost,
                                                 stream), "MoE expert cache save");
                    ram.key[keep] = victim;
                    ram.uses[keep] = c.heat[slot];
                    ram.stamp[keep] = g_moe_stream_clock;
                    ram.slot_of[victim] = (uint32_t)keep;
                }
            }
            const auto t0 = std::chrono::steady_clock::now();
            const char *source = NULL;
            c.heat[slot] = ram_hit ? ram.uses[source_slot] : 0;
            if (ram_hit) {
                /* The helper's save of the expert into this slot may still be queued. */
                ok = ok && moe_stream_ram_settle(source_slot);
                source = ram.at(source_slot);
                ram.slot_of.erase(key);
                ram.key[source_slot] = moe_stream_no_key;
                ram.released.push_back(source_slot);
            } else {
                if (c.stage_used == moe_stream_stage_slots) {
                    ok = ok && cuda_ok(cudaStreamSynchronize(stream), "MoE expert cache staging wait");
                    c.stage_used = 0;
                    /* On the helper this is its stream: its promotion copies are done too. */
                    if (int here = -1; cudaGetDevice(&here) == cudaSuccess && here == g_moe_stream.helper_device)
                        moe_stream_promote_drained();
                }
                char *staged = c.stage + (uint64_t)c.stage_used++ * c.slot_bytes;
                const char *map = (const char *)t->model_map;
                ok = ok && moe_stream_file_read(staged, map, t->gate_offset + (uint64_t)expert * c.gate_bytes,
                                                c.gate_bytes) &&
                     moe_stream_file_read(staged + c.gate_bytes, map, t->up_offset + (uint64_t)expert * c.gate_bytes,
                                          c.gate_bytes) &&
                     moe_stream_file_read(staged + 2 * c.gate_bytes, map,
                                          t->down_offset + (uint64_t)expert * c.down_bytes, c.down_bytes);
                source = staged;
            }
            const auto t1 = std::chrono::steady_clock::now();
            ok = ok && cuda_ok(cudaMemcpyAsync(at, source, c.slot_bytes, cudaMemcpyHostToDevice, stream),
                               "MoE expert cache load");
            c.misses++;
            c.from_ram += ram_hit;
            c.host_s += std::chrono::duration<double>(t1 - t0).count();
            c.device_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            if (c.key[slot] != moe_stream_no_key) c.slot_of.erase(c.key[slot]);
            c.key[slot] = ok ? key : moe_stream_no_key;
            c.generation++;
            if (ok) {
                c.slot_of[key] = (uint32_t)slot;
            } else {
                /* The slot ends up empty: one filled fewer, whether the
                 * victim was counted above or held an expert. */
                c.filled--;
                slot = -1;
            }
        }
    }
    if (slot >= 0) {
        c.stamp[slot] = g_moe_stream_clock;
        c.heat[slot]++;
    }
    return slot;
}

/* The fewest uses of any expert the device could give up in this batch. */
static uint32_t moe_stream_coldest(const moe_stream_cache &c) {
    uint32_t coldest = UINT32_MAX;
    for (uint32_t i = 0; i < c.slots; i++)
        if (c.stamp[i] < g_moe_stream_clock && c.heat[i] < coldest) coldest = c.heat[i];
    return coldest;
}

/* The RAM tier slot holding (layer, expert) for the CPU, read from the file
 * first when it is not there; -1 when every slot is in use by this batch. */
static int moe_stream_ram_hold(moe_stream_ram &ram, const moe_stream_cache &c,
                               const ds4_gpu_stream_expert_table *t, uint32_t expert) {
    const uint64_t key = (uint64_t)t->layer << 32 | expert;
    int slot = -1;
    if (const auto found = ram.slot_of.find(key); found != ram.slot_of.end()) {
        slot = (int)found->second;
    } else {
        if (!ram.free_slots.empty()) {
            slot = (int)ram.free_slots.back();
            ram.free_slots.pop_back();
        } else {
            uint64_t oldest = g_moe_stream_clock;
            for (uint32_t i = 0; i < ram.slots; i++) {
                if (ram.key[i] != moe_stream_no_key && ram.stamp[i] < oldest) {
                    oldest = ram.stamp[i];
                    slot = (int)i;
                }
            }
            if (slot >= 0) ram.slot_of.erase(ram.key[slot]);
        }
        if (slot >= 0) {
            char *at = ram.at((uint32_t)slot);
            const char *map = (const char *)t->model_map;
            const auto t0 = std::chrono::steady_clock::now();
            if (!moe_stream_file_read(at, map, t->gate_offset + (uint64_t)expert * c.gate_bytes, c.gate_bytes) ||
                !moe_stream_file_read(at + c.gate_bytes, map, t->up_offset + (uint64_t)expert * c.gate_bytes,
                                      c.gate_bytes) ||
                !moe_stream_file_read(at + 2 * c.gate_bytes, map, t->down_offset + (uint64_t)expert * c.down_bytes,
                                      c.down_bytes)) {
                fprintf(stderr, "ds4: CUDA MoE RAM tier could not read layer %u expert %u from the model file\n",
                        t->layer, expert);
                exit(1);
            }
            ram.file_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            ram.file_reads++;
            ram.key[slot] = key;
            ram.uses[slot] = 0;
            ram.slot_of[key] = (uint32_t)slot;
        }
    }
    if (slot >= 0) {
        ram.stamp[slot] = g_moe_stream_clock;
        ram.uses[slot]++;
    }
    return slot;
}

static uint32_t moe_cpu_block_bytes(unsigned type) {
    return type == 10 ? 84u : type == 12 ? 144u : type == 16 ? 66u : 0u;
}

/* The CPU's share of the open batch, into g_moe_cpu.out: gate and up of each
 * expert against its token's input, SiLU, then down.  No CUDA calls, so it
 * runs while the devices work on theirs. */
static bool moe_stream_cpu_compute(const moe_stream_cache &c, unsigned down_type, unsigned NS, unsigned M) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t n = (uint32_t)g_moe_stream.cpu_pairs.size();
    const unsigned gate_type = g_moe_stream.mid_type, K = g_moe_stream.mid_K, F = g_moe_stream.mid_M;
    const uint64_t gate_row = c.gate_bytes / F, down_row = c.down_bytes / M;
    const uint32_t gate_block = moe_cpu_block_bytes(gate_type), down_block = moe_cpu_block_bytes(down_type);
    const uint32_t down_in = down_block ? (uint32_t)(down_row / down_block) * 256u : 0u;
    if (!gate_block || !down_block || K % 256u || gate_row != (uint64_t)K / 256u * gate_block || down_in < F) {
        fprintf(stderr, "ds4: CUDA MoE CPU path cannot evaluate expert types %u/%u of %u/%u inputs\n",
                gate_type, down_type, K, F);
        return false;
    }
    const uint64_t xq_bytes = (uint64_t)K / 256u * g_moe_cpu.xq_block_bytes;
    const uint64_t midq_bytes = (uint64_t)down_in / 256u * g_moe_cpu.xq_block_bytes;
    const moe_stream_ram &ram = g_moe_stream_ram;
    g_moe_cpu.xq.resize(n * xq_bytes);
    g_moe_cpu.midq.resize(n * midq_bytes);
    g_moe_cpu.gate.resize((uint64_t)n * F);
    g_moe_cpu.up.resize((uint64_t)n * F);
    g_moe_cpu.mid.assign((uint64_t)n * down_in, 0.0f);
    if (!moe_stream_grow(g_moe_cpu.out, g_moe_cpu.out_cap, (uint64_t)n * M * 4, true, "MoE CPU output")) return false;
    const uint32_t gate_chunks = (F + 63u) / 64u, down_chunks = (M + 127u) / 128u;
    /* Quantising a vector is microseconds: done here, a round of its own
     * cost more in handing it out than it saved. */
    for (uint32_t i = 0; i < n; i++)
        g_moe_cpu.quantize(g_moe_stream.host_x + g_moe_stream.cpu_pairs[i] / NS * K, &g_moe_cpu.xq[i * xq_bytes], K);
    g_moe_cpu_pool.run(n * gate_chunks, [&](uint32_t job) {
        const uint32_t i = job / gate_chunks, first = job % gate_chunks * 64u, end = first + 64u < F ? first + 64u : F;
        const char *expert = ram.at(g_moe_stream.cpu_slots[i]);
        float *gate = &g_moe_cpu.gate[(uint64_t)i * F], *up = &g_moe_cpu.up[(uint64_t)i * F];
        g_moe_cpu.rows(gate_type, expert, gate_row, K, first, end, &g_moe_cpu.xq[i * xq_bytes], gate);
        g_moe_cpu.rows(gate_type, expert + c.gate_bytes, gate_row, K, first, end, &g_moe_cpu.xq[i * xq_bytes], up);
        for (uint32_t r = first; r < end; r++)
            g_moe_cpu.mid[(uint64_t)i * down_in + r] = gate[r] / (1.0f + expf(-gate[r])) * up[r];
    });
    for (uint32_t i = 0; i < n; i++)
        g_moe_cpu.quantize(&g_moe_cpu.mid[(uint64_t)i * down_in], &g_moe_cpu.midq[i * midq_bytes], down_in);
    g_moe_cpu_pool.run(n * down_chunks, [&](uint32_t job) {
        const uint32_t i = job / down_chunks, first = job % down_chunks * 128u, end = first + 128u < M ? first + 128u : M;
        g_moe_cpu.rows(down_type, ram.at(g_moe_stream.cpu_slots[i]) + 2 * c.gate_bytes, down_row, down_in, first, end,
                       &g_moe_cpu.midq[i * midq_bytes], g_moe_cpu.out + (uint64_t)i * M);
    });
    g_moe_stream.cpu_done = true;
    g_moe_cpu.pairs += n;
    g_moe_cpu.layers++;
    g_moe_cpu.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

/* What the helper and the CPU computed reaches the primary as one block, a
 * pair to a row, and one kernel puts the rows in place: a copy per row was a
 * queue entry each on a device behind a slow link, eight to a layer. */
static bool moe_stream_ext_open(moe_stream_cache &c, uint64_t pairs, unsigned M) {
    const uint64_t bytes = pairs * M * 4;
    const bool ok = moe_stream_grow(c.host_ext, c.host_ext_cap, bytes, true, "MoE outside rows staging") &&
                    moe_stream_grow(c.ext, c.ext_cap, bytes, false, "MoE outside rows");
    /* host_ext is the source of the previous layer's merge copy, queued
     * before this layer's selection read-back: the wait behind that read-back
     * (moe_stream_wait in _begin, or the event in moe_stream_settle) has run
     * it, so the memset finds it free. */
    if (ok) memset(c.host_ext, 0, bytes);
    c.ext_used = false;
    return ok;
}

static bool moe_stream_ext_merge(moe_stream_cache &c, float *part, unsigned stride, unsigned T, unsigned NS,
                                 unsigned M) {
    bool ok = true;
    if (c.ext_used) {
        ok = cuda_ok(cudaMemcpyAsync(c.ext, c.host_ext, (uint64_t)T * NS * M * 4, cudaMemcpyHostToDevice,
                                     cuda_decode_stream()), "MoE outside rows write");
        if (ok) {
            moe_merge<<<dim3((M + 255) / 256, NS, T), 256, 0, cuda_decode_stream()>>>(part, c.ext, c.selected, NS,
                                                                                    stride, M);
            ok = launched() != 0;
        }
    }
    return ok;
}

/* The CPU's rows on their way into the primary's part. */
static bool moe_stream_cpu_write(moe_stream_cache &c, unsigned M) {
    for (size_t i = 0; i < g_moe_stream.cpu_pairs.size(); i++)
        memcpy(c.host_ext + g_moe_stream.cpu_pairs[i] * M, g_moe_cpu.out + (uint64_t)i * M, (uint64_t)M * 4);
    c.ext_used = c.ext_used || !g_moe_stream.cpu_pairs.empty();
    return true;
}

/* The helper's gate and up over its share, started once the primary's own
 * kernel is queued; x_host is the layer input read back from the primary. */
static bool moe_stream_helper_mid(const float *x_host, unsigned type, unsigned T, unsigned NS,
                                  unsigned K, unsigned M) {
    moe_stream_cache &h = *g_moe_stream.helper;
    const auto t0 = std::chrono::steady_clock::now();
    if (!cuda_ok(cudaSetDevice(g_moe_stream.helper_device), "MoE helper device")) return false;
    bool ok = moe_stream_grow(h.hx, h.hx_cap, (uint64_t)T * K * 4, false, "MoE helper input") &&
        moe_stream_grow(h.hmid, h.hmid_cap, (uint64_t)T * NS * M * 4, false, "MoE helper mid") &&
        cuda_ok(cudaMemcpyAsync(h.hx, x_host, (uint64_t)T * K * 4, cudaMemcpyHostToDevice, 0),
                "MoE helper input copy") &&
        moe_mv_dispatch(h.hmid, h.hx, h.selected, h.gate, h.up, NULL, NULL, type, UINT_MAX, h.slots,
                        T, NS, K, M, false, h.slot_bytes);
    ok = cuda_ok(cudaSetDevice(g_moe_stream.device), "MoE primary device") && ok;
    h.hop_s[3] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

/* A RAM expert that has earned a helper slot is evaluated by the CPU the time
 * it earns it and changes places afterwards, once the helper's results for
 * the layer are in: its upload and the save of the expert it replaces then run
 * while the primary is busy with the next layer, where swapping at once held
 * the layer up for milliseconds.  Both copies are page-locked at the host end,
 * so queueing them does not block: the upload goes through a small ring, the
 * save into the RAM slot the upload left, which is why only experts in the
 * page-locked part of the tier are promoted this way. */
static std::vector<std::pair<uint64_t, uint32_t>> g_moe_promote;   /* key, RAM slot */
static std::vector<uint32_t> g_moe_ram_in_flight;                  /* RAM slots a queued save writes to */
static char *g_moe_promote_stage;
static uint32_t g_moe_promote_staged;
static const uint32_t moe_promote_stage_slots = 8;

/* The helper's stream has run dry: nothing queued reads the ring or writes
 * the tier any more. */
static void moe_stream_promote_drained(void) {
    g_moe_promote_staged = 0;
    g_moe_ram_in_flight.clear();
}

/* Before a RAM slot is read on any other stream: waits out a save the
 * helper may still have queued into it. */
static bool moe_stream_ram_settle(uint32_t ram_slot) {
    bool ok = true;
    if (std::find(g_moe_ram_in_flight.begin(), g_moe_ram_in_flight.end(), ram_slot) != g_moe_ram_in_flight.end()) {
        int here = -1;
        ok = cuda_ok(cudaGetDevice(&here), "MoE device") &&
             cuda_ok(cudaSetDevice(g_moe_stream.helper_device), "MoE helper device") &&
             cuda_ok(cudaStreamSynchronize(0), "MoE promotion save wait");
        if (ok) moe_stream_promote_drained();
        ok = cuda_ok(cudaSetDevice(here), "MoE primary device") && ok;
    }
    return ok;
}

static bool moe_stream_promote_flush(void) {
    bool ok = true;
    if (!g_moe_promote.empty()) {
        moe_stream_cache &h = g_moe_stream_caches[g_moe_stream.helper_device];
        moe_stream_ram &ram = g_moe_stream_ram;
        ok = cuda_ok(cudaSetDevice(g_moe_stream.helper_device), "MoE helper device") &&
             (g_moe_promote_stage ||
              cuda_ok(cudaMallocHost((void **)&g_moe_promote_stage, (uint64_t)moe_promote_stage_slots * h.slot_bytes),
                      "MoE promotion staging"));
        for (size_t i = 0; i < g_moe_promote.size() && ok; i++) {
            const uint64_t key = g_moe_promote[i].first;
            const uint32_t ram_slot = g_moe_promote[i].second;
            int slot = -1;
            uint32_t coldest = UINT32_MAX;
            for (uint32_t s = 0; s < h.slots; s++) {
                if (h.stamp[s] < g_moe_stream_clock && h.heat[s] < coldest) {
                    coldest = h.heat[s];
                    slot = (int)s;
                }
            }
            if (slot >= 0 && coldest < ram.uses[ram_slot]) {
                if (g_moe_promote_staged == moe_promote_stage_slots) {
                    ok = cuda_ok(cudaStreamSynchronize(0), "MoE promotion staging wait");
                    moe_stream_promote_drained();
                }
                char *staged = g_moe_promote_stage + (uint64_t)g_moe_promote_staged++ * h.slot_bytes;
                char *at = h.gate + (uint64_t)slot * h.slot_bytes;
                const uint64_t victim = h.key[slot];
                memcpy(staged, ram.at(ram_slot), h.slot_bytes);
                /* The save goes first and the load after it; if either copy
                 * is refused neither tier is told anything, so both still
                 * name what their bytes hold. */
                const bool saved = victim == moe_stream_no_key ||
                    cuda_ok(cudaMemcpyAsync(ram.at(ram_slot), at, h.slot_bytes, cudaMemcpyDeviceToHost, 0),
                            "MoE promotion save");
                const bool loaded = saved &&
                    cuda_ok(cudaMemcpyAsync(at, staged, h.slot_bytes, cudaMemcpyHostToDevice, 0), "MoE promotion load");
                ok = ok && loaded;
                if (saved && !loaded && victim != moe_stream_no_key) {
                    /* The save alone ran: the RAM slot has the victim now, nothing has the promoted expert. */
                    ram.slot_of.erase(key);
                    h.slot_of.erase(victim);
                    ram.key[ram_slot] = victim;
                    ram.uses[ram_slot] = h.heat[slot];
                    ram.stamp[ram_slot] = g_moe_stream_clock;
                    ram.slot_of[victim] = ram_slot;
                    g_moe_ram_in_flight.push_back(ram_slot);
                    h.key[slot] = moe_stream_no_key;
                    h.heat[slot] = 0;
                    h.stamp[slot] = 0;
                    h.filled--;
                    h.generation++;
                } else if (loaded) {
                    const uint32_t heat = ram.uses[ram_slot];
                    ram.slot_of.erase(key);
                    if (victim != moe_stream_no_key) {
                        h.slot_of.erase(victim);
                        ram.key[ram_slot] = victim;
                        ram.uses[ram_slot] = h.heat[slot];
                        ram.stamp[ram_slot] = g_moe_stream_clock;
                        ram.slot_of[victim] = ram_slot;
                        g_moe_ram_in_flight.push_back(ram_slot);
                    } else {
                        ram.key[ram_slot] = moe_stream_no_key;
                        ram.released.push_back(ram_slot);
                        h.filled++;
                    }
                    h.key[slot] = key;
                    h.slot_of[key] = (uint32_t)slot;
                    h.heat[slot] = heat;
                    h.stamp[slot] = g_moe_stream_clock;
                    h.generation++;
                    h.misses++;
                    h.from_ram++;
                }
            }
        }
        g_moe_promote.clear();
        ok = cuda_ok(cudaSetDevice(g_moe_stream.device), "MoE primary device") && ok;
    }
    return ok;
}

/* A helper that computes a few experts every few milliseconds looks idle to
 * its driver, which drops it to its lowest clocks: an RTX 4070 Laptop served
 * decoded tokens at 210 MHz of its 3105 and took over a millisecond per
 * layer.  DS4_CUDA_MOE_STREAM_KEEPALIVE=N keeps one small kernel of N
 * thousand iterations running on a stream of its own on the current device,
 * started again whenever the last one has ended, so the device reports load.
 * For the helper only: the graph's device is busy enough to hold its clocks,
 * and the same kernel there doubled the time its queue took (22 to 43 ms a
 * token on an RTX 5060 Ti). */
static void moe_stream_keep_awake(void) {
    static const uint64_t thousands = moe_stream_env_mib("DS4_CUDA_MOE_STREAM_KEEPALIVE", 0) >> 20;
    struct awake { cudaStream_t stream = NULL; unsigned *sink = NULL; bool tried = false; };
    static std::unordered_map<int, awake> by_device;
    int device = -1;
    if (thousands && cudaGetDevice(&device) == cudaSuccess) {
        awake &here = by_device[device];
        if (!here.tried) {
            here.tried = true;
            if (cudaStreamCreateWithFlags(&here.stream, cudaStreamNonBlocking) != cudaSuccess ||
                cudaMalloc((void **)&here.sink, sizeof(*here.sink)) != cudaSuccess) here.stream = NULL;
        }
        if (here.stream && cudaStreamQuery(here.stream) == cudaSuccess)
            keep_awake<<<1, 32, 0, here.stream>>>((unsigned)(thousands * 1000u), here.sink);
        (void)cudaGetLastError();
    }
}

/* The helper's down projection, then its rows into the primary's part, queued
 * behind the primary's own down kernel, which wrote zeros there. */
static bool moe_stream_helper_down(float *part, unsigned stride, unsigned type, unsigned T, unsigned NS,
                                   unsigned K, unsigned M) {
    moe_stream_cache &h = *g_moe_stream.helper;
    const uint64_t bytes = (uint64_t)T * NS * M * 4;
    const auto t0 = std::chrono::steady_clock::now();
    if (!cuda_ok(cudaSetDevice(g_moe_stream.helper_device), "MoE helper device")) return false;
    bool ok = moe_stream_grow(h.hpart, h.hpart_cap, bytes, false, "MoE helper output") &&
        moe_stream_grow(h.host_part, h.host_part_cap, bytes, true, "MoE helper output staging") &&
        moe_mv_dispatch(h.hpart, h.hmid, h.selected, h.down, NULL, NULL, NULL, type, UINT_MAX, h.slots,
                        T, NS, K, M, true, h.slot_bytes);
    if (ok && !g_moe_stream.cpu_pairs.empty()) ok = moe_stream_cpu_compute(*g_moe_stream.cache, type, NS, M);
    const auto t1 = std::chrono::steady_clock::now();
    ok = ok && cuda_ok(cudaMemcpyAsync(h.host_part, h.hpart, bytes, cudaMemcpyDeviceToHost, 0),
                       "MoE helper output read") &&
        moe_stream_wait(0, "MoE helper wait");
    if (ok) moe_stream_promote_drained();
    moe_stream_keep_awake();
    const auto t2 = std::chrono::steady_clock::now();
    ok = cuda_ok(cudaSetDevice(g_moe_stream.device), "MoE primary device") && ok;
    moe_stream_cache &c = g_moe_stream_caches[g_moe_stream.device];
    for (size_t i = 0; i < g_moe_stream.helper_pairs.size() && ok; i++) {
        const uint64_t pair = g_moe_stream.helper_pairs[i];
        memcpy(c.host_ext + pair * M, h.host_part + pair * M, (uint64_t)M * 4);
    }
    c.ext_used = c.ext_used || !g_moe_stream.helper_pairs.empty();
    h.hop_s[0] += std::chrono::duration<double>(t1 - t0).count();
    h.hop_s[1] += std::chrono::duration<double>(t2 - t1).count();
    h.hop_s[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    h.hops++;
    return ok;
}

}

namespace qwen4_cuda {
/* One whole layer's experts for a large batch, as the file lays them out:
 * all gates, all ups, all downs (see ds4_gpu_qwen4_moe_stream_layer). */
/* What a device keeps for whole-layer batches: the layer buffer and, for the
 * caller that has no list tensors of its own, the grouped kernels' scratch. */
struct moe_layer_device {
    char *buffer = NULL;
    uint64_t cap = 0;
    bool unavailable = false;
    std::vector<bool> runs_layer;         /* layers seen on this device; a share of its cache is per such layer */
    uint32_t layers = 0, last_layer = 0;
    uint32_t assigned = 0;                /* layers the graph placed here, known before any is seen */
    bool counted = false;                 /* a pass over this device's layers has wrapped round */
    bool in_arena = false;                /* buffer is the tail of this device's expert cache, not its own */
    bool arena_tried = false;
    int *lists = NULL, *counts = NULL, *none = NULL;
    unsigned *tiles = NULL;
    uint64_t lists_cap = 0, counts_cap = 0, none_cap = 0, tiles_cap = 0;
    /* When this device runs another one's batches (_stream_layer_remote):
     * the batch's input, selection and weights, and what it computes. */
    float *rx = NULL, *rw = NULL, *rmid = NULL, *rpart = NULL, *rsum = NULL, *ones = NULL;
    int *rsel = NULL;
    uint64_t rx_cap = 0, rw_cap = 0, rmid_cap = 0, rpart_cap = 0, rsum_cap = 0, rsel_cap = 0, ones_cap = 0;
    bool remote_unavailable = false;
};
/* Page-locked staging of a remote batch on its way between the devices. */
static struct {
    float *x, *w, *sum;
    int *sel;
    uint64_t x_cap, w_cap, sum_cap, sel_cap;
} g_moe_remote;
static std::unordered_map<int, moe_layer_device> g_moe_layer_devices;
static struct {
    char *buffer;                         /* the current device's, while a layer is open */
    uint64_t gate_off, up_off, down_off, gate_bytes;
    bool active, seen, complete;
    bool remote;                          /* the layer being loaded serves another device's batch */
    /* A pass leaves the experts its rows used most where single tokens find
     * them: the top of each layer in the device cache, copied on the device,
     * the next in the RAM tier, copied as the layer's bytes go by.  Which
     * those are is only known from an earlier pass, hence the counters. */
    uint32_t last_layer, layers, forwards;
    /* Where moe_layer_read drops each expert of the layer it is reading as
     * the bytes go by, -1 for nowhere: a RAM tier slot, or a slot of the
     * device cache that keeps a fixed share of the ranking. */
    std::vector<int> ram_slot, fixed_slot;
    moe_stream_cache *fixed_cache;
    std::vector<bool> ram_done, fixed_done;   /* per layer: its experts were handed out */
} g_moe_layer;
static std::unordered_map<uint64_t, uint32_t> g_moe_layer_heat;   /* rows that used (layer << 32 | expert) */
/* The graph's layers are spread over the devices (ds4_gpu_qwen4_moe_stream_set_device_layers):
 * each device's cache serves its own layers, so no device helps another. */
static bool g_moe_split;

#ifdef _WIN32
/* Faulting the mapping in page by page reads at a fraction of what the
 * drive does; these go to the device through page-locked buffers filled by
 * large unbuffered reads, each read overlapping the previous buffer's copy. */
struct moe_layer_stage {
    char *stage[4];
    cudaEvent_t copied[4];
    unsigned next;
    bool usable;
};
/* The buffers and their events belong to one device each: an event cannot be
 * recorded on another device's stream, and a buffer must not be refilled
 * while a copy from it to some other device is still queued. */
static struct {
    HANDLE file;
    DWORD open_error;                     /* why file is invalid, for the one report */
    bool tried;
    std::unordered_map<int, moe_layer_stage> by_device;
    double read_s, wait_s;                /* in ReadFile and waiting for a buffer, for the stats line */
    uint64_t read_bytes;
} g_moe_layer_reader;
static const uint64_t moe_layer_stage_bytes = 32ull << 20, moe_layer_sector = 4096;

static moe_layer_stage *moe_layer_stage_here(void) {
    int device = -1;
    moe_layer_stage *here = NULL;
    if (cudaGetDevice(&device) == cudaSuccess) {
        const bool made = g_moe_layer_reader.by_device.count(device) != 0;
        moe_layer_stage &s = g_moe_layer_reader.by_device[device];
        if (!made) {
            s.usable = true;
            for (unsigned i = 0; i < 4 && s.usable; i++)
                s.usable = cudaMallocHost((void **)&s.stage[i], moe_layer_stage_bytes + moe_layer_sector) ==
                               cudaSuccess &&
                           cudaEventCreateWithFlags(&s.copied[i], cudaEventDisableTiming) == cudaSuccess;
            (void)cudaGetLastError();
        }
        if (s.usable) here = &s;
    }
    return here;
}

static bool moe_layer_reader_open(void) {
    if (!g_moe_layer_reader.tried) {
        g_moe_layer_reader.tried = true;
        g_moe_layer_reader.file = INVALID_HANDLE_VALUE;
        if (g_model_fd >= 0)
            g_moe_layer_reader.file = ReOpenFile((HANDLE)_get_osfhandle(g_model_fd), GENERIC_READ,
                                                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                                 FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN);
        if (g_moe_layer_reader.file == INVALID_HANDLE_VALUE) g_moe_layer_reader.open_error = GetLastError();
        (void)cudaGetLastError();
    }
    return g_moe_layer_reader.file != INVALID_HANDLE_VALUE && moe_layer_stage_here() != NULL;
}

static bool moe_stream_file_read(char *dst, const char *map, uint64_t offset, uint64_t bytes) {
    static char *buffer = NULL;
    static uint64_t cap = 0;
    const uint64_t start = offset & ~(moe_layer_sector - 1);
    const uint64_t span = (offset - start + bytes + moe_layer_sector - 1) & ~(moe_layer_sector - 1);
    static bool reported = false;
    bool done = false;
    const char *failed = NULL;
    DWORD error = 0;
    (void)moe_layer_reader_open();
    if (g_moe_layer_reader.file == INVALID_HANDLE_VALUE) {
        failed = "opening an unbuffered handle";
        error = g_moe_layer_reader.open_error;
    } else {
        if (cap < span) {
            if (buffer) (void)VirtualFree(buffer, 0, MEM_RELEASE);
            buffer = (char *)VirtualAlloc(NULL, span, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            cap = buffer ? span : 0;
            if (!buffer) error = GetLastError();
        }
        if (!buffer) {
            failed = "VirtualAlloc";
        } else {
            OVERLAPPED where = {};
            where.Offset = (DWORD)start;
            where.OffsetHigh = (DWORD)(start >> 32);
            DWORD got = 0;
            if (!ReadFile(g_moe_layer_reader.file, buffer, (DWORD)span, &got, &where) &&
                GetLastError() != ERROR_HANDLE_EOF) {
                failed = "ReadFile";
                error = GetLastError();
            } else if (got < offset - start + bytes) {
                failed = "a short read";
            } else {
                memcpy(dst, buffer + (offset - start), bytes);
                done = true;
            }
        }
    }
    /* Without the unbuffered handle the mapping still has it, slowly: 5 to
     * 9 ms an expert, so the first time is reported. */
    if (!done) {
        if (!reported) {
            reported = true;
            fprintf(stderr, "ds4: CUDA MoE expert read from the model file failed at %s (Windows error %lu); "
                            "reading through the mapping instead\n", failed, (unsigned long)error);
        }
        memcpy(dst, map + offset, bytes);
    }
    return true;
}

/* Reads experts first_expert.. of one tensor, bytes in all, from file_off. */
static bool moe_layer_read(char *device, uint64_t file_off, uint64_t bytes, cudaStream_t stream,
                           uint64_t expert_bytes, uint64_t slot_part, uint32_t first_expert = 0) {
    moe_layer_stage *here = moe_layer_stage_here();
    bool ok = here != NULL;
    for (uint64_t done = 0; done < bytes && ok;) {
        const unsigned b = here->next++ % 4u;
        const uint64_t at = file_off + done, start = at & ~(moe_layer_sector - 1);
        const uint64_t want = bytes - done < moe_layer_stage_bytes ? bytes - done : moe_layer_stage_bytes;
        const uint64_t span = (at - start + want + moe_layer_sector - 1) & ~(moe_layer_sector - 1);
        OVERLAPPED where = {};
        where.Offset = (DWORD)start;
        where.OffsetHigh = (DWORD)(start >> 32);
        DWORD got = 0;
        /* The buffer's last copy must have run before it is filled again. */
        const char *data = here->stage[b] + (at - start);
        const auto t0 = std::chrono::steady_clock::now();
        ok = cuda_ok(cudaEventSynchronize(here->copied[b]), "MoE layer staging wait");
        const auto t1 = std::chrono::steady_clock::now();
        ok = ok && (ReadFile(g_moe_layer_reader.file, here->stage[b], (DWORD)span, &got, &where) ||
              GetLastError() == ERROR_HANDLE_EOF) && got >= at - start + want;
        g_moe_layer_reader.wait_s += std::chrono::duration<double>(t1 - t0).count();
        g_moe_layer_reader.read_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        g_moe_layer_reader.read_bytes += want;
        ok = ok &&
             (!device || cuda_ok(cudaMemcpyAsync(device + done, data, want, cudaMemcpyHostToDevice, stream),
                                 "MoE layer load"));
        for (uint64_t local = done / expert_bytes;
             ok && first_expert + local < g_moe_layer.ram_slot.size() && local * expert_bytes < done + want; local++) {
            const uint64_t e = first_expert + local;
            const uint64_t first = local * expert_bytes > done ? local * expert_bytes : done;
            const uint64_t end = (local + 1) * expert_bytes < done + want ? (local + 1) * expert_bytes : done + want;
            const uint64_t in_slot = slot_part + (first - local * expert_bytes);
            if (const int ram_slot = g_moe_layer.ram_slot[e]; ram_slot >= 0)
                memcpy(g_moe_stream_ram.at((uint32_t)ram_slot) + in_slot, data + (first - done), end - first);
            if (const int fixed_slot = g_moe_layer.fixed_slot[e]; fixed_slot >= 0) {
                const moe_stream_cache &c = *g_moe_layer.fixed_cache;
                ok = cuda_ok(cudaMemcpyAsync(c.gate + (uint64_t)fixed_slot * c.slot_bytes + in_slot, data + (first - done),
                                             end - first, cudaMemcpyHostToDevice, stream), "MoE fixed expert load");
            }
        }
        ok = ok && cuda_ok(cudaEventRecord(here->copied[b], stream), "MoE layer staging mark");
        done += want;
    }
    return ok;
}
#endif

#ifdef _WIN32
/* Reading a layer takes about as long as computing on it, so the next
 * layer's experts are read by a second thread while the devices work on the
 * current one: a pass then costs the longer of the two, not their sum.  Two
 * page-locked regions alternate, since the device copies out of one are still
 * queued when the next read starts. */
struct moe_layer_ahead {
    std::mutex lock;
    std::condition_variable wake, done;
    HANDLE file = INVALID_HANDLE_VALUE;
    char *region[2] = {NULL, NULL};
    uint64_t region_bytes = 0;
    bool started = false, unavailable = false;
    bool wanted = false, busy = false, held = false;
    ds4_gpu_stream_expert_table want = {}, have = {};
    int have_region = 1;
    uint64_t have_at[3] = {0, 0, 0};      /* gate, up and down inside the region */
    std::vector<ds4_gpu_stream_expert_table> tables;   /* by layer, as passes show them */
    std::vector<bool> known;

    bool read(char *to, uint64_t file_off, uint64_t bytes, uint64_t *data_at, uint64_t base) {
        const uint64_t start = file_off & ~(moe_layer_sector - 1);
        const uint64_t span = (file_off - start + bytes + moe_layer_sector - 1) & ~(moe_layer_sector - 1);
        bool ok = true;
        for (uint64_t done_bytes = 0; done_bytes < span && ok;) {
            const uint64_t piece = span - done_bytes < moe_layer_stage_bytes ? span - done_bytes : moe_layer_stage_bytes;
            OVERLAPPED where = {};
            where.Offset = (DWORD)(start + done_bytes);
            where.OffsetHigh = (DWORD)((start + done_bytes) >> 32);
            DWORD got = 0;
            ok = (ReadFile(file, to + base + done_bytes, (DWORD)piece, &got, &where) ||
                  GetLastError() == ERROR_HANDLE_EOF) && (got == piece || done_bytes + got >= file_off - start + bytes);
            done_bytes += piece;
        }
        *data_at = base + (file_off - start);
        return ok;
    }
    void loop() {
        std::unique_lock<std::mutex> hold(lock);
        for (;;) {
            wake.wait(hold, [this] { return wanted; });
            const ds4_gpu_stream_expert_table t = want;
            /* The region the layer before the current one was copied out
             * of: those copies have run, because every whole-layer batch
             * ends in a host wait on the copy stream (moe_layer_keep's
             * counts read) before the next layer asks for this read. */
            const int into = 1 - have_region;
            wanted = false;
            busy = true;
            held = false;
            hold.unlock();
            const uint64_t gate = (uint64_t)t.n_total_expert * t.gate_expert_bytes;
            const uint64_t down = (uint64_t)t.n_total_expert * t.down_expert_bytes;
            const uint64_t slack = 2 * moe_layer_sector;
            uint64_t at[3] = {0, 0, 0};
            const bool ok = read(region[into], t.gate_offset, gate, &at[0], 0) &&
                            read(region[into], t.up_offset, gate, &at[1], gate + slack) &&
                            read(region[into], t.down_offset, down, &at[2], 2 * (gate + slack));
            hold.lock();
            busy = false;
            held = ok;
            have = t;
            have_region = into;
            have_at[0] = at[0];
            have_at[1] = at[1];
            have_at[2] = at[2];
            done.notify_all();
        }
    }
};
static moe_layer_ahead g_moe_ahead;

/* The layer's experts if they were read ahead: host addresses of its gate, up
 * and down tensors, valid until the layer after next is asked for. */
static bool moe_layer_ahead_take(const ds4_gpu_stream_expert_table *t, const char *parts[3]) {
    moe_layer_ahead &a = g_moe_ahead;
    bool taken = false;
    if (a.started) {
        std::unique_lock<std::mutex> hold(a.lock);
        a.done.wait(hold, [&a] { return !a.busy && !a.wanted; });
        taken = a.held && a.have.layer == t->layer && a.have.gate_offset == t->gate_offset &&
                a.have.down_offset == t->down_offset;
        a.held = false;
        for (int i = 0; i < 3 && taken; i++) parts[i] = a.region[a.have_region] + a.have_at[i];
    }
    return taken;
}

/* Notes this layer's place in the file and has the one after it read. */
static void moe_layer_ahead_next(const ds4_gpu_stream_expert_table *t) {
    moe_layer_ahead &a = g_moe_ahead;
    const uint64_t gate = (uint64_t)t->n_total_expert * t->gate_expert_bytes;
    const uint64_t bytes = 2 * gate + (uint64_t)t->n_total_expert * t->down_expert_bytes + 6 * moe_layer_sector;
    if (!a.unavailable && !a.started && g_model_fd >= 0) {
        a.file = ReOpenFile((HANDLE)_get_osfhandle(g_model_fd), GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN);
        a.unavailable = a.file == INVALID_HANDLE_VALUE ||
                        cudaMallocHost((void **)&a.region[0], bytes) != cudaSuccess ||
                        cudaMallocHost((void **)&a.region[1], bytes) != cudaSuccess;
        (void)cudaGetLastError();
        if (!a.unavailable) {
            a.region_bytes = bytes;
            a.started = true;
            std::thread([&a] { a.loop(); }).detach();
        }
    }
    if (a.started && bytes <= a.region_bytes) {
        std::unique_lock<std::mutex> hold(a.lock);
        if (a.tables.size() <= t->layer) {
            a.tables.resize(t->layer + 1);
            a.known.resize(t->layer + 1, false);
        }
        a.tables[t->layer] = *t;
        a.known[t->layer] = true;
        /* The next layer seen so far, or the first one for the pass after this. */
        int next = -1;
        for (uint32_t l = t->layer + 1; l < a.known.size() && next < 0; l++)
            if (a.known[l]) next = (int)l;
        for (uint32_t l = 0; l < t->layer && next < 0; l++)
            if (a.known[l]) next = (int)l;
        if (next >= 0 && !a.busy) {
            a.want = a.tables[next];
            a.wanted = true;
            a.wake.notify_one();
        }
    }
}
#endif

/* The use counts outlive the process in <model>.expert-uses, written at exit.
 * A device that is slow to load, an external one above all, is given the
 * most used experts of that ranking once and then left alone. */
static std::string g_moe_stats_file;
static bool g_moe_stats_loaded;
static const char moe_stats_magic[8] = {'D', 'S', '4', 'X', 'U', 'S', 'E', '1'};

static void moe_stats_save(void) {
    const std::string scratch = g_moe_stats_file + ".tmp";
    FILE *out = g_moe_layer_heat.empty() ? NULL : fopen(scratch.c_str(), "wb");
    if (out) {
        const uint32_t header[2] = {g_moe_layer.layers, (uint32_t)g_moe_layer_heat.size()};
        bool ok = fwrite(moe_stats_magic, 8, 1, out) == 1 && fwrite(header, sizeof(header), 1, out) == 1;
        for (const auto &entry : g_moe_layer_heat)
            ok = ok && fwrite(&entry.first, 8, 1, out) == 1 && fwrite(&entry.second, 4, 1, out) == 1;
        ok = fclose(out) == 0 && ok;
        if (ok) {
            (void)remove(g_moe_stats_file.c_str());
            ok = rename(scratch.c_str(), g_moe_stats_file.c_str()) == 0;
        }
        if (!ok) fprintf(stderr, "ds4: could not write expert use counts to %s\n", g_moe_stats_file.c_str());
    }
}

static void moe_stats_load(void) {
    if (FILE *in = fopen(g_moe_stats_file.c_str(), "rb"); in != NULL) {
        char magic[8];
        uint32_t header[2] = {0, 0};
        bool ok = fread(magic, 8, 1, in) == 1 && !memcmp(magic, moe_stats_magic, 8) &&
                  fread(header, sizeof(header), 1, in) == 1 && header[0] > 0;
        std::unordered_map<uint64_t, uint32_t> counts;
        for (uint32_t i = 0; ok && i < header[1]; i++) {
            uint64_t key = 0;
            uint32_t uses = 0;
            ok = fread(&key, 8, 1, in) == 1 && fread(&uses, 4, 1, in) == 1;
            counts[key] = uses;
        }
        fclose(in);
        if (ok) {
            g_moe_layer_heat = std::move(counts);
            g_moe_layer.layers = header[0];
            g_moe_layer.complete = true;
            g_moe_stats_loaded = !g_moe_layer_heat.empty();
            fprintf(stderr, "ds4: expert use counts for %u experts read from %s\n", header[1],
                    g_moe_stats_file.c_str());
        } else {
            fprintf(stderr, "ds4: %s is not a complete expert use count file; ignored\n", g_moe_stats_file.c_str());
        }
    }
}

/* The layer's experts by rows that used them so far, most used first. */
static std::vector<std::pair<uint32_t, uint32_t>> moe_layer_ranked(uint32_t layer, uint32_t experts) {
    std::vector<std::pair<uint32_t, uint32_t>> ranked;
    for (uint32_t e = 0; e < experts; e++)
        if (const auto heat = g_moe_layer_heat.find((uint64_t)layer << 32 | e); heat != g_moe_layer_heat.end())
            ranked.emplace_back(heat->second, e);
    std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    return ranked;
}

static bool moe_layer_held(uint64_t key) {
    bool held = g_moe_stream_ram.slot_of.count(key) != 0;
    for (const auto &entry : g_moe_stream_caches) held = held || entry.second.slot_of.count(key) != 0;
    return held;
}

/* Every pass over the layers reports here.  The layer count is only trusted,
 * and shares only derived from it, once a pass has wrapped round or the count
 * came with the stored ranking. */
static void moe_layer_visit(uint32_t layer) {
    const bool wrapped = g_moe_layer.seen && layer <= g_moe_layer.last_layer;
    g_moe_layer.forwards += wrapped;
    g_moe_layer.complete = g_moe_layer.complete || wrapped;
    g_moe_layer.seen = true;
    g_moe_layer.last_layer = layer;
    if (layer >= g_moe_layer.layers) g_moe_layer.layers = layer + 1;
    int device = -1;
    if (cudaGetDevice(&device) == cudaSuccess) {
        moe_layer_device &here = g_moe_layer_devices[device];
        if (here.runs_layer.size() <= layer) here.runs_layer.resize(layer + 1, false);
        here.counted = here.counted || (here.layers && layer <= here.last_layer);
        here.last_layer = layer;
        here.layers += !here.runs_layer[layer];
        here.runs_layer[layer] = true;
    }
}

/* How many experts of each of its layers a device's cache of `slots` gets.
 * A device that has not been through all of its layers yet, as the graph's
 * device at the first token after a prompt that ran its experts elsewhere,
 * is taken to run every layer: counting only the ones it has seen would hand
 * the whole cache to the first few. */
static uint32_t moe_layer_share(uint32_t slots) {
    int device = -1;
    uint32_t layers = 0;
    if (g_moe_layer.complete && cudaGetDevice(&device) == cudaSuccess) {
        const moe_layer_device &here = g_moe_layer_devices[device];
        layers = here.assigned ? here.assigned : here.counted ? here.layers : g_moe_layer.layers;
    }
    return layers ? slots / layers : 0;
}

static bool moe_layer_pending(uint32_t layer, const moe_stream_cache &c) {
    const bool ram = g_moe_stream_ram.slots && (g_moe_layer.ram_done.size() <= layer || !g_moe_layer.ram_done[layer]);
    const bool fixed = g_moe_stats_loaded && c.slots && !c.provisional &&
                       (g_moe_layer.fixed_done.size() <= layer || !g_moe_layer.fixed_done[layer]);
    return g_moe_layer.complete && (ram || fixed);
}

/* Gives each expert of a layer its home, once per layer, for moe_layer_read
 * to fill from one sequential read: with a stored ranking the top of it goes
 * to the device cache c, to stay; the rest goes to the RAM tier as far as its
 * share reaches, so that later nothing comes from the disk.  c only takes
 * part once it is sized.  Returns whether anything was given a slot. */
static bool moe_layer_plan(const ds4_gpu_stream_expert_table *t, moe_stream_cache *c, bool prompt_pass) {
    moe_stream_ram &ram = g_moe_stream_ram;
    const uint32_t layer = t->layer, layers = g_moe_layer.complete ? g_moe_layer.layers : 0;
    g_moe_layer.ram_slot.assign(t->n_total_expert, -1);
    g_moe_layer.fixed_slot.assign(t->n_total_expert, -1);
    g_moe_layer.fixed_cache = c;
    if (g_moe_layer.ram_done.size() <= layer) {
        g_moe_layer.ram_done.resize(layer + 1, false);
        g_moe_layer.fixed_done.resize(layer + 1, false);
    }
    const bool sized = c && c->slots && !c->provisional;
    /* The fixed share is the graph's device's; one that only helps keeps a
     * cache that changes. */
    const bool want_fixed = sized && g_moe_stats_loaded && layers && !g_moe_layer.remote &&
                            !g_moe_layer.fixed_done[layer];
    const bool want_ram = ram.slots && layers && !g_moe_layer.ram_done[layer];
    /* The ranking's top positions are the device's even before it can take
     * them, so they are not put in RAM in the meantime. */
    const uint32_t fixed_share = g_moe_stats_loaded && layers && sized && !g_moe_layer.remote ?
                                 moe_layer_share(c->slots) : 0;
    /* What is free goes evenly to the layers still to come, so the tier ends
     * up full whatever the earlier layers left over. */
    /* Slots given up to a device are only freed by the next token's selection,
     * and a prompt pass has none: free them here, behind the copies out of them. */
    if (want_ram && !ram.released.empty() && moe_stream_wait(cuda_decode_stream(), "MoE RAM slot release")) {
        ram.free_slots.insert(ram.free_slots.end(), ram.released.begin(), ram.released.end());
        ram.released.clear();
    }
    uint32_t layers_to_come = 0;
    for (uint32_t i = 0; i < layers; i++) layers_to_come += i >= g_moe_layer.ram_done.size() || !g_moe_layer.ram_done[i];
    uint32_t ram_left = want_ram && layers_to_come ? (uint32_t)(ram.free_slots.size() / layers_to_come) : 0;
    /* The top of a layer's ranking is what a device takes over once it is
     * sized.  Out of page-locked memory that copy runs at the link's rate,
     * out of the rest at a quarter of it (13 s for Qwen3.8's 6.8 GiB), so
     * the locked slots, the lowest numbers, go to the top of every layer. */
    uint32_t locked_left = 0;
    if (ram_left) {
        std::sort(ram.free_slots.begin(), ram.free_slots.end(), std::greater<uint32_t>());
        for (const uint32_t slot : ram.free_slots) locked_left += slot < ram.locked_slots;
        locked_left /= layers_to_come;
    }
    uint32_t next_free = 0;
    bool any = false, ram_moved = false;
    std::vector<std::pair<uint32_t, uint32_t>> order = moe_layer_ranked(layer, t->n_total_expert);
    std::vector<bool> ranked(t->n_total_expert, false);
    for (const auto &entry : order) ranked[entry.second] = true;
    for (uint32_t e = 0; e < t->n_total_expert; e++)
        if (!ranked[e]) order.emplace_back(0u, e);
    for (size_t position = 0; position < order.size() && (want_fixed || want_ram); position++) {
        const uint32_t expert = order[position].second;
        const uint64_t key = (uint64_t)layer << 32 | expert;
        const auto in_ram = ram.slot_of.find(key);
        if (want_fixed && position < fixed_share && in_ram != ram.slot_of.end()) {
            /* It went to RAM while this device was not sized yet: move it. */
            while (next_free < c->slots && c->key[next_free] != moe_stream_no_key) next_free++;
            if (next_free < c->slots && moe_stream_ram_settle(in_ram->second) &&
                cuda_ok(cudaMemcpyAsync(c->gate + (uint64_t)next_free * c->slot_bytes, ram.at(in_ram->second),
                                        c->slot_bytes, cudaMemcpyHostToDevice, cuda_decode_stream()),
                        "MoE fixed expert move")) {
                c->key[next_free] = key;
                c->slot_of[key] = next_free;
                c->heat[next_free] = order[position].first;
                c->stamp[next_free] = 0;
                c->filled++;
                c->generation++;
                ram.key[in_ram->second] = moe_stream_no_key;
                ram.released.push_back(in_ram->second);
                ram.slot_of.erase(in_ram);
                ram_moved = true;
            }
        } else if (want_fixed && position < fixed_share && !c->slot_of.count(key)) {
            /* The top of the ranking belongs here even when a prompt pass on
             * another device kept it there: that device gives it up, and it
             * comes from the file once more. */
            while (next_free < c->slots && c->key[next_free] != moe_stream_no_key) next_free++;
            if (next_free < c->slots) {
                for (auto &entry : g_moe_stream_caches) {
                    moe_stream_cache &other = entry.second;
                    if (const auto theirs = other.slot_of.find(key); &other != c && theirs != other.slot_of.end()) {
                        other.key[theirs->second] = moe_stream_no_key;
                        other.heat[theirs->second] = 0;
                        other.stamp[theirs->second] = 0;
                        other.filled--;
                        other.slot_of.erase(theirs);
                        other.generation++;
                    }
                }
                c->key[next_free] = key;
                c->slot_of[key] = next_free;
                c->heat[next_free] = order[position].first;
                c->stamp[next_free] = 0;
                c->filled++;
                c->generation++;
                g_moe_layer.fixed_slot[expert] = (int)next_free;
                any = true;
            }
        } else if (!moe_layer_held(key)) {
            if (position >= fixed_share && ram_left && !ram.free_slots.empty()) {
                const bool locked = locked_left && ram.free_slots.back() < ram.locked_slots;
                const uint32_t slot = locked ? ram.free_slots.back() : ram.free_slots.front();
                if (locked) ram.free_slots.pop_back();
                else ram.free_slots.erase(ram.free_slots.begin());
                locked_left -= locked;
                ram.key[slot] = key;
                ram.slot_of[key] = slot;
                ram.uses[slot] = order[position].first;
                ram.stamp[slot] = 0;
                g_moe_layer.ram_slot[expert] = (int)slot;
                ram_left--;
                any = true;
            }
        }
    }
    g_moe_layer.fixed_done[layer] = g_moe_layer.fixed_done[layer] || want_fixed;
    /* The tier was handed the top of the ranking while the device had no size
     * and has just given it up: the slots that frees go to the rest of the
     * layer on the next pass, or the CPU reads those from the file a use at a
     * time (1.7 ms each on Kolibri-1 Q2).  Only in a prompt pass: while
     * decoding, a refill reads the whole layer for it (0.7 GiB a layer on
     * Qwen3.8, 20 s over the first tokens), where a use at a time reads
     * just what is needed. */
    g_moe_layer.ram_done[layer] = (g_moe_layer.ram_done[layer] || want_ram) && !(ram_moved && prompt_pass);
    return any;
}

static const char *moe_layer_weight(uint64_t off) {
    const char *at = NULL;
    if (g_moe_layer.active) {
        if (off == g_moe_layer.gate_off) at = g_moe_layer.buffer;
        else if (off == g_moe_layer.up_off) at = g_moe_layer.buffer + g_moe_layer.gate_bytes;
        else if (off == g_moe_layer.down_off) at = g_moe_layer.buffer + 2 * g_moe_layer.gate_bytes;
    }
    return at;
}
}

extern "C" int ds4_gpu_qwen4_moe_stream_layer(const ds4_gpu_stream_expert_table *t) {
    using namespace qwen4_cuda;
    int resident = 0, device = -1;
    g_moe_layer.buffer = NULL;
    if (g_ssd_streaming_mode && t && t->model_map && cudaGetDevice(&device) == cudaSuccess &&
        !g_moe_layer_devices[device].unavailable) {
        moe_layer_device &here = g_moe_layer_devices[device];
        const uint64_t gate = (uint64_t)t->n_total_expert * t->gate_expert_bytes;
        const uint64_t down = (uint64_t)t->n_total_expert * t->down_expert_bytes;
        moe_stream_cache &own = g_moe_stream_caches[device];
        if (here.in_arena && !own.tail_slots) {
            /* The cache the buffer lived in is gone. */
            here.buffer = NULL;
            here.cap = 0;
            here.in_arena = false;
        }
        /* A device that only runs another one's expert stage has nothing else
         * to load, so its cache can be sized before the first layer, with the
         * layer buffer as its tail: the cache then takes everything the
         * device has, and the buffer's bytes hold experts between batches. */
        if (g_moe_layer.remote && !own.slots && !here.arena_tried) {
            here.arena_tried = true;
            const uint64_t slot_bytes = 2 * t->gate_expert_bytes + t->down_expert_bytes;
            const uint32_t tail = (uint32_t)((2 * gate + down + slot_bytes - 1) / slot_bytes);
            if (!here.in_arena) (void)cudaFree(here.buffer);
            here.buffer = NULL;
            here.cap = 0;
            own.resized = true;
            if (moe_stream_cache_init(own, device, t->gate_expert_bytes, t->down_expert_bytes, tail + 16,
                                      t->n_total_expert)) {
                own.tail_slots = tail;
                here.buffer = own.gate + (uint64_t)(own.slots - tail) * slot_bytes;
                here.cap = 2 * gate + down;
                here.in_arena = true;
                fprintf(stderr, "ds4: CUDA device %d loads whole layers into the last %u slots of its expert cache\n",
                        device, tail);
            } else {
                own.resized = false;
            }
        }
        if (own.tail_slots) moe_stream_tail_reserve(own);
        if (here.cap < 2 * gate + down) {
            if (!here.in_arena) (void)cudaFree(here.buffer);
            here.in_arena = false;
            here.buffer = NULL;
            here.cap = 0;
            void *grown = NULL;
            if (cudaMalloc(&grown, 2 * gate + down) == cudaSuccess) {
                here.buffer = (char *)grown;
                here.cap = 2 * gate + down;
                fprintf(stderr, "ds4: CUDA device %d takes large batches a whole layer of experts at a time "
                                "(%.2f GiB buffer)\n", device, (double)here.cap / 1073741824.0);
            } else {
                (void)cudaGetLastError();
                here.unavailable = true;
                fprintf(stderr, "ds4: no memory on device %d for a whole layer of experts (%.2f GiB); large "
                                "batches keep the per-token expert path\n", device,
                        (double)(2 * gate + down) / 1073741824.0);
            }
        }
        g_moe_layer.buffer = here.buffer;
        if (g_moe_layer.buffer) {
            const char *map = (const char *)t->model_map;
            const cudaStream_t stream = cuda_decode_stream();
            bool ok = false;
            moe_layer_visit(t->layer);
            g_moe_layer.ram_slot.clear();
            g_moe_layer.fixed_slot.clear();
#ifdef _WIN32
            if (moe_layer_reader_open()) {
                int device = 0;
                (void)cudaGetDevice(&device);
                moe_stream_ram_init(g_moe_stream_ram, 2 * t->gate_expert_bytes + t->down_expert_bytes);
                moe_stream_cache &c = g_moe_stream_caches[device];
                const bool handing_out = moe_layer_plan(t, &c, true);
                /* A layer read ahead goes to the device as it is, and what the
                 * plan hands to RAM and to the fixed set is copied out of the
                 * same region: sending such a pass through the reader below
                 * gave up the read-ahead and cost it 20 s more on Qwen3.8. */
                const char *ahead[3] = {NULL, NULL, NULL};
                const bool was_ahead = moe_layer_ahead_take(t, ahead);
                moe_layer_ahead_next(t);
                if (was_ahead) {
                    /* What this device's cache holds is still copied on the
                     * device; the read-ahead supplies the runs in between.  A
                     * fixed slot given out just now holds nothing yet. */
                    const uint64_t expert[3] = {t->gate_expert_bytes, t->gate_expert_bytes, t->down_expert_bytes};
                    const uint64_t part[3] = {0, t->gate_expert_bytes, 2 * t->gate_expert_bytes};
                    char *to[3] = {g_moe_layer.buffer, g_moe_layer.buffer + gate, g_moe_layer.buffer + 2 * gate};
                    const auto cached_slot = [&](uint32_t e) {
                        const auto slot = c.slot_of.find((uint64_t)t->layer << 32 | e);
                        return slot != c.slot_of.end() && g_moe_layer.fixed_slot[e] < 0 ? (int)slot->second : -1;
                    };
                    ok = true;
                    for (uint32_t e = 0; e < t->n_total_expert && ok;) {
                        const int slot = cached_slot(e);
                        uint32_t end = e + 1;
                        if (slot < 0)
                            while (end < t->n_total_expert && cached_slot(end) < 0) end++;
                        for (int i = 0; i < 3 && ok; i++) {
                            const char *from = slot >= 0 ? c.gate + (uint64_t)slot * c.slot_bytes + part[i]
                                                         : ahead[i] + (uint64_t)e * expert[i];
                            ok = cuda_ok(cudaMemcpyAsync(to[i] + (uint64_t)e * expert[i], from,
                                                         (uint64_t)(end - e) * expert[i],
                                                         slot >= 0 ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                                                         stream), "MoE layer load");
                        }
                        e = end;
                    }
                    for (uint32_t e = 0; e < t->n_total_expert && ok && handing_out; e++) {
                        const int ram_slot = g_moe_layer.ram_slot[e], fixed_slot = g_moe_layer.fixed_slot[e];
                        for (int i = 0; i < 3 && ok && (ram_slot >= 0 || fixed_slot >= 0); i++) {
                            const char *from = ahead[i] + (uint64_t)e * expert[i];
                            if (ram_slot >= 0) memcpy(g_moe_stream_ram.at((uint32_t)ram_slot) + part[i], from, expert[i]);
                            if (fixed_slot >= 0)
                                ok = cuda_ok(cudaMemcpyAsync(c.gate + (uint64_t)fixed_slot * c.slot_bytes + part[i], from,
                                                             expert[i], cudaMemcpyHostToDevice, stream),
                                             "MoE fixed expert load");
                        }
                    }
                } else {
                /* An expert this device's cache holds is copied from there, on
                 * the device; one in the page-locked part of the RAM tier
                 * comes from RAM; runs of the others are read from the file.
                 * The rest of the RAM tier is not a source: copies from memory
                 * that is not page-locked measured a quarter of the rate. */
                const moe_stream_ram &ram = g_moe_stream_ram;
                std::vector<int> cached(t->n_total_expert, -1), in_ram(t->n_total_expert, -1);
                for (uint32_t e = 0; e < t->n_total_expert; e++) {
                    const uint64_t key = (uint64_t)t->layer << 32 | e;
                    if (const auto slot = c.slot_of.find(key); slot != c.slot_of.end() && g_moe_layer.fixed_slot[e] < 0)
                        cached[e] = (int)slot->second;
                    else if (const auto slot = ram.slot_of.find(key);
                             slot != ram.slot_of.end() && g_moe_layer.ram_slot[e] < 0 &&
                             slot->second < ram.locked_slots)
                        in_ram[e] = (int)slot->second;
                }
                /* With DS4_CUDA_MOE_STREAM_STATS: where a pass took its experts from. */
                static uint64_t from_cache, from_ram, from_file;
                static const bool report = getenv("DS4_CUDA_MOE_STREAM_STATS") != NULL;
                if (report && t->layer == 0 && from_cache + from_ram + from_file) {
                    fprintf(stderr, "ds4: CUDA MoE layer pass took %llu experts from device caches, %llu from RAM, "
                                    "%llu from the file (%.1f GiB in %.1f s of reads, %.1f s waiting for buffers)\n",
                            (unsigned long long)from_cache, (unsigned long long)from_ram,
                            (unsigned long long)from_file, (double)g_moe_layer_reader.read_bytes / 1073741824.0,
                            g_moe_layer_reader.read_s, g_moe_layer_reader.wait_s);
                    from_cache = from_ram = from_file = 0;
                    g_moe_layer_reader.read_s = g_moe_layer_reader.wait_s = 0.0;
                    g_moe_layer_reader.read_bytes = 0;
                }
                bool settled = true;
                for (uint32_t e = 0; e < t->n_total_expert; e++) {
                    from_cache += cached[e] >= 0;
                    from_ram += in_ram[e] >= 0;
                    from_file += cached[e] < 0 && in_ram[e] < 0;
                    /* A save queued on the helper may still be writing a RAM slot read here. */
                    if (in_ram[e] >= 0) settled = settled && moe_stream_ram_settle((uint32_t)in_ram[e]);
                }
                const auto load = [&](char *to, uint64_t file_off, uint64_t expert_bytes, uint64_t part) {
                    bool loaded = true;
                    for (uint32_t e = 0; e < t->n_total_expert && loaded;) {
                        if (cached[e] >= 0 || in_ram[e] >= 0) {
                            const char *from = cached[e] >= 0 ? c.gate + (uint64_t)cached[e] * c.slot_bytes + part
                                                              : ram.at((uint32_t)in_ram[e]) + part;
                            loaded = cuda_ok(cudaMemcpyAsync(to + (uint64_t)e * expert_bytes, from, expert_bytes,
                                                             cached[e] >= 0 ? cudaMemcpyDeviceToDevice
                                                                            : cudaMemcpyHostToDevice, stream),
                                             "MoE layer gather");
                            e++;
                        } else {
                            uint32_t end = e;
                            while (end < t->n_total_expert && cached[end] < 0 && in_ram[end] < 0) end++;
                            loaded = moe_layer_read(to + (uint64_t)e * expert_bytes, file_off + (uint64_t)e * expert_bytes,
                                                    (uint64_t)(end - e) * expert_bytes, stream, expert_bytes, part, e);
                            e = end;
                        }
                    }
                    return loaded;
                };
                ok = settled && load(g_moe_layer.buffer, t->gate_offset, t->gate_expert_bytes, 0) &&
                     load(g_moe_layer.buffer + gate, t->up_offset, t->gate_expert_bytes, t->gate_expert_bytes) &&
                     load(g_moe_layer.buffer + 2 * gate, t->down_offset, t->down_expert_bytes,
                          2 * t->gate_expert_bytes);
                }
            } else
#endif
            ok =
                cuda_ok(cudaMemcpyAsync(g_moe_layer.buffer, map + t->gate_offset, gate, cudaMemcpyHostToDevice,
                                        stream), "MoE layer gate load") &&
                cuda_ok(cudaMemcpyAsync(g_moe_layer.buffer + gate, map + t->up_offset, gate, cudaMemcpyHostToDevice,
                                        stream), "MoE layer up load") &&
                cuda_ok(cudaMemcpyAsync(g_moe_layer.buffer + 2 * gate, map + t->down_offset, down,
                                        cudaMemcpyHostToDevice, stream), "MoE layer down load");
            g_moe_layer.gate_off = t->gate_offset;
            g_moe_layer.up_off = t->up_offset;
            g_moe_layer.down_off = t->down_offset;
            g_moe_layer.gate_bytes = gate;
            g_moe_layer.active = ok;
            resident = ok;
        }
    }
    return resident;
}

extern "C" void ds4_gpu_qwen4_moe_stream_input(const ds4_gpu_tensor *x, uint32_t width) {
    qwen4_cuda::g_moe_stream.input = x;
    qwen4_cuda::g_moe_stream.input_width = width;
}

extern "C" void ds4_gpu_qwen4_moe_stream_set_stats_path(const char *model_path) {
    using namespace qwen4_cuda;
    if (model_path && model_path[0] && g_moe_stats_file.empty()) {
        g_moe_stats_file = std::string(model_path) + ".expert-uses";
        moe_stats_load();
        atexit(moe_stats_save);
    }
}

/* After the tiled kernels of a layer brought in by _stream_layer: counts the
 * rows that used each expert and copies the layer's most used ones from the
 * layer buffer into the device's expert cache, so the tokens generated after
 * the prompt do not start on an empty cache. */
static int moe_layer_keep(const ds4_gpu_stream_expert_table *t, const int *counts, uint32_t skip = 0) {
    using namespace qwen4_cuda;
    int device = -1;
    bool ok = g_moe_layer.active && t && counts &&
        cuda_ok(cudaGetDevice(&device), "MoE layer keep device") &&
        moe_stream_grow(g_moe_stream.host_ids, g_moe_stream.host_ids_cap, (uint64_t)t->n_total_expert * 4, true,
                        "MoE layer counts staging") &&
        cuda_ok(cudaMemcpyAsync(g_moe_stream.host_ids, counts, (uint64_t)t->n_total_expert * 4,
                                cudaMemcpyDeviceToHost, cuda_decode_stream()), "MoE layer counts read") &&
        moe_stream_wait(cuda_decode_stream(), "MoE layer counts wait");
    for (uint32_t e = 0; ok && e < t->n_total_expert; e++)
        if (g_moe_stream.host_ids[e] > 0) g_moe_layer_heat[(uint64_t)t->layer << 32 | e] += (uint32_t)g_moe_stream.host_ids[e];
    /* The cache can only be sized once a whole forward has loaded the dense
     * weights, so the first pass of a session keeps nothing. */
    if (ok && g_moe_layer.forwards) {
        moe_stream_cache &c = g_moe_stream_caches[device];
        /* No room for the cache now only means nothing is kept: generation
         * sizes it again when it starts.  One attempt per forward. */
        static uint64_t refused = UINT64_MAX;
        if (!c.slots && refused != g_moe_layer.forwards) {
            c.resized = true;
            if (!moe_stream_cache_init(c, device, t->gate_expert_bytes, t->down_expert_bytes, 16, t->n_total_expert)) {
                c.resized = false;
                refused = g_moe_layer.forwards;
            }
        }
        const auto ranked = moe_layer_ranked(t->layer, t->n_total_expert);
        const uint32_t share = c.slots && !c.provisional ?
                               moe_layer_share(c.slots - (c.tail_busy ? c.tail_slots : 0)) : 0;
        const uint64_t gate_all = g_moe_layer.gate_bytes;
        /* skip: the top of the ranking is another device's to pin, so keeping
         * it here would only be given up again at the first token. */
        for (size_t i = skip; ok && i < ranked.size() && i < (size_t)skip + share; i++) {
            const uint32_t expert = ranked[i].second;
            const uint64_t key = (uint64_t)t->layer << 32 | expert;
            bool on_device = false;
            for (const auto &entry : g_moe_stream_caches) on_device = on_device || entry.second.slot_of.count(key) != 0;
            if (!on_device) {
                int slot = -1;
                uint32_t coldest = ranked[i].first;
                /* With a ranking from an earlier run the set is fixed: only free slots. */
                for (uint32_t s = 0; s < c.slots && (slot < 0 || c.key[slot] != moe_stream_no_key); s++) {
                    if (c.key[s] == moe_stream_no_key || (!g_moe_stats_loaded && c.heat[s] < coldest)) {
                        coldest = c.key[s] == moe_stream_no_key ? 0 : c.heat[s];
                        slot = (int)s;
                    }
                }
                if (slot >= 0) {
                    char *at = c.gate + (uint64_t)slot * c.slot_bytes;
                    const cudaStream_t stream = cuda_decode_stream();
                    ok = cuda_ok(cudaMemcpyAsync(at, g_moe_layer.buffer + (uint64_t)expert * c.gate_bytes, c.gate_bytes,
                                                 cudaMemcpyDeviceToDevice, stream), "MoE layer keep gate") &&
                         cuda_ok(cudaMemcpyAsync(at + c.gate_bytes,
                                                 g_moe_layer.buffer + gate_all + (uint64_t)expert * c.gate_bytes,
                                                 c.gate_bytes, cudaMemcpyDeviceToDevice, stream), "MoE layer keep up") &&
                         cuda_ok(cudaMemcpyAsync(at + 2 * c.gate_bytes,
                                                 g_moe_layer.buffer + 2 * gate_all + (uint64_t)expert * c.down_bytes,
                                                 c.down_bytes, cudaMemcpyDeviceToDevice, stream), "MoE layer keep down");
                    if (c.key[slot] != moe_stream_no_key) c.slot_of.erase(c.key[slot]);
                    else c.filled++;
                    c.key[slot] = key;
                    c.slot_of[key] = (uint32_t)slot;
                    c.heat[slot] = ranked[i].first;
                    c.stamp[slot] = 0;
                    c.generation++;
                    /* The RAM tier is handed its experts on the first pass,
                     * before this cache has a size, and takes them from the
                     * top: one that belongs here leaves it again, or this
                     * cache would stay empty and fill a miss at a time once
                     * tokens are generated. */
                    moe_stream_ram &ram = g_moe_stream_ram;
                    if (const auto in_ram = ram.slot_of.find(key); in_ram != ram.slot_of.end()) {
                        ram.key[in_ram->second] = moe_stream_no_key;
                        ram.free_slots.push_back(in_ram->second);
                        ram.slot_of.erase(in_ram);
                        if (t->layer < g_moe_layer.ram_done.size()) g_moe_layer.ram_done[t->layer] = false;
                    }
                }
            }
        }
    }
    return ok;
}

namespace qwen4_cuda {
static uint64_t g_moe_reclaim_bytes;
}

extern "C" void ds4_gpu_qwen4_moe_stream_set_reclaim(uint64_t bytes) {
    qwen4_cuda::g_moe_reclaim_bytes = bytes;
}

extern "C" void ds4_gpu_qwen4_moe_stream_set_device_layers(int logical_tier, uint32_t layers) {
    using namespace qwen4_cuda;
    if (logical_tier >= 0 && logical_tier < g_n_gpus) {
        g_moe_layer_devices[g_gpu[logical_tier].device_id].assigned = layers;
        g_moe_split = true;
    }
}

extern "C" void ds4_gpu_qwen4_moe_stream_keep_awake(void) {
    qwen4_cuda::moe_stream_keep_awake();
}

extern "C" void ds4_gpu_qwen4_moe_stream_release(void) {
    using namespace qwen4_cuda;
    int device = -1;
    if (cudaGetDevice(&device) == cudaSuccess && g_moe_stream_caches.count(device)) {
        moe_stream_cache &c = g_moe_stream_caches[device];
        if (c.slots) {
            (void)cudaStreamSynchronize(cuda_decode_stream());
            moe_stream_cache_release(c);
            /* Sized for good when it comes back, and its share of each layer
             * handed out again. */
            c.resized = true;
            std::fill(g_moe_layer.fixed_done.begin(), g_moe_layer.fixed_done.end(), false);
            g_moe_stream.cache = NULL;
        }
    }
}

extern "C" int ds4_gpu_qwen4_moe_stream_layer_keep(const ds4_gpu_stream_expert_table *t,
        const ds4_gpu_tensor *counts) {
    return t && qwen4_cuda::tensor(counts, (uint64_t)t->n_total_expert * 4) ?
           moe_layer_keep(t, (const int *)counts->ptr) : 0;
}

/* For a graph without list tensors and with the shared expert as the last
 * slot (Kolibri-1): mid and part of a whole batch as the slot kernels would
 * leave them, but with the routed experts through the tiled kernels on a
 * layer brought in whole.  The slot kernels run first with an empty selection:
 * that computes the shared slot and zeroes the routed ones, which the tiled
 * kernels then fill.  Returns 1 when done, 0 when the caller should use the
 * slot path (streaming off, small batch, no room), -1 on failure. */
extern "C" int ds4_gpu_qwen4_moe_stream_layer_run(const ds4_gpu_stream_expert_table *t,
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, ds4_gpu_tensor *part, const ds4_gpu_tensor *selected,
        const void *map, uint64_t size, uint32_t T, uint32_t NS, uint32_t K, uint32_t F, uint32_t E,
        uint32_t gate_type, uint32_t down_type, uint64_t sgo, uint64_t suo, uint64_t sdo, uint32_t st) {
    using namespace qwen4_cuda;
    int done = 0, device = -1;
    const uint32_t NE = t ? t->n_total_expert : 0, NO = NS + 1;
    /* st == UINT_MAX: the caller computes the shared expert itself and only
     * the routed slots are filled; the tiled kernels write every one of them. */
    const bool shared_slot = st != UINT_MAX;
    if (g_ssd_streaming_mode && t && NE && NE <= 512 && tensor(mid, (uint64_t)T * NO * F * 4) &&
        tensor(x, (uint64_t)T * K * 4) && tensor(part, (uint64_t)T * NO * E * 4) &&
        tensor(selected, (uint64_t)T * NS * 4) && cudaGetDevice(&device) == cudaSuccess) {
        moe_layer_device &here = g_moe_layer_devices[device];
        const uint64_t pairs = (uint64_t)T * NS;
        const bool fresh = here.none_cap < pairs * 4;
        const bool room = !here.unavailable &&
            moe_stream_grow(here.lists, here.lists_cap, (uint64_t)NE * T * 4, false, "MoE layer lists") &&
            moe_stream_grow(here.counts, here.counts_cap, (uint64_t)NE * 4, false, "MoE layer counts") &&
            moe_stream_grow(here.tiles, here.tiles_cap, ((uint64_t)NE + 1) * 4, false, "MoE layer tiles") &&
            moe_stream_grow(here.none, here.none_cap, pairs * 4, false, "MoE layer empty selection");
        if (!room) {
            (void)cudaGetLastError();
            here.unavailable = true;
        } else if (ds4_gpu_qwen4_moe_stream_layer(t) > 0) {
            const cudaStream_t stream = cuda_decode_stream();
            const char *sg = NULL, *su = NULL, *sd = NULL;
            if (shared_slot) {
                const uint64_t sb = row_bytes(st, K) * F, sdb = row_bytes(st, F) * E;
                sg = weight(map, size, sgo, sb);
                su = weight(map, size, suo, sb);
                sd = weight(map, size, sdo, sdb);
            }
            const char *gate = g_moe_layer.buffer, *up = gate + g_moe_layer.gate_bytes, *down = up + g_moe_layer.gate_bytes;
            bool ok = (!shared_slot || (sg && su && sd)) &&
                (!fresh || cuda_ok(cudaMemsetAsync(here.none, 0xff, pairs * 4, stream), "MoE layer empty selection"));
            ok = ok && (!shared_slot || moe_mv_dispatch((float *)mid->ptr, (const float *)x->ptr, here.none, gate, up,
                                                        sg, su, gate_type, st, NE, T, NS, K, F, false));
            if (ok) {
                expert_lists<<<1, 256, 0, stream>>>(here.lists, here.counts, (const int *)selected->ptr, T * NS, NE, T);
                ok = launched();
            }
            ok = ok && matrix_dispatch((float *)mid->ptr, (const float *)x->ptr, gate, up, here.lists, here.counts,
                                       gate_type, NE, T, NS, NO, K, F, T, false, here.tiles) &&
                 (!shared_slot || moe_mv_dispatch((float *)part->ptr, (const float *)mid->ptr, here.none, down, NULL,
                                                  sd, NULL, down_type, st, NE, T, NS, F, E, true)) &&
                 matrix_dispatch((float *)part->ptr, (const float *)mid->ptr, down, NULL, here.lists, here.counts,
                                 down_type, NE, T, NS, NO, F, E, T, true, here.tiles) &&
                 moe_layer_keep(t, here.counts);
            done = ok ? 1 : -1;
        }
    }
    return done;
}

extern "C" int ds4_gpu_qwen4_moe_stream_layer_remote(const ds4_gpu_stream_expert_table *t,
        ds4_gpu_tensor *routed, const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights, uint32_t T, uint32_t NS, uint32_t K, uint32_t F, uint32_t E,
        uint32_t gate_type, uint32_t down_type) {
    using namespace qwen4_cuda;
    static const bool wanted = getenv("DS4_CUDA_MOE_STREAM_HELPER") != NULL && !g_moe_split;
    int done = 0, device = -1, count = 0, helper = -1;
    const auto entered = std::chrono::steady_clock::now();
    const uint32_t NE = t ? t->n_total_expert : 0;
    const uint64_t pairs = (uint64_t)T * NS;
    if (wanted && g_ssd_streaming_mode && NE && NE <= 512 && cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceCount(&count) == cudaSuccess && tensor(routed, (uint64_t)T * E * 4) &&
        tensor(x, (uint64_t)T * K * 4) && tensor(selected, pairs * 4) && tensor(weights, pairs * 4)) {
        for (int d = 0; d < count && helper < 0; d++)
            if (d != device) helper = d;
    }
    /* With a stored ranking this device pins the top of each layer when
     * generation starts; how many that will be, from what is free on it now. */
    static uint32_t pinned_here = 0;
    static uint64_t sized_at = UINT64_MAX;
    if (helper >= 0 && g_moe_stats_loaded && g_moe_layer.layers && sized_at != g_moe_layer.forwards) {
        sized_at = g_moe_layer.forwards;
        const moe_stream_cache &own = g_moe_stream_caches[device];
        const uint64_t slot_bytes = 2 * t->gate_expert_bytes + t->down_expert_bytes;
        const uint64_t reserve = moe_stream_env_mib("DS4_CUDA_MOE_STREAM_RESERVE_MIB", 512ull << 20);
        size_t free_bytes = 0, total_bytes = 0;
        uint64_t slots = own.slots && !own.provisional ? own.slots : 0;
        if (!slots && cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            /* Plus what the graph gives up when it starts decoding. */
            free_bytes += (size_t)g_moe_reclaim_bytes;
            slots = moe_stream_env_mib("DS4_CUDA_MOE_STREAM_CACHE_MIB", free_bytes > reserve ? free_bytes - reserve : 0) /
                    slot_bytes;
        }
        if (pinned_here != (uint32_t)(slots / g_moe_layer.layers))
            fprintf(stderr, "ds4: CUDA device %d is expected to pin %llu experts a layer (%.2f of %.2f GiB free); "
                            "device %d keeps the ranks below\n", device,
                    (unsigned long long)(slots / g_moe_layer.layers), (double)free_bytes / 1073741824.0,
                    (double)total_bytes / 1073741824.0, helper);
        pinned_here = (uint32_t)(slots / g_moe_layer.layers);
    }
    if (helper >= 0 && !g_moe_layer_devices[helper].remote_unavailable) {
        moe_layer_device &there = g_moe_layer_devices[helper];
        const cudaStream_t stream = cuda_decode_stream();
        bool ok = moe_stream_grow(g_moe_remote.x, g_moe_remote.x_cap, (uint64_t)T * K * 4, true, "MoE remote input") &&
            moe_stream_grow(g_moe_remote.sel, g_moe_remote.sel_cap, pairs * 4, true, "MoE remote selection") &&
            moe_stream_grow(g_moe_remote.w, g_moe_remote.w_cap, pairs * 4, true, "MoE remote weights") &&
            moe_stream_grow(g_moe_remote.sum, g_moe_remote.sum_cap, (uint64_t)T * E * 4, true, "MoE remote output") &&
            cuda_ok(cudaMemcpyAsync(g_moe_remote.x, x->ptr, (uint64_t)T * K * 4, cudaMemcpyDeviceToHost, stream),
                    "MoE remote input read") &&
            cuda_ok(cudaMemcpyAsync(g_moe_remote.sel, selected->ptr, pairs * 4, cudaMemcpyDeviceToHost, stream),
                    "MoE remote selection read") &&
            cuda_ok(cudaMemcpyAsync(g_moe_remote.w, weights->ptr, pairs * 4, cudaMemcpyDeviceToHost, stream),
                    "MoE remote weights read") &&
            moe_stream_wait(stream, "MoE remote input wait") &&
            cuda_ok(cudaSetDevice(helper), "MoE remote device");
        const bool room = ok &&
            moe_stream_grow(there.rx, there.rx_cap, (uint64_t)T * K * 4, false, "MoE remote input") &&
            moe_stream_grow(there.rsel, there.rsel_cap, pairs * 4, false, "MoE remote selection") &&
            moe_stream_grow(there.rw, there.rw_cap, pairs * 4, false, "MoE remote weights") &&
            moe_stream_grow(there.rmid, there.rmid_cap, pairs * F * 4, false, "MoE remote mid") &&
            moe_stream_grow(there.rpart, there.rpart_cap, pairs * E * 4, false, "MoE remote part") &&
            moe_stream_grow(there.rsum, there.rsum_cap, (uint64_t)T * E * 4, false, "MoE remote sum") &&
            moe_stream_grow(there.lists, there.lists_cap, (uint64_t)NE * T * 4, false, "MoE remote lists") &&
            moe_stream_grow(there.counts, there.counts_cap, (uint64_t)NE * 4, false, "MoE remote counts") &&
            moe_stream_grow(there.tiles, there.tiles_cap, ((uint64_t)NE + 1) * 4, false, "MoE remote tiles");
        g_moe_layer.remote = true;
        const bool loading = room && ds4_gpu_qwen4_moe_stream_layer(t) > 0;
        g_moe_layer.remote = false;
        if (ok && !room) {
            (void)cudaGetLastError();
            there.remote_unavailable = true;
            fprintf(stderr, "ds4: no memory on device %d to run large batches there; they stay on device %d\n",
                    helper, device);
        } else if (loading) {
            /* With DS4_CUDA_MOE_STREAM_STATS: seconds per pass spent loading
             * layers and computing on them, at the price of a wait in between. */
            static const bool timed = getenv("DS4_CUDA_MOE_STREAM_STATS") != NULL;
            static double load_s, compute_s;
            static auto mark = std::chrono::steady_clock::now();
            if (timed) {
                if (t->layer == 0 && load_s + compute_s > 0.0) {
                    fprintf(stderr, "ds4: CUDA MoE remote pass: %.1f s loading layers, %.1f s computing\n", load_s,
                            compute_s);
                    load_s = compute_s = 0.0;
                }
                (void)cudaStreamSynchronize(0);
                load_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - entered).count();
                mark = std::chrono::steady_clock::now();
            }
            /* Everything below is queued on the second device's stream, in order. */
            const char *gate = g_moe_layer.buffer, *up = gate + g_moe_layer.gate_bytes, *down = up + g_moe_layer.gate_bytes;
            ok = cuda_ok(cudaMemcpyAsync(there.rx, g_moe_remote.x, (uint64_t)T * K * 4, cudaMemcpyHostToDevice, 0),
                         "MoE remote input write") &&
                 cuda_ok(cudaMemcpyAsync(there.rsel, g_moe_remote.sel, pairs * 4, cudaMemcpyHostToDevice, 0),
                         "MoE remote selection write") &&
                 cuda_ok(cudaMemcpyAsync(there.rw, g_moe_remote.w, pairs * 4, cudaMemcpyHostToDevice, 0),
                         "MoE remote weights write");
            if (ok) {
                expert_lists<<<1, 256, 0, cuda_decode_stream()>>>(there.lists, there.counts, there.rsel, T * NS, NE, T);
                ok = launched();
            }
            ok = ok && matrix_dispatch(there.rmid, there.rx, gate, up, there.lists, there.counts, gate_type, NE, T, NS,
                                       NS, K, F, T, false, there.tiles) &&
                 matrix_dispatch(there.rpart, there.rmid, down, NULL, there.lists, there.counts, down_type, NE, T, NS,
                                 NS, F, E, T, true, there.tiles);
            if (ok) {
                moe_reduce<<<dim3((E + 255) / 256, T), 256, 0, cuda_decode_stream()>>>(there.rsum, NULL, NULL,
                    there.rpart, there.rw, NULL, NULL, NS, NS, E, 0);
                ok = launched();
            }
            ok = ok && moe_layer_keep(t, there.counts, g_moe_stats_loaded ? pinned_here : 0) &&
                 cuda_ok(cudaMemcpyAsync(g_moe_remote.sum, there.rsum, (uint64_t)T * E * 4, cudaMemcpyDeviceToHost, 0),
                         "MoE remote output read") &&
                 moe_stream_wait(0, "MoE remote wait");
            if (timed) {
                compute_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - mark).count();
                mark = std::chrono::steady_clock::now();
            }
            g_moe_layer.active = false;
            done = ok ? 1 : -1;
        }
        const bool back = cuda_ok(cudaSetDevice(device), "MoE primary device");
        if (done > 0)
            done = back && cuda_ok(cudaMemcpyAsync(routed->ptr, g_moe_remote.sum, (uint64_t)T * E * 4,
                                                   cudaMemcpyHostToDevice, stream), "MoE remote output write") ? 1 : -1;
    }
    return done;
}

extern "C" int ds4_gpu_qwen4_moe_reduce_routed_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *routed,
        const ds4_gpu_tensor *sg, const ds4_gpu_tensor *sh, ds4_gpu_tensor *R, const ds4_gpu_tensor *inj,
        uint32_t T, uint32_t D, uint32_t hc) {
    using namespace qwen4_cuda;
    int device = -1, launched_ok = 0;
    if (T && D && hc <= 4 && tensor(out, (uint64_t)T * D * 4) && tensor(routed, (uint64_t)T * D * 4) &&
        tensor(sg, (uint64_t)T * 4) && tensor(sh, (uint64_t)T * D * 4) &&
        (!hc || (tensor(R, (uint64_t)T * hc * D * 4) && tensor(inj, (uint64_t)T * hc * hc * 8 * 4))) &&
        cudaGetDevice(&device) == cudaSuccess) {
        /* The kernel weighs every slot; the sum is one slot of weight one. */
        moe_layer_device &here = g_moe_layer_devices[device];
        bool ok = true;
        if (here.ones_cap < (uint64_t)T * 4) {
            const std::vector<float> ones(T, 1.0f);
            ok = moe_stream_grow(here.ones, here.ones_cap, (uint64_t)T * 4, false, "MoE reduce weights") &&
                 cuda_ok(cudaMemcpy(here.ones, ones.data(), (uint64_t)T * 4, cudaMemcpyHostToDevice),
                         "MoE reduce weights fill");
        }
        if (ok) {
            moe_reduce<<<dim3((D + 255) / 256, T), 256, 0, cuda_decode_stream()>>>((float *)out->ptr,
                hc ? (float *)R->ptr : NULL, hc ? (const float *)inj->ptr : NULL, (const float *)routed->ptr,
                here.ones, (const float *)sg->ptr, (const float *)sh->ptr, 1, 1, D, hc);
            launched_ok = launched();
        }
    }
    return launched_ok;
}

extern "C" void ds4_gpu_qwen4_moe_stream_set_cpu(qwen4_cuda::moe_cpu_quantize_fn quantize,
        qwen4_cuda::moe_cpu_rows_fn rows, uint32_t xq_block_bytes) {
    qwen4_cuda::g_moe_cpu.quantize = quantize;
    qwen4_cuda::g_moe_cpu.rows = rows;
    qwen4_cuda::g_moe_cpu.xq_block_bytes = xq_block_bytes;
}

namespace qwen4_cuda {

/* What handing out cost while decoding, for DS4_CUDA_MOE_STREAM_STATS. */
static struct {
    uint64_t whole_layers, from_file;
    double seconds;
} g_moe_handout;

/* The first visit to a layer reads it once and hands its experts out: to
 * this device its share of the stored ranking, which with a helper it then
 * keeps (see moe_stream_place), and to the RAM tier the rest. */
static void moe_stream_handout(const ds4_gpu_stream_expert_table *t, moe_stream_cache &c, bool &ok) {
    moe_layer_visit(t->layer);
    moe_stream_ram_init(g_moe_stream_ram, c.slot_bytes);
#ifdef _WIN32
    const auto entered = std::chrono::steady_clock::now();
    if (!c.provisional && moe_layer_pending(t->layer, c) && moe_layer_reader_open() && moe_layer_plan(t, &c, false)) {
        const cudaStream_t stream = cuda_decode_stream();
        const uint64_t gate = (uint64_t)t->n_total_expert * t->gate_expert_bytes;
        bool for_ram = false;
        for (uint32_t e = 0; e < t->n_total_expert && !for_ram; e++) for_ram = g_moe_layer.ram_slot[e] >= 0;
        g_moe_handout.whole_layers += for_ram;
        for (uint32_t e = 0; e < t->n_total_expert && !for_ram; e++) g_moe_handout.from_file += g_moe_layer.fixed_slot[e] >= 0;
        if (for_ram) {
            ok = ok && moe_layer_read(NULL, t->gate_offset, gate, stream, t->gate_expert_bytes, 0) &&
                 moe_layer_read(NULL, t->up_offset, gate, stream, t->gate_expert_bytes, t->gate_expert_bytes) &&
                 moe_layer_read(NULL, t->down_offset, (uint64_t)t->n_total_expert * t->down_expert_bytes, stream,
                                t->down_expert_bytes, 2 * t->gate_expert_bytes);
        } else {
            /* Only a few experts for the fixed set come from the file: not
             * worth a pass over the whole layer, which a prompt pass before
             * this already made for the RAM tier. */
            const char *map = (const char *)t->model_map;
            for (uint32_t e = 0; e < t->n_total_expert && ok; e++) {
                if (const int slot = g_moe_layer.fixed_slot[e]; slot >= 0) {
                    if (c.stage_used == moe_stream_stage_slots) {
                        ok = cuda_ok(cudaStreamSynchronize(stream), "MoE fixed expert staging wait");
                        c.stage_used = 0;
                    }
                    char *staged = c.stage + (uint64_t)c.stage_used++ * c.slot_bytes;
                    ok = ok &&
                         moe_stream_file_read(staged, map, t->gate_offset + (uint64_t)e * c.gate_bytes, c.gate_bytes) &&
                         moe_stream_file_read(staged + c.gate_bytes, map, t->up_offset + (uint64_t)e * c.gate_bytes,
                                              c.gate_bytes) &&
                         moe_stream_file_read(staged + 2 * c.gate_bytes, map,
                                              t->down_offset + (uint64_t)e * c.down_bytes, c.down_bytes) &&
                         cuda_ok(cudaMemcpyAsync(c.gate + (uint64_t)slot * c.slot_bytes, staged, c.slot_bytes,
                                                 cudaMemcpyHostToDevice, stream), "MoE fixed expert load");
                }
            }
            ok = ok && cuda_ok(cudaStreamSynchronize(stream), "MoE fixed expert wait");
            c.stage_used = 0;
        }
        g_moe_handout.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - entered).count();
    }
    g_moe_layer.ram_slot.clear();
    g_moe_layer.fixed_slot.clear();
#else
    (void)ok;
#endif
}

/* DS4_CUDA_MOE_STREAM_STATS=N reports misses and their cost every N uses. */
static void moe_stream_stats(moe_stream_cache &c, int device, moe_stream_cache *h) {
    static const uint64_t stats_every = moe_stream_env_mib("DS4_CUDA_MOE_STREAM_STATS", 0) >> 20;
    if (stats_every && c.uses >= stats_every) {
        fprintf(stderr, "ds4: CUDA MoE expert cache on device %d: %llu of %llu uses missed, "
                        "%.3f ms read + %.3f ms upload per miss, %llu of them from RAM, %llu run on the helper; "
                        "%u of %u slots filled\n",
                device, (unsigned long long)c.misses, (unsigned long long)c.uses,
                c.misses ? 1000.0 * c.host_s / (double)c.misses : 0.0,
                c.misses ? 1000.0 * c.device_s / (double)c.misses : 0.0,
                (unsigned long long)c.from_ram, (unsigned long long)c.helped, c.filled, c.slots);
        if (g_moe_handout.seconds > 0.0) {
            fprintf(stderr, "ds4: CUDA MoE hand-out while decoding: %.1f s for %llu whole layers read for the RAM tier "
                            "and %llu experts read for this device\n", g_moe_handout.seconds,
                    (unsigned long long)g_moe_handout.whole_layers, (unsigned long long)g_moe_handout.from_file);
            g_moe_handout = {};
        }
        fprintf(stderr, "ds4: CUDA MoE per-layer wait: %.3f ms over %llu layers\n",
                g_moe_stream.waits ? 1000.0 * g_moe_stream.wait_s / (double)g_moe_stream.waits : 0.0,
                (unsigned long long)g_moe_stream.waits);
        g_moe_stream.wait_s = 0.0;
        g_moe_stream.waits = 0;
        if (g_moe_prof.layers && g_moe_layer.layers) {
            const double tokens = (double)g_moe_prof.layers / (double)g_moe_layer.layers;
            double total = 0.0;
            for (const double seconds : g_moe_prof.seconds) total += seconds;
            fprintf(stderr, "ds4: CUDA MoE token profile, ms per token over %.0f tokens: total %.1f = outside the "
                            "expert stage %.1f + device catching up %.1f + selection and loads %.1f + caller "
                            "between expert calls %.1f + mid %.1f + down %.1f\n", tokens, 1e3 * total / tokens,
                    1e3 * g_moe_prof.seconds[moe_prof_outside] / tokens,
                    1e3 * g_moe_prof.seconds[moe_prof_wait] / tokens,
                    1e3 * g_moe_prof.seconds[moe_prof_prepare] / tokens,
                    1e3 * g_moe_prof.seconds[moe_prof_gap] / tokens, 1e3 * g_moe_prof.seconds[moe_prof_mid] / tokens,
                    1e3 * g_moe_prof.seconds[moe_prof_down] / tokens);
            for (double &seconds : g_moe_prof.seconds) seconds = 0.0;
            g_moe_prof.layers = 0;
        }
        if (g_moe_cpu.pairs) {
            fprintf(stderr, "ds4: CUDA MoE CPU path: %llu uses in %llu layers, %.3f ms each, %.3f ms of thread time "
                            "each on %zu workers\n", (unsigned long long)g_moe_cpu.pairs,
                    (unsigned long long)g_moe_cpu.layers, 1000.0 * g_moe_cpu.seconds / (double)g_moe_cpu.pairs,
                    1e-6 * (double)g_moe_cpu_pool.busy_ns.exchange(0) / (double)g_moe_cpu.pairs,
                    g_moe_cpu_pool.workers.size());
            g_moe_cpu.pairs = g_moe_cpu.layers = 0;
            g_moe_cpu.seconds = 0.0;
            fprintf(stderr, "ds4: CUDA MoE RAM tier: %zu of %u slots hold an expert; %llu experts read from the file "
                            "for the CPU, %.3f ms each\n",
                    g_moe_stream_ram.slot_of.size(), g_moe_stream_ram.slots,
                    (unsigned long long)g_moe_stream_ram.file_reads,
                    g_moe_stream_ram.file_reads ? 1e3 * g_moe_stream_ram.file_s / (double)g_moe_stream_ram.file_reads
                                                : 0.0);
            g_moe_stream_ram.file_reads = 0;
            g_moe_stream_ram.file_s = 0.0;
        }
        if (h && h->hops) {
            fprintf(stderr, "ds4: CUDA MoE helper, ms per layer over %llu layers: mid %.3f, down launch %.3f, wait "
                            "and read %.3f, write back %.3f\n", (unsigned long long)h->hops,
                    1000.0 * h->hop_s[3] / (double)h->hops, 1000.0 * h->hop_s[0] / (double)h->hops,
                    1000.0 * h->hop_s[1] / (double)h->hops, 1000.0 * h->hop_s[2] / (double)h->hops);
            h->hops = 0;
            h->hop_s[0] = h->hop_s[1] = h->hop_s[2] = h->hop_s[3] = 0.0;
        }
        c.uses = c.misses = c.helped = c.from_ram = 0;
        c.host_s = c.device_s = 0.0;
    }
}

/* The selection is on the host: the bookkeeping of a layer's expert stage.
 * Every routed expert is stamped and loaded, sent to the helper or left to
 * the CPU, and the slot numbers go to the device, unless the kernels map
 * the ids themselves (table, see moe_stream_table_row). */
static bool moe_stream_place(const ds4_gpu_stream_expert_table *t, moe_stream_cache &c, moe_stream_cache *h,
                             int device, int helper_device, uint64_t pairs, bool table) {
    int *ids = g_moe_stream.host_ids;
    g_moe_stream_clock++;
    /* The staging ring and the released RAM slots were sources of copies
     * queued before this selection's read-back; the wait behind it
     * (moe_stream_wait or the event) has run them. */
    c.stage_used = 0;
    g_moe_stream_ram.free_slots.insert(g_moe_stream_ram.free_slots.end(), g_moe_stream_ram.released.begin(),
                                       g_moe_stream_ram.released.end());
    g_moe_stream_ram.released.clear();
    bool ok = true;
    int *helper_ids = h ? g_moe_stream.host_helper_ids : NULL;
    for (uint64_t i = 0; helper_ids && i < pairs; i++) helper_ids[i] = -1;
    g_moe_stream.helper_pairs.clear();
    /* What this batch uses and a device already holds is stamped before any
     * load picks a victim: a save in flight on one device's stream must not
     * be the source of a load on the other's.  With the table the layer was
     * handed out in _begin, ahead of the kernels. */
    if (!table) moe_stream_handout(t, c, ok);
    for (uint64_t i = 0; i < pairs; i++) {
        if (ids[i] >= 0 && (uint32_t)ids[i] < t->n_total_expert) {
            const uint64_t key = (uint64_t)t->layer << 32 | (uint32_t)ids[i];
            g_moe_layer_heat[key]++;
            if (const auto mine = c.slot_of.find(key); mine != c.slot_of.end())
                c.stamp[mine->second] = g_moe_stream_clock;
            if (h)
                if (const auto theirs = h->slot_of.find(key); theirs != h->slot_of.end())
                    h->stamp[theirs->second] = g_moe_stream_clock;
        }
    }
    static const bool cpu_wanted = getenv("DS4_CUDA_MOE_STREAM_CPU") != NULL;
    moe_stream_ram_init(g_moe_stream_ram, c.slot_bytes);
    const bool cpu_mode = cpu_wanted && g_moe_cpu.rows && !c.provisional && g_moe_stream_ram.slots >= pairs;
    g_moe_stream.cpu_pairs.clear();
    g_moe_promote.clear();
    g_moe_stream.cpu_slots.clear();
    g_moe_stream.cpu_done = false;
    g_moe_stream_by_use = cpu_mode;
    /* Counts halve now and then, so what was hot many queries ago gives way. */
    if (cpu_mode && g_moe_stream_clock % 4096 == 0) {
        for (auto &entry : g_moe_stream_caches)
            for (uint32_t &uses : entry.second.heat) uses >>= 1;
        for (uint32_t &uses : g_moe_stream_ram.uses) uses >>= 1;
    }
    /* A swap is two experts over the bus, so only so many per batch, and only
     * for an expert with a full percent more uses than the one it replaces
     * (2020 against 2000), and at least four more while the counts are still
     * small: two of similar use would trade places for ever. */
    uint64_t promotions = pairs / 8 + 1;
    const uint32_t promotion_floor = 4;
    for (uint64_t i = 0; i < pairs && ok; i++) {
        /* An id outside the layer stays negative, which the kernels skip. */
        if (ids[i] >= 0 && (uint32_t)ids[i] < t->n_total_expert) {
            const uint64_t key = (uint64_t)t->layer << 32 | (uint32_t)ids[i];
            const bool on_device = c.slot_of.count(key) || (h && h->slot_of.count(key));
            /* With a stored ranking this device's set is fixed, full or not:
             * only the helper's changes. */
            const bool fixed = g_moe_stats_loaded && h;
            const bool devices_full = (fixed || c.filled == c.slots) && (!h || h->filled == h->slots);
            if (cpu_mode && !on_device && devices_full) {
                /* The devices are to hold what is used most: a RAM expert
                 * used more than a device's least used one takes its place. */
                const int ram_slot = moe_stream_ram_hold(g_moe_stream_ram, c, t, (uint32_t)ids[i]);
                ok = ram_slot >= 0;
                /* A save queued on the helper may still be writing this slot. */
                ok = ok && moe_stream_ram_settle((uint32_t)ram_slot);
                const uint32_t heat = ok ? g_moe_stream_ram.uses[ram_slot] : 0;
                const uint32_t mine = fixed ? UINT32_MAX : moe_stream_coldest(c);
                const uint32_t theirs = h ? moe_stream_coldest(*h) : UINT32_MAX;
                const uint32_t coldest = mine < theirs ? mine : theirs;
                const bool hotter = coldest != UINT32_MAX && heat >= coldest + promotion_floor &&
                                    (uint64_t)heat * 100u >= (uint64_t)coldest * 101u;
                if (!ok || !promotions || !hotter) {
                    g_moe_stream.cpu_pairs.push_back(i);
                    g_moe_stream.cpu_slots.push_back((uint32_t)ram_slot);
                    ids[i] = -1;
                } else if (theirs < mine) {
                    /* The CPU has it this once; it changes places with the
                     * helper's least used after the layer (moe_stream_promote_flush). */
                    promotions--;
                    if ((uint32_t)ram_slot < g_moe_stream_ram.locked_slots)
                        g_moe_promote.emplace_back(key, (uint32_t)ram_slot);
                    g_moe_stream.cpu_pairs.push_back(i);
                    g_moe_stream.cpu_slots.push_back((uint32_t)ram_slot);
                    ids[i] = -1;
                } else {
                    promotions--;
                    ids[i] = moe_stream_slot(c, t, (uint32_t)ids[i]);
                    ok = ids[i] >= 0;
                }
            /* An expert neither device holds goes to the one whose least
             * recently used slot is the older. */
            } else if (h && !c.slot_of.count(key) &&
                (h->slot_of.count(key) || g_moe_stats_loaded || moe_stream_oldest(*h) < moe_stream_oldest(c))) {
                helper_ids[i] = ids[i];
                g_moe_stream.helper_pairs.push_back(i);
                ids[i] = -1;
            } else {
                ids[i] = moe_stream_slot(c, t, (uint32_t)ids[i]);
                ok = ids[i] >= 0;
            }
        } else {
            ids[i] = -1;
        }
    }
    if (ok && !g_moe_stream.helper_pairs.empty()) {
        ok = cuda_ok(cudaSetDevice(helper_device), "MoE helper device");
        h->stage_used = 0;
        for (size_t i = 0; i < g_moe_stream.helper_pairs.size() && ok; i++) {
            int &id = helper_ids[g_moe_stream.helper_pairs[i]];
            id = moe_stream_slot(*h, t, (uint32_t)id);
            ok = id >= 0;
        }
        if (ok && h->selected_cap < pairs) {
            void *grown = NULL;
            ok = cuda_ok(cudaMalloc(&grown, pairs * 4), "MoE helper selection");
            if (ok) {
                (void)cudaFree(h->selected);
                h->selected = (int *)grown;
                h->selected_cap = pairs;
            }
        }
        ok = ok && cuda_ok(cudaMemcpyAsync(h->selected, helper_ids, pairs * 4, cudaMemcpyHostToDevice, 0),
                           "MoE helper selection write");
        ok = cuda_ok(cudaSetDevice(device), "MoE primary device") && ok;
        c.helped += g_moe_stream.helper_pairs.size();
        c.misses += h->misses;
        c.from_ram += h->from_ram;
        h->from_ram = 0;
        c.host_s += h->host_s;
        c.device_s += h->device_s;
        h->misses = 0;
        h->host_s = h->device_s = 0.0;
    }
    c.uses += pairs;
    moe_stream_stats(c, device, h);
    if (!table)
        ok = ok && cuda_ok(cudaMemcpyAsync(c.selected, ids, pairs * 4, cudaMemcpyHostToDevice, cuda_decode_stream()),
                           "MoE expert cache selection write");
    g_moe_stream.helper = g_moe_stream.helper_pairs.empty() ? NULL : h;
    return ok;
}

/* Settles what _begin deferred, once the kernels are queued: waits for the
 * selection, then the bookkeeping.  That may load experts the kernels have
 * already run without, as zero rows like any expert the device lacks; redo
 * says so, and the caller runs them again behind the loads, which is the
 * ordering with the wait first for that one layer. */
static bool moe_stream_settle(bool *redo) {
    moe_stream_cache &c = g_moe_stream_caches[g_moe_stream.device];
    const auto wait_from = std::chrono::steady_clock::now();
    bool ok = moe_stream_event_wait(c.selected_event, "MoE expert cache selection wait");
    g_moe_stream.wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_from).count();
    g_moe_stream.waits++;
    moe_prof_lap(moe_prof_wait);
    g_moe_stream.deferred = false;
    ok = ok && moe_stream_place(&g_moe_stream.table, c, g_moe_stream.candidate, g_moe_stream.device,
                                g_moe_stream.helper_device, (uint64_t)g_moe_stream.T * g_moe_stream.NS, true);
    *redo = c.generation != g_moe_stream.uploaded;
    moe_prof_lap(moe_prof_prepare);
    return ok;
}

}

extern "C" void ds4_gpu_qwen4_moe_stream_end(void) {
    using namespace qwen4_cuda;
    /* A layer whose down kernel never ran still settles, so the clock and
     * the stamps stay in step with what was selected. */
    if (g_moe_stream.deferred) {
        bool redo = false;
        (void)moe_stream_settle(&redo);
    }
    g_moe_stream.cache = NULL;
    g_moe_stream.helper = NULL;
    g_moe_stream.cpu_pairs.clear();
    g_moe_layer.active = false;
}

/* Queues the read-back of `selected`, which the kernels that produced it
 * precede, and makes every selected expert resident on the current device
 * for the mid and down wrappers: before they are queued, after the host
 * has waited (moe_stream_place), or for a single token after they are
 * (moe_stream_settle).  A no-op when SSD streaming is off. */
extern "C" int ds4_gpu_qwen4_moe_stream_begin(const ds4_gpu_stream_expert_table *t,
        const ds4_gpu_tensor *selected, uint32_t T, uint32_t NS) {
    using namespace qwen4_cuda;
    ds4_gpu_qwen4_moe_stream_end();
    if (!g_ssd_streaming_mode) return 1;
    const uint64_t pairs = (uint64_t)T * NS;
    int device = -1;
    if (!t || !t->model_map || !t->n_total_expert || !t->gate_expert_bytes || !t->down_expert_bytes ||
        !pairs || !tensor(selected, pairs * 4) ||
        !cuda_ok(cudaGetDevice(&device), "MoE expert cache device")) {
        fprintf(stderr, "ds4: CUDA MoE expert cache got an invalid expert table or selection\n");
        return 0;
    }
    moe_stream_cache &c = g_moe_stream_caches[device];
    const uint32_t min_slots = pairs < t->n_total_expert ? (uint32_t)pairs : t->n_total_expert;
    /* A layer number that does not go up starts the next forward. */
    if (c.slots && c.provisional && t->layer <= c.last_layer) {
        moe_stream_cache_release(c);
        c.resized = true;
    }
    c.last_layer = t->layer;
    if (!c.slots && !moe_stream_cache_init(c, device, t->gate_expert_bytes, t->down_expert_bytes,
                                           min_slots, t->n_total_expert)) return 0;
    if (c.gate_bytes != t->gate_expert_bytes || c.down_bytes != t->down_expert_bytes || c.slots < min_slots) {
        fprintf(stderr, "ds4: CUDA MoE expert cache on device %d holds %u experts of %llu+%llu bytes; "
                        "layer %u needs %u of %llu+%llu at once\n",
                device, c.slots, (unsigned long long)c.gate_bytes, (unsigned long long)c.down_bytes, t->layer,
                min_slots, (unsigned long long)t->gate_expert_bytes, (unsigned long long)t->down_expert_bytes);
        return 0;
    }
    const ds4_gpu_tensor *input = g_moe_stream.input;
    const uint64_t input_bytes = (uint64_t)T * g_moe_stream.input_width * 4;
    g_moe_stream.input = NULL;
    g_moe_stream.input_ready = false;
    /* Single tokens only: a batch is a different path with other costs. */
    if (T != 1) g_moe_prof.running = false;
    moe_prof_lap(moe_prof_outside);
    g_moe_prof.running = T == 1;
    g_moe_prof.layers += T == 1;
    bool ok = true;
    static const int helper_device = [device]() {
        int count = 0;
        if (!getenv("DS4_CUDA_MOE_STREAM_HELPER") || cudaGetDeviceCount(&count) != cudaSuccess) return -1;
        if (g_moe_split) {
            fprintf(stderr, "ds4: DS4_CUDA_MOE_STREAM_HELPER is ignored: the layers are split over the devices "
                            "and each device's expert cache serves its own layers\n");
            return -1;
        }
        for (int d = 0; d < count; d++)
            if (d != device) return d;
        return -1;
    }();
    /* Not while the primary's arena is provisional: the helper would take the
     * whole first forward and keep the hottest experts a hop away. */
    moe_stream_cache *h = helper_device >= 0 && helper_device != device && !c.provisional ?
                          &g_moe_stream_caches[helper_device] : NULL;
    /* No large batch is open now: the layer buffers' slots hold experts again. */
    if (c.tail_busy) moe_stream_tail_release(c);
    if (h && h->tail_busy) moe_stream_tail_release(*h);
    if (h && !h->slots) {
        h->resized = true;
        ok = cuda_ok(cudaSetDevice(helper_device), "MoE helper device") &&
             moe_stream_cache_init(*h, helper_device, t->gate_expert_bytes, t->down_expert_bytes,
                                   min_slots, t->n_total_expert);
        ok = cuda_ok(cudaSetDevice(device), "MoE primary device") && ok;
    }
    ok = ok && (!h || moe_stream_grow(g_moe_stream.host_helper_ids, g_moe_stream.host_helper_ids_cap, pairs * 4,
                                      true, "MoE helper selection staging"));
    if (ok && c.selected_cap < pairs) {
        void *grown = NULL;
        ok = cuda_ok(cudaMalloc(&grown, pairs * 4), "MoE expert cache selection");
        if (ok) {
            (void)cudaFree(c.selected);
            c.selected = (int *)grown;
            c.selected_cap = pairs;
        }
    }
    /* The layer input is wanted on the host by a helper for nearly every
     * layer, so with one it rides on the selection's wait; the CPU route
     * alone wants it rarely and reads it then (moe_stream_fetch_input). */
    const bool read_input = input && tensor(input, input_bytes) && h;
    /* A batch and the provisional arena map the selection on the host before
     * the kernels are queued.  A single token's kernels map the router's ids
     * through the slot table and run while the host does the bookkeeping
     * (moe_stream_settle); whatever changes the table meanwhile goes out
     * ahead of them here. */
    const cudaStream_t stream = cuda_decode_stream();
    const bool deferred = T == 1 && !c.provisional;
    if (ok && deferred) {
        moe_stream_handout(t, c, ok);
        ok = ok && moe_stream_table_row(c, t->layer, t->n_total_expert, stream) &&
             (c.selected_event ||
              cuda_ok(cudaEventCreateWithFlags(&c.selected_event, cudaEventDisableTiming), "MoE selection event"));
    }
    const auto wait_from = std::chrono::steady_clock::now();
    ok = ok &&
         moe_stream_grow(g_moe_stream.host_ids, g_moe_stream.host_ids_cap, pairs * 4, true,
                         "MoE expert cache selection staging") &&
         cuda_ok(cudaMemcpyAsync(g_moe_stream.host_ids, selected->ptr, pairs * 4, cudaMemcpyDeviceToHost, stream),
                 "MoE expert cache selection read") &&
         (!read_input ||
          (moe_stream_grow(g_moe_stream.host_x, g_moe_stream.host_x_cap, input_bytes, true,
                           "MoE expert input staging") &&
           cuda_ok(cudaMemcpyAsync(g_moe_stream.host_x, input->ptr, input_bytes, cudaMemcpyDeviceToHost, stream),
                   "MoE expert input read")));
    g_moe_stream.input_ready = ok && read_input;
    if (ok && deferred) {
        ok = cuda_ok(cudaEventRecord(c.selected_event, stream), "MoE selection mark");
        g_moe_stream.deferred = ok;
        g_moe_stream.table = *t;
        g_moe_stream.T = T;
        g_moe_stream.NS = NS;
        g_moe_stream.candidate = h;
        g_moe_stream.uploaded = c.generation;
        g_moe_stream.mid.queued = false;
    } else if (ok) {
        ok = moe_stream_wait(stream, "MoE expert cache selection wait");
        g_moe_stream.wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_from).count();
        g_moe_stream.waits++;
        moe_prof_lap(moe_prof_wait);
        ok = ok && moe_stream_place(t, c, h, device, helper_device, pairs, false);
    }
    if (ok) {
        g_moe_stream.device = device;
        g_moe_stream.helper_device = helper_device;
        g_moe_stream.cache = &c;
        g_moe_stream.gate_off = t->gate_offset;
        g_moe_stream.up_off = t->up_offset;
        g_moe_stream.down_off = t->down_offset;
    }
    moe_prof_lap(moe_prof_prepare);
    return ok;
}

extern "C" int ds4_gpu_qwen4_moe_mid_tensor(ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t go, uint64_t uo,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t sgo, uint64_t suo, uint32_t st) {
    using namespace qwen4_cuda;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(mid, (uint64_t)T*NO*M*4) ||
        !tensor(x, (uint64_t)T*K*4) || !tensor(sel, (uint64_t)T*NS*4)) return 0;
    const uint64_t expert_bytes = expert_row_bytes(type, K)*M, sb = row_bytes(st, K)*M;
    const moe_stream_cache *stream = g_moe_stream.cache;
    const bool streamed = stream && go == g_moe_stream.gate_off && uo == g_moe_stream.up_off;
    if (streamed && (stream->gate_bytes != expert_bytes || stream->selected_cap < (uint64_t)T*NS)) {
        fprintf(stderr, "ds4: CUDA MoE expert cache holds %llu-byte gate experts, the kernel expects %llu\n",
                (unsigned long long)stream->gate_bytes, (unsigned long long)expert_bytes);
        return 0;
    }
    const char *g = streamed ? stream->gate : weight(map,size,go,expert_bytes*NE);
    const char *u = streamed ? stream->up : weight(map,size,uo,expert_bytes*NE);
    const char *sg = st != UINT_MAX ? weight(map,size,sgo,sb) : NULL;
    const char *su = st != UINT_MAX ? weight(map,size,suo,sb) : NULL;
    if (!g || !u || (st != UINT_MAX && (!sg || !su))) return 0;
    /* Deferred (see _begin): the kernel maps the router's ids through the
     * slot table; the helper's share is only known once _down_tensor has
     * settled, which starts the helper then. */
    const bool deferred = streamed && g_moe_stream.deferred;
    moe_stream_cache *helper = streamed && !deferred ? g_moe_stream.helper : NULL;
    if (streamed && !deferred && (helper || !g_moe_stream.cpu_pairs.empty()) &&
        !moe_stream_fetch_input((const float *)x->ptr, (uint64_t)T*K*4)) return 0;
    g_moe_stream.mid_type = type;
    g_moe_stream.mid_K = K;
    g_moe_stream.mid_M = M;
    moe_prof_lap(moe_prof_gap);
    const int *table = deferred ? stream->table + (uint64_t)g_moe_stream.table.layer * stream->table_experts : NULL;
    const int done = moe_mv_dispatch((float *)mid->ptr,(const float *)x->ptr,
                                     streamed && !deferred ? stream->selected : (const int *)sel->ptr,
                                     g,u,sg,su,type,st,streamed ? stream->slots : NE,T,NS,K,M,false,
                                     streamed ? stream->slot_bytes : 0,table,deferred ? stream->selected : NULL,
                                     deferred ? stream->table_experts : 0) &&
                     (!helper || moe_stream_helper_mid(g_moe_stream.host_x,type,T,NS,K,M));
    if (deferred) {
        g_moe_stream.mid.queued = done != 0;
        g_moe_stream.mid.out = (float *)mid->ptr;
        g_moe_stream.mid.x = (const float *)x->ptr;
        g_moe_stream.mid.sel = (const int *)sel->ptr;
        g_moe_stream.mid.g = g;
        g_moe_stream.mid.u = u;
        g_moe_stream.mid.sg = sg;
        g_moe_stream.mid.su = su;
        g_moe_stream.mid.type = type;
        g_moe_stream.mid.st = st;
        g_moe_stream.mid.K = K;
        g_moe_stream.mid.M = M;
    }
    moe_prof_lap(moe_prof_mid);
    return done;
}

extern "C" int ds4_gpu_qwen4_moe_down_tensor(ds4_gpu_tensor *part, const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t off,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t so, uint32_t st) {
    using namespace qwen4_cuda;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(part,(uint64_t)T*NO*M*4) ||
        !tensor(mid,(uint64_t)T*NO*K*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    const uint64_t expert_bytes = expert_row_bytes(type,K)*M;
    const moe_stream_cache *stream = g_moe_stream.cache;
    const bool streamed = stream && off == g_moe_stream.down_off;
    if (streamed && (stream->down_bytes != expert_bytes || stream->selected_cap < (uint64_t)T*NS)) {
        fprintf(stderr, "ds4: CUDA MoE expert cache holds %llu-byte down experts, the kernel expects %llu\n",
                (unsigned long long)stream->down_bytes, (unsigned long long)expert_bytes);
        return 0;
    }
    const char *w = streamed ? stream->down : weight(map,size,off,expert_bytes*NE);
    const char *sw = st != UINT_MAX ? weight(map,size,so,row_bytes(st,K)*M) : NULL;
    if (!w || (st != UINT_MAX && !sw)) return 0;
    const bool deferred = streamed && g_moe_stream.deferred;
    moe_prof_lap(moe_prof_gap);
    const uint64_t layer_row = deferred ? (uint64_t)g_moe_stream.table.layer * stream->table_experts : 0;
    const int *table = deferred ? stream->table + layer_row : NULL;
    int done = moe_mv_dispatch((float *)part->ptr,(const float *)mid->ptr,
                               streamed && !deferred ? stream->selected : (const int *)sel->ptr,
                               w,NULL,sw,NULL,type,st,streamed ? stream->slots : NE,T,NS,K,M,true,
                               streamed ? stream->slot_bytes : 0,table,NULL,deferred ? stream->table_experts : 0);
    if (done && deferred) {
        bool redo = false;
        done = moe_stream_settle(&redo);
        const auto &m = g_moe_stream.mid;
        if (done && redo) {
            /* An expert loaded just now was a zero row to the kernels: the
             * row goes out again and they run again behind the loads. */
            moe_stream_cache &own = g_moe_stream_caches[g_moe_stream.device];
            if (!m.queued) {
                fprintf(stderr, "ds4: CUDA MoE expert cache loaded experts after the mid kernel, which was not queued "
                                "through it\n");
                done = 0;
            } else {
                done = moe_stream_table_row(own,g_moe_stream.table.layer,own.table_experts,cuda_decode_stream()) &&
                       moe_mv_dispatch(m.out,m.x,m.sel,m.g,m.u,m.sg,m.su,m.type,m.st,own.slots,T,NS,m.K,m.M,false,
                                       own.slot_bytes,own.table + layer_row,own.selected,own.table_experts) &&
                       moe_mv_dispatch((float *)part->ptr,(const float *)mid->ptr,(const int *)sel->ptr,w,NULL,sw,NULL,
                                       type,st,own.slots,T,NS,K,M,true,own.slot_bytes,own.table + layer_row,NULL,
                                       own.table_experts);
            }
        }
        if (done && (g_moe_stream.helper || !g_moe_stream.cpu_pairs.empty()))
            done = m.queued && moe_stream_fetch_input(m.x,(uint64_t)T*m.K*4) &&
                   (!g_moe_stream.helper || moe_stream_helper_mid(g_moe_stream.host_x,m.type,T,NS,m.K,m.M));
    }
    const bool cpu = streamed && !g_moe_stream.cpu_pairs.empty();
    const bool outside = streamed && (cpu || g_moe_stream.helper);
    if (done && outside) {
        moe_stream_cache &own = g_moe_stream_caches[g_moe_stream.device];
        done = moe_stream_ext_open(own,(uint64_t)T*NS,M) &&
               (!g_moe_stream.helper || moe_stream_helper_down((float *)part->ptr,NO,type,T,NS,K,M)) &&
               (!cpu || ((g_moe_stream.cpu_done || moe_stream_cpu_compute(*stream,type,NS,M)) &&
                         moe_stream_cpu_write(own,M))) &&
               moe_stream_ext_merge(own,(float *)part->ptr,NO,T,NS,M) &&
               moe_stream_promote_flush();
    }
    moe_prof_lap(moe_prof_down);
    return done;
}

extern "C" int ds4_gpu_qwen4_moe_reduce_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *part,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *sg, const ds4_gpu_tensor *sh,
        ds4_gpu_tensor *R, const ds4_gpu_tensor *inj, uint32_t T, uint32_t NS, uint32_t stride,
        uint32_t D, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!T || !NS || !D || stride < NS + (sg && !sh) || hc > 4 ||
        !tensor(out,(uint64_t)T*D*4) || !tensor(part,(uint64_t)T*stride*D*4) ||
        !tensor(weights,(uint64_t)T*NS*4) || (sg && !tensor(sg,(uint64_t)T*4)) ||
        (sh && !tensor(sh,(uint64_t)T*D*4)) ||
        (hc && (!tensor(R,(uint64_t)T*hc*D*4) || !tensor(inj,(uint64_t)T*hc*hc*8*4)))) return 0;
    moe_reduce<<<dim3((D+255)/256,T),256,0,cuda_decode_stream()>>>((float *)out->ptr,
        hc ? (float *)R->ptr : NULL, hc ? (const float *)inj->ptr : NULL, (const float *)part->ptr,
        (const float *)weights->ptr, sg ? (const float *)sg->ptr : NULL, sh ? (const float *)sh->ptr : NULL,
        NS,stride,D,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_moe_build_lists_tensor(ds4_gpu_tensor *lists, ds4_gpu_tensor *counts,
        const ds4_gpu_tensor *sel, uint32_t T, uint32_t NS, uint32_t NE, uint32_t cap) {
    using namespace qwen4_cuda;
    if (!T || !NS || NS > NE || !NE || NE > 512 || cap < T || (uint64_t)T*NS > INT_MAX ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    expert_lists<<<1,256,0,cuda_decode_stream()>>>((int *)lists->ptr,(int *)counts->ptr,(const int *)sel->ptr,T*NS,NE,cap);
    return launched();
}

static int qwen4_moe_mm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t o0, uint64_t o1, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap, bool down) {
    using namespace qwen4_cuda;
    if (!T || !NE || !NS || NS > NE || NO < NS || !K || !M || cap < T ||
        !tensor(out,(uint64_t)T*NO*M*4) || !tensor(x,(uint64_t)T*(down ? NO : 1)*K*4) ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4)) return 0;
    const uint64_t bytes = expert_row_bytes(type,K)*M*NE;
    const char *l0 = moe_layer_weight(o0), *l1 = down ? NULL : moe_layer_weight(o1);
    const char *w0 = l0 ? l0 : weight(map,size,o0,bytes), *w1 = down ? NULL : l1 ? l1 : weight(map,size,o1,bytes);
    if (!w0 || (!down && !w1)) return 0;
    return matrix_dispatch((float *)out->ptr,(const float *)x->ptr,w0,w1,(const int *)lists->ptr,
        (const int *)counts->ptr,type,NE,T,NS,NO,K,M,cap,down);
}

extern "C" int ds4_gpu_qwen4_moe_mm_mid_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t go, uint64_t uo, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_moe_mm(out,x,lists,counts,map,size,go,uo,type,NE,T,NS,NO,K,M,cap,false);
}

extern "C" int ds4_gpu_qwen4_moe_mm_down_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_moe_mm(out,x,lists,counts,map,size,off,0,type,NE,T,NS,NO,K,M,cap,true);
}

extern "C" int ds4_gpu_qwen4_gdn_front_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *mixed, ds4_gpu_tensor *ga, ds4_gpu_tensor *gb,
        const void *map, uint64_t size, uint64_t co, uint64_t ao, uint64_t bo, uint64_t so, uint64_t dto,
        uint32_t type, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D, uint32_t CK, uint32_t K,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_cuda;
    const uint64_t C = ((uint64_t)2*Hk+Hv)*D, hb = (CK-1)*C*4;
    if (!T || !Hk || !Hv || !D || !K || C > UINT_MAX || CK < 2 || CK > 4 ||
        !tensor(qkv,(uint64_t)T*C*4) || !tensor(state,hb) ||
        (snap && (st >= T || !tensor(snap,hb))) || (snap2 && (st2 >= T || !tensor(snap2,hb)))) return 0;
    const char *w = weight(map,size,co,C*CK*4);
    if (!w || !ds4_gpu_qwen4_dense_mm_tensor(ga,mixed,map,size,ao,type,T,K,Hv) ||
              !ds4_gpu_qwen4_dense_mm_tensor(gb,mixed,map,size,bo,type,T,K,Hv)) return 0;
    conv<<<(C+255)/256,256,0,cuda_decode_stream()>>>((float *)qkv->ptr,(float *)state->ptr,(const float *)w,
        T,C,CK,true,snap ? (float *)snap->ptr : NULL,st,snap2 ? (float *)snap2->ptr : NULL,st2);
    return launched() && ds4_gpu_qwen4_gdn_prep_tensor(qkv,ga,gb,map,size,so,dto,T,Hk,Hv,D);
}

/* None of the "fused" entry points below is one launch on CUDA; the combine
 * one even adds a copy and a pointer swap that keeps a layer from being
 * replayed as a graph.  Off unless DS4_QWEN4_DECODE_FUSIONS asks for them. */
extern "C" int ds4_gpu_qwen4_decode_fusions_enabled(void) {
    static const int on = getenv("DS4_QWEN4_DECODE_FUSIONS") != NULL;
    return on;
}

extern "C" int ds4_gpu_qwen4_hc_combine_norm_tensor(ds4_gpu_tensor *next, const ds4_gpu_tensor *blk,
        const ds4_gpu_tensor *oldinj, ds4_gpu_tensor *xn, ds4_gpu_tensor *inj, const ds4_gpu_tensor *R,
        const void *map, uint64_t size, uint64_t go, uint64_t io, uint32_t type,
        uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    const uint64_t bytes = (uint64_t)T*E*hc*4;
    if (!qwen4_cuda::tensor(next,bytes) || !qwen4_cuda::tensor(R,bytes) || next->ptr == R->ptr ||
        !inj || !oldinj || inj->ptr == oldinj->ptr) return 0;
    if (!cuda_ok(cudaMemcpyAsync(next->ptr,R->ptr,bytes,cudaMemcpyDeviceToDevice,cuda_decode_stream()),"Qwen HC copy")) return 0;
    return ds4_gpu_qwen4_hc_combine_tensor(next,blk,oldinj,T,E,hc) &&
           ds4_gpu_qwen4_hc_norm_tensor(xn,inj,next,map,size,go,io,type,T,E,hc,ni,eps);
}

extern "C" int ds4_gpu_qwen4_mtp_stage_tensor(ds4_gpu_tensor *cat, const ds4_gpu_tensor *e,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t eo, uint64_t ho,
        uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_cuda;
    if (!E || !hc || hc > 4 || !tensor(cat,(uint64_t)(hc+1)*2*E*4) ||
        !tensor(e,(uint64_t)E*4) || !tensor(R,(uint64_t)hc*E*4)) return 0;
    const char *ge = weight(map,size,eo,(uint64_t)E*4), *gh = weight(map,size,ho,(uint64_t)hc*E*4);
    if (!ge || !gh) return 0;
    mtp_stage<<<hc+1,256,0,cuda_decode_stream()>>>((float *)cat->ptr,(const float *)e->ptr,
        (const float *)R->ptr,(const float *)ge,(const float *)gh,E,hc,eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_mtp_combine_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *proj,
        uint32_t E, uint32_t hc) {
    using namespace qwen4_cuda;
    if (!E || !hc || hc > 4 || !tensor(out,(uint64_t)hc*E*4) || !tensor(proj,(uint64_t)(hc+1)*E*4)) return 0;
    mtp_combine<<<((uint64_t)E*hc+255)/256,256,0,cuda_decode_stream()>>>((float *)out->ptr,(const float *)proj->ptr,E,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_argmax_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *logits, uint32_t N) {
    using namespace qwen4_cuda;
    const uint64_t blocks = ((uint64_t)N+4095)/4096;
    if (!N || !tensor(out,4) || !tensor(logits,(uint64_t)N*4) || !tensor(scratch,blocks*sizeof(max_pair))) return 0;
    argmax<false><<<blocks,256,0,cuda_decode_stream()>>>((int *)out->ptr,(max_pair *)scratch->ptr,(const float *)logits->ptr,N);
    argmax<true><<<1,256,0,cuda_decode_stream()>>>((int *)out->ptr,(max_pair *)scratch->ptr,NULL,blocks);
    return launched();
}

extern "C" int ds4_gpu_qwen4_vision_encode(float *out, const float *patches, const float *pos_embed,
        uint32_t N, uint32_t grid_w, const void *map, uint64_t size, const ds4_qwen4_vision_weights *w) {
    using namespace qwen4_cuda;
    if (!out || !patches || !pos_embed || !w || !map || !N || N > 65535 || ds4_gpu_commands_active()) return 0;
    const unsigned E = w->n_embd, FF = w->n_ff, H = w->n_head, D = H ? E/H : 0;
    const unsigned P = w->n_patch, M = w->n_merge, O = w->n_out;
    if (M != 2 || !grid_w || grid_w%M || N%grid_w || (N/grid_w)%M ||
        (D != 64 && D != 72) || H*D != E || !FF || !O || !P || P > 32 || E%32) return 0;
    const unsigned IP = 3*P*P, ME = E*M*M, merged = N/(M*M);
    std::vector<ds4_gpu_tensor *> buffers;
    auto alloc = [&](uint64_t n) {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n*4);
        buffers.push_back(t);
        return t;
    };
    ds4_gpu_tensor *patch = alloc((uint64_t)N*IP), *a0 = alloc((uint64_t)N*E), *a1 = alloc((uint64_t)N*E);
    ds4_gpu_tensor *pos = alloc((uint64_t)N*E), *x = alloc((uint64_t)N*E), *tmp = alloc((uint64_t)N*E);
    ds4_gpu_tensor *qkv = alloc((uint64_t)N*3*E), *q = alloc((uint64_t)N*E), *k = alloc((uint64_t)N*E);
    ds4_gpu_tensor *v = alloc((uint64_t)N*E), *attn = alloc((uint64_t)N*E), *ffn = alloc((uint64_t)N*FF);
    ds4_gpu_tensor *m0 = alloc((uint64_t)merged*ME), *res = alloc((uint64_t)merged*O);
    bool ok = true, active = false;
    for (const auto b : buffers) if (!b) ok = false;
    auto ptr = [](ds4_gpu_tensor *t) { return (float *)t->ptr; };
    auto wf = [&](uint64_t off, unsigned n) { return (const float *)weight(map,size,off,(uint64_t)n*4); };
    auto mm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t off, unsigned type,
                  unsigned rows, unsigned K, unsigned M) {
        return ds4_gpu_qwen4_dense_mm_tensor(dst,src,map,size,off,type,rows,K,M);
    };
    auto norm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t wo, uint64_t bo) {
        const float *wgt = wf(wo,E), *bias = wf(bo,E);
        if (!wgt || !bias) return 0;
        vis_norm<<<N,256,0,cuda_decode_stream()>>>(ptr(dst),(const float *)src->ptr,wgt,bias,E,w->eps);
        return launched();
    };
    auto add = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t bo, unsigned rows,
                   unsigned width, unsigned mode) {
        const float *bias = wf(bo,width);
        if (!bias) return 0;
        vis_add<<<((uint64_t)rows*width+255)/256,256,0,cuda_decode_stream()>>>(ptr(dst),
            src ? (const float *)src->ptr : NULL,bias,rows,width,mode);
        return launched();
    };
    do {
        if (!ok) break;
        ok = ds4_gpu_tensor_write(patch,0,patches,(uint64_t)N*IP*4) &&
             ds4_gpu_tensor_write(pos,0,pos_embed,(uint64_t)N*E*4);
        if (!ok || !(active = ds4_gpu_begin_commands())) { ok = false; break; }
        ok = mm(a0,patch,w->patch_w0,w->patch_type,N,IP,E) && mm(a1,patch,w->patch_w1,w->patch_type,N,IP,E);
        const float *pb = wf(w->patch_b,E);
        if (!ok || !pb) { ok = false; break; }
        vis_patch<<<((uint64_t)N*E+255)/256,256,0,cuda_decode_stream()>>>(ptr(x),ptr(a0),ptr(a1),pb,ptr(pos),N,E);
        ok = launched();
        for (unsigned l = 0; ok && l < DS4_QWEN4_VISION_LAYERS; l++) {
            const auto &lw = w->layer[l];
            ok = norm(tmp,x,lw.ln1_w,lw.ln1_b) && mm(qkv,tmp,lw.qkv_w,lw.qkv_type,N,E,3*E);
            const float *qb = wf(lw.qkv_b,3*E);
            if (!ok || !qb) { ok = false; break; }
            vis_qkv<<<N,256,0,cuda_decode_stream()>>>(ptr(q),ptr(k),ptr(v),ptr(qkv),qb,H,D,grid_w);
            vis_attention<<<dim3(N,H),32,0,cuda_decode_stream()>>>(ptr(attn),ptr(q),ptr(k),ptr(v),N,H,D);
            ok = launched() && mm(tmp,attn,lw.out_w,lw.out_type,N,E,E) && add(x,tmp,lw.out_b,N,E,2) &&
                 norm(tmp,x,lw.ln2_w,lw.ln2_b) && mm(ffn,tmp,lw.up_w,lw.up_type,N,E,FF) &&
                 add(ffn,NULL,lw.up_b,N,FF,0) && mm(tmp,ffn,lw.down_w,lw.down_type,N,FF,E) &&
                 add(x,tmp,lw.down_b,N,E,2);
        }
        ok = ok && norm(tmp,x,w->post_ln_w,w->post_ln_b) && mm(m0,tmp,w->mm0_w,w->mm0_type,merged,ME,ME) &&
             add(m0,NULL,w->mm0_b,merged,ME,1) && mm(res,m0,w->mm2_w,w->mm2_type,merged,ME,O) &&
             add(res,NULL,w->mm2_b,merged,O,2);
    } while (false);
    if (active && !ds4_gpu_end_commands()) ok = false;
    if (ok) ok = ds4_gpu_tensor_read(res,0,out,(uint64_t)merged*O*4);
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) ds4_gpu_tensor_free(*it);
    return ok;
}

extern "C" int ds4_gpu_qwen4_attn_decode_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc,
        const ds4_gpu_tensor *sel, const ds4_gpu_tensor *count, ds4_gpu_tensor *partial,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t pos0, bool sparse, uint32_t stride, float scale) {
    using namespace qwen4_cuda;
    const uint64_t n = (uint64_t)T * H * D * 4, cb = ((uint64_t)pos0 + T) * Hkv * ds4_gpu_qwen4_kv_row_bytes(D);
    if (!T || !H || !Hkv || H % Hkv || (D != 32 && D != 128 && D != 256) ||
        !tensor(out, n) || !tensor(q, n) || !tensor(gate, n) || !tensor(kc, cb) || !tensor(vc, cb) ||
        (sparse && (!stride || !tensor(sel, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4)))) return 0;
    const unsigned keys = sparse ? stride : pos0 + T;
    if (!partial && T >= 32 && D == 256 && H/Hkv <= 16 &&
        !((uintptr_t)kc->ptr&15) && !((uintptr_t)vc->ptr&15) &&
        ds4_cuda_attn_tokentile_arch_ok() && !g_quality_mode && !getenv("DS4_QWEN4_NO_ATTN_MM")) {
        const auto kernel = g_qwen4_kv_q8 ? attention_group<true> : attention_group<false>;
        kernel<<<dim3(Hkv,T),256,0,cuda_decode_stream()>>>(
            (float *)out->ptr,(const float *)q->ptr,(const float *)gate->ptr,(const __half *)kc->ptr,(const __half *)vc->ptr,
            sparse ? (const int *)sel->ptr : NULL,sparse ? (const unsigned *)count->ptr : NULL,
            H,Hkv,pos0,stride,sparse,scale);
        return launched();
    }
    const unsigned splits = partial ? std::min(64u, (keys + 31) / 32) : 1;
    if (partial && !tensor(partial, (uint64_t)T * H * splits * (D + 2) * 4)) return 0;
    const dim3 grid((H + 3) / 4, T, splits);
#define QWEN_ATTN_FMT(DIM, Q8) attention<DIM, Q8><<<grid, 128, 0, cuda_decode_stream()>>>((float *)out->ptr, \
        partial ? (float *)partial->ptr : NULL, (const float *)q->ptr, (const float *)gate->ptr, \
        (const __half *)kc->ptr, (const __half *)vc->ptr, sparse ? (const int *)sel->ptr : NULL, \
        sparse ? (const unsigned *)count->ptr : NULL, H, Hkv, pos0, stride, sparse, splits, (keys + splits - 1) / splits, scale)
#define QWEN_ATTN(DIM) if (g_qwen4_kv_q8) { QWEN_ATTN_FMT(DIM, true); } else { QWEN_ATTN_FMT(DIM, false); }
    if (D == 32) { QWEN_ATTN(32); }
    else if (D == 128) { QWEN_ATTN(128); }
    else { QWEN_ATTN(256); }
#undef QWEN_ATTN
#undef QWEN_ATTN_FMT
    if (!launched()) return 0;
    if (splits > 1) attn_merge<<<dim3(H, T), D, 0, cuda_decode_stream()>>>((float *)out->ptr,
        (const float *)partial->ptr, (const float *)gate->ptr, H, D, splits);
    return launched();
}
