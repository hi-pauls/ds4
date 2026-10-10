/* Checks the Kolibri-1 CUDA kernels (ds4_kolibri1_cuda.cuh) against a host
 * reference at the released model's widths: 48/4 GQA heads of 128, a
 * 513-token window on a ring of window + chunk rows, 384 experts top-6 and a
 * 2560-wide residual, the K/V caches as f32 and as q8_0 rows.  Needs only the
 * CUDA toolkit, not the ds4 backend:
 *   nvcc -O2 -o tests/test_kolibri1_cuda tests/test_kolibri1_cuda.cu */

#define DS4_KOLIBRI1_CUDA_KERNELS_ONLY
#include "../ds4_kolibri1_cuda.cuh"

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace kolibri1_cuda;

static const unsigned H = 48, HKV = 4, D = 128, WINDOW = 513, NE = 384, NS = 6, DIM = 2560;
static const float EPS = 1.0e-6f, ROPE_BASE = 10000.0f;
static int g_failures;

static void check_cuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "kolibri1-cuda: %s: %s\n", what, cudaGetErrorString(e));
        exit(2);
    }
}

static std::vector<float> random_values(size_t n, unsigned seed, float scale) {
    std::vector<float> v(n);
    srand(seed);
    for (size_t i = 0; i < n; i++) v[i] = scale * ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
    return v;
}

template<class T>
static T *upload(const std::vector<T> &v) {
    T *p = NULL;
    check_cuda(cudaMalloc(&p, v.size() * sizeof(T)), "cudaMalloc");
    check_cuda(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}

template<class T>
static std::vector<T> download(const T *p, size_t n) {
    std::vector<T> v(n);
    check_cuda(cudaDeviceSynchronize(), "kernel");
    check_cuda(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

static void report(const char *name, double max_err, double tol) {
    const bool ok = max_err <= tol;
    if (!ok) g_failures++;
    printf("kolibri1-cuda: %-34s max|d| %.3e %s\n", name, max_err, ok ? "OK" : "FAIL");
}

static double max_diff(const std::vector<float> &a, const std::vector<float> &b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) m = fmax(m, fabs((double)a[i] - (double)b[i]));
    return m;
}

/* K/V cache formats under test: f32 rows; q8 rows built from exact int8
 * values and half scales, whose dequantization the kernels reproduce bit for
 * bit; q8 rows the host quantized from random floats, which the reference
 * keeps unquantized, so the output error is that of the format. */
enum kv_format { KV_F32, KV_Q8_EXACT, KV_Q8_ROUNDED };
static const unsigned RB = D + D / 16;   /* q8 bytes per (row, kv head): 128 values, 4 half scales */

static float q8_scale(const std::vector<uint8_t> &c, size_t row_head, unsigned b) {
    __half h;
    memcpy(&h, &c[row_head * RB + D + 2 * b], 2);
    return __half2float(h);
}

static float q8_value(const std::vector<uint8_t> &c, size_t row_head, unsigned d) {
    return q8_scale(c, row_head, d / 32) * (float)(int8_t)c[row_head * RB + d];
}

/* kv_store_q8 on the host: a block's scale is amax / 127 rounded to half,
 * the values rounded against that scale. */
static void q8_quantize_row(std::vector<uint8_t> &c, size_t row_head, const float *v) {
    for (unsigned b = 0; b < D / 32; b++) {
        float amax = 0;
        for (unsigned i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[b * 32 + i]));
        const __half dh = __float2half_rn(amax / 127.0f);
        const float d = __half2float(dh), id = d > 0 ? 1.0f / d : 0;
        memcpy(&c[row_head * RB + D + 2 * b], &dh, 2);
        for (unsigned i = 0; i < 32; i++)
            c[row_head * RB + b * 32 + i] = (uint8_t)(int8_t)fminf(fmaxf(roundf(v[b * 32 + i] * id), -127.0f), 127.0f);
    }
}

/* A q8 cache of rows * HKV row heads and the floats it stands for: EXACT
 * draws int8 values and half scales directly, ROUNDED quantizes the given
 * floats and leaves them as the reference. */
static std::vector<uint8_t> q8_cache(std::vector<float> &values, size_t rows, kv_format fmt, unsigned seed) {
    std::vector<uint8_t> c(rows * HKV * RB, 0);
    srand(seed);
    for (size_t rh = 0; rh < rows * HKV; rh++) {
        if (fmt == KV_Q8_ROUNDED) q8_quantize_row(c, rh, &values[rh * D]);
        else {
            for (unsigned b = 0; b < D / 32; b++) {
                const __half dh = __float2half_rn(0.002f + 0.01f * (float)rand() / (float)RAND_MAX);
                memcpy(&c[rh * RB + D + 2 * b], &dh, 2);
            }
            for (unsigned d = 0; d < D; d++) c[rh * RB + d] = (uint8_t)(int8_t)(rand() % 255 - 127);
            for (unsigned d = 0; d < D; d++) values[rh * D + d] = q8_value(c, rh, d);
        }
    }
    return c;
}

static std::vector<float> q8_dequantize(const std::vector<uint8_t> &c) {
    std::vector<float> v(c.size() / RB * D);
    for (size_t rh = 0; rh < c.size() / RB; rh++)
        for (unsigned d = 0; d < D; d++) v[rh * D + d] = q8_value(c, rh, d);
    return v;
}

/* Normed (and maybe roped) head row, the host twin of attn_prep. */
static void ref_head(float *dst, const float *src, const float *gamma, unsigned pos, bool rope) {
    double ss = 0;
    for (unsigned i = 0; i < D; i++) ss += (double)src[i] * src[i];
    const double r = 1.0 / sqrt(ss / D + EPS);
    for (unsigned i = 0; i < D; i++) dst[i] = (float)(src[i] * r * gamma[i]);
    if (!rope) return;
    for (unsigned i = 0; i < D / 2; i++) {
        const double theta = (double)pos * pow(ROPE_BASE, -2.0 * i / D);
        const double x0 = dst[i], x1 = dst[i + D / 2];
        dst[i] = (float)(x0 * cos(theta) - x1 * sin(theta));
        dst[i + D / 2] = (float)(x0 * sin(theta) + x1 * cos(theta));
    }
}

/* q8 writes the caches as q8 rows; they come back dequantized and are held to
 * the format's tolerance, the queries to the f32 one. */
static void test_attn_prep(unsigned T, unsigned pos0, unsigned cache_rows, bool rope, bool q8 = false) {
    const unsigned rows = cache_rows ? cache_rows : pos0 + T;
    const auto q = random_values((size_t)T * H * D, 11, 2.0f), k = random_values((size_t)T * HKV * D, 12, 2.0f);
    const auto v = random_values((size_t)T * HKV * D, 13, 2.0f);
    const auto gq = random_values(D, 14, 1.0f), gk = random_values(D, 15, 1.0f);
    std::vector<float> zero((size_t)rows * HKV * D, 0.0f);
    const std::vector<uint8_t> zero_q8((size_t)rows * HKV * RB, 0);
    float *dq = upload(q), *dk = upload(k), *dv = upload(v), *dgq = upload(gq), *dgk = upload(gk);
    float *qout = upload(std::vector<float>(q.size(), 0.0f));
    float *kc = q8 ? (float *)upload(zero_q8) : upload(zero), *vc = q8 ? (float *)upload(zero_q8) : upload(zero);
    attn_prep<<<dim3(H + HKV + 1, T), 32>>>(make_attn_prep_args(T, H, HKV, D, pos0, cache_rows, rope, ROPE_BASE, EPS, q8),
                                            dq, dk, dv, dgq, dgk, qout, kc, vc);
    const auto gq_out = download(qout, q.size());
    const auto gkc = q8 ? q8_dequantize(download((const uint8_t *)kc, zero_q8.size())) : download(kc, zero.size());
    const auto gvc = q8 ? q8_dequantize(download((const uint8_t *)vc, zero_q8.size())) : download(vc, zero.size());
    std::vector<float> rq(q.size()), rkc = zero, rvc = zero;
    for (unsigned t = 0; t < T; t++) {
        const unsigned pos = pos0 + t, row = cache_rows ? pos % cache_rows : pos;
        for (unsigned h = 0; h < H; h++)
            ref_head(&rq[((size_t)t * H + h) * D], &q[((size_t)t * H + h) * D], gq.data(), pos, rope);
        for (unsigned h = 0; h < HKV; h++) {
            ref_head(&rkc[((size_t)row * HKV + h) * D], &k[((size_t)t * HKV + h) * D], gk.data(), pos, rope);
            for (unsigned i = 0; i < D; i++) rvc[((size_t)row * HKV + h) * D + i] = v[((size_t)t * HKV + h) * D + i];
        }
    }
    char name[64];
    snprintf(name, sizeof(name), "attn_prep%s T=%u pos0=%u ring=%u", q8 ? " q8" : "", T, pos0, cache_rows);
    if (q8) {
        report(name, max_diff(gq_out, rq), 1e-4);
        snprintf(name, sizeof(name), "attn_prep q8 caches T=%u pos0=%u ring=%u", T, pos0, cache_rows);
        report(name, fmax(max_diff(gkc, rkc), max_diff(gvc, rvc)), 2e-2);
    } else
        report(name, fmax(max_diff(gq_out, rq), fmax(max_diff(gkc, rkc), max_diff(gvc, rvc))), 1e-4);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dgq); cudaFree(dgk); cudaFree(qout); cudaFree(kc); cudaFree(vc);
}

static void test_dequant(unsigned rows) {
    std::vector<float> ref((size_t)rows * HKV * D);
    const auto c = q8_cache(ref, rows, KV_Q8_EXACT, 19);
    uint8_t *dc = upload(c);
    float *out = upload(std::vector<float>(ref.size(), 0.0f));
    kv_dequant_q8<<<(unsigned)((ref.size() / 4 + 255) / 256), 256>>>(out, dc, (unsigned)ref.size(), D);
    report("kv_dequant_q8", max_diff(download(out, ref.size()), ref), 0.0);
    cudaFree(dc); cudaFree(out);
}

/* attn_softmax_piece against a one-shot softmax: the host plays the V
 * product of each piece on the columns the kernel leaves, so the state,
 * the row rescaling and the final normalization are what is checked. */
static void test_softmax_piece(unsigned T, unsigned pos0, unsigned piece) {
    const unsigned n_keys = pos0 + T, h = 5;
    const auto s = random_values((size_t)T * n_keys, 51, 3.0f), v = random_values((size_t)n_keys * D, 52, 1.0f);
    std::vector<float> acc((size_t)T * H * D, 7.0f), ref((size_t)T * D);
    float *ml = upload(std::vector<float>((size_t)T * H * 2, 0.0f)), *out = upload(acc), *col = upload(std::vector<float>((size_t)T * piece, 0.0f));
    for (unsigned k0 = 0; k0 < n_keys; k0 += piece) {
        const unsigned keys = std::min(piece, n_keys - k0);
        std::vector<float> sc((size_t)T * keys);
        for (unsigned t = 0; t < T; t++)
            for (unsigned i = 0; i < keys; i++) sc[(size_t)t * keys + i] = s[(size_t)t * n_keys + k0 + i];
        check_cuda(cudaMemcpy(col, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
        attn_softmax_piece<<<T, 256>>>(col, ml, out, keys, k0, pos0, H, h, D);
        const auto p = download(col, sc.size()), o = download(out, acc.size());
        for (unsigned t = 0; t < T; t++)
            for (unsigned d = 0; d < D; d++) {
                double a = o[((size_t)t * H + h) * D + d];
                for (unsigned i = 0; i < keys; i++) a += (double)p[(size_t)t * keys + i] * v[((size_t)k0 + i) * D + d];
                acc[((size_t)t * H + h) * D + d] = (float)a;
            }
        check_cuda(cudaMemcpy(out, acc.data(), acc.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
    }
    for (unsigned t = 0; t < T; t++) {
        const unsigned n = pos0 + t + 1;
        double mx = -1e300, l = 0;
        std::vector<double> w(n);
        for (unsigned i = 0; i < n; i++) mx = fmax(mx, s[(size_t)t * n_keys + i]);
        for (unsigned i = 0; i < n; i++) l += w[i] = exp(s[(size_t)t * n_keys + i] - mx);
        for (unsigned d = 0; d < D; d++) {
            double a = 0;
            for (unsigned i = 0; i < n; i++) a += w[i] * v[(size_t)i * D + d];
            ref[(size_t)t * D + d] = (float)(a / l);
        }
    }
    std::vector<float> got((size_t)T * D);
    for (unsigned t = 0; t < T; t++)
        for (unsigned d = 0; d < D; d++) got[(size_t)t * D + d] = acc[((size_t)t * H + h) * D + d];
    char name[64];
    snprintf(name, sizeof(name), "softmax_piece T=%u pos0=%u piece=%u", T, pos0, piece);
    report(name, max_diff(got, ref), 2e-5);
    cudaFree(ml); cudaFree(out); cudaFree(col);
}

/* The kernels the backend picks, so DS4_KOLIBRI1_OLD_ATTN=1 checks the
 * previous q8 tiled kernel and merge instead. */
template<bool Q8>
static void launch_attn(bool tiles, const attn_args &a, unsigned T, const float *q, const float *kc, const float *vc,
                        float *out, float *part) {
    const auto q8_tiles = attn_old_kernels() ? attn_decode_tiles_q8 : attn_decode_tiles_q8_pipe;
    if (tiles) (Q8 ? q8_tiles : attn_decode_tiles)<<<dim3(a.n_splits, HKV, T), 32 * ATTN_NSG>>>(a, q, kc, vc, out, part);
    else attn_decode<Q8><<<dim3(a.n_splits, HKV, T), 32 * ATTN_NSG>>>(a, q, kc, vc, out, part);
}

static void launch_merge(const attn_args &a, unsigned T, const float *part, float *out) {
    if (attn_old_kernels()) attn_merge<<<dim3(H, T), 32>>>(a, part, out);
    else attn_merge_wide<<<dim3(H, T), ATTN_MERGE_THREADS>>>(a, part, out);
}

/* tiles takes attn_decode_tiles, which only does full context with splits. */
static void test_attn(unsigned T, unsigned pos0, unsigned window, unsigned cache_rows, bool split, bool tiles = false,
                      kv_format fmt = KV_F32) {
    const unsigned rows = cache_rows ? cache_rows : pos0 + T;
    const float scale = 1.0f / sqrtf((float)D);
    const auto q = random_values((size_t)T * H * D, 21, 1.0f);
    auto kc = random_values((size_t)rows * HKV * D, 22, 1.0f), vc = random_values((size_t)rows * HKV * D, 23, 1.0f);
    const attn_args a = tiles ? make_attn_tile_args(T, H, HKV, D, pos0, scale)
                              : make_attn_args(T, H, HKV, D, pos0, window, cache_rows, split, scale);
    float *dq = upload(q), *dk, *dv, *out = upload(std::vector<float>(q.size(), 0.0f));
    if (fmt == KV_F32) {
        dk = upload(kc);
        dv = upload(vc);
    } else {
        dk = (float *)upload(q8_cache(kc, rows, fmt, 24));
        dv = (float *)upload(q8_cache(vc, rows, fmt, 25));
    }
    float *part = upload(std::vector<float>((size_t)T * H * ATTN_MAX_SPLITS * (2 + D), 0.0f));
    if (fmt == KV_F32) launch_attn<false>(tiles, a, T, dq, dk, dv, out, a.n_splits > 1 ? part : NULL);
    else launch_attn<true>(tiles, a, T, dq, dk, dv, out, a.n_splits > 1 ? part : NULL);
    if (a.n_splits > 1) launch_merge(a, T, part, out);
    const auto got = download(out, q.size());
    std::vector<float> ref(q.size());
    for (unsigned t = 0; t < T; t++) {
        const unsigned pos = pos0 + t, n = window ? std::min(pos + 1, window) : pos + 1, start = pos + 1 - n;
        for (unsigned h = 0; h < H; h++) {
            const unsigned kvh = h / (H / HKV);
            std::vector<double> s(n);
            double mx = -1e300, l = 0;
            for (unsigned i = 0; i < n; i++) {
                const unsigned p = start + i, row = cache_rows ? p % cache_rows : p;
                double dot = 0;
                for (unsigned d = 0; d < D; d++)
                    dot += (double)q[((size_t)t * H + h) * D + d] * scale * kc[((size_t)row * HKV + kvh) * D + d];
                s[i] = dot;
                mx = fmax(mx, dot);
            }
            for (unsigned i = 0; i < n; i++) l += s[i] = exp(s[i] - mx);
            for (unsigned d = 0; d < D; d++) {
                double o = 0;
                for (unsigned i = 0; i < n; i++) {
                    const unsigned p = start + i, row = cache_rows ? p % cache_rows : p;
                    o += s[i] * vc[((size_t)row * HKV + kvh) * D + d];
                }
                ref[((size_t)t * H + h) * D + d] = (float)(o / l);
            }
        }
    }
    char name[64];
    snprintf(name, sizeof(name), "attn%s%s T=%u pos0=%u win=%u ring=%u splits=%u", tiles ? " tiles" : "",
             fmt == KV_F32 ? "" : fmt == KV_Q8_EXACT ? " q8" : " q8~", T, pos0, window, cache_rows, a.n_splits);
    /* the q8 tiled kernel carries the query as int8 hi/lo, about 15 bits */
    report(name, max_diff(got, ref), fmt == KV_Q8_ROUNDED ? 2e-2 : fmt == KV_Q8_EXACT && tiles ? 2e-4 : 2e-5);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(out); cudaFree(part);
}

/* DS4_TEST_KOLIBRI_BENCH=1: microseconds per launch (kernel plus merge) of a
 * decode kernel at T=1 in one cache format, the split layout the backend
 * uses. */
static void bench_attn(unsigned pos0, unsigned window, unsigned cache_rows, bool tiles, kv_format fmt) {
    const unsigned T = 1, rows = cache_rows ? cache_rows : pos0 + T, launches = 200;
    const float scale = 1.0f / sqrtf((float)D);
    const auto q = random_values((size_t)T * H * D, 21, 1.0f);
    std::vector<float> kc((size_t)rows * HKV * D), vc((size_t)rows * HKV * D);
    const attn_args a = tiles ? make_attn_tile_args(T, H, HKV, D, pos0, scale)
                              : make_attn_args(T, H, HKV, D, pos0, window, cache_rows, true, scale);
    float *dq = upload(q), *dk, *dv, *out = upload(std::vector<float>(q.size(), 0.0f));
    if (fmt == KV_F32) {
        dk = upload(random_values(kc.size(), 22, 1.0f));
        dv = upload(random_values(vc.size(), 23, 1.0f));
    } else {
        dk = (float *)upload(q8_cache(kc, rows, KV_Q8_EXACT, 24));
        dv = (float *)upload(q8_cache(vc, rows, KV_Q8_EXACT, 25));
    }
    float *part = upload(std::vector<float>((size_t)T * H * ATTN_MAX_SPLITS * (2 + D), 0.0f));
    cudaEvent_t t0, t1;
    check_cuda(cudaEventCreate(&t0), "event");
    check_cuda(cudaEventCreate(&t1), "event");
    for (unsigned i = 0; i < launches + 10; i++) {
        if (i == 10) check_cuda(cudaEventRecord(t0), "event");
        if (fmt == KV_F32) launch_attn<false>(tiles, a, T, dq, dk, dv, out, a.n_splits > 1 ? part : NULL);
        else launch_attn<true>(tiles, a, T, dq, dk, dv, out, a.n_splits > 1 ? part : NULL);
        if (a.n_splits > 1) launch_merge(a, T, part, out);
    }
    check_cuda(cudaEventRecord(t1), "event");
    check_cuda(cudaEventSynchronize(t1), "bench");
    float ms = 0;
    check_cuda(cudaEventElapsedTime(&ms, t0, t1), "event");
    printf("kolibri1-cuda: bench %-6s %s%s keys=%6u splits=%3u %9.1f us/launch\n", tiles ? "tiles" : "decode",
           fmt == KV_F32 ? "f32" : "q8 ", attn_old_kernels() ? " (old)" : "", window ? window : pos0 + T, a.n_splits,
           1000.0 * ms / launches);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(out); cudaFree(part);
}

static void test_router(unsigned T, bool ties) {
    auto logits = random_values((size_t)T * NE, 31, 4.0f), bias = random_values(NE, 32, 1.0f);
    if (ties) {
        for (auto &x : logits) x = 0.25f;
        for (auto &x : bias) x = 0.0f;
    }
    float *dl = upload(logits), *db = upload(bias), *dw = upload(std::vector<float>((size_t)T * NS, 0.0f));
    int *ds = upload(std::vector<int>((size_t)T * NS, -1));
    router_topk<<<T, 32 * ROUTER_NSG>>>(dl, db, ds, dw, T, NE, NS, 1.0f);
    const auto sel = download(ds, (size_t)T * NS);
    const auto w = download(dw, (size_t)T * NS);
    unsigned mismatched = 0;
    double werr = 0;
    for (unsigned t = 0; t < T; t++) {
        std::vector<bool> taken(NE, false);
        for (unsigned r = 0; r < NS; r++) {
            int best = -1;
            for (unsigned e = 0; e < NE; e++) {
                const float sc = logits[(size_t)t * NE + e] + bias[e];
                if (!taken[e] && (best < 0 || sc > logits[(size_t)t * NE + best] + bias[best])) best = (int)e;
            }
            taken[best] = true;
            if (sel[(size_t)t * NS + r] != best) mismatched++;
            const double sig = 1.0 / (1.0 + exp(-(double)logits[(size_t)t * NE + best]));
            werr = fmax(werr, fabs(w[(size_t)t * NS + r] - sig));
        }
    }
    char name[64];
    snprintf(name, sizeof(name), "router T=%u%s mismatched=%u", T, ties ? " ties" : "", mismatched);
    report(name, mismatched ? 1.0 : werr, 1e-6);
    cudaFree(dl); cudaFree(db); cudaFree(dw); cudaFree(ds);
}

static void test_moe_sum(unsigned T) {
    const auto part = random_values((size_t)T * (NS + 1) * DIM, 41, 1.0f), w = random_values((size_t)T * NS, 42, 1.0f);
    float *dp = upload(part), *dw = upload(w), *out = upload(std::vector<float>((size_t)T * DIM, 0.0f));
    moe_sum<<<dim3((DIM + 255) / 256, T), 256>>>(dp, dw, out, T, NS, DIM);
    const auto got = download(out, (size_t)T * DIM);
    std::vector<float> ref((size_t)T * DIM);
    for (unsigned t = 0; t < T; t++)
        for (unsigned d = 0; d < DIM; d++) {
            double acc = part[((size_t)t * (NS + 1) + NS) * DIM + d];
            for (unsigned s = 0; s < NS; s++) acc += (double)w[(size_t)t * NS + s] * part[((size_t)t * (NS + 1) + s) * DIM + d];
            ref[(size_t)t * DIM + d] = (float)acc;
        }
    report("moe_sum", max_diff(got, ref), 1e-5);
    cudaFree(dp); cudaFree(dw); cudaFree(out);
}

int main(void) {
    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, 0), "device");
    printf("kolibri1-cuda: %s\n", prop.name);
    const unsigned chunk = 8, ring = WINDOW + chunk;
    if (getenv("DS4_TEST_KOLIBRI_BENCH")) {
        const unsigned keys[] = { 30000, 65536, 131072 };
        for (unsigned i = 0; i < 3; i++) {
            bench_attn(keys[i] - 1, 0, 0, true, KV_F32);
            bench_attn(keys[i] - 1, 0, 0, true, KV_Q8_EXACT);
            bench_attn(keys[i] - 1, 0, 0, false, KV_F32);
            bench_attn(keys[i] - 1, 0, 0, false, KV_Q8_EXACT);
        }
        bench_attn(30000, WINDOW, ring, false, KV_F32);
        bench_attn(30000, WINDOW, ring, false, KV_Q8_EXACT);
        return 0;
    }
    test_attn_prep(chunk, 1000, ring, true);
    test_attn_prep(3, 40, 0, false);
    test_attn(1, 1000, 0, 0, true);
    test_attn(1, 1000, WINDOW, ring, true);
    test_attn(1, 1000, WINDOW, ring, false);
    test_attn(1, 30000, WINDOW, ring, true);
    test_attn(2, 300, WINDOW, ring, true);
    test_attn(chunk, 1200, WINDOW, ring, false);
    test_attn(chunk, 30, 0, 0, false);
    test_attn(1, 20, 0, 0, true, true);
    test_attn(1, 40, 0, 0, true, true);
    test_attn(2, 1000, 0, 0, true, true);
    test_attn(1, 29999, 0, 0, true, true);
    test_attn(1, 30000, 0, 0, true, true);
    /* q8 rows: exact rows at the f32 tolerance, rounded ones at the format's */
    test_attn_prep(chunk, 1000, ring, true, true);
    test_attn_prep(3, 40, 0, false, true);
    test_dequant(1000);
    test_softmax_piece(5, 20, 8);
    test_softmax_piece(3, 100, 64);
    test_attn(1, 1000, 0, 0, true, false, KV_Q8_EXACT);
    test_attn(1, 1000, WINDOW, ring, true, false, KV_Q8_EXACT);
    test_attn(1, 30000, WINDOW, ring, true, false, KV_Q8_EXACT);
    test_attn(2, 300, WINDOW, ring, true, false, KV_Q8_EXACT);
    test_attn(chunk, 1200, WINDOW, ring, false, false, KV_Q8_EXACT);
    test_attn(chunk, 30, 0, 0, false, false, KV_Q8_EXACT);
    test_attn(1, 20, 0, 0, true, true, KV_Q8_EXACT);
    test_attn(1, 40, 0, 0, true, true, KV_Q8_EXACT);
    test_attn(2, 1000, 0, 0, true, true, KV_Q8_EXACT);
    test_attn(1, 29999, 0, 0, true, true, KV_Q8_EXACT);
    test_attn(1, 30000, 0, 0, true, true, KV_Q8_EXACT);
    test_attn(1, 30000, WINDOW, ring, true, false, KV_Q8_ROUNDED);
    test_attn(1, 30000, 0, 0, true, true, KV_Q8_ROUNDED);
    test_router(17, false);
    test_router(2, true);
    test_moe_sum(3);
    printf("kolibri1-cuda: %s\n", g_failures ? "FAILED" : "all kernels match the host reference");
    return g_failures ? 1 : 0;
}
