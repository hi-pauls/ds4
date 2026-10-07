/* Checks the Kolibri-1 CUDA kernels (ds4_kolibri1_cuda.cuh) against a host
 * reference at the released model's widths: 48/4 GQA heads of 128, a
 * 513-token window on a ring of window + chunk rows, 384 experts top-6 and a
 * 2560-wide residual.  Needs only the CUDA toolkit, not the ds4 backend:
 *   nvcc -O2 -o tests/test_kolibri1_cuda tests/test_kolibri1_cuda.cu */

#define DS4_KOLIBRI1_CUDA_KERNELS_ONLY
#include "../ds4_kolibri1_cuda.cuh"

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

static void test_attn_prep(unsigned T, unsigned pos0, unsigned cache_rows, bool rope) {
    const unsigned rows = cache_rows ? cache_rows : pos0 + T;
    const auto q = random_values((size_t)T * H * D, 11, 2.0f), k = random_values((size_t)T * HKV * D, 12, 2.0f);
    const auto v = random_values((size_t)T * HKV * D, 13, 2.0f);
    const auto gq = random_values(D, 14, 1.0f), gk = random_values(D, 15, 1.0f);
    std::vector<float> zero((size_t)rows * HKV * D, 0.0f);
    float *dq = upload(q), *dk = upload(k), *dv = upload(v), *dgq = upload(gq), *dgk = upload(gk);
    float *qout = upload(std::vector<float>(q.size(), 0.0f)), *kc = upload(zero), *vc = upload(zero);
    attn_prep<<<dim3(H + HKV + 1, T), 32>>>(make_attn_prep_args(T, H, HKV, D, pos0, cache_rows, rope, ROPE_BASE, EPS),
                                            dq, dk, dv, dgq, dgk, qout, kc, vc);
    const auto gq_out = download(qout, q.size()), gkc = download(kc, zero.size()), gvc = download(vc, zero.size());
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
    snprintf(name, sizeof(name), "attn_prep T=%u pos0=%u ring=%u", T, pos0, cache_rows);
    report(name, fmax(max_diff(gq_out, rq), fmax(max_diff(gkc, rkc), max_diff(gvc, rvc))), 1e-4);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dgq); cudaFree(dgk); cudaFree(qout); cudaFree(kc); cudaFree(vc);
}

static void test_attn(unsigned T, unsigned pos0, unsigned window, unsigned cache_rows, bool split) {
    const unsigned rows = cache_rows ? cache_rows : pos0 + T;
    const float scale = 1.0f / sqrtf((float)D);
    const auto q = random_values((size_t)T * H * D, 21, 1.0f);
    const auto kc = random_values((size_t)rows * HKV * D, 22, 1.0f), vc = random_values((size_t)rows * HKV * D, 23, 1.0f);
    const attn_args a = make_attn_args(T, H, HKV, D, pos0, window, cache_rows, split, scale);
    float *dq = upload(q), *dk = upload(kc), *dv = upload(vc), *out = upload(std::vector<float>(q.size(), 0.0f));
    float *part = upload(std::vector<float>((size_t)T * H * ATTN_MAX_SPLITS * (2 + D), 0.0f));
    attn_decode<<<dim3(a.n_splits, HKV, T), 32 * ATTN_NSG>>>(a, dq, dk, dv, out, a.n_splits > 1 ? part : NULL);
    if (a.n_splits > 1) attn_merge<<<dim3(H, T), 32>>>(a, part, out);
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
    snprintf(name, sizeof(name), "attn T=%u pos0=%u win=%u ring=%u splits=%u", T, pos0, window, cache_rows, a.n_splits);
    report(name, max_diff(got, ref), 2e-5);
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
    test_attn_prep(chunk, 1000, ring, true);
    test_attn_prep(3, 40, 0, false);
    test_attn(1, 1000, 0, 0, true);
    test_attn(1, 1000, WINDOW, ring, true);
    test_attn(1, 1000, WINDOW, ring, false);
    test_attn(chunk, 1200, WINDOW, ring, false);
    test_attn(chunk, 30, 0, 0, false);
    test_router(17, false);
    test_router(2, true);
    test_moe_sum(3);
    printf("kolibri1-cuda: %s\n", g_failures ? "FAILED" : "all kernels match the host reference");
    return g_failures ? 1 : 0;
}
