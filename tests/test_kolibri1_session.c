/* Kolibri-1 session-continuation and save/restore gate.
 *
 * Continuation: a session sync that extends an existing checkpoint must
 * produce the same next-token logits as a fresh prefill of the full prompt.
 * This covers the CPU reference continuation path, which has its own
 * per-token forward and must not fall through to the DeepSeek decode code,
 * and the chunked-prefill continuation of the default backend.
 *
 * Save/restore: a prefill snapshot must round-trip through both
 * ds4_session_save_snapshot/load_snapshot (ds4-bench frontiers) and the file
 * payload path (ds4-agent /save, server disk KV caches) and land on the same
 * logits a fresh prefill produces for the same prefix.  The prompt crosses the
 * sliding window (the 46-token fixture crosses the 33-row fixture window; the
 * GPU ring is forced small with DS4_KOLIBRI1_PREFILL_CHUNK=1) so the sliding
 * layer mapping is exercised on both backends.
 *
 * Usage: tests/test_kolibri1_session [--cpu] [--ctx N] [--tokens N] [model.gguf]
 */
#include "../ds4.c"

#include <math.h>

static ds4_backend test_default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static void push_ids(ds4_tokens *t, const int *ids, int count) {
    for (int i = 0; i < count; i++) ds4_tokens_push(t, ids[i]);
}

static int test_token_id(int i, int vocab) {
    return (int)(((uint32_t)i * 7919u + 13u) % (uint32_t)vocab);
}

static float test_max_abs_diff(const float *a, const float *b, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > max_abs) max_abs = d;
    }
    return max_abs;
}

/* Prefill -> save snapshot and file payload -> decode past the frontier ->
 * restore both into sessions and compare with a fresh prefill, including one
 * further decode step so the restored sliding cache is read. */
static int run_save_restore(ds4_engine *e, bool cpu, int ctx, int n_tokens) {
    const int vocab = ds4_engine_vocab_size(e);
    float *ref_logits = malloc((size_t)vocab * sizeof(ref_logits[0]));
    float *live_logits = malloc((size_t)vocab * sizeof(live_logits[0]));
    float *file_logits = malloc((size_t)vocab * sizeof(file_logits[0]));
    ds4_session *live = NULL, *ref = NULL, *file = NULL;
    ds4_session_snapshot snap = {0};
    ds4_tokens prompt = {0};
    FILE *fp = NULL;
    int rc = 1;
    char err[192] = {0};

    if (!ref_logits || !live_logits || !file_logits) {
        fprintf(stderr, "test_kolibri1_session: out of memory\n");
        goto done;
    }
    for (int i = 0; i < n_tokens; i++) ds4_tokens_push(&prompt, test_token_id(i, vocab));

    if (ds4_session_create(&live, e, ctx) != 0 ||
        ds4_session_create(&ref, e, ctx) != 0 ||
        ds4_session_create(&file, e, ctx) != 0) {
        fprintf(stderr, "test_kolibri1_session: session create failed\n");
        goto done;
    }
    if (ds4_session_sync(ref, &prompt, err, sizeof(err)) != 0 ||
        ds4_session_sync(live, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "test_kolibri1_session: save/restore prefill failed: %s\n", err);
        goto done;
    }

    const uint64_t payload_bytes = ds4_session_payload_bytes(live);
    if (payload_bytes == 0 ||
        ds4_session_save_snapshot(live, &snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "test_kolibri1_session: snapshot save failed: %s\n",
                payload_bytes == 0 ? "no payload bytes" : err);
        goto done;
    }
    if (snap.len != payload_bytes) {
        fprintf(stderr, "test_kolibri1_session: snapshot size %llu != payload bytes %llu\n",
                (unsigned long long)snap.len, (unsigned long long)payload_bytes);
        goto done;
    }
    fp = tmpfile();
    if (!fp || ds4_session_save_payload(live, fp, err, sizeof(err)) != 0) {
        fprintf(stderr, "test_kolibri1_session: file payload save failed: %s\n", err);
        goto done;
    }
    if ((uint64_t)ftello(fp) != snap.len) {
        fprintf(stderr, "test_kolibri1_session: file payload size %lld != snapshot %llu\n",
                (long long)ftello(fp), (unsigned long long)snap.len);
        goto done;
    }

    /* Decode past the saved frontier: restoring over this advanced state is
     * the ds4-bench flow, and it must discard the extra rows. */
    for (int i = 0; i < 4; i++) {
        const int token = ds4_session_argmax(live);
        if (token < 0 || ds4_session_eval(live, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "test_kolibri1_session: save/restore decode failed: %s\n", err);
            goto done;
        }
    }
    if (ds4_session_load_snapshot(live, &snap, err, sizeof(err)) != 0 ||
        ds4_session_pos(live) != n_tokens) {
        fprintf(stderr, "test_kolibri1_session: snapshot restore failed: %s\n", err);
        goto done;
    }
    rewind(fp);
    if (ds4_session_load_payload(file, fp, snap.len, err, sizeof(err)) != 0 ||
        ds4_session_pos(file) != n_tokens) {
        fprintf(stderr, "test_kolibri1_session: file payload restore failed: %s\n", err);
        goto done;
    }

    if (ds4_session_copy_logits(ref, ref_logits, vocab) != vocab ||
        ds4_session_copy_logits(live, live_logits, vocab) != vocab ||
        ds4_session_copy_logits(file, file_logits, vocab) != vocab) {
        fprintf(stderr, "test_kolibri1_session: logits copy failed\n");
        goto done;
    }
    float max_live = test_max_abs_diff(live_logits, ref_logits, vocab);
    float max_file = test_max_abs_diff(file_logits, ref_logits, vocab);
    const int ref_argmax = ds4_session_argmax(ref);
    const int live_argmax = ds4_session_argmax(live);
    const int file_argmax = ds4_session_argmax(file);
    if (ref_argmax != live_argmax || ref_argmax != file_argmax ||
        max_live > 1.0e-5f || max_file > 1.0e-5f) {
        fprintf(stderr,
                "test_kolibri1_session: restore logits max|d| live %.3g file %.3g "
                "argmax live %d file %d ref %d\n",
                max_live, max_file, live_argmax, file_argmax, ref_argmax);
        goto done;
    }

    /* One more decode step reads the restored sliding rows, not just the
     * restored logits. */
    const int next = ref_argmax;
    if (ds4_session_eval(ref, next, err, sizeof(err)) != 0 ||
        ds4_session_eval(live, next, err, sizeof(err)) != 0 ||
        ds4_session_eval(file, next, err, sizeof(err)) != 0 ||
        ds4_session_copy_logits(ref, ref_logits, vocab) != vocab ||
        ds4_session_copy_logits(live, live_logits, vocab) != vocab ||
        ds4_session_copy_logits(file, file_logits, vocab) != vocab) {
        fprintf(stderr, "test_kolibri1_session: restored decode failed: %s\n", err);
        goto done;
    }
    max_live = test_max_abs_diff(live_logits, ref_logits, vocab);
    max_file = test_max_abs_diff(file_logits, ref_logits, vocab);
    const bool ok = ds4_session_argmax(ref) == ds4_session_argmax(live) &&
                    ds4_session_argmax(ref) == ds4_session_argmax(file) &&
                    max_live <= 1.0e-5f && max_file <= 1.0e-5f;
    printf("kolibri1-session %s: save/restore tokens=%d payload=%llu "
           "max|d| snapshot %.3g file %.3g argmax %d/%d/%d %s\n",
           cpu ? "cpu" : "default", n_tokens,
           (unsigned long long)snap.len, max_live, max_file,
           ds4_session_argmax(live), ds4_session_argmax(file), ds4_session_argmax(ref),
           ok ? "OK" : "FAIL");
    rc = ok ? 0 : 1;

done:
    if (fp) fclose(fp);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(live);
    ds4_session_free(ref);
    ds4_session_free(file);
    ds4_tokens_free(&prompt);
    free(ref_logits);
    free(live_logits);
    free(file_logits);
    return rc;
}

int main(int argc, char **argv) {
    bool cpu = false;
    int ctx = 512;
    int tokens = 46;
    const char *model = "gguf/Kolibri-1-mini.gguf";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cpu")) cpu = true;
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = atoi(argv[++i]);
        else model = argv[i];
    }
    if (ctx <= 0 || tokens <= 0) {
        fprintf(stderr, "test_kolibri1_session: invalid --ctx/--tokens\n");
        return 2;
    }

    ds4_engine_options opt = {0};
    opt.model_path = model;
    opt.backend = cpu ? DS4_BACKEND_CPU : test_default_backend();
    opt.context_size = ctx;
    opt.power_percent = 100;
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) {
        fprintf(stderr, "test_kolibri1_session: engine open failed for %s\n", model);
        return 2;
    }

    static const int ids[] = {11, 22, 33, 44, 55, 66};
    const int base_len = 3;
    const int full_len = 6;
    const int vocab = ds4_engine_vocab_size(e);
    float *live = malloc((size_t)vocab * sizeof(live[0]));
    float *fresh = malloc((size_t)vocab * sizeof(fresh[0]));
    ds4_tokens base = {0}, full = {0};
    ds4_session *cont = NULL, *ref = NULL;
    int rc = 1;
    char err[160];
    if (!live || !fresh) {
        fprintf(stderr, "test_kolibri1_session: out of memory\n");
        goto done;
    }
    push_ids(&base, ids, base_len);
    push_ids(&full, ids, full_len);

    if (ds4_session_create(&cont, e, ctx) != 0 ||
        ds4_session_create(&ref, e, ctx) != 0) {
        fprintf(stderr, "test_kolibri1_session: session create failed\n");
        goto done;
    }
    if (ds4_session_sync(cont, &base, err, sizeof(err)) != 0) {
        fprintf(stderr, "test_kolibri1_session: base sync failed: %s\n", err);
        goto done;
    }
    if (ds4_session_sync(cont, &full, err, sizeof(err)) != 0) {
        fprintf(stderr, "test_kolibri1_session: continuation sync failed: %s\n", err);
        goto done;
    }
    if (ds4_session_sync(ref, &full, err, sizeof(err)) != 0) {
        fprintf(stderr, "test_kolibri1_session: fresh sync failed: %s\n", err);
        goto done;
    }
    if (ds4_session_copy_logits(cont, live, vocab) != vocab ||
        ds4_session_copy_logits(ref, fresh, vocab) != vocab) {
        fprintf(stderr, "test_kolibri1_session: logits copy failed\n");
        goto done;
    }

    float max_abs = 0.0f;
    for (int i = 0; i < vocab; i++) {
        const float d = fabsf(live[i] - fresh[i]);
        if (d > max_abs) max_abs = d;
    }
    const int cont_argmax = ds4_session_argmax(cont);
    const int ref_argmax = ds4_session_argmax(ref);
    const bool ok = cont_argmax == ref_argmax && max_abs <= 1.0e-5f;
    printf("kolibri1-session %s: continuation vs fresh max|d| %.3g "
           "argmax %d/%d %s\n",
           cpu ? "cpu" : "default", max_abs, cont_argmax, ref_argmax,
           ok ? "OK" : "FAIL");
    rc = ok ? 0 : 1;

done:
    if (cont) ds4_session_free(cont);
    if (ref) ds4_session_free(ref);
    ds4_tokens_free(&base);
    ds4_tokens_free(&full);
    free(live);
    free(fresh);
    if (rc != 0) {
        ds4_engine_close(e);
        return rc;
    }

    /* A small prefill chunk forces the GPU sliding ring to window + 1 rows so
     * the 46-token prompt wraps it (the CPU window is DS4_N_SWA = 33 rows).
     * Keep an explicit chunk override for the real-artifact run. */
    if (!cpu && !getenv("DS4_KOLIBRI1_PREFILL_CHUNK")) {
        setenv("DS4_KOLIBRI1_PREFILL_CHUNK", "1", 1);
    }
    if (tokens >= ctx) {
        fprintf(stderr,
                "test_kolibri1_session: --tokens %d must stay below --ctx %d\n",
                tokens, ctx);
        ds4_engine_close(e);
        return 2;
    }
    rc = run_save_restore(e, cpu, ctx, tokens);
    ds4_engine_close(e);
    return rc;
}
