/* Experiment: cut a tool result out of a Qwen3.8 session without recomputing
 * what follows, and compare against a full recompute without it.
 *
 *   A  full context, tool result kept
 *   B  recompute with the tool result removed       (the exact answer)
 *   C  prefill with the tool result, cut it, ask    (the cheap answer)
 *
 * Every variant answers the same questions greedily; C and A are then fed B's
 * answer token by token to measure top-1 agreement and KL(B || variant).
 *
 * Usage: DS4_TEST_MODEL=<gguf> ./tests/test_qwen4_cut <tool result file> */
#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEN_TOKENS 160

/* reads a ':'-separated list of files into one text */
static char *read_file(const char *paths, size_t limit) {
    char *buf = malloc(limit + 1), *list = strdup(paths), *save = NULL;
    size_t n = 0;
    for (char *path = strtok_r(list, ":", &save); path && n < limit; path = strtok_r(NULL, ":", &save)) {
        FILE *fp = fopen(path, "rb");
        if (!fp) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
        n += fread(buf + n, 1, limit - n, fp);
        fclose(fp);
    }
    buf[n] = '\0';
    free(list);
    return buf;
}

static void append(ds4_engine *e, ds4_tokens *out, const char *text) {
    ds4_tokens t = {0};
    ds4_tokenize_rendered_chat(e, text, &t);
    for (int i = 0; i < t.len; i++) ds4_tokens_push(out, t.v[i]);
    ds4_tokens_free(&t);
}

static void fail(const char *what, const char *err) {
    fprintf(stderr, "test_qwen4_cut: %s: %s\n", what, err ? err : "");
    exit(1);
}

/* greedy answer of up to GEN_TOKENS tokens, stopping at end of turn */
static void generate(ds4_engine *e, ds4_session *s, ds4_tokens *out, int im_end) {
    char err[256];
    for (int i = 0; i < GEN_TOKENS; i++) {
        const int tok = ds4_session_argmax(s);
        if (tok < 0 || tok == im_end) break;
        ds4_tokens_push(out, tok);
        if (ds4_session_eval(s, tok, err, sizeof(err)) != 0) fail("eval", err);
    }
    (void)e;
}

static void print_text(ds4_engine *e, const char *label, const ds4_tokens *t) {
    printf("%s: ", label);
    for (int i = 0; i < t->len; i++) {
        size_t len = 0;
        char *text = ds4_token_text(e, t->v[i], &len);
        for (size_t c = 0; text && c < len; c++) putchar(text[c] == '\n' ? '/' : text[c]);
        free(text);
    }
    printf("\n");
}

/* feeds ref token by token, comparing each next-token distribution with the
 * one recorded for the reference */
static void score(ds4_session *s, const ds4_tokens *ref, float **ref_logits, int vocab,
                  const char *label) {
    char err[256];
    float *logits = malloc((size_t)vocab * sizeof(float));
    int agree = 0;
    double kl_sum = 0, kl_max = 0;
    for (int i = 0; i < ref->len; i++) {
        ds4_session_copy_logits(s, logits, vocab);
        int best = 0, ref_best = 0;
        double mx = -1e300, rmx = -1e300;
        for (int v = 0; v < vocab; v++) {
            if (logits[v] > mx) { mx = logits[v]; best = v; }
            if (ref_logits[i][v] > rmx) { rmx = ref_logits[i][v]; ref_best = v; }
        }
        double z = 0, rz = 0;
        for (int v = 0; v < vocab; v++) {
            z += exp(logits[v] - mx);
            rz += exp(ref_logits[i][v] - rmx);
        }
        double kl = 0;
        for (int v = 0; v < vocab; v++) {
            const double rp = exp(ref_logits[i][v] - rmx) / rz;
            if (rp > 1e-12) kl += rp * ((ref_logits[i][v] - rmx - log(rz)) - (logits[v] - mx - log(z)));
        }
        agree += best == ref_best;
        kl_sum += kl;
        if (kl > kl_max) kl_max = kl;
        if (ds4_session_eval(s, ref->v[i], err, sizeof(err)) != 0) fail("eval", err);
    }
    printf("%s vs B: top-1 agreement %d/%d, KL mean %.4f max %.4f\n", label, agree, ref->len,
           ref->len ? kl_sum / ref->len : 0.0, kl_max);
    free(logits);
}

int main(int argc, char **argv) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || argc < 2) fail("usage", "DS4_TEST_MODEL=<gguf> test_qwen4_cut <tool result file>");
    char *tool_text = read_file(argv[1], 400000);

    ds4_engine_options options = {
        .model_path = model,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
        .ssd_streaming = true,
#endif
        .context_size = 131072,
        .prefill_chunk = 1024,
        .n_threads = 1,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &options) != 0) fail("engine open", model);
    const int vocab = ds4_engine_vocab_size(e);
    ds4_tokens end_tok = {0};
    ds4_tokenize_rendered_chat(e, "<|im_end|>", &end_tok);
    const int im_end = end_tok.v[0];

    /* a: the task, with facts the questions ask about */
    ds4_tokens a = {0}, x = {0}, b = {0}, q = {0};
    append(e, &a, "<|im_start|>system\nYou are a coding assistant working in the DwarfStar repository.<|im_end|>\n"
                  "<|im_start|>user\nOur release codename is Bluefin and the target date is March 14. "
                  "Read README.md and tell me which models DwarfStar supports.<|im_end|>\n"
                  "<|im_start|>assistant\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"README.md\"}}\n"
                  "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n");
    /* x: the tool result */
    append(e, &x, tool_text);
    /* b: what followed, with facts of its own; with a second file, b holds
     * another tool result whose rows were computed while x was visible */
    append(e, &b, "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
                  "DwarfStar supports DeepSeek V4 Flash, GLM 5.3 Flash and Qwen3.8 Flash Next, each with its own "
                  "Metal graph rather than a generic GGUF runner.<|im_end|>\n");
    if (argc > 2) {
        char *second = read_file(argv[2], 400000);
        const size_t len = strlen(second);
        char *mid = strchr(second + len / 2, '\n');
        const size_t cut_at = mid ? (size_t)(mid - second) + 1 : len;
        static const char needle[] = "Maintainer note: the Qwen3.8 benchmark seed is 4471.\n";
        char *with_needle = malloc(len + sizeof(needle));
        memcpy(with_needle, second, cut_at);
        memcpy(with_needle + cut_at, needle, sizeof(needle) - 1);
        memcpy(with_needle + cut_at + sizeof(needle) - 1, second + cut_at, len - cut_at + 1);
        append(e, &b, "<|im_start|>user\nNow read the Qwen3.8 notes.<|im_end|>\n<|im_start|>assistant\n<tool_call>\n"
                      "{\"name\": \"read\", \"arguments\": {\"path\": \"docs/QWEN38_FLASH_NEXT.md\"}}\n"
                      "</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n");
        append(e, &b, with_needle);
        append(e, &b, "\n</tool_response><|im_end|>\n<|im_start|>assistant\nThe notes cover how Qwen3.8 Flash Next "
                      "runs in DwarfStar.<|im_end|>\n");
        free(with_needle);
        free(second);
    }
    append(e, &b, "<|im_start|>user\nGood. The benchmark machine is an M1 Max with 64 GB, and the reviewer is Dana. "
                  "Draft a one-line changelog entry for adding Qwen3.8 support.<|im_end|>\n"
                  "<|im_start|>assistant\nAdd Qwen3.8 Flash Next support with a resident Metal graph, q8 K/V and "
                  "a shared context pool.<|im_end|>\n");
    append(e, &q, "<|im_start|>user\nAnswer briefly, one line each: 1. What is the release codename and date? "
                  "2. Who is the reviewer and what is the benchmark machine? 3. Which models does DwarfStar "
                  "support? 4. What is the first heading of README.md? 5. What benchmark seed does the "
                  "maintainer note in the Qwen3.8 notes give?<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n");

    /* the cut: the tool result, rounded inward to whole indexer blocks */
    const int start = (a.len + 3) / 4 * 4, end = (a.len + x.len) / 4 * 4;
    ds4_tokens full = {0}, cut = {0}, prefix = {0};
    for (int i = 0; i < a.len; i++) ds4_tokens_push(&full, a.v[i]);
    for (int i = 0; i < x.len; i++) ds4_tokens_push(&full, x.v[i]);
    for (int i = 0; i < b.len; i++) ds4_tokens_push(&full, b.v[i]);
    for (int i = 0; i < full.len; i++) {
        ds4_tokens_push(&prefix, full.v[i]);
        if (i < start || i >= end) ds4_tokens_push(&cut, full.v[i]);
    }
    for (int i = 0; i < q.len; i++) {
        ds4_tokens_push(&full, q.v[i]);
        ds4_tokens_push(&cut, q.v[i]);
    }
    printf("a %d, tool result %d (cut [%d, %d) = %d), b %d, question %d tokens\n",
           a.len, x.len, start, end, end - start, b.len, q.len);

    ds4_session *s = NULL;
    char err[256];
    if (ds4_session_create(&s, e, 131072) != 0) fail("session", NULL);

    /* B first: its answer and distributions are the reference */
    ds4_tokens ans_b = {0};
    if (ds4_session_sync(s, &cut, err, sizeof(err)) != 0) fail("sync B", err);
    generate(e, s, &ans_b, im_end);
    print_text(e, "B recompute without tool result", &ans_b);
    float **ref_logits = malloc((size_t)(ans_b.len ? ans_b.len : 1) * sizeof(float *));
    if (ds4_session_sync(s, &cut, err, sizeof(err)) != 0) fail("sync B again", err);
    for (int i = 0; i < ans_b.len; i++) {
        ref_logits[i] = malloc((size_t)vocab * sizeof(float));
        ds4_session_copy_logits(s, ref_logits[i], vocab);
        if (ds4_session_eval(s, ans_b.v[i], err, sizeof(err)) != 0) fail("eval B", err);
    }

    /* C: prefill with the tool result, cut it, then ask */
    ds4_tokens ans_c = {0};
    if (ds4_session_sync(s, &prefix, err, sizeof(err)) != 0) fail("sync C prefix", err);
    if (ds4_session_qwen4_cut(s, start, end, err, sizeof(err)) != 0) fail("cut", err);
    if (ds4_session_sync(s, &cut, err, sizeof(err)) != 0) fail("sync C", err);
    if (ds4_session_common_prefix(s, &cut) != cut.len) fail("C", "the question was not appended to the cut session");
    generate(e, s, &ans_c, im_end);
    print_text(e, "C cut without recompute", &ans_c);
    if (ds4_session_sync(s, &prefix, err, sizeof(err)) != 0) fail("sync C prefix again", err);
    if (ds4_session_qwen4_cut(s, start, end, err, sizeof(err)) != 0) fail("cut again", err);
    if (ds4_session_sync(s, &cut, err, sizeof(err)) != 0) fail("sync C again", err);
    score(s, &ans_b, ref_logits, vocab, "C");

    /* A: tool result kept, as the scale for how much it matters at all */
    ds4_tokens ans_a = {0};
    if (ds4_session_sync(s, &full, err, sizeof(err)) != 0) fail("sync A", err);
    generate(e, s, &ans_a, im_end);
    print_text(e, "A full context", &ans_a);
    if (ds4_session_sync(s, &full, err, sizeof(err)) != 0) fail("sync A again", err);
    score(s, &ans_b, ref_logits, vocab, "A");

    ds4_session_free(s);
    ds4_engine_close(e);
    return 0;
}
