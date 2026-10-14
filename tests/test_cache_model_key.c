/* The conversation and system-prompt disk caches must never restore KV saved
 * by another model: servers of different models can share one directory.
 * The engine is stubbed; a snapshot is a fixed byte pattern. */
#include "../ds4_convcache.h"
#include "../ds4_syscache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const uint8_t pattern[64] = "kv bytes of the model that saved this conversation, padded....";
static int loads;

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    snap->ptr = malloc(sizeof(pattern));
    memcpy(snap->ptr, pattern, sizeof(pattern));
    snap->len = snap->cap = sizeof(pattern);
    return 0;
}

int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    if (snap->len != sizeof(pattern) || memcmp(snap->ptr, pattern, sizeof(pattern)) != 0) return 1;
    loads++;
    return 0;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

uint32_t ds4_gpu_stream_expert_cache_enum_residents(ds4_stream_expert_ref *out, uint32_t max_out) {
    (void)out; (void)max_out;
    return 0;
}

static int failures;

static void expect(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static int conv_disk_load(char *dir, uint64_t key, const char *text) {
    ds4_convcache cc;
    const ds4_convcache_opt opt = { .ram_budget_bytes = 1 << 20, .disk_budget_bytes = 1 << 20,
                                    .disk_dir = dir, .model_key = key };
    int token_count = 0;
    size_t text_len = 0;
    const int before = loads;
    expect(ds4_convcache_init(&cc, &opt), "convcache init");
    const bool ok = ds4_convcache_load_from_disk(&cc, text, strlen(text), (ds4_session *)&cc,
                                                 &token_count, &text_len, NULL, NULL, NULL, NULL,
                                                 NULL, 0);
    ds4_convcache_free(&cc);
    return ok && loads == before + 1;
}

int main(void) {
    char dir[] = "/tmp/ds4-cache-model-key-XXXXXX";
    if (!mkdtemp(dir)) {
        perror("mkdtemp");
        return 2;
    }
    const uint64_t model_a = 0x1111222233334444ULL, model_b = 0x5555666677778888ULL;
    const char *conversation = "<|user|>hello<|assistant|>hi there";

    ds4_convcache cc;
    const ds4_convcache_opt opt = { .ram_budget_bytes = 1 << 20, .disk_budget_bytes = 1 << 20,
                                    .disk_dir = dir, .model_key = model_a };
    expect(ds4_convcache_init(&cc, &opt), "convcache init");
    expect(ds4_convcache_store(&cc, NULL, (ds4_session *)&cc, conversation, strlen(conversation),
                               0, 2, 4096, 7, NULL, 0, NULL, 0, NULL, 0) >= 0, "convcache store");
    ds4_convcache_flush_to_disk(&cc);
    ds4_convcache_free(&cc);
    expect(conv_disk_load(dir, model_a, conversation), "conversation loads for the model that saved it");
    expect(!conv_disk_load(dir, model_b, conversation), "conversation must not load for another model");

    /* a snapshot larger than the whole RAM budget goes to disk alone and is
     * reported as stored; without a disk directory it is lost, and says so */
    const char *long_conversation = "<|user|>a 265k-token session<|assistant|>does not fit RAM";
    char err[160] = {0};
    const ds4_convcache_opt tiny = { .ram_budget_bytes = 16, .disk_budget_bytes = 1 << 20,
                                     .disk_dir = dir, .model_key = model_a };
    expect(ds4_convcache_init(&cc, &tiny), "convcache init, tiny RAM");
    expect(ds4_convcache_store(&cc, NULL, (ds4_session *)&cc, long_conversation, strlen(long_conversation),
                               0, 2, 4096, 7, NULL, 0, NULL, 0, err, sizeof(err)) == DS4_CONVCACHE_DISK_ONLY,
           "oversized snapshot is stored to disk only");
    ds4_convcache_free(&cc);
    expect(conv_disk_load(dir, model_a, long_conversation), "disk-only snapshot loads back");
    const ds4_convcache_opt no_disk = { .ram_budget_bytes = 16, .model_key = model_a };
    expect(ds4_convcache_init(&cc, &no_disk), "convcache init, no disk");
    expect(ds4_convcache_store(&cc, NULL, (ds4_session *)&cc, long_conversation, strlen(long_conversation),
                               0, 2, 4096, 7, NULL, 0, NULL, 0, err, sizeof(err)) == -1 &&
           strstr(err, "no disk copy") != NULL, "oversized snapshot without disk fails with its reason");
    ds4_convcache_free(&cc);

    const char *prompt = "You are a helpful assistant.";
    ds4_syscache sa, sb;
    const ds4_syscache_opt oa = { .disk_budget_bytes = 1 << 20, .disk_dir = dir, .model_key = model_a };
    const ds4_syscache_opt ob = { .disk_budget_bytes = 1 << 20, .disk_dir = dir, .model_key = model_b };
    ds4_syscache_init(&sa, &oa);
    ds4_syscache_init(&sb, &ob);
    expect(ds4_syscache_store(&sa, prompt, strlen(prompt), 5, (ds4_session *)&sa, NULL, 0),
           "syscache store");
    int tokens = 0;
    expect(ds4_syscache_has(&sa, prompt, strlen(prompt)), "system prompt present for its model");
    expect(!ds4_syscache_has(&sb, prompt, strlen(prompt)), "system prompt absent for another model");
    expect(!ds4_syscache_load(&sb, prompt, strlen(prompt), (ds4_session *)&sb, &tokens, NULL, 0),
           "system prompt must not load for another model");
    expect(ds4_syscache_load(&sa, prompt, strlen(prompt), (ds4_session *)&sa, &tokens, NULL, 0) &&
           tokens == 5, "system prompt loads for its model");

    char cmd[128];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", dir);
    if (system(cmd) != 0) fprintf(stderr, "could not remove %s\n", dir);
    if (failures == 0) printf("cache model key: ok\n");
    return failures == 0 ? 0 : 1;
}
