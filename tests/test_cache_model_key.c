/* The conversation and system-prompt disk caches must never restore KV saved
 * by another model: servers of different models can share one directory.
 * The engine is stubbed; a snapshot is a fixed byte pattern. */
#include "../ds4_convcache.h"
#include "../ds4_syscache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const uint8_t pattern[64] = "kv bytes of the model that saved this conversation, padded....";
static int loads;
/* when set, a snapshot opens like a session payload: header, then these tokens */
static const uint32_t *payload_ids;
static uint32_t payload_n;

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    if (payload_ids) {
        uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = { DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION };
        h[7] = payload_n;
        snap->len = snap->cap = sizeof(h) + payload_n * sizeof(uint32_t) + sizeof(pattern);
        snap->ptr = malloc(snap->len);
        memcpy(snap->ptr, h, sizeof(h));
        memcpy(snap->ptr + sizeof(h), payload_ids, payload_n * sizeof(uint32_t));
        memcpy(snap->ptr + sizeof(h) + payload_n * sizeof(uint32_t), pattern, sizeof(pattern));
    } else {
        snap->ptr = malloc(sizeof(pattern));
        memcpy(snap->ptr, pattern, sizeof(pattern));
        snap->len = snap->cap = sizeof(pattern);
    }
    return 0;
}

int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    uint32_t magic = 0;
    if (snap->len >= sizeof(magic)) memcpy(&magic, snap->ptr, sizeof(magic));
    if (magic != DS4_SESSION_PAYLOAD_MAGIC &&
        (snap->len != sizeof(pattern) || memcmp(snap->ptr, pattern, sizeof(pattern)) != 0)) return 1;
    loads++;
    return 0;
}

/* a payload the way the session writes one: header, (from, from_mtp), tokens */
static ds4_session_snapshot fake_payload(const uint32_t *ids, uint32_t from, uint32_t rows, bool delta) {
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = { DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION };
    h[7] = rows;
    const size_t extra = delta ? 2 * sizeof(uint32_t) : 0;
    ds4_session_snapshot snap = {0};
    snap.len = snap.cap = sizeof(h) + extra + (rows - from) * sizeof(uint32_t) + sizeof(pattern);
    snap.ptr = malloc(snap.len);
    const uint32_t start[2] = { from, 0 };
    memcpy(snap.ptr, h, sizeof(h));
    memcpy(snap.ptr + sizeof(h), start, extra);
    memcpy(snap.ptr + sizeof(h) + extra, ids + from, (rows - from) * sizeof(uint32_t));
    memcpy(snap.ptr + sizeof(h) + extra + (rows - from) * sizeof(uint32_t), pattern, sizeof(pattern));
    return snap;
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

    /* an entry keyed by the visible transcript is lined up by the tokens its
     * snapshot opens with, in RAM and after it went to disk */
    const uint32_t ids[] = { 904, 905, 1, 2, 900, 7, 8, 901, 3 };
    const char *visible = "<|user|>visible key, reasoning left out";
    const size_t visible_common = strlen("<|user|>visible");
    ds4_convcache_candidate cand[4];
    payload_ids = ids;
    payload_n = sizeof(ids) / sizeof(ids[0]);
    expect(ds4_convcache_init(&cc, &opt), "convcache init, payload snapshots");
    expect(ds4_convcache_store(&cc, NULL, (ds4_session *)&cc, visible, strlen(visible),
                               0, 2, 4096, 9, NULL, 0, NULL, 0, NULL, 0) >= 0, "visible-key store");
    int n = ds4_convcache_candidates(&cc, visible, strlen(visible), visible_common, cand, 4);
    expect(n == 1 && cand[0].ram_idx >= 0 && cand[0].tokens_len == (int)payload_n &&
           memcmp(cand[0].tokens, ids, sizeof(ids)) == 0, "RAM candidate carries the snapshot's tokens");
    ds4_convcache_candidates_free(cand, n);
    ds4_convcache_flush_to_disk(&cc);
    ds4_convcache_free(&cc);
    expect(ds4_convcache_init(&cc, &opt), "convcache init, disk only");
    n = ds4_convcache_candidates(&cc, visible, strlen(visible), visible_common, cand, 4);
    expect(n == 1 && cand[0].ram_idx == -1 && cand[0].tokens_len == (int)payload_n &&
           memcmp(cand[0].tokens, ids, sizeof(ids)) == 0, "disk candidate carries the snapshot's tokens");
    ds4_convcache_candidates_free(cand, n);
    ds4_convcache_free(&cc);
    payload_ids = NULL;

    /* an appendable entry: a base, then a record per turn.  Candidates see the
     * key and tokens of the newest record, a load applies base and records in
     * order, and a record cut short by a crash is left out */
    const uint32_t turn[] = { 904, 905, 1, 2, 900, 7, 901, 3, 904, 905, 4, 5, 6 };
    const char *key = "<|user|>append base", *next_key = "<|user|>append base and a second turn";
    char *path = NULL;
    expect(ds4_convcache_init(&cc, &opt), "convcache init, appendable");
    path = ds4_convcache_entry_path(&cc, key, strlen(key));
    ds4_session_snapshot raw = fake_payload(turn, 0, 8, false);
    expect(path && ds4_convcache_write(&cc, path, true, key, strlen(key), 8, NULL, 0, NULL, 0, &raw, err, sizeof(err)),
           "base written");
    ds4_session_snapshot_free(&raw);
    raw = fake_payload(turn, 8, 13, true);
    const uint8_t tool_memory[] = "tool memory of the second turn";
    expect(ds4_convcache_write(&cc, path, false, next_key, strlen(next_key), 13, NULL, 0, tool_memory,
                               sizeof(tool_memory), &raw, err, sizeof(err)), "record appended");
    ds4_session_snapshot_free(&raw);
    n = ds4_convcache_candidates(&cc, key, strlen(key), strlen(key), cand, 4);
    expect(n == 1 && cand[0].text_len == strlen(next_key) && cand[0].token_count == 13 &&
           cand[0].tokens_len == 13 && memcmp(cand[0].tokens, turn, sizeof(turn)) == 0,
           "candidate carries the record's key and tokens");
    const int loads_before = loads;
    uint8_t *aux = NULL;
    uint32_t aux_len = 0;
    expect(n == 1 && ds4_convcache_load_candidate(&cc, &cand[0], (ds4_session *)&cc, NULL, NULL, &aux, &aux_len,
                                                  err, sizeof(err)) &&
           loads == loads_before + 2 && aux_len == sizeof(tool_memory) && cc.loaded_path &&
           strcmp(cc.loaded_path, path) == 0, "load applies base and record and keeps the newest tool memory");
    free(aux);
    ds4_convcache_candidates_free(cand, n);
    ds4_convcache_free(&cc);

    expect(ds4_convcache_init(&cc, &opt), "convcache init, cut record");
    raw = fake_payload(turn, 0, 8, false);
    expect(ds4_convcache_write(&cc, path, true, key, strlen(key), 8, NULL, 0, NULL, 0, &raw, err, sizeof(err)),
           "base rewritten");
    ds4_session_snapshot_free(&raw);
    raw = fake_payload(turn, 8, 13, true);
    expect(ds4_convcache_write(&cc, path, false, next_key, strlen(next_key), 13, NULL, 0, NULL, 0, &raw, err,
                               sizeof(err)), "record appended again");
    ds4_session_snapshot_free(&raw);
    struct stat st;
    expect(stat(path, &st) == 0 && truncate(path, st.st_size - 5) == 0, "record cut short");
    n = ds4_convcache_candidates(&cc, key, strlen(key), strlen(key), cand, 4);
    expect(n == 1 && cand[0].text_len == strlen(key) && cand[0].tokens_len == 8,
           "a record cut short is left out, the base stays");
    ds4_convcache_candidates_free(cand, n);
    ds4_convcache_free(&cc);
    free(path);

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
