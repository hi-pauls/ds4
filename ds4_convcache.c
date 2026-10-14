/*
 * ds4_convcache.c — see ds4_convcache.h for design notes.
 */
#include "ds4_convcache.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>    /* unlink() */
#include <zstd.h>

/* Disk file layout for a .conv entry:
 *   [4-byte magic] [8-byte model_key] [4-byte text_len] [4-byte token_count]
 *   [8-byte uncompressed_snap_len]
 *   [4-byte n_residents]
 *   [n_residents * sizeof(ds4_stream_expert_ref) bytes: residents array]
 *   [4-byte aux_blob_len] [aux_blob_len bytes]
 *   [text_len bytes of rendered text]
 *   [ZSTD-compressed snapshot bytes to end of file]
 * The text follows the (variable-size) residents block; scan_disk_best_match
 * seeks past both header and residents to reach the text for prefix matching.
 * Servers of different models may share a directory: a file whose model_key
 * differs is never loaded, and CONY files, which carry no key, are rejected. */
#define DS4_CONV_FILE_MAGIC 0x434f4e5au  /* "CONZ" — + model_key */
#define DS4_CONV_TOKEN_COUNT_OFFSET 16   /* magic 4 + model_key 8 + text_len 4 */
/* Compression level: 1 is the fastest zstd preset. */
/* Compression level: 1 is the fastest zstd preset. Empirical on ~140 MiB
 * snapshots: ~0.9s encode / ~0.4s decode / ~40% ratio — cheap enough to
 * run inline on every store/load. */
#define DS4_CONV_ZSTD_LEVEL 1

/* -------- helpers ---------------------------------------------------- */

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Compress a snapshot in place: replace ptr/len/cap with a freshly-malloc'd
 * compressed buffer and return the original uncompressed length via
 * `*uncompressed_len_out`. The old buffer is freed. Returns true on success;
 * on failure the input snapshot is left untouched. */
static bool compress_snapshot_inplace(ds4_session_snapshot *snap,
                                      uint64_t *uncompressed_len_out,
                                      char *err, size_t errlen) {
    if (!snap || !snap->ptr || snap->len == 0) return false;
    size_t bound = ZSTD_compressBound((size_t)snap->len);
    uint8_t *cbuf = (uint8_t *)malloc(bound);
    if (!cbuf) {
        if (err && errlen) snprintf(err, errlen, "OOM compressing snapshot");
        return false;
    }
    size_t clen = ZSTD_compress(cbuf, bound, snap->ptr, (size_t)snap->len,
                                DS4_CONV_ZSTD_LEVEL);
    if (ZSTD_isError(clen)) {
        if (err && errlen) {
            snprintf(err, errlen, "zstd_compress: %s", ZSTD_getErrorName(clen));
        }
        free(cbuf);
        return false;
    }
    if (uncompressed_len_out) *uncompressed_len_out = snap->len;
    free(snap->ptr);
    snap->ptr = cbuf;
    snap->len = (uint64_t)clen;
    snap->cap = (uint64_t)clen;
    return true;
}

/* Decompress a compressed snapshot into a fresh ds4_session_snapshot suitable
 * for ds4_session_load_snapshot(). Caller must ds4_session_snapshot_free()
 * the result. Returns true on success. */
static bool decompress_snapshot(const ds4_session_snapshot *compressed,
                                uint64_t uncompressed_len,
                                ds4_session_snapshot *out,
                                char *err, size_t errlen) {
    if (!compressed || !compressed->ptr || compressed->len == 0 ||
        uncompressed_len == 0 || !out) return false;
    uint8_t *ubuf = (uint8_t *)malloc((size_t)uncompressed_len);
    if (!ubuf) {
        if (err && errlen) snprintf(err, errlen, "OOM decompressing snapshot");
        return false;
    }
    size_t ulen = ZSTD_decompress(ubuf, (size_t)uncompressed_len,
                                  compressed->ptr, (size_t)compressed->len);
    if (ZSTD_isError(ulen)) {
        if (err && errlen) {
            snprintf(err, errlen, "zstd_decompress: %s",
                     ZSTD_getErrorName(ulen));
        }
        free(ubuf);
        return false;
    }
    if (ulen != uncompressed_len) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "decompressed size mismatch: got %zu expected %llu",
                     ulen, (unsigned long long)uncompressed_len);
        }
        free(ubuf);
        return false;
    }
    out->ptr = ubuf;
    out->len = uncompressed_len;
    out->cap = uncompressed_len;
    return true;
}

/* Compose the on-disk path for a given SHA. Caller frees. */
static char *disk_path_for(const ds4_convcache *cc, const char *sha) {
    if (!cc || !cc->opt.disk_dir || !sha) return NULL;
    size_t n = strlen(cc->opt.disk_dir) + 1 + 40 + 5 + 1;
    char *p = (char *)malloc(n);
    if (!p) return NULL;
    snprintf(p, n, "%s/%s.conv", cc->opt.disk_dir, sha);
    return p;
}

/* Write a magic-tagged entry file atomically. Layout:
 *   [magic(u32)] [text_len(u32)] [token_count(u32)] [uncompressed_len(u64)]
 *   [n_residents(u32)] [n_residents * sizeof(ds4_stream_expert_ref) bytes]
 *   [aux_blob_len(u32)] [aux_blob_len bytes]
 *   [text bytes] [ZSTD-compressed snapshot bytes to EOF]
 * The snap_ptr/snap_len are ALREADY compressed by the caller.
 * Returns whether the file landed; disk backing is best-effort, so most
 * callers ignore it. */
static bool write_entry_atomic(const char *path,
                               uint64_t model_key,
                               const char *text, size_t text_len,
                               int token_count,
                               uint64_t uncompressed_len,
                               const ds4_stream_expert_ref *residents,
                               uint32_t residents_len,
                               const uint8_t *aux_blob,
                               uint32_t aux_blob_len,
                               const uint8_t *snap_ptr, uint64_t snap_len) {
    if (!path || !text || text_len == 0 || !snap_ptr || snap_len == 0 ||
        uncompressed_len == 0) return false;
    size_t plen = strlen(path);
    char *tmp = (char *)malloc(plen + 5);
    if (!tmp) return false;
    snprintf(tmp, plen + 5, "%s.tmp", path);
    bool landed = false;
    FILE *fp = fopen(tmp, "wb");
    if (fp) {
        uint32_t magic = DS4_CONV_FILE_MAGIC;
        uint32_t tl = (uint32_t)text_len;
        uint32_t tc = (uint32_t)(token_count < 0 ? 0 : token_count);
        uint64_t ul = uncompressed_len;
        uint32_t nr = residents ? residents_len : 0u;
        size_t resb = (size_t)nr * sizeof(ds4_stream_expert_ref);
        uint32_t ab = aux_blob ? aux_blob_len : 0u;
        bool ok = fwrite(&magic, sizeof(magic), 1, fp) == 1 &&
                  fwrite(&model_key, sizeof(model_key), 1, fp) == 1 &&
                  fwrite(&tl,    sizeof(tl),    1, fp) == 1 &&
                  fwrite(&tc,    sizeof(tc),    1, fp) == 1 &&
                  fwrite(&ul,    sizeof(ul),    1, fp) == 1 &&
                  fwrite(&nr,    sizeof(nr),    1, fp) == 1 &&
                  (resb == 0 || fwrite(residents, 1, resb, fp) == resb) &&
                  fwrite(&ab,    sizeof(ab),    1, fp) == 1 &&
                  (ab   == 0 || fwrite(aux_blob, 1, ab, fp) == ab) &&
                  fwrite(text, 1, text_len, fp) == text_len &&
                  fwrite(snap_ptr, 1, (size_t)snap_len, fp) == (size_t)snap_len;
        ok = fclose(fp) == 0 && ok;
        landed = ok && rename(tmp, path) == 0;
        if (!landed) unlink(tmp);
    }
    free(tmp);
    return landed;
}

/* Delete the least recently written .conv files until the directory fits the
 * disk budget again. keep_path, when given, is the file just written and is
 * never deleted, so an entry larger than the whole budget still survives
 * until the next write. mtime is the recency key: files are only ever
 * rewritten whole, when their entry is evicted from RAM or flushed. */
static void trim_disk_to_budget(const ds4_convcache *cc, const char *keep_path) {
    if (!cc || cc->opt.disk_budget_bytes == 0 || !cc->opt.disk_dir) return;
    for (;;) {
        DIR *dir = opendir(cc->opt.disk_dir);
        if (!dir) return;
        uint64_t total = 0;
        char *oldest = NULL;
        time_t oldest_mtime = 0;
        struct dirent *de;
        while ((de = readdir(dir)) != NULL) {
            const size_t n = strlen(de->d_name);
            if (n < 6 || strcmp(de->d_name + n - 5, ".conv") != 0) continue;
            const size_t plen = strlen(cc->opt.disk_dir) + 1 + n + 1;
            char *path = (char *)malloc(plen);
            if (!path) continue;
            snprintf(path, plen, "%s/%s", cc->opt.disk_dir, de->d_name);
            struct stat st;
            if (stat(path, &st) != 0) { free(path); continue; }
            total += (uint64_t)st.st_size;
            if ((keep_path && strcmp(path, keep_path) == 0) ||
                (oldest && st.st_mtime >= oldest_mtime)) {
                free(path);
                continue;
            }
            free(oldest);
            oldest = path;
            oldest_mtime = st.st_mtime;
        }
        closedir(dir);
        const bool over = total > cc->opt.disk_budget_bytes && oldest != NULL;
        const bool removed = over && unlink(oldest) == 0;
        free(oldest);
        if (!removed) return;
    }
}

/* Read just the text-header portion of a .conv file. Returns malloc'd text
 * (caller frees) on success, NULL on any error. Sets *text_len_out to the
 * length. Leaves the FILE* positioned just past the text (i.e. at the start
 * of the compressed snapshot bytes) if `keep_open` is non-NULL. */
static char *read_entry_text(const char *path,
                             uint64_t model_key,
                             size_t *text_len_out,
                             FILE **keep_open) {
    if (keep_open) *keep_open = NULL;
    if (text_len_out) *text_len_out = 0;
    if (!path) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    uint32_t magic = 0, text_len = 0, token_count = 0;
    uint64_t file_model_key = 0;
    uint64_t uncompressed_len = 0;
    uint32_t n_residents = 0;
    if (fread(&magic,           sizeof(magic),           1, fp) != 1 ||
        fread(&file_model_key,  sizeof(file_model_key),  1, fp) != 1 ||
        fread(&text_len,        sizeof(text_len),        1, fp) != 1 ||
        fread(&token_count,     sizeof(token_count),     1, fp) != 1 ||
        fread(&uncompressed_len, sizeof(uncompressed_len), 1, fp) != 1 ||
        fread(&n_residents,     sizeof(n_residents),     1, fp) != 1 ||
        magic != DS4_CONV_FILE_MAGIC || file_model_key != model_key ||
        text_len == 0 || text_len > (uint32_t)(64 * 1024 * 1024))
    {
        fclose(fp);
        return NULL;
    }
    /* Skip past the residents block; caller can re-read it explicitly if it
     * needs the resident set (load_from_disk does). */
    off_t resb = (off_t)n_residents * (off_t)sizeof(ds4_stream_expert_ref);
    if (fseeko(fp, resb, SEEK_CUR) != 0) { fclose(fp); return NULL; }
    /* Skip past the aux_blob length + bytes too. */
    uint32_t aux_blob_len = 0;
    if (fread(&aux_blob_len, sizeof(aux_blob_len), 1, fp) != 1) {
        fclose(fp); return NULL;
    }
    if (fseeko(fp, (off_t)aux_blob_len, SEEK_CUR) != 0) {
        fclose(fp); return NULL;
    }
    char *text = (char *)malloc((size_t)text_len + 1);
    if (!text) { fclose(fp); return NULL; }
    if (fread(text, 1, text_len, fp) != text_len) {
        free(text); fclose(fp); return NULL;
    }
    text[text_len] = '\0';
    if (text_len_out) *text_len_out = text_len;
    if (keep_open) *keep_open = fp;
    else fclose(fp);
    return text;
}

/* Grow the entry array to at least `min_cap`. */
static bool ensure_cap(ds4_convcache *cc, int min_cap) {
    if (cc->cap >= min_cap) return true;
    int new_cap = cc->cap ? cc->cap * 2 : 4;
    while (new_cap < min_cap) new_cap *= 2;
    ds4_convcache_entry *ne =
        (ds4_convcache_entry *)realloc(cc->entries,
                                       (size_t)new_cap * sizeof(*ne));
    if (!ne) return false;
    cc->entries = ne;
    cc->cap = new_cap;
    return true;
}

/* Remove entry at index i and its snapshot from RAM. If `write_to_disk` and
 * disk backing is configured, first serialise it. */
static void evict_entry(ds4_convcache *cc, int i, bool write_to_disk) {
    ds4_convcache_entry *e = &cc->entries[i];
    if (write_to_disk && cc->opt.disk_budget_bytes > 0 && cc->opt.disk_dir) {
        char *path = disk_path_for(cc, e->sha);
        if (path) {
            write_entry_atomic(path, cc->opt.model_key,
                               e->text, e->text_len, e->token_count,
                               e->uncompressed_len,
                               e->residents, e->residents_len,
                               e->aux_blob, e->aux_blob_len,
                               e->snap.ptr, e->snap.len);
            trim_disk_to_budget(cc, path);
            free(path);
        }
    }
    if (e->snap.len <= cc->ram_used_bytes) {
        cc->ram_used_bytes -= e->snap.len;
    } else {
        cc->ram_used_bytes = 0;
    }
    ds4_session_snapshot_free(&e->snap);
    free(e->text);
    e->text = NULL;
    free(e->residents);
    e->residents = NULL;
    e->residents_len = 0;
    free(e->aux_blob);
    e->aux_blob = NULL;
    e->aux_blob_len = 0;
    if (i < cc->len - 1) {
        memmove(&cc->entries[i], &cc->entries[i + 1],
                (size_t)(cc->len - i - 1) * sizeof(cc->entries[0]));
    }
    cc->len--;
}

/* True if `entry_text` (up to entry_len bytes) is a byte-prefix of `text`
 * (up to text_len bytes). */
static bool entry_text_is_prefix_of(const char *entry_text, size_t entry_len,
                                    const char *text,       size_t text_len) {
    if (!entry_text || !text) return false;
    if (entry_len > text_len) return false;
    return memcmp(entry_text, text, entry_len) == 0;
}

/* LRU pick excluding entries whose text is a byte-prefix of the pinned
 * text prefix (i.e., we protect any entry we might want to load next).
 * Returns -1 if nothing evictable. */
static int find_lru_victim(const ds4_convcache *cc,
                           const char *pinned_prefix,
                           size_t      pinned_prefix_len) {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < cc->len; i++) {
        const ds4_convcache_entry *e = &cc->entries[i];
        if (pinned_prefix && pinned_prefix_len > 0 &&
            entry_text_is_prefix_of(e->text, e->text_len,
                                    pinned_prefix, pinned_prefix_len))
        {
            continue;  /* pinned — this entry could be loaded next */
        }
        if (e->last_hit_ns < oldest) {
            oldest = e->last_hit_ns;
            victim = i;
        }
    }
    return victim;
}

/* -------- public API ------------------------------------------------- */

bool ds4_convcache_init(ds4_convcache *cc, const ds4_convcache_opt *opt) {
    if (!cc || !opt) return false;
    memset(cc, 0, sizeof(*cc));
    if (opt->ram_budget_bytes == 0) return false;
    cc->opt.ram_budget_bytes = opt->ram_budget_bytes;
    cc->opt.disk_budget_bytes = opt->disk_budget_bytes;
    cc->opt.model_key = opt->model_key;
    if (opt->disk_dir) {
        cc->opt.disk_dir = strdup(opt->disk_dir);
        if (!cc->opt.disk_dir) return false;
        /* Best-effort create the dir; if it exists we don't care. */
        (void)mkdir(cc->opt.disk_dir, 0700);
    }
    return true;
}

void ds4_convcache_free(ds4_convcache *cc) {
    if (!cc) return;
    for (int i = 0; i < cc->len; i++) {
        ds4_session_snapshot_free(&cc->entries[i].snap);
        free(cc->entries[i].text);
        free(cc->entries[i].residents);
        free(cc->entries[i].aux_blob);
    }
    free(cc->entries);
    free(cc->opt.disk_dir);
    memset(cc, 0, sizeof(*cc));
}

int ds4_convcache_find_prefix(ds4_convcache *cc,
                              const char *text_prefix,
                              size_t text_prefix_len) {
    if (!cc || !text_prefix || text_prefix_len == 0) return -1;
    int best = -1;
    size_t best_common = 0;
    for (int i = 0; i < cc->len; i++) {
        const ds4_convcache_entry *e = &cc->entries[i];
        if (!e->text) continue;
        /* Two useful cases:
         *   (a) entry.text is a byte-prefix of incoming — entry's KV is a
         *       start-of-conversation for a longer request (e.g. real chat
         *       continuation with a new user turn appended).
         *   (b) incoming is a byte-prefix of entry.text — the entry has extra
         *       tokens (generated response) beyond what the incoming request
         *       asks for; still usable by loading the snap and rewinding the
         *       session to the incoming token length.
         * The common prefix length (min of the two texts, when they match up
         * to that point) is what determines usefulness — prefer the entry
         * with the longest common prefix. */
        size_t min_len = e->text_len < text_prefix_len ? e->text_len
                                                       : text_prefix_len;
        if (min_len == 0) continue;
        if (memcmp(e->text, text_prefix, min_len) != 0) continue;
        if (min_len > best_common) {
            best = i;
            best_common = min_len;
        }
    }
    if (best >= 0) cc->entries[best].last_hit_ns = now_ns();
    return best;
}

int ds4_convcache_store(ds4_convcache *cc,
                        ds4_engine *engine,
                        ds4_session *session,
                        const char *text, size_t text_len,
                        int model_id, int quant_bits, int ctx_size,
                        int token_count,
                        const char *pinned_text_prefix,
                        size_t pinned_text_prefix_len,
                        const uint8_t *aux_blob,
                        uint32_t aux_blob_len,
                        char *err, size_t errlen) {
    (void)engine;
    if (!cc || !session || !text || text_len == 0) return -1;
    if (cc->opt.ram_budget_bytes == 0) return -1;

    /* Entry SHA — used only for the disk-backed filename. We store a short
     * printable-hex digest based on a djb2 hash; a cryptographic SHA isn't
     * required for cache identity because the RAM lookup path uses direct
     * byte-prefix comparison on the stored text. Seeding it with the model
     * key keeps two models' copies of one conversation in separate files. */
    char sha[41] = {0};
    uint64_t h = 5381 ^ cc->opt.model_key;
    for (size_t i = 0; i < text_len; i++) h = ((h << 5) + h) ^ (uint8_t)text[i];
    snprintf(sha, sizeof(sha), "%016llx%016llx%08x",
             (unsigned long long)h,
             (unsigned long long)(h ^ 0xdeadbeefcafebabeULL),
             (unsigned)(text_len & 0xffffffff));

    /* Snapshot first — we need its size to know how much room to make. */
    ds4_session_snapshot snap = {0};
    char save_err[160] = {0};
    if (ds4_session_save_snapshot(session, &snap, save_err, sizeof(save_err)) != 0) {
        if (err && errlen) snprintf(err, errlen, "snapshot: %s", save_err);
        return -1;
    }

    /* Compress before checking the RAM budget — compressed size is what
     * actually occupies RAM, and it's typically ~40% of the raw snapshot. */
    uint64_t uncompressed_len = 0;
    char comp_err[160] = {0};
    if (!compress_snapshot_inplace(&snap, &uncompressed_len,
                                   comp_err, sizeof(comp_err)))
    {
        if (err && errlen) snprintf(err, errlen, "compress: %s", comp_err);
        ds4_session_snapshot_free(&snap);
        return -1;
    }

    /* Snapshot the current streaming-expert-cache residency. Small (~40 KB
     * for 5000 experts) so this always fits alongside the KV state; both
     * are restored together on load so the routing hits after a swap don't
     * pay SSD-read latency for experts the outgoing session had warm. */
    ds4_stream_expert_ref *residents = NULL;
    uint32_t residents_len = 0;
    uint32_t rtotal = ds4_gpu_stream_expert_cache_enum_residents(NULL, 0);
    if (rtotal > 0) {
        residents = (ds4_stream_expert_ref *)
            malloc((size_t)rtotal * sizeof(*residents));
        if (residents) {
            residents_len =
                ds4_gpu_stream_expert_cache_enum_residents(residents, rtotal);
            if (residents_len > rtotal) residents_len = rtotal;
        }
    }

    /* Doesn't fit RAM at all → write to disk if backing exists, then bail. */
    if (snap.len > cc->opt.ram_budget_bytes) {
        bool landed = false;
        if (cc->opt.disk_budget_bytes > 0 && cc->opt.disk_dir) {
            char *path = disk_path_for(cc, sha);
            if (path) {
                landed = write_entry_atomic(path, cc->opt.model_key, text, text_len, token_count,
                                            uncompressed_len,
                                            residents, residents_len,
                                            aux_blob, aux_blob_len,
                                            snap.ptr, snap.len);
                trim_disk_to_budget(cc, path);
                free(path);
            }
        }
        if (!landed && err && errlen) {
            snprintf(err, errlen,
                     "compressed snapshot %llu bytes exceeds ram budget %llu bytes, no disk copy",
                     (unsigned long long)snap.len,
                     (unsigned long long)cc->opt.ram_budget_bytes);
        }
        free(residents);
        ds4_session_snapshot_free(&snap);
        return landed ? DS4_CONVCACHE_DISK_ONLY : -1;
    }

    /* Evict until it fits, protecting entries that are prefixes of the
     * incoming (would-be-loaded) text. */
    while (cc->ram_used_bytes + snap.len > cc->opt.ram_budget_bytes) {
        int v = find_lru_victim(cc, pinned_text_prefix, pinned_text_prefix_len);
        if (v < 0) {
            free(residents);
            ds4_session_snapshot_free(&snap);
            if (err && errlen) {
                snprintf(err, errlen,
                         "cannot fit snapshot; only pinned entries remain");
            }
            return -1;
        }
        evict_entry(cc, v, true);
    }

    if (!ensure_cap(cc, cc->len + 1)) {
        free(residents);
        ds4_session_snapshot_free(&snap);
        return -1;
    }

    /* Copy the text bytes — they're owned by the entry from here on. */
    char *text_copy = (char *)malloc(text_len + 1);
    if (!text_copy) {
        free(residents);
        ds4_session_snapshot_free(&snap);
        if (err && errlen) snprintf(err, errlen, "OOM copying text");
        return -1;
    }
    memcpy(text_copy, text, text_len);
    text_copy[text_len] = '\0';

    ds4_convcache_entry *e = &cc->entries[cc->len++];
    memset(e, 0, sizeof(*e));
    snprintf(e->sha, sizeof(e->sha), "%s", sha);
    e->model_id    = model_id;
    e->quant_bits  = quant_bits;
    e->ctx_size    = ctx_size;
    e->token_count = token_count;
    e->text        = text_copy;
    e->text_len    = text_len;
    e->last_hit_ns       = now_ns();
    e->snap              = snap;               /* transfer ownership */
    e->uncompressed_len  = uncompressed_len;
    e->residents         = residents;          /* transfer ownership */
    e->residents_len     = residents_len;
    if (aux_blob && aux_blob_len > 0) {
        e->aux_blob = (uint8_t *)malloc(aux_blob_len);
        if (e->aux_blob) {
            memcpy(e->aux_blob, aux_blob, aux_blob_len);
            e->aux_blob_len = aux_blob_len;
        }
    }
    cc->ram_used_bytes  += snap.len;
    return cc->len - 1;
}

bool ds4_convcache_load(ds4_convcache *cc, int idx,
                        ds4_session *session,
                        ds4_stream_expert_ref **out_residents,
                        uint32_t *out_residents_len,
                        uint8_t **out_aux_blob,
                        uint32_t *out_aux_blob_len,
                        char *err, size_t errlen) {
    if (out_residents) *out_residents = NULL;
    if (out_residents_len) *out_residents_len = 0;
    if (out_aux_blob) *out_aux_blob = NULL;
    if (out_aux_blob_len) *out_aux_blob_len = 0;
    if (!cc || !session) return false;
    if (idx < 0 || idx >= cc->len) return false;

    ds4_convcache_entry *e = &cc->entries[idx];

    /* Decompress into a temporary snapshot for ds4_session_load_snapshot();
     * the entry itself holds compressed bytes. */
    ds4_session_snapshot tmp = {0};
    char dec_err[160] = {0};
    if (!decompress_snapshot(&e->snap, e->uncompressed_len,
                             &tmp, dec_err, sizeof(dec_err)))
    {
        if (err && errlen) snprintf(err, errlen, "decompress: %s", dec_err);
        return false;
    }
    char load_err[160] = {0};
    int rc = ds4_session_load_snapshot(session, &tmp,
                                       load_err, sizeof(load_err));
    ds4_session_snapshot_free(&tmp);
    if (rc != 0) {
        if (err && errlen) snprintf(err, errlen, "load: %s", load_err);
        return false;
    }

    /* Transfer residents / aux_blob ownership if caller asked for them;
     * teardown below won't try to free again because we NULL them here. */
    if (out_residents && out_residents_len && e->residents_len > 0) {
        *out_residents     = e->residents;
        *out_residents_len = e->residents_len;
        e->residents       = NULL;
        e->residents_len   = 0;
    }
    if (out_aux_blob && out_aux_blob_len && e->aux_blob_len > 0) {
        *out_aux_blob     = e->aux_blob;
        *out_aux_blob_len = e->aux_blob_len;
        e->aux_blob       = NULL;
        e->aux_blob_len   = 0;
    }

    if (e->snap.len <= cc->ram_used_bytes) {
        cc->ram_used_bytes -= e->snap.len;
    } else {
        cc->ram_used_bytes = 0;
    }
    ds4_session_snapshot_free(&e->snap);
    free(e->text);
    e->text = NULL;
    free(e->residents);
    e->residents = NULL;
    e->residents_len = 0;
    free(e->aux_blob);
    e->aux_blob = NULL;
    e->aux_blob_len = 0;
    if (idx < cc->len - 1) {
        memmove(&cc->entries[idx], &cc->entries[idx + 1],
                (size_t)(cc->len - idx - 1) * sizeof(cc->entries[0]));
    }
    cc->len--;
    return true;
}

/* Scan the disk_dir for any .conv file whose stored text has a common
 * byte-prefix with the incoming text. Returns the malloc'd path of the
 * best-matching file (longest common prefix wins) and via out params the
 * common prefix length + the token_count from the file header. Caller
 * frees the returned path. Returns NULL if no candidate matches. */
static char *scan_disk_best_match(const ds4_convcache *cc,
                                  const char *incoming_text,
                                  size_t incoming_text_len,
                                  size_t *best_common_out,
                                  int    *best_token_count_out) {
    if (!cc || !cc->opt.disk_dir || !incoming_text || incoming_text_len == 0) {
        return NULL;
    }
    DIR *d = opendir(cc->opt.disk_dir);
    if (!d) return NULL;
    char *best_path = NULL;
    size_t best_common = 0;
    int best_token_count = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *name = ent->d_name;
        size_t nlen = strlen(name);
        if (nlen < 6 || strcmp(name + nlen - 5, ".conv") != 0) continue;
        size_t path_len = strlen(cc->opt.disk_dir) + 1 + nlen + 1;
        char *path = (char *)malloc(path_len);
        if (!path) continue;
        snprintf(path, path_len, "%s/%s", cc->opt.disk_dir, name);
        FILE *fp = NULL;
        size_t text_len = 0;
        char *text = read_entry_text(path, cc->opt.model_key, &text_len, &fp);
        if (!text) { free(path); continue; }
        int token_count = 0;
        long saved = ftell(fp);
        if (fseeko(fp, DS4_CONV_TOKEN_COUNT_OFFSET, SEEK_SET) == 0) {
            uint32_t tc = 0;
            if (fread(&tc, sizeof(tc), 1, fp) == 1) token_count = (int)tc;
        }
        (void)fseek(fp, saved, SEEK_SET);
        fclose(fp);
        size_t min_len = text_len < incoming_text_len ? text_len
                                                      : incoming_text_len;
        if (min_len > best_common &&
            memcmp(text, incoming_text, min_len) == 0)
        {
            free(best_path);
            best_path = path;
            best_common = min_len;
            best_token_count = token_count;
        } else {
            free(path);
        }
        free(text);
    }
    closedir(d);
    if (best_common_out) *best_common_out = best_common;
    if (best_token_count_out) *best_token_count_out = best_token_count;
    return best_path;
}

/* Loads one .conv file into the session; takes ownership of path. */
static bool load_disk_file(const ds4_convcache *cc,
                           char *path,
                           int best_token_count,
                           ds4_session *session,
                           int *loaded_token_count_out,
                           size_t *loaded_text_len_out,
                           ds4_stream_expert_ref **out_residents,
                           uint32_t *out_residents_len,
                           uint8_t **out_aux_blob,
                           uint32_t *out_aux_blob_len,
                           char *err, size_t errlen) {
    /* Open, read the full header (incl. residents block), then the compressed
     * snapshot bytes past the text. */
    FILE *fp = fopen(path, "rb");
    if (!fp) { free(path); return false; }
    uint32_t magic = 0, text_len = 0, token_count = 0;
    uint64_t file_model_key = 0;
    uint64_t uncompressed_len = 0;
    uint32_t n_residents = 0;
    if (fread(&magic,            sizeof(magic),            1, fp) != 1 ||
        fread(&file_model_key,   sizeof(file_model_key),   1, fp) != 1 ||
        fread(&text_len,         sizeof(text_len),         1, fp) != 1 ||
        fread(&token_count,      sizeof(token_count),      1, fp) != 1 ||
        fread(&uncompressed_len, sizeof(uncompressed_len), 1, fp) != 1 ||
        fread(&n_residents,      sizeof(n_residents),      1, fp) != 1 ||
        magic != DS4_CONV_FILE_MAGIC || file_model_key != cc->opt.model_key ||
        uncompressed_len == 0)
    {
        fclose(fp); free(path); return false;
    }
    ds4_stream_expert_ref *residents = NULL;
    if (n_residents > 0) {
        residents = (ds4_stream_expert_ref *)
            malloc((size_t)n_residents * sizeof(*residents));
        if (!residents ||
            fread(residents, sizeof(*residents), n_residents, fp) != n_residents)
        {
            free(residents); fclose(fp); free(path); return false;
        }
    }
    uint32_t aux_blob_len = 0;
    uint8_t *aux_blob = NULL;
    if (fread(&aux_blob_len, sizeof(aux_blob_len), 1, fp) != 1) {
        free(residents); fclose(fp); free(path); return false;
    }
    if (aux_blob_len > 0) {
        aux_blob = (uint8_t *)malloc(aux_blob_len);
        if (!aux_blob ||
            fread(aux_blob, 1, aux_blob_len, fp) != aux_blob_len)
        {
            free(aux_blob); free(residents); fclose(fp); free(path); return false;
        }
    }
    if (fseeko(fp, (off_t)text_len, SEEK_CUR) != 0)
    {
        free(aux_blob); free(residents); fclose(fp); free(path); return false;
    }
    long snap_start = ftell(fp);
    if (fseek(fp, 0, SEEK_END) != 0) { free(aux_blob); free(residents); fclose(fp); free(path); return false; }
    long file_end = ftell(fp);
    long snap_bytes = file_end - snap_start;
    if (snap_bytes <= 0 || fseek(fp, snap_start, SEEK_SET) != 0) {
        free(aux_blob); free(residents); fclose(fp); free(path); return false;
    }
    uint8_t *cbuf = (uint8_t *)malloc((size_t)snap_bytes);
    if (!cbuf) { free(aux_blob); free(residents); fclose(fp); free(path); return false; }
    size_t nread = fread(cbuf, 1, (size_t)snap_bytes, fp);
    fclose(fp);
    free(path);
    if (nread != (size_t)snap_bytes) { free(aux_blob); free(residents); free(cbuf); return false; }

    ds4_session_snapshot compressed = {
        .ptr = cbuf,
        .len = (uint64_t)snap_bytes,
        .cap = (uint64_t)snap_bytes,
    };
    ds4_session_snapshot uncompressed = {0};
    char dec_err[160] = {0};
    bool dec_ok = decompress_snapshot(&compressed, uncompressed_len,
                                      &uncompressed, dec_err, sizeof(dec_err));
    ds4_session_snapshot_free(&compressed);
    if (!dec_ok) {
        free(aux_blob); free(residents);
        if (err && errlen) snprintf(err, errlen, "decompress: %s", dec_err);
        return false;
    }
    char load_err[160] = {0};
    int rc = ds4_session_load_snapshot(session, &uncompressed,
                                       load_err, sizeof(load_err));
    ds4_session_snapshot_free(&uncompressed);
    if (rc != 0) {
        free(aux_blob); free(residents);
        if (err && errlen) snprintf(err, errlen, "load: %s", load_err);
        return false;
    }
    if (loaded_token_count_out) *loaded_token_count_out = best_token_count;
    if (loaded_text_len_out) *loaded_text_len_out = text_len;
    /* Transfer residents / aux_blob to caller (or free if unwanted). */
    if (out_residents && out_residents_len && residents) {
        *out_residents     = residents;
        *out_residents_len = n_residents;
    } else {
        free(residents);
    }
    if (out_aux_blob && out_aux_blob_len && aux_blob) {
        *out_aux_blob     = aux_blob;
        *out_aux_blob_len = aux_blob_len;
    } else {
        free(aux_blob);
    }
    return true;
}

bool ds4_convcache_load_from_disk(ds4_convcache *cc,
                                  const char *incoming_text,
                                  size_t incoming_text_len,
                                  ds4_session *session,
                                  int *loaded_token_count_out,
                                  size_t *loaded_text_len_out,
                                  ds4_stream_expert_ref **out_residents,
                                  uint32_t *out_residents_len,
                                  uint8_t **out_aux_blob,
                                  uint32_t *out_aux_blob_len,
                                  char *err, size_t errlen) {
    if (out_residents) *out_residents = NULL;
    if (out_residents_len) *out_residents_len = 0;
    if (out_aux_blob) *out_aux_blob = NULL;
    if (out_aux_blob_len) *out_aux_blob_len = 0;
    if (!cc || !session || !incoming_text) return false;
    if (cc->opt.disk_budget_bytes == 0 || !cc->opt.disk_dir) return false;

    size_t best_common = 0;
    int best_token_count = 0;
    char *path = scan_disk_best_match(cc, incoming_text, incoming_text_len,
                                      &best_common, &best_token_count);
    if (!path) return false;
    return load_disk_file(cc, path, best_token_count, session, loaded_token_count_out, loaded_text_len_out,
                          out_residents, out_residents_len, out_aux_blob, out_aux_blob_len, err, errlen);
}

static size_t common_prefix_len(const char *a, size_t a_len, const char *b, size_t b_len) {
    const size_t n = a_len < b_len ? a_len : b_len;
    size_t k = 0;
    while (k < n && a[k] == b[k]) k++;
    return k;
}

/* Decompresses from fp until len bytes of output have been written to dst. */
static bool zstd_read(ZSTD_DStream *ds, FILE *fp, ZSTD_inBuffer *in, uint8_t *in_buf, size_t in_cap,
                      void *dst, size_t len) {
    ZSTD_outBuffer out = { dst, len, 0 };
    bool ok = true;
    while (ok && out.pos < out.size) {
        if (in->pos == in->size) {
            in->size = fread(in_buf, 1, in_cap, fp);
            in->pos = 0;
            ok = in->size > 0;
        }
        if (ok) ok = !ZSTD_isError(ZSTD_decompressStream(ds, &out, in));
    }
    return ok;
}

/* A session payload opens with DS4_SESSION_PAYLOAD_U32_FIELDS header words,
 * the eighth the token count, and then the tokens (ds4_session_save_payload);
 * only that much of the compressed snapshot at fp is decompressed, about 1 MB
 * for a 265k-token session of several GB. */
static bool snapshot_tokens(FILE *fp, int **tokens_out, int *len_out) {
    bool ok = false;
    *tokens_out = NULL;
    *len_out = 0;
    ZSTD_DStream *ds = ZSTD_createDStream();
    uint8_t *in_buf = (uint8_t *)malloc(1 << 16);
    if (ds && in_buf) {
        ZSTD_inBuffer in = { in_buf, 0, 0 };
        uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
        if (zstd_read(ds, fp, &in, in_buf, 1 << 16, h, sizeof(h)) &&
            h[0] == DS4_SESSION_PAYLOAD_MAGIC && h[1] == DS4_SESSION_PAYLOAD_VERSION &&
            h[7] > 0 && h[7] <= (uint32_t)(16 * 1024 * 1024)) {
            int *tokens = (int *)malloc((size_t)h[7] * sizeof(int));
            ok = tokens && zstd_read(ds, fp, &in, in_buf, 1 << 16, tokens, (size_t)h[7] * sizeof(int));
            if (ok) {
                *tokens_out = tokens;
                *len_out = (int)h[7];
            } else {
                free(tokens);
            }
        }
    }
    free(in_buf);
    ZSTD_freeDStream(ds);
    return ok;
}

static int candidate_cmp_longest_first(const void *a, const void *b) {
    const ds4_convcache_candidate *x = (const ds4_convcache_candidate *)a;
    const ds4_convcache_candidate *y = (const ds4_convcache_candidate *)b;
    return x->common < y->common ? 1 : x->common > y->common ? -1 : 0;
}

int ds4_convcache_candidates(const ds4_convcache *cc,
                             const char *text, size_t text_len, size_t min_common,
                             ds4_convcache_candidate *out, int max) {
    int n = 0;
    if (cc && text && out && max > 0) {
        for (int i = 0; i < cc->len && n < max; i++) {
            const ds4_convcache_entry *e = &cc->entries[i];
            const size_t common = e->text ? common_prefix_len(e->text, e->text_len, text, text_len) : 0;
            if (common >= min_common && common > 0) {
                char *copy = (char *)malloc(e->text_len + 1);
                if (copy) {
                    memcpy(copy, e->text, e->text_len);
                    copy[e->text_len] = '\0';
                    out[n] = (ds4_convcache_candidate){ i, NULL, copy, e->text_len, common, e->token_count,
                                                        NULL, 0 };
                    FILE *snap = fmemopen(e->snap.ptr, (size_t)e->snap.len, "rb");
                    if (snap) {
                        (void)snapshot_tokens(snap, &out[n].tokens, &out[n].tokens_len);
                        fclose(snap);
                    }
                    n++;
                }
            }
        }
        DIR *d = cc->opt.disk_dir && cc->opt.disk_budget_bytes ? opendir(cc->opt.disk_dir) : NULL;
        struct dirent *ent;
        while (d && n < max && (ent = readdir(d)) != NULL) {
            const size_t nlen = strlen(ent->d_name);
            if (nlen >= 6 && strcmp(ent->d_name + nlen - 5, ".conv") == 0) {
                const size_t path_len = strlen(cc->opt.disk_dir) + 1 + nlen + 1;
                char *path = (char *)malloc(path_len);
                FILE *fp = NULL;
                size_t entry_len = 0;
                char *entry = NULL;
                if (path) {
                    snprintf(path, path_len, "%s/%s", cc->opt.disk_dir, ent->d_name);
                    entry = read_entry_text(path, cc->opt.model_key, &entry_len, &fp);
                }
                const size_t common = entry ? common_prefix_len(entry, entry_len, text, text_len) : 0;
                int token_count = 0, *tokens = NULL, tokens_len = 0;
                if (fp) {
                    /* read_entry_text leaves fp at the snapshot */
                    if (entry && common >= min_common && common > 0) (void)snapshot_tokens(fp, &tokens, &tokens_len);
                    uint32_t tc = 0;
                    if (fseeko(fp, DS4_CONV_TOKEN_COUNT_OFFSET, SEEK_SET) == 0 && fread(&tc, sizeof(tc), 1, fp) == 1)
                        token_count = (int)tc;
                    fclose(fp);
                }
                if (entry && common >= min_common && common > 0) {
                    out[n++] = (ds4_convcache_candidate){ -1, path, entry, entry_len, common, token_count,
                                                          tokens, tokens_len };
                } else {
                    free(entry);
                    free(path);
                }
            }
        }
        if (d) closedir(d);
        qsort(out, (size_t)n, sizeof(out[0]), candidate_cmp_longest_first);
    }
    return n;
}

void ds4_convcache_candidates_free(ds4_convcache_candidate *c, int n) {
    for (int i = 0; c && i < n; i++) {
        free(c[i].path);
        free(c[i].text);
        free(c[i].tokens);
        c[i].path = c[i].text = NULL;
        c[i].tokens = NULL;
    }
}

bool ds4_convcache_load_candidate(ds4_convcache *cc,
                                  const ds4_convcache_candidate *c,
                                  ds4_session *session,
                                  ds4_stream_expert_ref **out_residents,
                                  uint32_t *out_residents_len,
                                  uint8_t **out_aux_blob,
                                  uint32_t *out_aux_blob_len,
                                  char *err, size_t errlen) {
    bool ok = false;
    if (out_residents) *out_residents = NULL;
    if (out_residents_len) *out_residents_len = 0;
    if (out_aux_blob) *out_aux_blob = NULL;
    if (out_aux_blob_len) *out_aux_blob_len = 0;
    if (cc && c && session) {
        if (c->ram_idx >= 0) {
            ok = ds4_convcache_load(cc, c->ram_idx, session, out_residents, out_residents_len,
                                    out_aux_blob, out_aux_blob_len, err, errlen);
        } else if (c->path) {
            char *path = strdup(c->path);
            ok = path && load_disk_file(cc, path, c->token_count, session, NULL, NULL, out_residents,
                                        out_residents_len, out_aux_blob, out_aux_blob_len, err, errlen);
        }
    }
    return ok;
}

bool ds4_convcache_disk_has_prefix(const ds4_convcache *cc,
                                   const char *incoming_text,
                                   size_t incoming_text_len) {
    if (!cc) return false;
    size_t common = 0;
    char *path = scan_disk_best_match(cc, incoming_text, incoming_text_len,
                                      &common, NULL);
    if (!path) return false;
    free(path);
    return common > 0;
}

void ds4_convcache_flush_to_disk(ds4_convcache *cc) {
    if (!cc) return;
    if (cc->opt.disk_budget_bytes == 0 || !cc->opt.disk_dir) return;
    for (int i = 0; i < cc->len; i++) {
        ds4_convcache_entry *e = &cc->entries[i];
        char *path = disk_path_for(cc, e->sha);
        if (!path) continue;
        write_entry_atomic(path, cc->opt.model_key,
                           e->text, e->text_len, e->token_count,
                           e->uncompressed_len,
                           e->residents, e->residents_len,
                           e->aux_blob, e->aux_blob_len,
                           e->snap.ptr, e->snap.len);
        free(path);
    }
    trim_disk_to_budget(cc, NULL);
}

uint64_t ds4_convcache_ram_used(const ds4_convcache *cc) {
    return cc ? cc->ram_used_bytes : 0;
}

int ds4_convcache_len(const ds4_convcache *cc) {
    return cc ? cc->len : 0;
}
