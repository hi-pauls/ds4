/*
 * ds4_convcache.c — see ds4_convcache.h for design notes.
 */
#include "ds4_convcache.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
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
 *   CONZ: [ZSTD-compressed snapshot bytes to end of file]
 *   CONA: [8-byte base frame length] [ZSTD frame of the snapshot]
 *         then appended records, each the state a later turn reached and
 *         the whole key it answers to (a finished turn renders differently
 *         once the next one follows it, so keys do not extend each other):
 *         [4-byte DLTA] [4-byte key_len] [key text]
 *         [4-byte token_count] [4-byte aux_len] [aux bytes]
 *         [8-byte uncompressed delta length] [8-byte frame length]
 *         [ZSTD frame of a delta payload (ds4_session_save_delta_snapshot)]
 * Servers of different models may share a directory: a file whose model_key
 * differs is never loaded, and CONY files, which carry no key, are rejected. */
#define DS4_CONV_FILE_MAGIC 0x434f4e5au  /* "CONZ" — + model_key */
#define DS4_CONV_FILE_MAGIC_APPEND 0x434f4e41u  /* "CONA" */
#define DS4_CONV_RECORD_MAGIC 0x444c5441u  /* "DLTA" */
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

static void entry_sha(const ds4_convcache *cc, const char *text, size_t text_len, char sha[41]) {
    uint64_t h = 5381 ^ cc->opt.model_key;
    for (size_t i = 0; i < text_len; i++) h = ((h << 5) + h) ^ (uint8_t)text[i];
    snprintf(sha, 41, "%016llx%016llx%08x",
             (unsigned long long)h,
             (unsigned long long)(h ^ 0xdeadbeefcafebabeULL),
             (unsigned)(text_len & 0xffffffff));
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

/* Writes a CONA entry (layout at the top) with no records yet, atomically.
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
        uint32_t magic = DS4_CONV_FILE_MAGIC_APPEND;
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
                  fwrite(&snap_len, sizeof(snap_len), 1, fp) == 1 &&
                  fwrite(snap_ptr, 1, (size_t)snap_len, fp) == (size_t)snap_len;
        ok = fclose(fp) == 0 && ok;
        landed = ok && rename(tmp, path) == 0;
        if (!landed) unlink(tmp);
    }
    free(tmp);
    return landed;
}

/* Appends one record to a CONA file.  A record that does not land whole is
 * cut off again, and a crash part way leaves one conv_file_read skips. */
static bool append_record(const char *path, const char *key, size_t key_len, int token_count,
                          const uint8_t *aux, uint32_t aux_len, uint64_t raw_len,
                          const uint8_t *frame, uint64_t frame_len) {
    bool landed = false;
    const int fd = path && key && key_len ? open(path, O_WRONLY | O_APPEND) : -1;
    struct stat st;
    if (fd >= 0 && fstat(fd, &st) == 0) {
        FILE *fp = fdopen(fd, "ab");
        if (fp) {
            const uint32_t magic = DS4_CONV_RECORD_MAGIC, kl = (uint32_t)key_len;
            const uint32_t tc = (uint32_t)(token_count < 0 ? 0 : token_count), al = aux ? aux_len : 0u;
            landed = fwrite(&magic, sizeof(magic), 1, fp) == 1 && fwrite(&kl, sizeof(kl), 1, fp) == 1 &&
                     fwrite(key, 1, kl, fp) == kl && fwrite(&tc, sizeof(tc), 1, fp) == 1 &&
                     fwrite(&al, sizeof(al), 1, fp) == 1 && (al == 0 || fwrite(aux, 1, al, fp) == al) &&
                     fwrite(&raw_len, sizeof(raw_len), 1, fp) == 1 &&
                     fwrite(&frame_len, sizeof(frame_len), 1, fp) == 1 &&
                     fwrite(frame, 1, (size_t)frame_len, fp) == (size_t)frame_len;
            landed = fflush(fp) == 0 && landed;
            if (!landed) (void)ftruncate(fd, st.st_size);
            fclose(fp);
        } else {
            close(fd);
        }
    } else if (fd >= 0) {
        close(fd);
    }
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

/* One .conv file as read back: the key, token count and tool memory of its
 * newest state (the last record's, else the base's), residents, and where each
 * compressed frame sits.  A CONZ file is a base alone; a CONA file is a base
 * followed by appended delta records (append_record), of which a record cut
 * short by a crash is left out with everything behind it. */
typedef struct {
    off_t at;
    uint64_t len;
    uint64_t raw_len;
} conv_frame;

typedef struct {
    char *text;
    size_t text_len;
    int token_count;
    ds4_stream_expert_ref *residents;
    uint32_t n_residents;
    uint8_t *aux;
    uint32_t aux_len;
    conv_frame base;
    conv_frame *records;
    int n_records;
} conv_file;

static void conv_file_free(conv_file *f) {
    free(f->text);
    free(f->residents);
    free(f->aux);
    free(f->records);
    memset(f, 0, sizeof(*f));
}

static bool read_exact(FILE *fp, void *dst, size_t len) {
    return len == 0 || fread(dst, 1, len, fp) == len;
}

/* Reads one appended record's metadata and steps past its frame; false at
 * the end of the file or at a record the file does not hold whole. */
static bool read_record(FILE *fp, off_t file_end, conv_file *f) {
    uint32_t magic = 0, key_len = 0, token_count = 0, aux_len = 0;
    uint64_t raw_len = 0, frame_len = 0;
    char *key = NULL;
    uint8_t *aux = NULL;
    bool ok = read_exact(fp, &magic, sizeof(magic)) && magic == DS4_CONV_RECORD_MAGIC &&
              read_exact(fp, &key_len, sizeof(key_len)) && key_len > 0 && key_len <= (uint32_t)(64 * 1024 * 1024);
    if (ok) {
        key = (char *)malloc((size_t)key_len + 1);
        ok = key && read_exact(fp, key, key_len) &&
             read_exact(fp, &token_count, sizeof(token_count)) &&
             read_exact(fp, &aux_len, sizeof(aux_len)) && aux_len <= (uint32_t)(64 * 1024 * 1024);
    }
    if (ok && aux_len) {
        aux = (uint8_t *)malloc(aux_len);
        ok = aux && read_exact(fp, aux, aux_len);
    }
    ok = ok && read_exact(fp, &raw_len, sizeof(raw_len)) && read_exact(fp, &frame_len, sizeof(frame_len));
    const off_t at = ok ? ftello(fp) : -1;
    ok = ok && at >= 0 && frame_len > 0 && (uint64_t)(file_end - at) >= frame_len;
    conv_frame *records = ok ? (conv_frame *)realloc(f->records, (size_t)(f->n_records + 1) * sizeof(conv_frame)) : NULL;
    ok = ok && records && fseeko(fp, (off_t)frame_len, SEEK_CUR) == 0;
    if (records) f->records = records;
    if (ok) {
        key[key_len] = '\0';
        free(f->text);
        f->text = key;
        f->text_len = key_len;
        key = NULL;
        f->token_count = (int)token_count;
        f->records[f->n_records++] = (conv_frame){ at, frame_len, raw_len };
        if (aux_len) {
            free(f->aux);
            f->aux = aux;
            f->aux_len = aux_len;
            aux = NULL;
        }
    }
    free(key);
    free(aux);
    return ok;
}

/* Parses a .conv file of this model; false and nothing kept on any error. */
static bool conv_file_read(const char *path, uint64_t model_key, conv_file *f) {
    memset(f, 0, sizeof(*f));
    FILE *fp = path ? fopen(path, "rb") : NULL;
    uint32_t magic = 0, text_len = 0, token_count = 0, n_residents = 0, aux_len = 0;
    uint64_t file_model_key = 0, uncompressed_len = 0;
    off_t file_end = -1;
    bool ok = fp && fseeko(fp, 0, SEEK_END) == 0 && (file_end = ftello(fp)) > 0 && fseeko(fp, 0, SEEK_SET) == 0 &&
              read_exact(fp, &magic, sizeof(magic)) &&
              (magic == DS4_CONV_FILE_MAGIC || magic == DS4_CONV_FILE_MAGIC_APPEND) &&
              read_exact(fp, &file_model_key, sizeof(file_model_key)) && file_model_key == model_key &&
              read_exact(fp, &text_len, sizeof(text_len)) && text_len > 0 &&
              text_len <= (uint32_t)(64 * 1024 * 1024) &&
              read_exact(fp, &token_count, sizeof(token_count)) &&
              read_exact(fp, &uncompressed_len, sizeof(uncompressed_len)) && uncompressed_len > 0 &&
              read_exact(fp, &n_residents, sizeof(n_residents)) && n_residents <= (uint32_t)(1u << 24);
    if (ok && n_residents) {
        f->residents = (ds4_stream_expert_ref *)malloc((size_t)n_residents * sizeof(*f->residents));
        ok = f->residents && read_exact(fp, f->residents, (size_t)n_residents * sizeof(*f->residents));
        f->n_residents = n_residents;
    }
    ok = ok && read_exact(fp, &aux_len, sizeof(aux_len)) && aux_len <= (uint32_t)(64 * 1024 * 1024);
    if (ok && aux_len) {
        f->aux = (uint8_t *)malloc(aux_len);
        ok = f->aux && read_exact(fp, f->aux, aux_len);
        f->aux_len = aux_len;
    }
    if (ok) {
        f->text = (char *)malloc((size_t)text_len + 1);
        ok = f->text && read_exact(fp, f->text, text_len);
        if (ok) f->text[text_len] = '\0';
        f->text_len = text_len;
        f->token_count = (int)token_count;
    }
    uint64_t base_len = 0;
    if (ok && magic == DS4_CONV_FILE_MAGIC_APPEND) ok = read_exact(fp, &base_len, sizeof(base_len));
    const off_t base_at = ok ? ftello(fp) : -1;
    if (ok && magic == DS4_CONV_FILE_MAGIC) base_len = (uint64_t)(file_end - base_at);
    ok = ok && base_at >= 0 && base_len > 0 && (uint64_t)(file_end - base_at) >= base_len;
    if (ok) {
        f->base = (conv_frame){ base_at, base_len, uncompressed_len };
        if (magic == DS4_CONV_FILE_MAGIC_APPEND && fseeko(fp, (off_t)base_len, SEEK_CUR) == 0)
            while (read_record(fp, file_end, f)) {}
    }
    if (fp) fclose(fp);
    if (!ok) conv_file_free(f);
    return ok;
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
    free(cc->loaded_path);
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
    entry_sha(cc, text, text_len, sha);

    /* Snapshot first — we need its size to know how much room to make. */
    const uint64_t t_start = now_ns();
    ds4_session_snapshot snap = {0};
    char save_err[160] = {0};
    if (ds4_session_save_snapshot(session, &snap, save_err, sizeof(save_err)) != 0) {
        if (err && errlen) snprintf(err, errlen, "snapshot: %s", save_err);
        return -1;
    }
    const uint64_t t_snap = now_ns();

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
    const uint64_t t_comp = now_ns();
    fprintf(stderr, "ds4: convcache store snapshot %.2fs, compress %.2fs, %llu -> %llu bytes\n",
            (double)(t_snap - t_start) / 1e9, (double)(t_comp - t_snap) / 1e9,
            (unsigned long long)uncompressed_len, (unsigned long long)snap.len);

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
                const uint64_t t_write = now_ns();
                landed = write_entry_atomic(path, cc->opt.model_key, text, text_len, token_count,
                                            uncompressed_len,
                                            residents, residents_len,
                                            aux_blob, aux_blob_len,
                                            snap.ptr, snap.len);
                fprintf(stderr, "ds4: convcache store write %.2fs\n", (double)(now_ns() - t_write) / 1e9);
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
        conv_file f;
        if (!conv_file_read(path, cc->opt.model_key, &f)) { free(path); continue; }
        size_t min_len = f.text_len < incoming_text_len ? f.text_len
                                                        : incoming_text_len;
        if (min_len > best_common &&
            memcmp(f.text, incoming_text, min_len) == 0)
        {
            free(best_path);
            best_path = path;
            best_common = min_len;
            best_token_count = f.token_count;
        } else {
            free(path);
        }
        conv_file_free(&f);
    }
    closedir(d);
    if (best_common_out) *best_common_out = best_common;
    if (best_token_count_out) *best_token_count_out = best_token_count;
    return best_path;
}

/* Loads one .conv file into the session, base and records; takes ownership of
 * path and keeps it as the cache's loaded_path on success. */
static bool load_disk_file(ds4_convcache *cc,
                           char *path,
                           ds4_session *session,
                           int *loaded_token_count_out,
                           size_t *loaded_text_len_out,
                           ds4_stream_expert_ref **out_residents,
                           uint32_t *out_residents_len,
                           uint8_t **out_aux_blob,
                           uint32_t *out_aux_blob_len,
                           char *err, size_t errlen) {
    conv_file f;
    bool ok = conv_file_read(path, cc->opt.model_key, &f);
    FILE *fp = ok ? fopen(path, "rb") : NULL;
    ok = fp != NULL;
    /* the base, then every record in the order the turns appended them */
    for (int k = -1; ok && k < f.n_records; k++) {
        const conv_frame *frame = k < 0 ? &f.base : &f.records[k];
        ds4_session_snapshot compressed = { (uint8_t *)malloc((size_t)frame->len), frame->len, frame->len };
        ds4_session_snapshot uncompressed = {0};
        char step_err[160] = {0};
        ok = compressed.ptr && fseeko(fp, frame->at, SEEK_SET) == 0 &&
             read_exact(fp, compressed.ptr, (size_t)frame->len);
        if (ok && !decompress_snapshot(&compressed, frame->raw_len, &uncompressed, step_err, sizeof(step_err))) {
            if (err && errlen) snprintf(err, errlen, "decompress %s: %s", k < 0 ? "base" : "record", step_err);
            ok = false;
        }
        if (ok && ds4_session_load_snapshot(session, &uncompressed, step_err, sizeof(step_err)) != 0) {
            if (err && errlen) snprintf(err, errlen, "load %s %d: %s", k < 0 ? "base" : "record", k + 1, step_err);
            ok = false;
        }
        ds4_session_snapshot_free(&compressed);
        ds4_session_snapshot_free(&uncompressed);
    }
    if (fp) fclose(fp);
    if (ok) {
        if (loaded_token_count_out) *loaded_token_count_out = f.token_count;
        if (loaded_text_len_out) *loaded_text_len_out = f.text_len;
        if (out_residents && out_residents_len && f.residents) {
            *out_residents = f.residents;
            *out_residents_len = f.n_residents;
            f.residents = NULL;
        }
        if (out_aux_blob && out_aux_blob_len && f.aux) {
            *out_aux_blob = f.aux;
            *out_aux_blob_len = f.aux_len;
            f.aux = NULL;
        }
        free(cc->loaded_path);
        cc->loaded_path = path;
        path = NULL;
    }
    conv_file_free(&f);
    free(path);
    return ok;
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
    return load_disk_file(cc, path, session, loaded_token_count_out, loaded_text_len_out,
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
 * a delta has its start row and MTP row between header and tokens, and only
 * the tokens from that row on.  Only that much of the compressed frame at fp
 * is decompressed, about 1 MB for a 265k-token session of several GB.  The
 * tokens are appended to *tokens; a delta that does not start where they end
 * is refused. */
static bool snapshot_tokens(FILE *fp, bool delta, int **tokens, int *len) {
    bool ok = false;
    ZSTD_DStream *ds = ZSTD_createDStream();
    uint8_t *in_buf = (uint8_t *)malloc(1 << 16);
    if (ds && in_buf) {
        ZSTD_inBuffer in = { in_buf, 0, 0 };
        uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS], from[2] = {0, 0};
        if (zstd_read(ds, fp, &in, in_buf, 1 << 16, h, sizeof(h)) &&
            h[0] == DS4_SESSION_PAYLOAD_MAGIC && h[1] == DS4_SESSION_PAYLOAD_VERSION &&
            h[7] <= (uint32_t)(16 * 1024 * 1024) &&
            (!delta || zstd_read(ds, fp, &in, in_buf, 1 << 16, from, sizeof(from))) &&
            from[0] == (uint32_t)*len && h[7] > from[0]) {
            const size_t added = (size_t)(h[7] - from[0]);
            int *grown = (int *)realloc(*tokens, ((size_t)*len + added) * sizeof(int));
            if (grown) *tokens = grown;
            ok = grown && zstd_read(ds, fp, &in, in_buf, 1 << 16, grown + *len, added * sizeof(int));
            if (ok) *len += (int)added;
        }
    }
    free(in_buf);
    ZSTD_freeDStream(ds);
    return ok;
}

/* The tokens a file's newest state holds: its base's, then every record's. */
static void conv_file_tokens(const char *path, const conv_file *f, int **tokens, int *len) {
    FILE *fp = fopen(path, "rb");
    bool ok = fp && fseeko(fp, f->base.at, SEEK_SET) == 0 && snapshot_tokens(fp, false, tokens, len);
    for (int k = 0; ok && k < f->n_records; k++)
        ok = fseeko(fp, f->records[k].at, SEEK_SET) == 0 && snapshot_tokens(fp, true, tokens, len);
    if (!ok) {
        free(*tokens);
        *tokens = NULL;
        *len = 0;
    }
    if (fp) fclose(fp);
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
                        (void)snapshot_tokens(snap, false, &out[n].tokens, &out[n].tokens_len);
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
                conv_file f = {0};
                bool read = false;
                if (path) {
                    snprintf(path, path_len, "%s/%s", cc->opt.disk_dir, ent->d_name);
                    read = conv_file_read(path, cc->opt.model_key, &f);
                }
                const size_t common = read ? common_prefix_len(f.text, f.text_len, text, text_len) : 0;
                if (read && common >= min_common && common > 0) {
                    int *tokens = NULL, tokens_len = 0;
                    conv_file_tokens(path, &f, &tokens, &tokens_len);
                    out[n++] = (ds4_convcache_candidate){ -1, path, f.text, f.text_len, common, f.token_count,
                                                          tokens, tokens_len };
                    f.text = NULL;
                } else {
                    free(path);
                }
                conv_file_free(&f);
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
            ok = path && load_disk_file(cc, path, session, NULL, NULL, out_residents,
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

char *ds4_convcache_entry_path(const ds4_convcache *cc, const char *text, size_t text_len) {
    char sha[41] = {0};
    char *path = NULL;
    if (cc && text && text_len && cc->opt.disk_dir && cc->opt.disk_budget_bytes) {
        entry_sha(cc, text, text_len, sha);
        path = disk_path_for(cc, sha);
    }
    return path;
}

bool ds4_convcache_write(const ds4_convcache *cc, const char *path, bool base, const char *text, size_t text_len,
                         int token_count, const ds4_stream_expert_ref *residents, uint32_t residents_len,
                         const uint8_t *aux, uint32_t aux_len, ds4_session_snapshot *raw, char *err, size_t errlen) {
    bool landed = false;
    uint64_t raw_len = 0;
    char comp_err[160] = {0};
    const uint64_t t0 = now_ns();
    if (!cc || !path || !raw || !raw->ptr) {
        if (err && errlen) snprintf(err, errlen, "nothing to write");
    } else if (!compress_snapshot_inplace(raw, &raw_len, comp_err, sizeof(comp_err))) {
        if (err && errlen) snprintf(err, errlen, "compress: %s", comp_err);
    } else {
        const uint64_t t1 = now_ns();
        landed = base ? write_entry_atomic(path, cc->opt.model_key, text, text_len, token_count, raw_len,
                                           residents, residents_len, aux, aux_len, raw->ptr, raw->len)
                      : append_record(path, text, text_len, token_count, aux, aux_len, raw_len, raw->ptr, raw->len);
        if (!landed && err && errlen) snprintf(err, errlen, "%s %s failed", base ? "writing" : "appending to", path);
        if (landed) trim_disk_to_budget(cc, path);
        fprintf(stderr, "ds4: convcache %s compress %.2fs write %.2fs, %llu -> %llu bytes\n",
                base ? "base" : "record", (double)(t1 - t0) / 1e9, (double)(now_ns() - t1) / 1e9,
                (unsigned long long)raw_len, (unsigned long long)raw->len);
    }
    return landed;
}

uint64_t ds4_convcache_ram_used(const ds4_convcache *cc) {
    return cc ? cc->ram_used_bytes : 0;
}

int ds4_convcache_len(const ds4_convcache *cc) {
    return cc ? cc->len : 0;
}
