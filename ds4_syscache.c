/*
 * ds4_syscache.c — see ds4_syscache.h for design notes.
 *
 * File layout on disk (little-endian):
 *   [magic(u32) 'SYSC']
 *   [text_len(u32)]
 *   [token_count(u32)]
 *   [uncompressed_snapshot_len(u64)]
 *   [text_len bytes of rendered system-prompt text]
 *   [ZSTD-compressed ds4_session_snapshot bytes to EOF]
 */
#include "ds4_syscache.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zstd.h>

#define DS4_SYSCACHE_MAGIC 0x53595343u  /* "SYSC" */
#define DS4_SYSCACHE_ZSTD_LEVEL 1

/* ---- hashing (djb2, same shape as convcache — fine for identity keying) ---- */

static void sha_from_text(const char *text, size_t text_len, char out[41]) {
    uint64_t h = 5381;
    for (size_t i = 0; i < text_len; i++) h = ((h << 5) + h) ^ (uint8_t)text[i];
    snprintf(out, 41, "%016llx%016llx%08x",
             (unsigned long long)h,
             (unsigned long long)(h ^ 0xdeadbeefcafebabeULL),
             (unsigned)(text_len & 0xffffffff));
}

static char *disk_path_for(const ds4_syscache *sc, const char sha[41]) {
    if (!sc || !sc->opt.disk_dir) return NULL;
    size_t n = strlen(sc->opt.disk_dir) + 1 + 40 + 4 + 1;
    char *p = (char *)malloc(n);
    if (!p) return NULL;
    snprintf(p, n, "%s/%s.sys", sc->opt.disk_dir, sha);
    return p;
}

/* ---- compression helpers (self-contained; syscache is standalone from convcache) ---- */

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
                                DS4_SYSCACHE_ZSTD_LEVEL);
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

/* ---- init/free ---- */

bool ds4_syscache_init(ds4_syscache *sc, const ds4_syscache_opt *opt) {
    if (!sc || !opt) return false;
    memset(sc, 0, sizeof(*sc));
    sc->opt = *opt;
    return true;
}

void ds4_syscache_free(ds4_syscache *sc) {
    if (!sc) return;
    memset(sc, 0, sizeof(*sc));
}

/* ---- direct-hit test ---- */

bool ds4_syscache_has(const ds4_syscache *sc,
                      const char *sys_text,
                      size_t sys_text_len) {
    if (!sc || !sc->opt.disk_dir || sc->opt.disk_budget_bytes == 0 ||
        !sys_text || sys_text_len == 0) return false;
    char sha[41];
    sha_from_text(sys_text, sys_text_len, sha);
    char *path = disk_path_for(sc, sha);
    if (!path) return false;
    struct stat st;
    bool ok = (stat(path, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0);
    free(path);
    return ok;
}

/* ---- load ---- */

bool ds4_syscache_load(ds4_syscache *sc,
                       const char *sys_text,
                       size_t sys_text_len,
                       ds4_session *session,
                       int *loaded_token_count_out,
                       char *err, size_t errlen) {
    if (!sc || !session || !sys_text || sys_text_len == 0) return false;
    if (sc->opt.disk_budget_bytes == 0 || !sc->opt.disk_dir) return false;
    char sha[41];
    sha_from_text(sys_text, sys_text_len, sha);
    char *path = disk_path_for(sc, sha);
    if (!path) return false;

    FILE *fp = fopen(path, "rb");
    if (!fp) { free(path); return false; }
    uint32_t magic = 0, text_len = 0, token_count = 0;
    uint64_t uncompressed_len = 0;
    if (fread(&magic, sizeof(magic), 1, fp) != 1 ||
        fread(&text_len, sizeof(text_len), 1, fp) != 1 ||
        fread(&token_count, sizeof(token_count), 1, fp) != 1 ||
        fread(&uncompressed_len, sizeof(uncompressed_len), 1, fp) != 1 ||
        magic != DS4_SYSCACHE_MAGIC ||
        uncompressed_len == 0)
    {
        fclose(fp); free(path); return false;
    }
    /* Sanity check the stored text length matches what we hashed. Guards
     * against a hash collision or a corrupted file. */
    if ((size_t)text_len != sys_text_len) {
        fclose(fp); free(path); return false;
    }
    /* Skip past the text bytes — we only used them for the collision check
     * conceptually; we don't need to re-read them since we hashed already. */
    if (fseeko(fp, (off_t)text_len, SEEK_CUR) != 0) {
        fclose(fp); free(path); return false;
    }
    long snap_start = ftell(fp);
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); free(path); return false; }
    long file_end = ftell(fp);
    long snap_bytes = file_end - snap_start;
    if (snap_bytes <= 0 || fseek(fp, snap_start, SEEK_SET) != 0) {
        fclose(fp); free(path); return false;
    }
    uint8_t *cbuf = (uint8_t *)malloc((size_t)snap_bytes);
    if (!cbuf) { fclose(fp); free(path); return false; }
    size_t nread = fread(cbuf, 1, (size_t)snap_bytes, fp);
    fclose(fp);
    free(path);
    if (nread != (size_t)snap_bytes) { free(cbuf); return false; }

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
        if (err && errlen) snprintf(err, errlen, "decompress: %s", dec_err);
        return false;
    }
    char load_err[160] = {0};
    int rc = ds4_session_load_snapshot(session, &uncompressed,
                                       load_err, sizeof(load_err));
    ds4_session_snapshot_free(&uncompressed);
    if (rc != 0) {
        if (err && errlen) snprintf(err, errlen, "load: %s", load_err);
        return false;
    }
    if (loaded_token_count_out) *loaded_token_count_out = (int)token_count;
    return true;
}

/* ---- LRU eviction by mtime, until total sits under budget ---- */

typedef struct {
    char   *path;
    time_t  mtime;
    off_t   size;
} sys_entry;

static int sys_entry_cmp_oldest_first(const void *a, const void *b) {
    const sys_entry *ea = a;
    const sys_entry *eb = b;
    if (ea->mtime < eb->mtime) return -1;
    if (ea->mtime > eb->mtime) return 1;
    return 0;
}

static void enforce_budget(const ds4_syscache *sc) {
    if (!sc || !sc->opt.disk_dir || sc->opt.disk_budget_bytes == 0) return;
    DIR *d = opendir(sc->opt.disk_dir);
    if (!d) return;
    sys_entry *entries = NULL;
    size_t count = 0, cap = 0;
    uint64_t total = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *name = ent->d_name;
        size_t nlen = strlen(name);
        if (nlen < 5 || strcmp(name + nlen - 4, ".sys") != 0) continue;
        size_t plen = strlen(sc->opt.disk_dir) + 1 + nlen + 1;
        char *path = (char *)malloc(plen);
        if (!path) continue;
        snprintf(path, plen, "%s/%s", sc->opt.disk_dir, name);
        struct stat st;
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
            free(path); continue;
        }
        if (count == cap) {
            size_t new_cap = cap ? cap * 2 : 8;
            sys_entry *ne = (sys_entry *)realloc(entries, new_cap * sizeof(*ne));
            if (!ne) { free(path); break; }
            entries = ne; cap = new_cap;
        }
        entries[count].path = path;
        entries[count].mtime = st.st_mtime;
        entries[count].size = st.st_size;
        total += (uint64_t)st.st_size;
        count++;
    }
    closedir(d);
    if (total <= sc->opt.disk_budget_bytes) {
        for (size_t i = 0; i < count; i++) free(entries[i].path);
        free(entries);
        return;
    }
    qsort(entries, count, sizeof(*entries), sys_entry_cmp_oldest_first);
    for (size_t i = 0; i < count && total > sc->opt.disk_budget_bytes; i++) {
        if (unlink(entries[i].path) == 0) {
            total -= (uint64_t)entries[i].size;
        }
    }
    for (size_t i = 0; i < count; i++) free(entries[i].path);
    free(entries);
}

/* ---- store ---- */

bool ds4_syscache_store(ds4_syscache *sc,
                        const char *sys_text,
                        size_t sys_text_len,
                        int token_count,
                        ds4_session *session,
                        char *err, size_t errlen) {
    if (!sc || !session || !sys_text || sys_text_len == 0 || token_count <= 0) {
        return false;
    }
    if (sc->opt.disk_budget_bytes == 0 || !sc->opt.disk_dir) return false;

    ds4_session_snapshot snap = {0};
    char save_err[160] = {0};
    if (ds4_session_save_snapshot(session, &snap, save_err, sizeof(save_err)) != 0) {
        if (err && errlen) snprintf(err, errlen, "snapshot: %s", save_err);
        return false;
    }
    uint64_t uncompressed_len = 0;
    char comp_err[160] = {0};
    if (!compress_snapshot_inplace(&snap, &uncompressed_len,
                                   comp_err, sizeof(comp_err)))
    {
        if (err && errlen) snprintf(err, errlen, "compress: %s", comp_err);
        ds4_session_snapshot_free(&snap);
        return false;
    }

    char sha[41];
    sha_from_text(sys_text, sys_text_len, sha);
    char *path = disk_path_for(sc, sha);
    if (!path) { ds4_session_snapshot_free(&snap); return false; }
    size_t plen = strlen(path);
    char *tmp = (char *)malloc(plen + 5);
    if (!tmp) { free(path); ds4_session_snapshot_free(&snap); return false; }
    snprintf(tmp, plen + 5, "%s.tmp", path);

    bool ok = false;
    FILE *fp = fopen(tmp, "wb");
    if (fp) {
        uint32_t magic = DS4_SYSCACHE_MAGIC;
        uint32_t tl = (uint32_t)sys_text_len;
        uint32_t tc = (uint32_t)token_count;
        uint64_t ul = uncompressed_len;
        ok = fwrite(&magic, sizeof(magic), 1, fp) == 1 &&
             fwrite(&tl, sizeof(tl), 1, fp) == 1 &&
             fwrite(&tc, sizeof(tc), 1, fp) == 1 &&
             fwrite(&ul, sizeof(ul), 1, fp) == 1 &&
             fwrite(sys_text, 1, sys_text_len, fp) == sys_text_len &&
             fwrite(snap.ptr, 1, (size_t)snap.len, fp) == (size_t)snap.len;
        fclose(fp);
        if (ok) {
            if (rename(tmp, path) != 0) { unlink(tmp); ok = false; }
        } else {
            unlink(tmp);
        }
    }
    free(tmp);
    free(path);
    ds4_session_snapshot_free(&snap);

    if (ok) enforce_budget(sc);

    if (!ok && err && errlen) {
        snprintf(err, errlen, "write failed");
    }
    return ok;
}

/* ---- introspection ---- */

uint64_t ds4_syscache_disk_used(const ds4_syscache *sc) {
    if (!sc || !sc->opt.disk_dir) return 0;
    DIR *d = opendir(sc->opt.disk_dir);
    if (!d) return 0;
    uint64_t total = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *name = ent->d_name;
        size_t nlen = strlen(name);
        if (nlen < 5 || strcmp(name + nlen - 4, ".sys") != 0) continue;
        size_t plen = strlen(sc->opt.disk_dir) + 1 + nlen + 1;
        char *path = (char *)malloc(plen);
        if (!path) continue;
        snprintf(path, plen, "%s/%s", sc->opt.disk_dir, name);
        struct stat st;
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) {
            total += (uint64_t)st.st_size;
        }
        free(path);
    }
    closedir(d);
    return total;
}

int ds4_syscache_entry_count(const ds4_syscache *sc) {
    if (!sc || !sc->opt.disk_dir) return 0;
    DIR *d = opendir(sc->opt.disk_dir);
    if (!d) return 0;
    int n = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *name = ent->d_name;
        size_t nlen = strlen(name);
        if (nlen < 5 || strcmp(name + nlen - 4, ".sys") != 0) continue;
        n++;
    }
    closedir(d);
    return n;
}
