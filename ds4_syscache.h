/*
 * ds4_syscache.h — System-prompt KV cache for ds4-server.
 *
 * A tiny disk-backed cache keyed by the SHA-1 of the rendered system-prompt
 * text (system message content + tool schemas + template tags — whatever is
 * always emitted at the start of every request from a given client config).
 *
 * Lookup is O(1) filename check: hash the incoming request's rendered
 * system portion, probe `<disk_dir>/<sha1>.sys`. On hit, load the snapshot
 * and the caller rewinds live to the stored token_count before forward-
 * prefilling the user turn.
 *
 * Store is one-shot per unique system prompt: when a request cold-prefills
 * for the first time under a system prompt we haven't seen before, we
 * snapshot the live state and write it. Subsequent requests with the same
 * system prompt land in the direct-hit path and skip cold prefill entirely.
 *
 * Separate budget from ds4_convcache so full-conversation churn can't evict
 * system-prompt entries the operator actually wants warm.
 */
#ifndef DS4_SYSCACHE_H
#define DS4_SYSCACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4.h"

typedef struct ds4_syscache_opt {
    uint64_t disk_budget_bytes;   /* 0 disables the cache */
    char    *disk_dir;            /* owned by caller; must outlive the cache */
    uint64_t model_key;           /* the model the KV belongs to; entries of
                                     another key are never loaded */
} ds4_syscache_opt;

typedef struct ds4_syscache {
    ds4_syscache_opt opt;
} ds4_syscache;

/* Init / free — no runtime state beyond opt, but this shape mirrors convcache
 * so future in-RAM optimisations can drop in without an API change. */
bool ds4_syscache_init(ds4_syscache *sc, const ds4_syscache_opt *opt);
void ds4_syscache_free(ds4_syscache *sc);

/* Direct-hit test. Hashes `sys_text` and probes the filesystem. Returns true
 * if a matching file exists. Cheap (single stat()). */
bool ds4_syscache_has(const ds4_syscache *sc,
                      const char *sys_text,
                      size_t sys_text_len);

/* Load the snapshot for `sys_text` into `session` if one exists. On success
 * `*loaded_token_count_out` is set to the entry's stored token count so the
 * caller can rewind live to the appropriate position before forward-
 * prefilling. Returns false if no entry exists or on I/O error. */
bool ds4_syscache_load(ds4_syscache *sc,
                       const char *sys_text,
                       size_t sys_text_len,
                       ds4_session *session,
                       int *loaded_token_count_out,
                       char *err, size_t errlen);

/* Snapshot the current live session and store it under the hash of
 * `sys_text`. `token_count` is the position within the session where the
 * system prompt ends — used on load to know where to rewind. If a matching
 * file already exists, it is overwritten. If the disk budget is exceeded
 * after this write, the oldest entries (by mtime) are deleted until the
 * total sits under budget. Returns true on success. */
bool ds4_syscache_store(ds4_syscache *sc,
                        const char *sys_text,
                        size_t sys_text_len,
                        int token_count,
                        ds4_session *session,
                        char *err, size_t errlen);

/* Introspection ------------------------------------------------------- */

/* Total bytes used across all .sys files in the cache directory. */
uint64_t ds4_syscache_disk_used(const ds4_syscache *sc);

/* Number of .sys files present. */
int ds4_syscache_entry_count(const ds4_syscache *sc);

#endif
