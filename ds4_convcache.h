/*
 * ds4_convcache.h — RAM-first conversation cache for ds4-server.
 *
 * An alternative to the prefix-hashed disk cache (ds4_kvstore) that is
 * designed for a single interactive user rather than a many-user API server:
 *
 *   - The active conversation lives in the ds4_session (live KV) as normal.
 *   - When the incoming request does not extend the live conversation, the
 *     live snapshot is moved into this RAM cache and the incoming
 *     conversation (if we have it) is loaded into the live session.
 *   - Writes to disk happen only on eviction (RAM budget exhausted) or on
 *     server shutdown — not per turn, not per request.
 *
 * Entries are keyed by the SHA-1 of the rendered prompt text, matching the
 * existing kvstore convention so on-disk snapshots stay interchangeable.
 *
 * Thread model: caller synchronises access. ds4-server serialises
 * generate_job on a single worker, so no locking is needed inside this
 * module beyond what ds4_session itself provides.
 */
#ifndef DS4_CONVCACHE_H
#define DS4_CONVCACHE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "ds4.h"
#include "ds4_gpu.h"   /* ds4_stream_expert_ref */

typedef struct ds4_convcache_entry {
    char sha[41];                 /* SHA-1 hex of the rendered live text at save */
    int  model_id;
    int  quant_bits;
    int  ctx_size;
    int  token_count;             /* live tokens at time of store */
    uint64_t last_hit_ns;         /* monotonic ns for LRU */
    char    *text;                /* owned; rendered live text at save time */
    size_t   text_len;
    ds4_session_snapshot snap;    /* owned; free with ds4_session_snapshot_free.
                                     Holds the ZSTD-compressed snapshot bytes;
                                     .len is the compressed size (also what
                                     counts against the RAM budget). */
    uint64_t uncompressed_len;    /* original snapshot size — allocation size
                                     needed when decompressing on load */
    ds4_stream_expert_ref *residents; /* owned; expert-cache residency at save
                                     time, sorted by priority desc. Small
                                     (~40 KB for 5000 residents). NULL if the
                                     backend doesn't support enum_residents. */
    uint32_t residents_len;
    uint8_t *aux_blob;            /* owned; opaque per-conversation blob the
                                     caller supplied at store time (currently
                                     used for the server's tool_memory snapshot
                                     so tool-call DSML restores together with
                                     KV). NULL if none. */
    uint32_t aux_blob_len;
} ds4_convcache_entry;

typedef struct ds4_convcache_opt {
    uint64_t ram_budget_bytes;    /* required; 0 disables the cache entirely */
    uint64_t disk_budget_bytes;   /* 0 = no disk tier */
    char    *disk_dir;            /* owned by caller; must outlive the cache */
    uint64_t model_key;           /* the model the KV belongs to; disk entries of
                                     another key are never loaded */
} ds4_convcache_opt;

typedef struct ds4_convcache {
    ds4_convcache_opt   opt;
    ds4_convcache_entry *entries;
    int                 len;
    int                 cap;
    uint64_t            ram_used_bytes;
} ds4_convcache;

/* Init / free ---------------------------------------------------------- */

/* Zero-init followed by opt copy. Safe to call on a zeroed struct.
 * Returns true on success. Does not create disk_dir. */
bool ds4_convcache_init(ds4_convcache *cc, const ds4_convcache_opt *opt);

/* Frees all in-RAM snapshots and the entry table. Disk files are left in
 * place — call ds4_convcache_flush_to_disk() first if you want persistence. */
void ds4_convcache_free(ds4_convcache *cc);

/* Lookup ---------------------------------------------------------------- */

/* Return the index of the LONGEST entry whose stored text starts with the
 * given text_prefix. The intended usage is: pass the incoming request's
 * rendered prompt text — the returned entry, when loaded, will have live
 * tokens covering AT LEAST the incoming prompt (plus possibly generated
 * tokens beyond). Caller should rewind the session to the incoming prompt's
 * token length after ds4_convcache_load(). Returns -1 if no entry matches.
 * Updates the winning entry's last_hit_ns to now. */
int ds4_convcache_find_prefix(ds4_convcache *cc,
                              const char *text_prefix,
                              size_t text_prefix_len);

/* Store the live session as a new entry keyed by sha ------------------- */

/* Snapshots the current live session and adds it to the RAM cache. `text`
 * / `text_len` is the key the live state answers to — the rendered live text,
 * or, when the live state holds hidden reasoning the client does not replay,
 * the visible transcript up to the live frontier. It serves both as the
 * entry's identity (SHA-1 keyed off it) and for prefix-match lookups by
 * ds4_convcache_find_prefix(). If the RAM budget would be exceeded, oldest
 * entries are evicted first (to disk when disk_budget_bytes > 0, otherwise
 * dropped). If `pinned_text_prefix` is non-NULL, any entry whose text is a
 * byte-prefix of `pinned_text_prefix` (i.e., an entry we'd want to load
 * next) is protected from eviction.
 *
 * Returns the entry index on success, or -1 on failure. */
int ds4_convcache_store(ds4_convcache *cc,
                        ds4_engine *engine,
                        ds4_session *session,
                        const char *text,
                        size_t text_len,
                        int model_id,
                        int quant_bits,
                        int ctx_size,
                        int token_count,
                        const char *pinned_text_prefix,
                        size_t pinned_text_prefix_len,
                        const uint8_t *aux_blob,
                        uint32_t aux_blob_len,
                        char *err, size_t errlen);

/* Load an in-RAM entry into the live session -------------------------- */

/* Applies entries[idx].snap to the given session. After success the entry
 * is removed from the RAM cache (its slot is freed). If `out_residents` is
 * non-NULL, ownership of the entry's residents array is transferred to the
 * caller (which must free()), and `*out_residents_len` is set. Pass NULL
 * to ignore residents (they'll be freed with the entry). */
bool ds4_convcache_load(ds4_convcache *cc,
                        int idx,
                        ds4_session *session,
                        ds4_stream_expert_ref **out_residents,
                        uint32_t *out_residents_len,
                        uint8_t **out_aux_blob,
                        uint32_t *out_aux_blob_len,
                        char *err, size_t errlen);

/* Scan the disk dir for the best-matching .conv entry (longest common
 * byte-prefix with `incoming_text`) and load its snapshot into the session.
 * On success `*loaded_token_count_out` is set to the entry's stored token
 * count and `*loaded_text_len_out` to the length of its stored key, so the
 * caller knows where the request's new text starts. If
 * `out_residents` is non-NULL, the residents block from the file is
 * malloc'd and returned (caller frees). */
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
                                  char *err, size_t errlen);

/* Disk index integration ---------------------------------------------- */

/* Cheap check: is there any .conv file with a non-zero common byte-prefix
 * with `incoming_text`? Iterates the disk dir but only reads each file's
 * text header. Returns false when disk backing is off. */
bool ds4_convcache_disk_has_prefix(const ds4_convcache *cc,
                                   const char *incoming_text,
                                   size_t incoming_text_len);

/* Shutdown ------------------------------------------------------------ */

/* Serialise every RAM entry to disk_dir, one file per entry, then free the
 * cache. No-op if disk_budget_bytes == 0. */
void ds4_convcache_flush_to_disk(ds4_convcache *cc);

/* Introspection (for tests / logging) --------------------------------- */

uint64_t ds4_convcache_ram_used(const ds4_convcache *cc);
int      ds4_convcache_len(const ds4_convcache *cc);

#endif
