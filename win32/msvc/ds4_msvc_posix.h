/* The POSIX calls and GCC spellings the CUDA sources use, for the Microsoft
 * compiler.
 *
 * nvcc on Windows only drives cl.exe, so ds4_cuda.cu and cuda/mmq are built
 * with it into ds4_cuda.dll while the rest of the engine is built with
 * MinGW.  nvcc force-includes this header, and win32/msvc holds stand-ins for
 * <unistd.h>, <pthread.h> and <sys/mman.h> that include it.
 *
 * Everything here is static: the DLL and the MinGW programs run on different
 * C runtimes, so the DLL keeps its own file descriptors and must not hand one
 * across. */
#ifndef DS4_MSVC_POSIX_H
#define DS4_MSVC_POSIX_H

#if defined(_MSC_VER)

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_NONSTDC_NO_WARNINGS
#define _CRT_NONSTDC_NO_WARNINGS
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

/* The CRT's off_t is a 32-bit long, and the sources cast file offsets beyond
 * 2 GiB to it; this claims the name before <sys/types.h> can. */
#define _OFF_T_DEFINED
typedef __int64 _off_t;
typedef __int64 off_t;

#include <windows.h>
#include <process.h>
#include <io.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <intrin.h>

#undef near
#undef far
#undef small
#undef interface
#undef ERROR
#undef IN
#undef OUT

#define __attribute__(x)
#define __restrict__ __restrict
#define __builtin_expect(value, expected) (value)
#define __builtin_popcount(x) ((int)__popcnt((unsigned)(x)))
#define __builtin_popcountll(x) ((int)__popcnt64((unsigned long long)(x)))
#define __builtin_prefetch(...) ((void)0)

static __forceinline int ds4_msvc_ctz(unsigned x) {
    unsigned long index = 0;
    _BitScanForward(&index, x);
    return (int)index;
}
static __forceinline int ds4_msvc_clz(unsigned x) {
    unsigned long index = 0;
    _BitScanReverse(&index, x);
    return 31 - (int)index;
}
static __forceinline int ds4_msvc_ctzll(unsigned long long x) {
    unsigned long index = 0;
    _BitScanForward64(&index, x);
    return (int)index;
}
static __forceinline int ds4_msvc_clzll(unsigned long long x) {
    unsigned long index = 0;
    _BitScanReverse64(&index, x);
    return 63 - (int)index;
}
#define __builtin_ctz(x) ds4_msvc_ctz((unsigned)(x))
#define __builtin_clz(x) ds4_msvc_clz((unsigned)(x))
#define __builtin_ctzll(x) ds4_msvc_ctzll((unsigned long long)(x))
#define __builtin_clzll(x) ds4_msvc_clzll((unsigned long long)(x))

typedef intptr_t ssize_t;
#ifndef SSIZE_MAX
#define SSIZE_MAX INTPTR_MAX
#endif
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#define isatty _isatty

/* ---- files ---------------------------------------------------------------- */

#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

static inline int ds4_msvc_open(const char *path, int flags, int mode = 0) {
    return _open(path, flags | _O_BINARY, mode);
}
#define open ds4_msvc_open
#define close _close

/* The CRT's stat has no st_blksize, which the loader reads for its direct-I/O
 * alignment.  As with the POSIX pair, the struct and the function share a name. */
struct ds4_msvc_stat {
    uint64_t st_size;
    unsigned st_mode;
    unsigned st_blksize;
    __time64_t st_mtime;
};
static inline void ds4_msvc_stat_from(struct ds4_msvc_stat *out, const struct _stat64 *in) {
    out->st_size = (uint64_t)in->st_size;
    out->st_mode = in->st_mode;
    out->st_blksize = 4096;
    out->st_mtime = in->st_mtime;
}
static inline int ds4_msvc_fstat(int fd, struct ds4_msvc_stat *out) {
    struct _stat64 raw;
    int result = _fstat64(fd, &raw);
    if (result == 0) ds4_msvc_stat_from(out, &raw);
    return result;
}
static inline int ds4_msvc_stat(const char *path, struct ds4_msvc_stat *out) {
    struct _stat64 raw;
    int result = _stat64(path, &raw);
    if (result == 0) ds4_msvc_stat_from(out, &raw);
    return result;
}
#define fstat ds4_msvc_fstat
#define stat ds4_msvc_stat

/* ReadFile takes a 32-bit length; tensors run to several GiB. */
static inline ssize_t pread(int fd, void *buf, size_t n, off_t off) {
    ssize_t result = -1;
    HANDLE file = (HANDLE)_get_osfhandle(fd);
    if (file == INVALID_HANDLE_VALUE) {
        errno = EBADF;
    } else {
        size_t done = 0;
        bool failed = false, eof = false;
        while (done < n && !failed && !eof) {
            OVERLAPPED at = {};
            uint64_t position = (uint64_t)off + done;
            DWORD want = (n - done > (1u << 30)) ? (1u << 30) : (DWORD)(n - done), got = 0;
            at.Offset = (DWORD)position;
            at.OffsetHigh = (DWORD)(position >> 32);
            if (ReadFile(file, (char *)buf + done, want, &got, &at)) {
                done += got;
                eof = got == 0;
            } else if (GetLastError() == ERROR_HANDLE_EOF) {
                eof = true;
            } else {
                failed = true;
            }
        }
        if (failed && done == 0) errno = EIO;
        else result = (ssize_t)done;
    }
    return result;
}

/* ---- memory mapping ------------------------------------------------------ */

#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_NORESERVE 0x4000
#define MAP_POPULATE 0x8000
#define MAP_FAILED ((void *)-1)
#define POSIX_MADV_NORMAL 0
#define POSIX_MADV_RANDOM 1
#define POSIX_MADV_SEQUENTIAL 2
#define POSIX_MADV_WILLNEED 3
#define POSIX_MADV_DONTNEED 4
#define MADV_NORMAL POSIX_MADV_NORMAL
#define MADV_RANDOM POSIX_MADV_RANDOM
#define MADV_SEQUENTIAL POSIX_MADV_SEQUENTIAL
#define MADV_WILLNEED POSIX_MADV_WILLNEED
#define MADV_DONTNEED POSIX_MADV_DONTNEED
#define MADV_HUGEPAGE 14

/* Same scheme as win32/ds4_win32.c: a view starts on a 64 KiB file boundary,
 * and munmap finds the real base through VirtualQuery. */
static inline void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    void *result = MAP_FAILED;
    (void)addr;
    if (len == 0) {
        errno = EINVAL;
    } else if (flags & MAP_ANONYMOUS) {
        DWORD protect = (prot & PROT_WRITE) ? PAGE_READWRITE : (prot & PROT_READ) ? PAGE_READONLY : PAGE_NOACCESS;
        void *p = VirtualAlloc(NULL, len, MEM_RESERVE | (prot != PROT_NONE ? MEM_COMMIT : 0), protect);
        if (p) result = p;
        else errno = ENOMEM;
    } else {
        HANDLE file = (HANDLE)_get_osfhandle(fd);
        if (file == INVALID_HANDLE_VALUE) {
            errno = EBADF;
        } else {
            bool writable = (prot & PROT_WRITE) != 0, shared = (flags & MAP_SHARED) != 0;
            DWORD protect = !writable ? PAGE_READONLY : shared ? PAGE_READWRITE : PAGE_WRITECOPY;
            DWORD access = !writable ? FILE_MAP_READ : shared ? FILE_MAP_WRITE : FILE_MAP_COPY;
            SYSTEM_INFO info;
            GetSystemInfo(&info);
            uint64_t delta = (uint64_t)off % info.dwAllocationGranularity;
            uint64_t base = (uint64_t)off - delta;
            HANDLE mapping = CreateFileMappingA(file, NULL, protect, 0, 0, NULL);
            if (mapping) {
                void *p = MapViewOfFile(mapping, access, (DWORD)(base >> 32), (DWORD)base, len + delta);
                CloseHandle(mapping);
                if (p) result = (char *)p + delta;
                else errno = ENOMEM;
            } else {
                errno = EACCES;
            }
        }
    }
    return result;
}

static inline int munmap(void *addr, size_t len) {
    (void)len;
    int result = -1;
    MEMORY_BASIC_INFORMATION info;
    if (addr && VirtualQuery(addr, &info, sizeof(info)) && info.AllocationBase) {
        BOOL ok = (info.Type == MEM_MAPPED) ? UnmapViewOfFile(info.AllocationBase)
                                            : VirtualFree(info.AllocationBase, 0, MEM_RELEASE);
        if (ok) result = 0;
    }
    if (result != 0) errno = EINVAL;
    return result;
}

static inline int posix_madvise(void *addr, size_t len, int advice) {
    if (advice == POSIX_MADV_WILLNEED) {
        WIN32_MEMORY_RANGE_ENTRY range = { addr, len };
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    }
    return 0;
}
#define madvise posix_madvise

/* ---- system and time ------------------------------------------------------ */

#define _SC_PAGESIZE 1
#define _SC_PAGE_SIZE _SC_PAGESIZE
#define _SC_NPROCESSORS_ONLN 2
#define _SC_PHYS_PAGES 4
#define _SC_AVPHYS_PAGES 5
static inline long long sysconf(int name) {
    long long result = -1;
    SYSTEM_INFO info;
    MEMORYSTATUSEX memory = {};
    memory.dwLength = sizeof(memory);
    GetSystemInfo(&info);
    if (name == _SC_PAGESIZE) result = info.dwPageSize;
    else if (name == _SC_NPROCESSORS_ONLN) result = info.dwNumberOfProcessors;
    else if (name == _SC_PHYS_PAGES && GlobalMemoryStatusEx(&memory)) result = (long long)(memory.ullTotalPhys / info.dwPageSize);
    else if (name == _SC_AVPHYS_PAGES && GlobalMemoryStatusEx(&memory)) result = (long long)(memory.ullAvailPhys / info.dwPageSize);
    return result;
}

#ifndef CLOCK_MONOTONIC
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#endif
static inline int clock_gettime(int clock, struct timespec *out) {
    if (clock == CLOCK_MONOTONIC) {
        LARGE_INTEGER ticks, frequency;
        QueryPerformanceCounter(&ticks);
        QueryPerformanceFrequency(&frequency);
        out->tv_sec = (time_t)(ticks.QuadPart / frequency.QuadPart);
        out->tv_nsec = (long)((ticks.QuadPart % frequency.QuadPart) * 1000000000ll / frequency.QuadPart);
    } else {
        timespec_get(out, TIME_UTC);
    }
    return 0;
}

static inline unsigned usleep(unsigned microseconds) {
    Sleep((microseconds + 999) / 1000);
    return 0;
}

/* ---- threads ---------------------------------------------------------------- */

/* Only creation and join: the one thread the CUDA backend starts itself. */
typedef HANDLE pthread_t;
struct ds4_msvc_thread_start {
    void *(*entry)(void *);
    void *argument;
};
static unsigned __stdcall ds4_msvc_thread_main(void *raw) {
    ds4_msvc_thread_start start = *(ds4_msvc_thread_start *)raw;
    free(raw);
    start.entry(start.argument);
    return 0;
}
static inline int pthread_create(pthread_t *thread, const void *attributes, void *(*entry)(void *), void *argument) {
    (void)attributes;
    int result = EAGAIN;
    ds4_msvc_thread_start *start = (ds4_msvc_thread_start *)malloc(sizeof(*start));
    if (start) {
        start->entry = entry;
        start->argument = argument;
        uintptr_t handle = _beginthreadex(NULL, 0, ds4_msvc_thread_main, start, 0, NULL);
        if (handle) {
            *thread = (HANDLE)handle;
            result = 0;
        } else {
            free(start);
        }
    }
    return result;
}
static inline int pthread_join(pthread_t thread, void **value) {
    if (value) *value = NULL;
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

#endif /* _MSC_VER */
#endif /* DS4_MSVC_POSIX_H */
