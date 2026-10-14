/* Windows implementations of the POSIX calls declared in ds4_win32.h. */
#include "ds4_win32.h"

#if defined(_WIN32) && !defined(__CYGWIN__)

/* PrefetchVirtualMemory is declared from Windows 8 on. */
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#include "ds4_win32_net.h"
#include <windows.h>
#include <stdarg.h>
#include <ctype.h>
#include <process.h>

/* The wrappers below are these functions; they call the real ones. */
#undef socket
#undef socketpair
#undef accept
#undef bind
#undef listen
#undef connect
#undef send
#undef recv
#undef shutdown
#undef setsockopt
#undef getsockopt
#undef getsockname
#undef getpeername
#undef poll
#undef select

/* Read by the MinGW-w64 startup code: every open() and fopen() without an
 * explicit mode, and the three standard streams, become binary. */
int _CRT_fmode = _O_BINARY;

static HANDLE fd_handle(int fd) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) errno = EBADF;
    return h;
}

/* ---- memory mapping ------------------------------------------------------ */

static DWORD allocation_granularity(void) {
    static DWORD granularity;
    if (!granularity) {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        granularity = info.dwAllocationGranularity;
    }
    return granularity;
}

/* A view starts on a 64 KiB boundary of the file, so an offset that is only
 * page aligned is mapped from the boundary below it and the pointer moved up.
 * munmap finds the real base again through VirtualQuery. */
void *mmap(void *addr, size_t len, int prot, int flags, int fd, int64_t off) {
    void *result = MAP_FAILED;
    if (len == 0) {
        errno = EINVAL;
    } else if (flags & MAP_ANONYMOUS) {
        DWORD protect = (prot & PROT_WRITE) ? PAGE_READWRITE : (prot & PROT_READ) ? PAGE_READONLY : PAGE_NOACCESS;
        /* Accessible memory is committed even under MAP_NORESERVE: Windows has
         * no overcommit, and a page that is only reserved faults on access. */
        DWORD type = MEM_RESERVE | (prot != PROT_NONE ? MEM_COMMIT : 0);
        void *p = VirtualAlloc((flags & MAP_FIXED) ? addr : NULL, len, type, protect);
        if (p) result = p;
        else errno = ENOMEM;
    } else {
        HANDLE file = fd_handle(fd);
        if (file != INVALID_HANDLE_VALUE) {
            int writable = (prot & PROT_WRITE) != 0;
            int shared = (flags & MAP_SHARED) != 0;
            DWORD protect = !writable ? PAGE_READONLY : shared ? PAGE_READWRITE : PAGE_WRITECOPY;
            DWORD access = !writable ? FILE_MAP_READ : shared ? FILE_MAP_WRITE : FILE_MAP_COPY;
            uint64_t delta = (uint64_t)off % allocation_granularity();
            uint64_t base = (uint64_t)off - delta;
            HANDLE mapping = CreateFileMappingA(file, NULL, protect, 0, 0, NULL);
            if (mapping) {
                void *want = (flags & MAP_FIXED) ? (char *)addr - delta : NULL;
                void *p = MapViewOfFileEx(mapping, access, (DWORD)(base >> 32), (DWORD)base, len + delta, want);
                /* The view keeps the mapping object alive. */
                CloseHandle(mapping);
                if (p) result = (char *)p + delta;
                else errno = (GetLastError() == ERROR_NOT_ENOUGH_MEMORY) ? ENOMEM : EINVAL;
            } else {
                errno = (GetLastError() == ERROR_ACCESS_DENIED) ? EACCES : EINVAL;
            }
        }
    }
    return result;
}

/* Windows cannot unmap part of a view.  The engine unmaps the tail of the
 * model, tables it reads from disk only, so that nothing faults them in; a
 * range that starts inside a view therefore stays mapped and only leaves the
 * working set.  It is unmapped with the view, by the call that names its base
 * (ds4_win32.c's own offset shift moves that by less than 64 KiB). */
int munmap(void *addr, size_t len) {
    int result = -1;
    MEMORY_BASIC_INFORMATION info;
    if (addr && VirtualQuery(addr, &info, sizeof(info)) && info.AllocationBase) {
        if ((size_t)((char *)addr - (char *)info.AllocationBase) >= allocation_granularity()) {
            VirtualUnlock(addr, len);
            result = 0;
        } else {
            BOOL ok = (info.Type == MEM_MAPPED) ? UnmapViewOfFile(info.AllocationBase)
                                                : VirtualFree(info.AllocationBase, 0, MEM_RELEASE);
            if (ok) result = 0;
        }
    }
    if (result != 0) errno = EINVAL;
    return result;
}

int madvise(void *addr, size_t len, int advice) {
    if (advice == MADV_WILLNEED) {
        WIN32_MEMORY_RANGE_ENTRY range = { addr, len };
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    } else if (advice == MADV_DONTNEED || advice == MADV_FREE) {
        /* Drops the pages from the working set.  A mapped file's pages come
         * back from the file, private ones from the page file, so the
         * contents survive: weaker than Linux for private memory, which the
         * engine only uses as a hint. */
        VirtualUnlock(addr, len);
    }
    return 0;
}

int posix_madvise(void *addr, size_t len, int advice) {
    return madvise(addr, len, advice);
}

/* VirtualLock fails until the process's minimum working set covers what is
 * locked, so the quota grows by the request and the lock is tried again. */
int mlock(const void *addr, size_t len) {
    int result = 0;
    if (!VirtualLock((void *)addr, len)) {
        SIZE_T min_size, max_size;
        HANDLE process = GetCurrentProcess();
        result = -1;
        if (GetProcessWorkingSetSize(process, &min_size, &max_size) &&
            SetProcessWorkingSetSize(process, min_size + len, max_size + len) &&
            VirtualLock((void *)addr, len))
            result = 0;
        else
            errno = ENOMEM;
    }
    return result;
}

int munlock(const void *addr, size_t len) {
    VirtualUnlock((void *)addr, len);
    return 0;
}

int msync(void *addr, size_t len, int flags) {
    (void)flags;
    int result = FlushViewOfFile(addr, len) ? 0 : -1;
    if (result != 0) errno = EINVAL;
    return result;
}

int mprotect(void *addr, size_t len, int prot) {
    DWORD old;
    DWORD protect = (prot & PROT_WRITE) ? PAGE_READWRITE : (prot & PROT_READ) ? PAGE_READONLY : PAGE_NOACCESS;
    int result = VirtualProtect(addr, len, protect, &old) ? 0 : -1;
    if (result != 0) errno = EINVAL;
    return result;
}

/* ---- files ---------------------------------------------------------------- */

/* ReadFile and WriteFile take 32-bit lengths, and the engine reads tensors of
 * several GiB in one call. */
#define IO_CHUNK (1u << 30)

ssize_t pread(int fd, void *buf, size_t n, int64_t off) {
    ssize_t result = -1;
    HANDLE file = fd_handle(fd);
    if (file != INVALID_HANDLE_VALUE) {
        size_t done = 0;
        int failed = 0, eof = 0;
        while (done < n && !failed && !eof) {
            OVERLAPPED at = { 0 };
            uint64_t position = (uint64_t)off + done;
            DWORD want = (n - done > IO_CHUNK) ? IO_CHUNK : (DWORD)(n - done), got = 0;
            at.Offset = (DWORD)position;
            at.OffsetHigh = (DWORD)(position >> 32);
            if (ReadFile(file, (char *)buf + done, want, &got, &at)) {
                done += got;
                eof = got == 0;
            } else if (GetLastError() == ERROR_HANDLE_EOF) {
                eof = 1;
            } else {
                failed = 1;
            }
        }
        if (failed && done == 0) errno = EIO;
        else result = (ssize_t)done;
    }
    return result;
}

ssize_t pwrite(int fd, const void *buf, size_t n, int64_t off) {
    ssize_t result = -1;
    HANDLE file = fd_handle(fd);
    if (file != INVALID_HANDLE_VALUE) {
        size_t done = 0;
        int failed = 0;
        while (done < n && !failed) {
            OVERLAPPED at = { 0 };
            uint64_t position = (uint64_t)off + done;
            DWORD want = (n - done > IO_CHUNK) ? IO_CHUNK : (DWORD)(n - done), put = 0;
            at.Offset = (DWORD)position;
            at.OffsetHigh = (DWORD)(position >> 32);
            if (WriteFile(file, (const char *)buf + done, want, &put, &at)) done += put;
            else failed = 1;
        }
        if (failed && done == 0) errno = (GetLastError() == ERROR_DISK_FULL) ? ENOSPC : EIO;
        else result = (ssize_t)done;
    }
    return result;
}

/* Windows files carry no Unix permission bits; access follows the directory. */
int fchmod(int fd, int mode) {
    (void)fd;
    (void)mode;
    return 0;
}

int fsync(int fd) {
    return _commit(fd);
}

int fdatasync(int fd) {
    return _commit(fd);
}

int flock(int fd, int op) {
    int result = -1;
    HANDLE file = fd_handle(fd);
    if (file != INVALID_HANDLE_VALUE) {
        OVERLAPPED whole = { 0 };
        BOOL ok;
        if (op & LOCK_UN) {
            ok = UnlockFileEx(file, 0, MAXDWORD, MAXDWORD, &whole);
        } else {
            DWORD flags = ((op & LOCK_EX) ? LOCKFILE_EXCLUSIVE_LOCK : 0) |
                          ((op & LOCK_NB) ? LOCKFILE_FAIL_IMMEDIATELY : 0);
            ok = LockFileEx(file, flags, 0, MAXDWORD, MAXDWORD, &whole);
        }
        if (ok) result = 0;
        else errno = (GetLastError() == ERROR_LOCK_VIOLATION) ? EWOULDBLOCK : EIO;
    }
    return result;
}

static int errno_of_last_error(void) {
    switch (GetLastError()) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND: return ENOENT;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION: return EACCES;
    case ERROR_ALREADY_EXISTS:
    case ERROR_FILE_EXISTS: return EEXIST;
    case ERROR_DISK_FULL: return ENOSPC;
    case ERROR_NOT_SAME_DEVICE: return EXDEV;
    default: return EIO;
    }
}

int ds4_w32_rename(const char *from, const char *to) {
    int result = MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED) ? 0 : -1;
    if (result != 0) errno = errno_of_last_error();
    return result;
}

int ds4_w32_mkdir(const char *path, ...) {
    return _mkdir(path);
}

char *mkdtemp(char *tmpl) {
    char *result = NULL;
    size_t len = strlen(tmpl);
    if (len < 6 || strcmp(tmpl + len - 6, "XXXXXX") != 0) {
        errno = EINVAL;
    } else {
        static const char letters[] = "abcdefghijklmnopqrstuvwxyz0123456789";
        unsigned seed = (unsigned)GetTickCount() ^ ((unsigned)_getpid() << 12);
        for (int attempt = 0; attempt < 256 && !result; attempt++) {
            for (int i = 0; i < 6; i++) {
                seed = seed * 1103515245u + 12345u;
                tmpl[len - 6 + i] = letters[(seed >> 16) % (sizeof(letters) - 1)];
            }
            if (_mkdir(tmpl) == 0) result = tmpl;
            else if (errno != EEXIST) attempt = 256;
        }
    }
    return result;
}

char *realpath(const char *path, char *resolved) {
    char *result = NULL;
    char *full = _fullpath(resolved, path, resolved ? MAX_PATH : 0);
    if (!full) {
        errno = ENAMETOOLONG;
    } else if (GetFileAttributesA(full) == INVALID_FILE_ATTRIBUTES) {
        if (!resolved) free(full);
        errno = ENOENT;
    } else {
        result = full;
    }
    return result;
}

int symlink(const char *target, const char *linkpath) {
    (void)target;
    (void)linkpath;
    errno = ENOSYS;
    return -1;
}

ssize_t readlink(const char *path, char *buf, size_t n) {
    (void)path;
    (void)buf;
    (void)n;
    errno = EINVAL;
    return -1;
}

int dprintf(int fd, const char *format, ...) {
    char stack[1024];
    char *text = stack;
    va_list args;
    va_start(args, format);
    int len = vsnprintf(stack, sizeof(stack), format, args);
    va_end(args);
    if (len >= (int)sizeof(stack)) {
        text = malloc((size_t)len + 1);
        if (text) {
            va_start(args, format);
            vsnprintf(text, (size_t)len + 1, format, args);
            va_end(args);
        }
    }
    int result = (len < 0 || !text) ? -1 : _write(fd, text, (unsigned)len);
    if (text != stack) free(text);
    return result;
}

/* ---- sockets ---------------------------------------------------------------- */

/* Which values are sockets, and which of those are non-blocking: a socket
 * handle and a CRT descriptor are both small integers and can coincide.
 * Handles are multiples of four, so the table is indexed by handle / 4. */
#define SOCKET_SLOTS 65536
#define SLOT_SOCKET 1
#define SLOT_NONBLOCKING 2
static unsigned char socket_slots[SOCKET_SLOTS];

static unsigned char *socket_slot(int fd) {
    return (fd > 0 && (fd & 3) == 0 && (fd >> 2) < SOCKET_SLOTS) ? &socket_slots[fd >> 2] : NULL;
}

static int is_socket(int fd) {
    unsigned char *slot = socket_slot(fd);
    return slot && (*slot & SLOT_SOCKET);
}

static int wsa_errno(int error) {
    switch (error) {
    case 0: return 0;
    case WSAEWOULDBLOCK: return EAGAIN;
    case WSAEINPROGRESS:
    case WSAEALREADY: return EINPROGRESS;
    case WSAEINTR: return EINTR;
    case WSAECONNRESET:
    case WSAENETRESET: return ECONNRESET;
    case WSAECONNABORTED: return ECONNABORTED;
    case WSAECONNREFUSED: return ECONNREFUSED;
    case WSAETIMEDOUT: return ETIMEDOUT;
    case WSAEADDRINUSE: return EADDRINUSE;
    case WSAEADDRNOTAVAIL: return EADDRNOTAVAIL;
    case WSAENOTCONN: return ENOTCONN;
    case WSAEISCONN: return EISCONN;
    case WSAESHUTDOWN: return EPIPE;
    case WSAEHOSTUNREACH: return EHOSTUNREACH;
    case WSAENETUNREACH: return ENETUNREACH;
    case WSAEMSGSIZE: return EMSGSIZE;
    case WSAENOBUFS: return ENOBUFS;
    case WSAEACCES: return EACCES;
    case WSAEMFILE: return EMFILE;
    case WSAENOTSOCK:
    case WSAEBADF: return EBADF;
    case WSAEINVAL:
    case WSAEFAULT: return EINVAL;
    case WSAEAFNOSUPPORT: return EAFNOSUPPORT;
    case WSAEPROTONOSUPPORT: return EPROTONOSUPPORT;
    case WSAEOPNOTSUPP: return EOPNOTSUPP;
    default: return EIO;
    }
}

static int socket_failed(void) {
    errno = wsa_errno(WSAGetLastError());
    return -1;
}

static int adopt_socket(SOCKET s) {
    int result = -1;
    unsigned char *slot = (s == INVALID_SOCKET) ? NULL : socket_slot((int)s);
    if (s == INVALID_SOCKET) {
        socket_failed();
    } else if (!slot) {
        closesocket(s);
        errno = EMFILE;
    } else {
        *slot = SLOT_SOCKET;
        result = (int)s;
    }
    return result;
}

__attribute__((constructor)) static void start_winsock(void) {
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);
}

/* DS4_WIN32_MAX_WORKING_SET_MIB caps the physical memory the process holds.
 * A model larger than RAM is mapped, and Windows lets the working set grow
 * with every page touched until the whole machine is short; under the cap the
 * pages it gives up stay in the file cache, where anyone can take them. */
__attribute__((constructor)) static void cap_working_set(void) {
    const char *env = getenv("DS4_WIN32_MAX_WORKING_SET_MIB");
    const unsigned long long mib = env && env[0] ? strtoull(env, NULL, 10) : 0;
    if (mib != 0 &&
        !SetProcessWorkingSetSizeEx(GetCurrentProcess(), 64u << 20, (SIZE_T)(mib << 20),
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE)) {
        fprintf(stderr, "ds4: could not cap the working set at %llu MiB (error %lu)\n", mib, GetLastError());
    }
}

/* Started by a service, the process has no window and Windows schedules it as
 * background work: slower cores, lower clock.  Qwen3.8's CPU experts then
 * take 0.26 ms each instead of 0.16.  Opting out of execution speed
 * throttling asks for the scheduling a foreground program gets. */
__attribute__((constructor)) static void refuse_power_throttling(void) {
    PROCESS_POWER_THROTTLING_STATE state;
    memset(&state, 0, sizeof(state));
    state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask = 0;
    if (!getenv("DS4_WIN32_KEEP_POWER_THROTTLING") &&
        !SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state))) {
        fprintf(stderr, "ds4: could not opt out of power throttling (error %lu)\n", GetLastError());
    }
}

int ds4_w32_socket(int domain, int type, int protocol) {
    return adopt_socket(socket(domain, type, protocol));
}

int ds4_w32_accept(int fd, struct sockaddr *addr, socklen_t *len) {
    return adopt_socket(accept((SOCKET)fd, addr, len));
}

/* Windows has no socketpair; two loopback TCP sockets connected to each other
 * behave the same for the wake-up and worker channels the server uses. */
int ds4_w32_socketpair(int domain, int type, int protocol, int pair[2]) {
    (void)domain;
    (void)protocol;
    int result = -1;
    struct sockaddr_in address;
    socklen_t length = sizeof(address);
    int listener = ds4_w32_socket(AF_INET, type, 0);
    int client = -1, server = -1;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (listener >= 0 && bind((SOCKET)listener, (struct sockaddr *)&address, sizeof(address)) == 0 &&
        listen((SOCKET)listener, 1) == 0 &&
        getsockname((SOCKET)listener, (struct sockaddr *)&address, &length) == 0 &&
        (client = ds4_w32_socket(AF_INET, type, 0)) >= 0 &&
        connect((SOCKET)client, (struct sockaddr *)&address, sizeof(address)) == 0 &&
        (server = ds4_w32_accept(listener, NULL, NULL)) >= 0) {
        BOOL no_delay = TRUE;
        setsockopt((SOCKET)client, IPPROTO_TCP, TCP_NODELAY, (const char *)&no_delay, sizeof(no_delay));
        setsockopt((SOCKET)server, IPPROTO_TCP, TCP_NODELAY, (const char *)&no_delay, sizeof(no_delay));
        pair[0] = client;
        pair[1] = server;
        result = 0;
    } else {
        int error = WSAGetLastError();
        if (client >= 0) close(client);
        errno = error ? wsa_errno(error) : errno;
    }
    if (listener >= 0) close(listener);
    return result;
}

int ds4_w32_bind(int fd, const struct sockaddr *addr, socklen_t len) {
    return bind((SOCKET)fd, addr, len) == 0 ? 0 : socket_failed();
}

int ds4_w32_listen(int fd, int backlog) {
    return listen((SOCKET)fd, backlog) == 0 ? 0 : socket_failed();
}

int ds4_w32_connect(int fd, const struct sockaddr *addr, socklen_t len) {
    int result = 0;
    if (connect((SOCKET)fd, addr, len) != 0) {
        /* A non-blocking connect in progress reports "would block" here. */
        int error = WSAGetLastError();
        errno = (error == WSAEWOULDBLOCK) ? EINPROGRESS : wsa_errno(error);
        result = -1;
    }
    return result;
}

/* Winsock has no MSG_DONTWAIT or MSG_NOSIGNAL; only these flags go through. */
#define WINSOCK_FLAGS (MSG_PEEK | MSG_OOB | MSG_WAITALL)

ssize_t ds4_w32_send(int fd, const void *buf, size_t n, int flags) {
    int sent = send((SOCKET)fd, buf, (n > IO_CHUNK) ? (int)IO_CHUNK : (int)n, flags & WINSOCK_FLAGS);
    return sent == SOCKET_ERROR ? socket_failed() : sent;
}

ssize_t ds4_w32_recv(int fd, void *buf, size_t n, int flags) {
    int got = recv((SOCKET)fd, buf, (n > IO_CHUNK) ? (int)IO_CHUNK : (int)n, flags & WINSOCK_FLAGS);
    return got == SOCKET_ERROR ? socket_failed() : got;
}

int ds4_w32_shutdown(int fd, int how) {
    return shutdown((SOCKET)fd, how) == 0 ? 0 : socket_failed();
}

/* Winsock takes socket timeouts as milliseconds in a DWORD, not a timeval. */
int ds4_w32_setsockopt(int fd, int level, int name, const void *value, socklen_t len) {
    int result = 0;
    DWORD milliseconds;
    if (level == SOL_SOCKET && (name == SO_RCVTIMEO || name == SO_SNDTIMEO) && len == (socklen_t)sizeof(struct timeval)) {
        const struct timeval *timeout = value;
        milliseconds = (DWORD)(timeout->tv_sec * 1000 + timeout->tv_usec / 1000);
        value = &milliseconds;
        len = sizeof(milliseconds);
    }
    /* SO_REUSEPORT is defined as 0 here: there is nothing to set. */
    if (!(level == SOL_SOCKET && name == 0) && setsockopt((SOCKET)fd, level, name, value, len) != 0)
        result = socket_failed();
    return result;
}

int ds4_w32_getsockopt(int fd, int level, int name, void *value, socklen_t *len) {
    int result = getsockopt((SOCKET)fd, level, name, value, len) == 0 ? 0 : socket_failed();
    if (result == 0 && level == SOL_SOCKET && name == SO_ERROR && *len >= (socklen_t)sizeof(int))
        *(int *)value = wsa_errno(*(int *)value);
    return result;
}

int ds4_w32_getsockname(int fd, struct sockaddr *addr, socklen_t *len) {
    return getsockname((SOCKET)fd, addr, len) == 0 ? 0 : socket_failed();
}

int ds4_w32_getpeername(int fd, struct sockaddr *addr, socklen_t *len) {
    return getpeername((SOCKET)fd, addr, len) == 0 ? 0 : socket_failed();
}

int ds4_w32_poll(struct pollfd *fds, unsigned long count, int timeout_ms) {
    int result;
    if (count == 0) {
        Sleep(timeout_ms < 0 ? INFINITE : (DWORD)timeout_ms);
        result = 0;
    } else {
        /* WSAPoll rejects the whole call for event bits it does not know. */
        for (unsigned long i = 0; i < count; i++) fds[i].events &= POLLIN | POLLOUT | POLLRDNORM | POLLWRNORM | POLLRDBAND;
        result = WSAPoll(fds, count, timeout_ms);
        if (result == SOCKET_ERROR) result = socket_failed();
    }
    return result;
}

static int set_is_empty(const fd_set *set) {
    return !set || set->fd_count == 0;
}

int ds4_w32_select(int nfds, fd_set *r, fd_set *w, fd_set *e, struct timeval *timeout) {
    (void)nfds;
    int result;
    if (set_is_empty(r) && set_is_empty(w) && set_is_empty(e)) {
        /* select() as a sleep: Winsock refuses a call without sockets. */
        Sleep(timeout ? (DWORD)(timeout->tv_sec * 1000 + timeout->tv_usec / 1000) : INFINITE);
        result = 0;
    } else {
        result = select(0, r, w, e, timeout);
        if (result == SOCKET_ERROR) result = socket_failed();
    }
    return result;
}

ssize_t read(int fd, void *buf, size_t n) {
    ssize_t result;
    if (is_socket(fd)) result = ds4_w32_recv(fd, buf, n, 0);
    else result = _read(fd, buf, (n > IO_CHUNK) ? IO_CHUNK : (unsigned)n);
    return result;
}

ssize_t write(int fd, const void *buf, size_t n) {
    ssize_t result;
    if (is_socket(fd)) result = ds4_w32_send(fd, buf, n, 0);
    else result = _write(fd, buf, (n > IO_CHUNK) ? IO_CHUNK : (unsigned)n);
    return result;
}

int close(int fd) {
    int result;
    if (is_socket(fd)) {
        *socket_slot(fd) = 0;
        result = closesocket((SOCKET)fd) == 0 ? 0 : socket_failed();
    } else {
        result = _close(fd);
    }
    return result;
}

ssize_t writev(int fd, const struct iovec *iov, int count) {
    ssize_t total = 0;
    int stop = 0;
    for (int i = 0; i < count && !stop; i++) {
        ssize_t put = write(fd, iov[i].iov_base, iov[i].iov_len);
        if (put < 0 && total == 0) total = -1;
        if (put > 0) total += put;
        stop = put < 0 || (size_t)put < iov[i].iov_len;
    }
    return total;
}

ssize_t readv(int fd, const struct iovec *iov, int count) {
    ssize_t total = 0;
    int stop = 0;
    for (int i = 0; i < count && !stop; i++) {
        ssize_t got = read(fd, iov[i].iov_base, iov[i].iov_len);
        if (got < 0 && total == 0) total = -1;
        if (got > 0) total += got;
        stop = got <= 0 || (size_t)got < iov[i].iov_len;
    }
    return total;
}

ssize_t sendmsg(int fd, const struct msghdr *message, int flags) {
    (void)flags;
    return writev(fd, message->msg_iov, message->msg_iovlen);
}

ssize_t recvmsg(int fd, struct msghdr *message, int flags) {
    (void)flags;
    message->msg_flags = 0;
    return readv(fd, message->msg_iov, message->msg_iovlen);
}

int fcntl(int fd, int cmd, ...) {
    int result = 0;
    unsigned char *slot = is_socket(fd) ? socket_slot(fd) : NULL;
    va_list args;
    va_start(args, cmd);
    if (cmd == F_GETFL) {
        result = (slot && (*slot & SLOT_NONBLOCKING)) ? O_NONBLOCK : 0;
    } else if (cmd == F_SETFL && slot) {
        u_long enabled = (va_arg(args, int) & O_NONBLOCK) != 0;
        if (ioctlsocket((SOCKET)fd, FIONBIO, &enabled) != 0) result = socket_failed();
        else *slot = (unsigned char)(SLOT_SOCKET | (enabled ? SLOT_NONBLOCKING : 0));
    } else if (cmd == F_GETFD) {
        /* Close-on-exec here is a handle no child inherits. */
        HANDLE handle = slot ? (HANDLE)(intptr_t)fd : (HANDLE)_get_osfhandle(fd);
        DWORD flags = 0;
        if (handle == INVALID_HANDLE_VALUE || !GetHandleInformation(handle, &flags)) {
            errno = EBADF;
            result = -1;
        } else {
            result = (flags & HANDLE_FLAG_INHERIT) ? 0 : FD_CLOEXEC;
        }
    } else if (cmd != F_SETFD && cmd != F_SETFL) {
        errno = EINVAL;
        result = -1;
    }
    va_end(args);
    return result;
}

/* ---- process and system --------------------------------------------------- */

long sysconf(int name) {
    long result = -1;
    SYSTEM_INFO info;
    MEMORYSTATUSEX memory;
    memset(&memory, 0, sizeof(memory));
    memory.dwLength = sizeof(memory);
    GetSystemInfo(&info);
    switch (name) {
    case _SC_PAGESIZE: result = (long)info.dwPageSize; break;
    case _SC_NPROCESSORS_ONLN:
    case _SC_NPROCESSORS_CONF: result = (long)info.dwNumberOfProcessors; break;
    case _SC_PHYS_PAGES:
        if (GlobalMemoryStatusEx(&memory)) result = (long)(memory.ullTotalPhys / info.dwPageSize);
        break;
    case _SC_AVPHYS_PAGES:
        if (GlobalMemoryStatusEx(&memory)) result = (long)(memory.ullAvailPhys / info.dwPageSize);
        break;
    default: errno = EINVAL;
    }
    return result;
}

int getpagesize(void) {
    return (int)sysconf(_SC_PAGESIZE);
}

int setenv(const char *name, const char *value, int overwrite) {
    int result = 0;
    if (overwrite || !getenv(name)) result = _putenv_s(name, value) ? -1 : 0;
    return result;
}

int unsetenv(const char *name) {
    return _putenv_s(name, "") ? -1 : 0;
}

/* ---- signals ---------------------------------------------------------------- */

#define SIGNAL_COUNT 32
static void (*signal_handlers[SIGNAL_COUNT])(int);

/* The CRT resets a handler to SIG_DFL before calling it; sigaction handlers
 * stay installed, so this puts itself back first. */
static void signal_trampoline(int sig) {
    signal(sig, signal_trampoline);
    void (*handler)(int) = (sig >= 0 && sig < SIGNAL_COUNT) ? signal_handlers[sig] : NULL;
    if (handler) handler(sig);
}

static int crt_delivers(int sig) {
    return sig == SIGINT || sig == SIGTERM || sig == SIGABRT || sig == SIGFPE || sig == SIGILL || sig == SIGSEGV;
}

int sigemptyset(sigset_t *set) {
    *set = 0;
    return 0;
}

int sigfillset(sigset_t *set) {
    *set = ~0ull;
    return 0;
}

int sigaddset(sigset_t *set, int sig) {
    *set |= 1ull << (sig & 63);
    return 0;
}

int sigprocmask(int how, const sigset_t *set, sigset_t *old) {
    (void)how;
    (void)set;
    if (old) *old = 0;
    return 0;
}

int sigaction(int sig, const struct sigaction *act, struct sigaction *old) {
    int result = 0;
    if (sig <= 0 || sig >= SIGNAL_COUNT) {
        errno = EINVAL;
        result = -1;
    } else {
        if (old) {
            memset(old, 0, sizeof(*old));
            old->sa_handler = signal_handlers[sig] ? signal_handlers[sig] : SIG_DFL;
        }
        if (act) {
            int custom = act->sa_handler != SIG_DFL && act->sa_handler != SIG_IGN;
            signal_handlers[sig] = custom ? act->sa_handler : NULL;
            if (crt_delivers(sig)) signal(sig, custom ? signal_trampoline : act->sa_handler);
        }
    }
    return result;
}

int kill(int pid, int sig) {
    int result = -1;
    if (pid == _getpid()) {
        result = sig ? raise(sig) : 0;
    } else {
        HANDLE process = OpenProcess(sig ? PROCESS_TERMINATE : PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
        if (!process) {
            errno = ESRCH;
        } else {
            if (!sig || TerminateProcess(process, 128 + (UINT)sig)) result = 0;
            else errno = EPERM;
            CloseHandle(process);
        }
    }
    return result;
}

/* ---- terminal ---------------------------------------------------------------- */

static HANDLE console_output(void) {
    return GetStdHandle(STD_OUTPUT_HANDLE);
}

int tcgetattr(int fd, struct termios *t) {
    int result = -1;
    HANDLE in = fd_handle(fd);
    DWORD in_mode, out_mode = 0;
    if (in != INVALID_HANDLE_VALUE && GetConsoleMode(in, &in_mode)) {
        GetConsoleMode(console_output(), &out_mode);
        memset(t, 0, sizeof(*t));
        t->c_iflag = ICRNL;
        t->c_oflag = OPOST;
        t->c_cflag = CS8;
        t->c_lflag = ((in_mode & ENABLE_ECHO_INPUT) ? ECHO : 0) | ((in_mode & ENABLE_LINE_INPUT) ? ICANON : 0) |
                     ((in_mode & ENABLE_PROCESSED_INPUT) ? ISIG : 0) | IEXTEN;
        t->ds4_w32_in_mode = in_mode;
        t->ds4_w32_out_mode = out_mode;
        result = 0;
    } else {
        errno = ENOTTY;
    }
    return result;
}

/* Cooked mode puts back exactly the console modes tcgetattr saved in the
 * struct; raw mode asks the console for VT input sequences, which is what a
 * Unix line editor expects to read. */
int tcsetattr(int fd, int when, const struct termios *t) {
    (void)when;
    int result = -1;
    HANDLE in = fd_handle(fd);
    DWORD current;
    if (in != INVALID_HANDLE_VALUE && GetConsoleMode(in, &current)) {
        DWORD in_mode = t->ds4_w32_in_mode, out_mode = t->ds4_w32_out_mode;
        if (!(t->c_lflag & ICANON)) {
            in_mode = ENABLE_VIRTUAL_TERMINAL_INPUT | ((t->c_lflag & ISIG) ? ENABLE_PROCESSED_INPUT : 0);
            out_mode |= ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        }
        if (SetConsoleMode(in, in_mode)) {
            SetConsoleMode(console_output(), out_mode);
            result = 0;
        }
    }
    if (result != 0) errno = ENOTTY;
    return result;
}

int ioctl(int fd, unsigned long request, ...) {
    int result = -1;
    va_list args;
    va_start(args, request);
    if (is_socket(fd)) {
        /* FIONREAD and FIONBIO, with Winsock's own request numbers. */
        result = ioctlsocket((SOCKET)fd, (long)request, va_arg(args, u_long *)) == 0 ? 0 : socket_failed();
    } else if (request == TIOCGWINSZ) {
        struct winsize *size = va_arg(args, struct winsize *);
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (GetConsoleScreenBufferInfo(console_output(), &info)) {
            memset(size, 0, sizeof(*size));
            size->ws_col = (unsigned short)(info.srWindow.Right - info.srWindow.Left + 1);
            size->ws_row = (unsigned short)(info.srWindow.Bottom - info.srWindow.Top + 1);
            result = 0;
        } else {
            errno = ENOTTY;
        }
    } else {
        errno = EINVAL;
    }
    va_end(args);
    return result;
}

/* ---- strings and time ------------------------------------------------------ */

struct qsort_r_call {
    int (*compare)(const void *, const void *, void *);
    void *context;
};

/* qsort_s hands the context over first. */
static int qsort_r_adapter(void *raw, const void *a, const void *b) {
    const struct qsort_r_call *call = raw;
    return call->compare(a, b, call->context);
}

void qsort_r(void *base, size_t count, size_t size,
             int (*compare)(const void *, const void *, void *), void *context) {
    struct qsort_r_call call = { compare, context };
    qsort_s(base, count, size, qsort_r_adapter, &call);
}

char *strndup(const char *s, size_t n) {
    size_t len = strnlen(s, n);
    char *copy = malloc(len + 1);
    if (copy) {
        memcpy(copy, s, len);
        copy[len] = '\0';
    }
    return copy;
}

char *strcasestr(const char *haystack, const char *needle) {
    char *result = NULL;
    size_t len = strlen(needle);
    for (const char *p = haystack; !result && (*p || !len); p++) {
        if (_strnicmp(p, needle, len) == 0) result = (char *)p;
        if (!*p) break;
    }
    return result;
}

void *memmem(const void *haystack, size_t hlen, const void *needle, size_t nlen) {
    void *result = NULL;
    if (nlen == 0) {
        result = (void *)haystack;
    } else if (nlen <= hlen) {
        const char *h = haystack;
        for (size_t i = 0; !result && i + nlen <= hlen; i++)
            if (h[i] == *(const char *)needle && memcmp(h + i, needle, nlen) == 0) result = (void *)(h + i);
    }
    return result;
}

char *strsep(char **stringp, const char *delim) {
    char *start = *stringp;
    if (start) {
        char *end = start + strcspn(start, delim);
        if (*end) {
            *end = '\0';
            *stringp = end + 1;
        } else {
            *stringp = NULL;
        }
    }
    return start;
}

ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    ssize_t result = -1;
    if (!lineptr || !n || !stream) {
        errno = EINVAL;
    } else {
        size_t len = 0;
        int c = EOF, failed = 0;
        while (!failed && (c = fgetc(stream)) != EOF) {
            if (len + 2 > *n) {
                size_t capacity = *n ? *n * 2 : 128;
                char *grown = realloc(*lineptr, capacity);
                if (grown) {
                    *lineptr = grown;
                    *n = capacity;
                } else {
                    failed = 1;
                }
            }
            if (!failed) (*lineptr)[len++] = (char)c;
            if (c == '\n') break;
        }
        if (!failed && len > 0) {
            (*lineptr)[len] = '\0';
            result = (ssize_t)len;
        }
    }
    return result;
}

/* A temporary file the CRT deletes on close stands in for the buffer.  A
 * stream opened for reading starts with the buffer's bytes; one opened for
 * writing is remembered here and copied into its buffer by ds4_w32_fclose,
 * with the NUL POSIX appends when there is room for it.  An open_memstream
 * stream gets its malloc'd buffer at fclose only, not at every fflush. */
#undef fclose
enum { fmem_slots = 16 };
static struct { FILE *file; char *buf; size_t size; char **out; size_t *outlen; } fmem_open[fmem_slots];
static SRWLOCK fmem_lock = SRWLOCK_INIT;

static FILE *fmem_register(FILE *file, char *buf, size_t size, char **out, size_t *outlen) {
    FILE *result = NULL;
    AcquireSRWLockExclusive(&fmem_lock);
    for (int i = 0; i < fmem_slots && !result; i++) {
        if (!fmem_open[i].file) {
            fmem_open[i].file = file;
            fmem_open[i].buf = buf;
            fmem_open[i].size = size;
            fmem_open[i].out = out;
            fmem_open[i].outlen = outlen;
            result = file;
        }
    }
    ReleaseSRWLockExclusive(&fmem_lock);
    if (!result) errno = EMFILE;
    return result;
}

FILE *fmemopen(void *buf, size_t size, const char *mode) {
    FILE *result = NULL;
    if (!buf || !mode || (mode[0] != 'r' && mode[0] != 'w')) {
        errno = EINVAL;
    } else {
        FILE *file = tmpfile();
        if (file && mode[0] == 'r') {
            if (fwrite(buf, 1, size, file) == size && fseek(file, 0, SEEK_SET) == 0) result = file;
        } else if (file) {
            result = fmem_register(file, buf, size, NULL, NULL);
        }
        if (file && !result) fclose(file);
    }
    return result;
}

FILE *open_memstream(char **ptr, size_t *sizeloc) {
    FILE *result = NULL;
    if (!ptr || !sizeloc) {
        errno = EINVAL;
    } else {
        FILE *file = tmpfile();
        if (file) result = fmem_register(file, NULL, 0, ptr, sizeloc);
        if (file && !result) fclose(file);
    }
    return result;
}

int ds4_w32_fclose(FILE *stream) {
    char *buf = NULL;
    size_t size = 0;
    char **out = NULL;
    size_t *outlen = NULL;
    AcquireSRWLockExclusive(&fmem_lock);
    for (int i = 0; stream && i < fmem_slots && !buf && !out; i++) {
        if (fmem_open[i].file == stream) {
            buf = fmem_open[i].buf;
            size = fmem_open[i].size;
            out = fmem_open[i].out;
            outlen = fmem_open[i].outlen;
            fmem_open[i].file = NULL;
        }
    }
    ReleaseSRWLockExclusive(&fmem_lock);
    int failed = 0;
    if (out) {
        /* ftell is 32-bit here; a snapshot can pass 2 GB. */
        const long long written = fflush(stream) == 0 ? _ftelli64(stream) : -1;
        char *copy = written >= 0 ? malloc((size_t)written + 1) : NULL;
        if (!copy || _fseeki64(stream, 0, SEEK_SET) != 0 ||
            fread(copy, 1, (size_t)written, stream) != (size_t)written) {
            free(copy);
            failed = 1;
            errno = ENOMEM;
        } else {
            copy[written] = '\0';
            *out = copy;
            *outlen = (size_t)written;
        }
    } else if (buf) {
        const long written = fflush(stream) == 0 ? ftell(stream) : -1;
        if (written < 0 || (size_t)written > size || fseek(stream, 0, SEEK_SET) != 0 ||
            fread(buf, 1, (size_t)written, stream) != (size_t)written) {
            failed = 1;
            errno = ENOSPC;
        } else if ((size_t)written < size) {
            buf[written] = '\0';
        }
    }
    return fclose(stream) != 0 || failed ? EOF : 0;
}

time_t timegm(struct tm *tm) {
    return _mkgmtime(tm);
}

struct tm *ds4_w32_localtime_r(const time_t *t, struct tm *out) {
    return localtime_s(out, t) == 0 ? out : NULL;
}

struct tm *ds4_w32_gmtime_r(const time_t *t, struct tm *out) {
    return gmtime_s(out, t) == 0 ? out : NULL;
}

#endif /* _WIN32 && !__CYGWIN__ */
