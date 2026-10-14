/* POSIX calls the engine uses that MinGW-w64 lacks, for the Windows build.
 *
 * The Makefile force-includes this header into every C file of a MinGW build
 * (-include win32/ds4_win32.h) and puts win32/include ahead of the system
 * headers, where empty stand-ins for <sys/mman.h> and friends live.  The
 * engine sources stay as they are.
 *
 * This header must not include <windows.h>: its macros (min, max, near, far,
 * small, ERROR, interface) collide with ordinary identifiers in the engine.
 * Everything Win32 stays inside win32/ds4_win32.c. */
#ifndef DS4_WIN32_H
#define DS4_WIN32_H

#if defined(_WIN32) && !defined(__CYGWIN__)

/* MinGW's printf family only understands %zu and %lld through its own
 * implementation, which this selects.  It has to come before any system
 * header. */
#ifndef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1
#endif

/* read(), write() and close() are replaced outright (see below), so the CRT's
 * own declarations, which take a 32-bit count, are parked under other names. */
#define read ds4_w32_crt_read
#define write ds4_w32_crt_write
#define close ds4_w32_crt_close

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <fcntl.h>
#include <io.h>
#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#undef read
#undef write
#undef close

#ifdef __cplusplus
extern "C" {
#endif

/* ---- open(2) flags without a Windows meaning ---------------------------- */

/* Binary mode needs no flag here: ds4_win32.c makes it the CRT's default,
 * because the engine reads GGUF and writes KV snapshots through plain open()
 * and fopen(), and text mode would rewrite 0x0A bytes. */
#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif
#ifndef O_NONBLOCK
#define O_NONBLOCK 0x4000
#endif
#ifndef O_NOCTTY
#define O_NOCTTY 0
#endif
#ifndef O_SYNC
#define O_SYNC 0
#endif

/* ---- memory mapping ------------------------------------------------------ */

#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_NORESERVE 0x4000
#define MAP_POPULATE 0x8000
#define MAP_HUGETLB 0
#define MAP_FAILED ((void *)-1)

#define MADV_NORMAL 0
#define MADV_RANDOM 1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_FREE 8
#define MADV_HUGEPAGE 14
#define MADV_NOHUGEPAGE 15
#define MADV_DONTDUMP 16
#define POSIX_MADV_NORMAL MADV_NORMAL
#define POSIX_MADV_RANDOM MADV_RANDOM
#define POSIX_MADV_SEQUENTIAL MADV_SEQUENTIAL
#define POSIX_MADV_WILLNEED MADV_WILLNEED
#define POSIX_MADV_DONTNEED MADV_DONTNEED

#define MS_ASYNC 1
#define MS_SYNC 4
#define MS_INVALIDATE 2

void *mmap(void *addr, size_t len, int prot, int flags, int fd, int64_t off);
int munmap(void *addr, size_t len);
int madvise(void *addr, size_t len, int advice);
int posix_madvise(void *addr, size_t len, int advice);
int mlock(const void *addr, size_t len);
int munlock(const void *addr, size_t len);
int msync(void *addr, size_t len, int flags);
int mprotect(void *addr, size_t len, int prot);

/* ---- files ---------------------------------------------------------------- */

/* The CRT's read() and write() take a 32-bit count and know nothing of
 * sockets.  ds4_win32.c defines these three itself, under their real names so
 * that no macro touches struct members called read or write: they take a
 * size_t, cap one transfer at 1 GiB like a short read, and route a socket to
 * Winsock (see ds4_win32_net.h). */
ssize_t read(int fd, void *buf, size_t n);
ssize_t write(int fd, const void *buf, size_t n);
int close(int fd);

ssize_t pread(int fd, void *buf, size_t n, int64_t off);
ssize_t pwrite(int fd, const void *buf, size_t n, int64_t off);
int fchmod(int fd, int mode);
int fsync(int fd);
int fdatasync(int fd);

#define LOCK_SH 1
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8
int flock(int fd, int op);

/* A POSIX rename replaces its target; the CRT's refuses to. */
int ds4_w32_rename(const char *from, const char *to);
#define rename ds4_w32_rename

int ds4_w32_mkdir(const char *path, ...);
#define mkdir ds4_w32_mkdir

char *mkdtemp(char *tmpl);
char *realpath(const char *path, char *resolved);
int symlink(const char *target, const char *linkpath);
ssize_t readlink(const char *path, char *buf, size_t n);
#define lstat stat

/* 64-bit offsets and sizes: off_t is 32 bits unless _FILE_OFFSET_BITS=64 is
 * set, which the Makefile does; these keep stat in step with it. */
#ifndef S_ISLNK
#define S_ISLNK(m) 0
#endif
#ifndef S_ISSOCK
#define S_ISSOCK(m) 0
#endif

/* ---- process and system --------------------------------------------------- */

#define _SC_PAGESIZE 1
#define _SC_PAGE_SIZE _SC_PAGESIZE
#define _SC_NPROCESSORS_ONLN 2
#define _SC_NPROCESSORS_CONF 3
#define _SC_PHYS_PAGES 4
#define _SC_AVPHYS_PAGES 5
long sysconf(int name);
int getpagesize(void);

int setenv(const char *name, const char *value, int overwrite);
int unsetenv(const char *name);

/* fcntl covers what the engine asks of it: F_SETFD/FD_CLOEXEC on files and
 * O_NONBLOCK on sockets.  Anything else fails with EINVAL. */
#define F_GETFD 1
#define F_SETFD 2
#define F_GETFL 3
#define F_SETFL 4
#define FD_CLOEXEC 1
int fcntl(int fd, int cmd, ...);

/* ---- signals ---------------------------------------------------------------- */

/* Windows delivers SIGINT, SIGTERM and SIGBREAK only.  The others get numbers
 * so that handlers can be installed for them; they are never raised. */
#ifndef SIGHUP
#define SIGHUP 1
#endif
#ifndef SIGQUIT
#define SIGQUIT 3
#endif
#ifndef SIGKILL
#define SIGKILL 9
#endif
#ifndef SIGPIPE
#define SIGPIPE 13
#endif
#ifndef SIGALRM
#define SIGALRM 14
#endif
#ifndef SIGUSR1
#define SIGUSR1 10
#endif
#ifndef SIGUSR2
#define SIGUSR2 12
#endif
#ifndef SIGCHLD
#define SIGCHLD 17
#endif
#ifndef SIGCONT
#define SIGCONT 18
#endif
#ifndef SIGSTOP
#define SIGSTOP 19
#endif
#ifndef SIGTSTP
#define SIGTSTP 20
#endif
#ifndef SIGWINCH
#define SIGWINCH 28
#endif

typedef unsigned long long ds4_w32_sigset_t;
#define sigset_t ds4_w32_sigset_t
#define SA_RESTART 0
#define SA_NODEFER 0
#define SA_RESETHAND 0
#define SA_SIGINFO 0
#define SIG_BLOCK 0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2
struct sigaction {
    void (*sa_handler)(int);
    sigset_t sa_mask;
    int sa_flags;
};
int sigemptyset(sigset_t *set);
int sigfillset(sigset_t *set);
int sigaddset(sigset_t *set, int sig);
int sigaction(int sig, const struct sigaction *act, struct sigaction *old);
int sigprocmask(int how, const sigset_t *set, sigset_t *old);
int kill(int pid, int sig);

/* ---- terminal ---------------------------------------------------------------- */

/* Enough of termios for linenoise's raw mode on a Windows console, which has
 * understood VT sequences in both directions since Windows 10. */
typedef unsigned int tcflag_t;
typedef unsigned char cc_t;
#define NCCS 20
struct termios {
    tcflag_t c_iflag, c_oflag, c_cflag, c_lflag;
    cc_t c_cc[NCCS];
    unsigned long ds4_w32_in_mode, ds4_w32_out_mode;
};
#define BRKINT 0x0001
#define ICRNL 0x0002
#define INPCK 0x0004
#define ISTRIP 0x0008
#define IXON 0x0010
#define OPOST 0x0001
#define CS8 0x0001
#define ECHO 0x0001
#define ICANON 0x0002
#define IEXTEN 0x0004
#define ISIG 0x0008
#define VMIN 0
#define VTIME 1
#define TCSANOW 0
#define TCSADRAIN 1
#define TCSAFLUSH 2
int tcgetattr(int fd, struct termios *t);
int tcsetattr(int fd, int when, const struct termios *t);

struct winsize {
    unsigned short ws_row, ws_col, ws_xpixel, ws_ypixel;
};
#define TIOCGWINSZ 0x5413
int ioctl(int fd, unsigned long request, ...);

int dprintf(int fd, const char *format, ...);

/* ---- strings and time ------------------------------------------------------ */

/* glibc's argument order: the comparison takes the context last. */
void qsort_r(void *base, size_t count, size_t size,
             int (*compare)(const void *, const void *, void *), void *context);
char *strndup(const char *s, size_t n);
char *strcasestr(const char *haystack, const char *needle);
void *memmem(const void *haystack, size_t hlen, const void *needle, size_t nlen);
char *strsep(char **stringp, const char *delim);
ssize_t getline(char **lineptr, size_t *n, FILE *stream);
FILE *fmemopen(void *buf, size_t size, const char *mode);
FILE *open_memstream(char **ptr, size_t *sizeloc);
/* What was written to an fmemopen stream reaches its buffer when the stream
 * is closed, so every fclose goes through the shim. */
int ds4_w32_fclose(FILE *stream);
#define fclose ds4_w32_fclose
time_t timegm(struct tm *tm);
#ifndef localtime_r
struct tm *ds4_w32_localtime_r(const time_t *t, struct tm *out);
struct tm *ds4_w32_gmtime_r(const time_t *t, struct tm *out);
#define localtime_r ds4_w32_localtime_r
#define gmtime_r ds4_w32_gmtime_r
#endif

#ifdef __cplusplus
}
#endif

#endif /* _WIN32 && !__CYGWIN__ */
#endif /* DS4_WIN32_H */
