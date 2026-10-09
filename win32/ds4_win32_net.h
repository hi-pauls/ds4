/* BSD sockets over Winsock for the Windows build.
 *
 * The stand-ins for <sys/socket.h>, <netinet/in.h>, <poll.h> and friends in
 * win32/include all include this header, so only the sources that use the
 * network see Winsock at all.
 *
 * Two things differ from Unix and are hidden here:
 *
 *   - A socket is a kernel handle, not a CRT file descriptor.  The read(),
 *     write() and close() wrappers in ds4_win32.h send a socket to recv(),
 *     send() and closesocket() and anything else to the CRT.
 *   - Winsock reports through WSAGetLastError(); the wrappers translate that
 *     into errno, which is what the engine tests.
 *
 * socketpair() is a connected pair of loopback TCP sockets, so whatever the
 * server waits on in poll() is a socket. */
#ifndef DS4_WIN32_NET_H
#define DS4_WIN32_NET_H

#if defined(_WIN32) && !defined(__CYGWIN__)

#include "ds4_win32.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <errno.h>

/* <windows.h> leftovers that are ordinary identifiers in the engine. */
#undef near
#undef far
#undef small
#undef interface
#undef ERROR
#undef IN
#undef OUT
#undef min
#undef max

#ifdef __cplusplus
extern "C" {
#endif

/* As on Linux, where the two are one value: the wrappers report EAGAIN. */
#undef EWOULDBLOCK
#define EWOULDBLOCK EAGAIN

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0x40
#endif
#ifndef SHUT_RD
#define SHUT_RD SD_RECEIVE
#define SHUT_WR SD_SEND
#define SHUT_RDWR SD_BOTH
#endif
#ifndef SO_REUSEPORT
#define SO_REUSEPORT 0
#endif
#ifndef AF_UNIX
#define AF_UNIX 1
#endif
#ifndef AF_LOCAL
#define AF_LOCAL AF_UNIX
#endif

typedef long suseconds_t;

struct iovec {
    void *iov_base;
    size_t iov_len;
};

struct msghdr {
    void *msg_name;
    socklen_t msg_namelen;
    struct iovec *msg_iov;
    int msg_iovlen;
    void *msg_control;
    socklen_t msg_controllen;
    int msg_flags;
};
ssize_t sendmsg(int fd, const struct msghdr *message, int flags);
ssize_t recvmsg(int fd, struct msghdr *message, int flags);

int ds4_w32_socket(int domain, int type, int protocol);
int ds4_w32_socketpair(int domain, int type, int protocol, int pair[2]);
int ds4_w32_accept(int fd, struct sockaddr *addr, socklen_t *len);
int ds4_w32_bind(int fd, const struct sockaddr *addr, socklen_t len);
int ds4_w32_listen(int fd, int backlog);
int ds4_w32_connect(int fd, const struct sockaddr *addr, socklen_t len);
ssize_t ds4_w32_send(int fd, const void *buf, size_t n, int flags);
ssize_t ds4_w32_recv(int fd, void *buf, size_t n, int flags);
int ds4_w32_shutdown(int fd, int how);
int ds4_w32_setsockopt(int fd, int level, int name, const void *value, socklen_t len);
int ds4_w32_getsockopt(int fd, int level, int name, void *value, socklen_t *len);
int ds4_w32_getsockname(int fd, struct sockaddr *addr, socklen_t *len);
int ds4_w32_getpeername(int fd, struct sockaddr *addr, socklen_t *len);
int ds4_w32_poll(struct pollfd *fds, unsigned long count, int timeout_ms);
int ds4_w32_select(int nfds, fd_set *r, fd_set *w, fd_set *e, struct timeval *timeout);
ssize_t writev(int fd, const struct iovec *iov, int count);
ssize_t readv(int fd, const struct iovec *iov, int count);

#define socket ds4_w32_socket
#define socketpair ds4_w32_socketpair
#define accept ds4_w32_accept
#define bind ds4_w32_bind
#define listen ds4_w32_listen
#define connect ds4_w32_connect
#define send ds4_w32_send
#define recv ds4_w32_recv
#define shutdown ds4_w32_shutdown
#define setsockopt ds4_w32_setsockopt
#define getsockopt ds4_w32_getsockopt
#define getsockname ds4_w32_getsockname
#define getpeername ds4_w32_getpeername
#define poll ds4_w32_poll
#define select ds4_w32_select

#ifdef __cplusplus
}
#endif

#endif /* _WIN32 && !__CYGWIN__ */
#endif /* DS4_WIN32_NET_H */
