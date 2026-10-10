// Windows shims for the test programs only; force-included after ds4_win32.h.
//
// The tests take page-aligned blocks from posix_memalign and give them back to
// free. The universal CRT has no posix_memalign, and what _aligned_malloc
// returns must go to _aligned_free: handing it to free corrupts the heap. So the
// blocks are remembered here and free is routed through a check, which leaves
// every other pointer to the real free.
#pragma once

#include <errno.h>
#include <stdlib.h>
#include <malloc.h>

enum { ds4_test_aligned_slots = 256 };
static void *ds4_test_aligned[ds4_test_aligned_slots];

static inline int ds4_test_posix_memalign(void **out, size_t alignment, size_t size)
{
    int result = ENOMEM;
    int slot = -1;
    for (int i = 0; i < ds4_test_aligned_slots && slot < 0; i++)
        if (!ds4_test_aligned[i]) slot = i;
    if (slot >= 0)
        if ((*out = _aligned_malloc(size, alignment)) != NULL)
        {
            ds4_test_aligned[slot] = *out;
            result = 0;
        }
    return result;
}

static inline void ds4_test_free(void *block)
{
    int slot = -1;
    for (int i = 0; block && i < ds4_test_aligned_slots && slot < 0; i++)
        if (ds4_test_aligned[i] == block) slot = i;
    if (slot >= 0)
    {
        ds4_test_aligned[slot] = NULL;
        _aligned_free(block);
    }
    else
        free(block);
}

#define posix_memalign(out, alignment, size) ds4_test_posix_memalign((out), (alignment), (size))
#define free(block) ds4_test_free(block)

// The tests also unlink a temporary file while they still hold it open. Windows
// allows that only when every open handle shares deletion and the delete asks
// for POSIX semantics (Windows 10 1709 and later): the name goes at once, the
// handles stay good. mkstemp from the toolchain opens without that sharing, so
// both ends are replaced.
#include <fcntl.h>
#include <io.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// windows.h claims these as macros; the tests use them as ordinary names.
#undef near
#undef far
#undef small
#undef min
#undef max
#undef IN
#undef OUT

static inline int ds4_test_mkstemp(char *name)
{
    int fd = -1;
    const size_t length = strlen(name);
    for (unsigned attempt = 0; attempt < 100u && fd < 0 && length >= 6; attempt++)
    {
        char *suffix = name + length - 6;
        unsigned seed = (unsigned)GetTickCount() * 2654435761u + GetCurrentProcessId() + attempt * 7919u;
        for (int i = 0; i < 6; i++, seed /= 36u)
            suffix[i] = "abcdefghijklmnopqrstuvwxyz0123456789"[seed % 36u];
        const HANDLE file = CreateFileA(name, GENERIC_READ | GENERIC_WRITE,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                                        CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file != INVALID_HANDLE_VALUE) fd = _open_osfhandle((intptr_t)file, _O_RDWR | _O_BINARY);
    }
    if (fd < 0) errno = EEXIST;
    return fd;
}

static inline int ds4_test_unlink(const char *name)
{
    int result = -1;
    struct { DWORD Flags; } disposition = { 0x1 | 0x2 };   // delete, POSIX semantics
    const HANDLE file = CreateFileA(name, DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file != INVALID_HANDLE_VALUE)
    {
        if (SetFileInformationByHandle(file, (FILE_INFO_BY_HANDLE_CLASS)21, &disposition, sizeof(disposition)))
            result = 0;
        CloseHandle(file);
    }
    if (result != 0) errno = EACCES;
    return result;
}

#define mkstemp(name) ds4_test_mkstemp(name)
#define unlink(name) ds4_test_unlink(name)
