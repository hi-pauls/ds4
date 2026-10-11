// What the CUDA C++ test programs use from POSIX and the MSVC runtime lacks;
// force-included after ds4_msvc_posix.h when nvcc builds them.
//
// The tests unlink a temporary file while they still hold it open. Windows
// allows that only when every open handle shares deletion and the delete asks
// for POSIX semantics (Windows 10 1709 and later), so mkstemp and unlink are
// both replaced, as win32/ds4_test_win32.h does for the MinGW tests.
#pragma once

// Every header that declares unlink has to be in before the macro below, or
// its declaration is renamed along with the calls.
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef near
#undef far
#undef small
#undef IN
#undef OUT

static inline int setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value) == 0 ? 0 : -1;
}

static inline int unsetenv(const char *name) {
    return _putenv_s(name, "") == 0 ? 0 : -1;
}

static inline int ds4_test_mkstemp(char *name) {
    int fd = -1;
    const size_t length = strlen(name);
    for (unsigned attempt = 0; attempt < 100u && fd < 0 && length >= 6; attempt++) {
        char *suffix = name + length - 6;
        unsigned seed = (unsigned)GetTickCount() * 2654435761u + GetCurrentProcessId() + attempt * 7919u;
        for (int i = 0; i < 6; i++, seed /= 36u) suffix[i] = "abcdefghijklmnopqrstuvwxyz0123456789"[seed % 36u];
        const HANDLE file = CreateFileA(name, GENERIC_READ | GENERIC_WRITE,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                                        CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file != INVALID_HANDLE_VALUE) fd = _open_osfhandle((intptr_t)file, _O_RDWR | _O_BINARY);
    }
    if (fd < 0) errno = EEXIST;
    return fd;
}

static inline int ds4_test_unlink(const char *name) {
    int result = -1;
    struct { DWORD Flags; } disposition = { 0x1 | 0x2 };   // delete, POSIX semantics
    const HANDLE file = CreateFileA(name, DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file != INVALID_HANDLE_VALUE) {
        if (SetFileInformationByHandle(file, (FILE_INFO_BY_HANDLE_CLASS)21, &disposition, sizeof(disposition)))
            result = 0;
        CloseHandle(file);
    }
    if (result != 0) errno = EACCES;
    return result;
}

#define mkstemp(name) ds4_test_mkstemp(name)
#define unlink(name) ds4_test_unlink(name)
