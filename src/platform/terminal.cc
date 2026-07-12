#include "jetstream/platform.hh"

#include <cstdio>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>
#undef ERROR
#undef FATAL
#elif !defined(JST_OS_BROWSER)
#include <unistd.h>
#endif

namespace Jetstream::Platform {

bool PrepareStandardOutputForAnsi() {
#if defined(JST_OS_IOS) || defined(JST_OS_BROWSER)
    return false;
#elif defined(JST_OS_WINDOWS)
    const int descriptor = _fileno(stdout);
    if (descriptor < 0 || _isatty(descriptor) == 0) {
        return false;
    }

    const auto nativeHandle = _get_osfhandle(descriptor);
    if (nativeHandle == -1) {
        return false;
    }

    const auto outputHandle = reinterpret_cast<HANDLE>(nativeHandle);
    DWORD mode = 0;
    if (!GetConsoleMode(outputHandle, &mode)) {
        return false;
    }

    constexpr DWORD requiredMode =
        ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    if ((mode & requiredMode) == requiredMode) {
        return true;
    }

    return SetConsoleMode(outputHandle, mode | requiredMode) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

}  // namespace Jetstream::Platform
