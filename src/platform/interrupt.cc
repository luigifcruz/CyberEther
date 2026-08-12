#include "jetstream/platform.hh"

#include <atomic>
#include <cerrno>
#include <cstdlib>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#else
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace Jetstream::Platform {

namespace {

std::atomic<void (*)() noexcept> interruptHandler{nullptr};
static_assert(decltype(interruptHandler)::is_always_lock_free);
bool interruptHandlerInstalled = false;

#if defined(JST_OS_WINDOWS)

HANDLE interruptOutput = INVALID_HANDLE_VALUE;
bool interruptConsoleAttached = false;

BOOL WINAPI ConsoleInterruptHandler(const DWORD signal) noexcept {
    if (signal != CTRL_C_EVENT) {
        return FALSE;
    }

    if (const auto handler = interruptHandler.load(std::memory_order_relaxed)) {
        handler();
        return TRUE;
    }

    return FALSE;
}

#else

using WriteFunction = ssize_t (*)(int, const void*, std::size_t);
using ExitFunction = void (*)(int);

WriteFunction interruptWrite = write;
ExitFunction interruptExit = std::_Exit;
int interruptOutput = -1;
struct sigaction previousInterruptAction = {};

void SignalInterruptHandler(int) noexcept {
    const int error = errno;
    if (const auto handler = interruptHandler.load(std::memory_order_relaxed)) {
        handler();
    }
    errno = error;
}

#endif

}  // namespace

bool InstallInterruptHandler(void (*handler)() noexcept) {
    if (!handler || interruptHandlerInstalled) {
        return false;
    }

    interruptHandler.store(handler, std::memory_order_relaxed);

#if defined(JST_OS_WINDOWS)
    if (GetConsoleCP() == 0) {
        if (!AttachConsole(ATTACH_PARENT_PROCESS)) {
            interruptHandler.store(nullptr, std::memory_order_relaxed);
            return false;
        }
        interruptConsoleAttached = true;
    }

    interruptOutput = CreateFileW(
        L"CONOUT$",
        GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (!SetConsoleCtrlHandler(ConsoleInterruptHandler, TRUE)) {
        if (interruptOutput != INVALID_HANDLE_VALUE) {
            (void)CloseHandle(interruptOutput);
            interruptOutput = INVALID_HANDLE_VALUE;
        }
        if (interruptConsoleAttached) {
            (void)FreeConsole();
            interruptConsoleAttached = false;
        }
        interruptHandler.store(nullptr, std::memory_order_relaxed);
        return false;
    }
#else
    interruptOutput = open("/dev/tty", O_WRONLY | O_NONBLOCK | O_CLOEXEC);

    struct sigaction action = {};
    action.sa_handler = SignalInterruptHandler;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, &previousInterruptAction) != 0) {
        if (interruptOutput >= 0) {
            (void)close(interruptOutput);
            interruptOutput = -1;
        }
        interruptHandler.store(nullptr, std::memory_order_relaxed);
        return false;
    }
#endif

    interruptHandlerInstalled = true;
    return true;
}

void UninstallInterruptHandler() {
    if (!interruptHandlerInstalled) {
        return;
    }

#if defined(JST_OS_WINDOWS)
    if (!SetConsoleCtrlHandler(ConsoleInterruptHandler, FALSE)) {
        return;
    }
#else
    if (sigaction(SIGINT, &previousInterruptAction, nullptr) != 0) {
        return;
    }
#endif

    interruptHandlerInstalled = false;
    interruptHandler.store(nullptr, std::memory_order_relaxed);

#if defined(JST_OS_WINDOWS)
    if (interruptOutput != INVALID_HANDLE_VALUE) {
        (void)CloseHandle(interruptOutput);
        interruptOutput = INVALID_HANDLE_VALUE;
    }
    if (interruptConsoleAttached) {
        (void)FreeConsole();
        interruptConsoleAttached = false;
    }
#else
    if (interruptOutput >= 0) {
        (void)close(interruptOutput);
        interruptOutput = -1;
    }
#endif
}

void WriteInterruptMessage(const char* message, const std::size_t size) noexcept {
#if defined(JST_OS_WINDOWS)
    if (interruptOutput == nullptr || interruptOutput == INVALID_HANDLE_VALUE) {
        return;
    }

    DWORD written = 0;
    const BOOL result = WriteFile(
        interruptOutput,
        message,
        static_cast<DWORD>(size),
        &written,
        nullptr
    );
#else
    if (interruptOutput < 0) {
        return;
    }
    const ssize_t result = interruptWrite(interruptOutput, message, size);
#endif
    (void)result;
}

[[noreturn]] void ForceTerminate(const I32 status) noexcept {
#if defined(JST_OS_WINDOWS)
    (void)TerminateProcess(GetCurrentProcess(), static_cast<UINT>(status));
#else
    interruptExit(status);
#endif
    std::_Exit(status);
}

}  // namespace Jetstream::Platform
