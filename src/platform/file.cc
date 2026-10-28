#include "jetstream/platform.hh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string_view>
#include <system_error>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <sddl.h>
#include <fcntl.h>
#include <io.h>
#undef ERROR
#undef FATAL
#else
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace Jetstream::Platform {

namespace {

std::FILE* OpenTemporaryFile(const std::filesystem::path& path, bool ownerOnly) {
#if defined(JST_OS_WINDOWS)
    if (!ownerOnly) {
        return _wfopen(path.c_str(), L"wb");
    }

    // Use a protected DACL granting access only to the file owner from creation.
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            L"D:P(A;;GA;;;OW)", SDDL_REVISION_1, &descriptor, nullptr)) {
        return nullptr;
    }
    SECURITY_ATTRIBUTES security = {sizeof(SECURITY_ATTRIBUTES), descriptor, FALSE};
    const HANDLE handle = CreateFileW(path.c_str(), GENERIC_WRITE, 0, &security,
                                      CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    LocalFree(descriptor);
    if (handle == INVALID_HANDLE_VALUE) {
        return nullptr;
    }

    const int fd = _open_osfhandle(reinterpret_cast<std::intptr_t>(handle), _O_WRONLY | _O_BINARY);
    if (fd < 0) {
        CloseHandle(handle);
    } else if (auto* file = _fdopen(fd, "wb")) {
        return file;
    } else {
        _close(fd);
    }
#else
    // Exclusive creation prevents reusing an inode another reader already opened.
    const int flags = O_WRONLY | O_CREAT | (ownerOnly ? O_EXCL : O_TRUNC);
    const int fd = open(path.c_str(), flags, ownerOnly ? 0600 : 0666);
    if (fd < 0) {
        return nullptr;
    }
    if (auto* file = fdopen(fd, "wb")) {
        return file;
    }
    close(fd);
#endif

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return nullptr;
}

}  // namespace

Result WriteFileAtomic(const std::string& destination, std::string_view bytes, bool ownerOnly) {
    const auto path = PathFromUtf8(destination);
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            JST_ERROR("[FILE] Cannot create directory '{}'.", PathToUtf8(parent));
            return Result::ERROR;
        }
    }

    auto temporary = path;
    temporary += ".tmp";
    const auto discard = [&]() {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
    };

    auto* file = OpenTemporaryFile(temporary, ownerOnly);
    if (!file) {
        JST_ERROR("[FILE] Cannot open temporary file '{}'.", PathToUtf8(temporary));
        return Result::ERROR;
    }

    const bool written = bytes.empty() || std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
    const bool closed = std::fclose(file) == 0;
    if (!written || !closed) {
        discard();
        JST_ERROR("[FILE] Failed to write temporary file '{}'.", PathToUtf8(temporary));
        return Result::ERROR;
    }

    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        discard();
        JST_ERROR("[FILE] Failed to replace '{}'.", PathToUtf8(path));
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

struct InputStream::Impl {
#if defined(JST_OS_WINDOWS)
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
    int originalFlags = 0;
    bool fifo = false;
#endif
};

InputStream::InputStream() : impl(std::make_unique<Impl>()) {}

InputStream::~InputStream() {
    close();
}

Result InputStream::open(const std::string& path, std::string& error) {
    close();

#if defined(JST_OS_WINDOWS)
    const HANDLE handle = CreateFileW(PathFromUtf8(path).c_str(), GENERIC_READ,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        error = std::system_category().message(static_cast<int>(GetLastError()));
        return Result::ERROR;
    }
    impl->handle = handle;
#else
    const int fd = ::open(PathFromUtf8(path).c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        error = std::strerror(errno);
        return Result::ERROR;
    }

    const int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        error = std::strerror(errno);
        ::close(fd);
        return Result::ERROR;
    }
    struct stat info = {};
    impl->fd = fd;
    impl->originalFlags = flags;
    impl->fifo = fstat(fd, &info) == 0 && S_ISFIFO(info.st_mode);
#endif

    return Result::SUCCESS;
}

Result InputStream::read(void* data, const U64 size, U64& received) {
    received = 0;
    if (!isOpen()) {
        return Result::ERROR;
    }

#if defined(JST_OS_WINDOWS)
    U64 capacity = std::min<U64>(size, MAXDWORD);
    if (GetFileType(impl->handle) == FILE_TYPE_PIPE) {
        DWORD available = 0;
        if (!PeekNamedPipe(impl->handle, nullptr, 0, nullptr, &available, nullptr)) {
            return GetLastError() == ERROR_BROKEN_PIPE ? Result::INCOMPLETE : Result::ERROR;
        }
        if (available == 0) {
            return Result::TIMEOUT;
        }
        capacity = std::min<U64>(capacity, available);
    }

    DWORD count = 0;
    if (!ReadFile(impl->handle, data, static_cast<DWORD>(capacity), &count, nullptr)) {
        return GetLastError() == ERROR_BROKEN_PIPE ? Result::INCOMPLETE : Result::ERROR;
    }
    if (count == 0) {
        return Result::INCOMPLETE;
    }
    received = count;
    return Result::SUCCESS;
#else
    const auto capacity = static_cast<std::size_t>(
        std::min<U64>(size, static_cast<U64>(std::numeric_limits<ssize_t>::max())));
    while (true) {
        const ssize_t count = ::read(impl->fd, data, capacity);
        if (count > 0) {
            received = static_cast<U64>(count);
            return Result::SUCCESS;
        }
        if (count == 0) {
            if (!impl->fifo) {
                return Result::INCOMPLETE;
            }
            pollfd descriptor = {impl->fd, POLLIN, 0};
            const bool hungUp = ::poll(&descriptor, 1, 0) > 0 && (descriptor.revents & POLLHUP);
            return hungUp ? Result::INCOMPLETE : Result::TIMEOUT;
        }
        if (errno == EINTR) {
            continue;
        }
        return (errno == EAGAIN || errno == EWOULDBLOCK) ? Result::TIMEOUT : Result::ERROR;
    }
#endif
}

Result InputStream::wait(const U64 timeoutMilliseconds) {
    if (!isOpen()) {
        return Result::ERROR;
    }

#if defined(JST_OS_WINDOWS)
    if (GetFileType(impl->handle) != FILE_TYPE_PIPE) {
        return Result::SUCCESS;
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeoutMilliseconds);
    while (true) {
        DWORD available = 0;
        if (!PeekNamedPipe(impl->handle, nullptr, 0, nullptr, &available, nullptr) || available > 0) {
            return Result::SUCCESS;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return Result::TIMEOUT;
        }
        Sleep(1);
    }
#else
    pollfd descriptor = {impl->fd, POLLIN, 0};
    const int timeout = static_cast<int>(std::min<U64>(timeoutMilliseconds,
                                                       std::numeric_limits<int>::max()));
    const int ready = ::poll(&descriptor, 1, timeout);
    if (ready == 0 || (ready < 0 && errno == EINTR)) {
        return Result::TIMEOUT;
    }
    return Result::SUCCESS;
#endif
}

void InputStream::close() {
#if defined(JST_OS_WINDOWS)
    if (impl->handle != INVALID_HANDLE_VALUE) {
        CloseHandle(impl->handle);
        impl->handle = INVALID_HANDLE_VALUE;
    }
#else
    if (impl->fd >= 0) {
        fcntl(impl->fd, F_SETFL, impl->originalFlags);
        ::close(impl->fd);
        impl->fd = -1;
    }
#endif
}

bool InputStream::isOpen() const {
#if defined(JST_OS_WINDOWS)
    return impl->handle != INVALID_HANDLE_VALUE;
#else
    return impl->fd >= 0;
#endif
}

}  // namespace Jetstream::Platform
