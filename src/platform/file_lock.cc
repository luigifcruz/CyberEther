#include "jetstream/platform.hh"

#include <cerrno>
#include <filesystem>
#include <memory>
#include <utility>

#if defined(JST_OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef ERROR
#undef FATAL
#elif !defined(JST_OS_BROWSER)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace Jetstream::Platform {

struct FileLock::Impl {
    std::string path;
    bool locked = false;

#if defined(JST_OS_WINDOWS)
    HANDLE handle = INVALID_HANDLE_VALUE;
#elif !defined(JST_OS_BROWSER)
    int fd = -1;
#endif

    static Result ensureParentDirectory(const std::string& path);
};

Result FileLock::Impl::ensureParentDirectory(const std::string& path) {
    const auto parent = PathFromUtf8(path).parent_path();
    if (parent.empty()) {
        return Result::SUCCESS;
    }

    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
        JST_ERROR("Failed to create file lock directory '{}'.", PathToUtf8(parent));
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

FileLock::FileLock() {
    impl = std::make_unique<Impl>();
}

FileLock::~FileLock() {
    release();
}

FileLock::FileLock(FileLock&&) noexcept = default;

FileLock& FileLock::operator=(FileLock&& other) noexcept {
    if (this != &other) {
        release();
        impl = std::move(other.impl);
    }

    return *this;
}

Result FileLock::acquire(const std::string& path, bool wait) {
    if (!impl) {
        impl = std::make_unique<Impl>();
    }

    auto& state = *impl;

    if (state.locked) {
        JST_ERROR("Cannot acquire file lock '{}' because this lock already owns '{}'.", path, state.path);
        return Result::ERROR;
    }

    JST_CHECK(Impl::ensureParentDirectory(path));

#if defined(JST_OS_WINDOWS)
    const auto lockPath = PathFromUtf8(path);
    HANDLE handle = CreateFileW(lockPath.c_str(),
                                GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr,
                                OPEN_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL,
                                nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        JST_ERROR("Failed to open file lock '{}' [Error: {}].", path, GetLastError());
        return Result::ERROR;
    }

    OVERLAPPED overlapped = {};
    DWORD flags = LOCKFILE_EXCLUSIVE_LOCK;
    if (!wait) {
        flags |= LOCKFILE_FAIL_IMMEDIATELY;
    }

    if (!LockFileEx(handle, flags, 0, MAXDWORD, MAXDWORD, &overlapped)) {
        const auto error = GetLastError();
        CloseHandle(handle);
        if (!wait && error == ERROR_LOCK_VIOLATION) {
            return Result::SKIP;
        }

        JST_ERROR("Failed to acquire file lock '{}' [Error: {}].", path, error);
        return Result::ERROR;
    }

    state.handle = handle;
#elif defined(JST_OS_BROWSER)
    (void)wait;
    JST_ERROR("File locks are not supported in this platform.");
    return Result::ERROR;
#else
    const int fd = open(path.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        JST_ERROR("Failed to open file lock '{}' [Errno: {}].", path, errno);
        return Result::ERROR;
    }

    const int operation = wait ? LOCK_EX : (LOCK_EX | LOCK_NB);
    while (flock(fd, operation) != 0) {
        if (errno == EINTR && wait) {
            continue;
        }

        const auto error = errno;
        close(fd);
        if (!wait && (error == EWOULDBLOCK || error == EAGAIN)) {
            return Result::SKIP;
        }

        JST_ERROR("Failed to acquire file lock '{}' [Errno: {}].", path, error);
        return Result::ERROR;
    }

    state.fd = fd;
#endif

    state.path = path;
    state.locked = true;
    return Result::SUCCESS;
}

void FileLock::release() {
    if (!impl || !impl->locked) {
        return;
    }

#if defined(JST_OS_WINDOWS)
    OVERLAPPED overlapped = {};
    (void)UnlockFileEx(impl->handle, 0, MAXDWORD, MAXDWORD, &overlapped);
    CloseHandle(impl->handle);
    impl->handle = INVALID_HANDLE_VALUE;
#elif !defined(JST_OS_BROWSER)
    (void)flock(impl->fd, LOCK_UN);
    close(impl->fd);
    impl->fd = -1;
#endif

    impl->path.clear();
    impl->locked = false;
}

bool FileLock::locked() const {
    return impl && impl->locked;
}

}  // namespace Jetstream::Platform
