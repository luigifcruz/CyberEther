#include "jetstream/platform.hh"

#include <cstdio>
#include <filesystem>
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

}  // namespace Jetstream::Platform
