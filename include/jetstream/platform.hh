#ifndef JETSTREAM_PLATFORM_HH
#define JETSTREAM_PLATFORM_HH

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"

namespace Jetstream::Platform {

enum class DynamicLibraryVisibility {
    Local,
    Global,
};

JETSTREAM_API std::filesystem::path PathFromUtf8(const std::string& path);
JETSTREAM_API std::string PathToUtf8(const std::filesystem::path& path);

JETSTREAM_API Result EnvironmentVariable(const std::string& name, std::string& value);
JETSTREAM_API Result EnvironmentPath(const std::string& name, std::filesystem::path& path);

JETSTREAM_API bool PrepareStandardOutputForAnsi();

#if defined(JST_OS_LINUX) || defined(JST_OS_MAC) || defined(JST_OS_WINDOWS)
JETSTREAM_API bool InstallInterruptHandler(void (*handler)() noexcept);
JETSTREAM_API void UninstallInterruptHandler();
JETSTREAM_API void WriteInterruptMessage(const char* message, std::size_t size) noexcept;
[[noreturn]] JETSTREAM_API void ForceTerminate(I32 status) noexcept;
#endif

#if !defined(JST_OS_BROWSER)
JETSTREAM_API bool ShutdownSocketRead(std::uintptr_t socket) noexcept;
#endif

JETSTREAM_API Result RunProcess(const std::string& executable,
                                const std::vector<std::string>& arguments,
                                std::string& output,
                                U64 timeoutMilliseconds = 0);

JETSTREAM_API void* OpenDynamicLibrary(const std::string& path,
                                       DynamicLibraryVisibility visibility,
                                       std::string& error);
JETSTREAM_API void CloseDynamicLibrary(void* handle);
JETSTREAM_API void* LoadDynamicLibrarySymbol(void* handle,
                                             const char* symbol,
                                             std::string& error);

class JETSTREAM_API FileLock {
 public:
    FileLock();
    ~FileLock();

    FileLock(FileLock&&) noexcept;
    FileLock& operator=(FileLock&&) noexcept;

    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

    Result acquire(const std::string& path, bool wait = true);
    void release();
    bool locked() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

JETSTREAM_API Result OpenUrl(const std::string& url);
JETSTREAM_API Result ConfigPath(std::string& path);
JETSTREAM_API Result CachePath(std::string& path);
JETSTREAM_API Result PickFile(std::string& path,
                              const std::vector<std::string>& extensions = {},
                              std::function<void(std::string)> callback = nullptr);
JETSTREAM_API Result PickFolder(std::string& path,
                                std::function<void(std::string)> callback = nullptr);
JETSTREAM_API Result SaveFile(std::string& path,
                              std::function<void(std::string)> callback = nullptr);

JETSTREAM_API bool IsFilePending();

}  // namespace Jetstream::Platform

#endif
