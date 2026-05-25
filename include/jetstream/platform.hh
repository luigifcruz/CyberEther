#ifndef JETSTREAM_PLATFORM_HH
#define JETSTREAM_PLATFORM_HH

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"

namespace Jetstream::Platform {

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
