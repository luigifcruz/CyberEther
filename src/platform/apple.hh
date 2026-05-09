#ifndef JETSTREAM_PLATFORM_APPLE_HH
#define JETSTREAM_PLATFORM_APPLE_HH

#include <string>
#include <vector>
#include <functional>

#include "jetstream/platform.hh"
#include "jetstream/macros.hh"

namespace Jetstream::Platform {

Result OpenUrl(const std::string& url);
Result ConfigPath(std::string& path);
Result CachePath(std::string& path);
Result PickFile(std::string& path,
                const std::vector<std::string>& extensions,
                std::function<void(std::string)> callback);
Result PickFolder(std::string& path,
                  std::function<void(std::string)> callback);
Result SaveFile(std::string& path,
                std::function<void(std::string)> callback);

}  // namespace Jetstream::Platform

#endif
