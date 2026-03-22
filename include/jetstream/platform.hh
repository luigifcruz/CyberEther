#ifndef JETSTREAM_PLATFORM_HH
#define JETSTREAM_PLATFORM_HH

#include <vector>
#include <functional>
#include <string>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"

namespace Jetstream::Platform {

Result OpenUrl(const std::string& url);
Result PickFile(std::string& path,
                const std::vector<std::string>& extensions = {},
                std::function<void(std::string)> callback = nullptr);
Result PickFolder(std::string& path,
                  std::function<void(std::string)> callback = nullptr);
Result SaveFile(std::string& path,
                std::function<void(std::string)> callback = nullptr);

bool IsFilePending();

}  // namespace Jetstream::Platform

#endif
