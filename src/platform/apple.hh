#ifndef JETSTREAM_PLATFORM_APPLE_HH
#define JETSTREAM_PLATFORM_APPLE_HH

#include <string>

#include "jetstream/platform.hh"
#include "jetstream/macros.hh"

namespace Jetstream::Platform {

Result OpenUrl(const std::string& url);
Result PickFile(std::string& path);
Result PickFolder(std::string& path);
Result SaveFile(std::string& path);

}  // namespace Jetstream::Platform

#endif