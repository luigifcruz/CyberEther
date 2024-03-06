#ifndef JETSTREAM_PLATFORM_HH
#define JETSTREAM_PLATFORM_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"

namespace Jetstream::Platform {

Result OpenUrl(const std::string& url);
Result PickFile(std::string& path);
Result PickFolder(std::string& path);
Result SaveFile(std::string& path);

}  // namespace Jetstream::Platform

#endif