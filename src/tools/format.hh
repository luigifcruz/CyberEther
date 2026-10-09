#ifndef JETSTREAM_TOOLS_FORMAT_HH
#define JETSTREAM_TOOLS_FORMAT_HH

#include "jetstream/logger.hh"
#include "jetstream/types.hh"

#include <chrono>
#include <ctime>
#include <string>

namespace Jetstream::Format {

inline std::string LocalTime(const char* format) {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm time{};
#if defined(_WIN32)
    localtime_s(&time, &now);
#else
    localtime_r(&now, &time);
#endif
    char value[64] = {};
    std::strftime(value, sizeof(value), format, &time);
    return value;
}

}  // namespace Jetstream::Format

#endif  // JETSTREAM_TOOLS_FORMAT_HH
