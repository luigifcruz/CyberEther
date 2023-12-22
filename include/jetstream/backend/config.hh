#ifndef JETSTREAM_BACKEND_CONFIG_HH
#define JETSTREAM_BACKEND_CONFIG_HH

#include "jetstream/logger.hh"
#include "jetstream/types.hh"

namespace Jetstream::Backend {

enum class PhysicalDeviceType : uint8_t {
    UNKNOWN     = 0,
    DISCRETE    = 1,
    INTEGRATED  = 2,
    OTHER       = 3,
};

inline std::ostream& operator<<(std::ostream& os, const PhysicalDeviceType& type) {
    switch (type) {
        case PhysicalDeviceType::UNKNOWN:
            os << "UNKNOWN";
            break;
        case PhysicalDeviceType::DISCRETE:
            os << "DISCRETE";
            break;
        case PhysicalDeviceType::INTEGRATED:
            os << "INTEGRATED";
            break;
        case PhysicalDeviceType::OTHER:
            os << "OTHER";
            break;
        default:
            os.setstate(std::ios_base::failbit);
            break;
    }
    return os;
}

struct Config {
    U64 deviceId = 0;
#ifdef JST_DEBUG_MODE
    bool validationEnabled = true;
#else
    bool validationEnabled = false;
#endif
    U64 stagingBufferSize = 64*1024*1024;
    bool headless = false;
};

}  // namespace Jetstream::Backend

template <> struct fmt::formatter<Jetstream::Backend::PhysicalDeviceType> : ostream_formatter {};

#endif
