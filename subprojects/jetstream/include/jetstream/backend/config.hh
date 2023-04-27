#ifndef JETSTREAM_BACKEND_CONFIG_HH
#define JETSTREAM_BACKEND_CONFIG_HH

#include <fmt/ostream.h>

#include "jetstream/types.hh"

namespace Jetstream::Backend {

// TODO: Add physical device limits API.

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
    U64 deviceId;
    bool validationEnabled = true;
};

}  // namespace Jetstream::Backend

template <> struct fmt::formatter<Jetstream::Backend::PhysicalDeviceType> : ostream_formatter {};

#endif
