#ifndef JETSTREAM_BACKEND_CONFIG_HH
#define JETSTREAM_BACKEND_CONFIG_HH

#include <optional>
#include <string>

#include "jetstream/logger.hh"
#include "jetstream/types.hh"

namespace Jetstream::Backend {

enum class PhysicalDeviceType : uint8_t {
    UNKNOWN     = 0,
    DISCRETE    = 1,
    INTEGRATED  = 2,
    OTHER       = 3,
    EXTERNAL    = 4,
    VIRTUAL     = 5,
    CPU         = 6,
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
        case PhysicalDeviceType::EXTERNAL:
            os << "EXTERNAL";
            break;
        case PhysicalDeviceType::VIRTUAL:
            os << "VIRTUAL";
            break;
        case PhysicalDeviceType::CPU:
            os << "CPU";
            break;
        default:
            os.setstate(std::ios_base::failbit);
            break;
    }
    return os;
}

struct PhysicalDevice {
    U64 index = 0;
    std::string name;
    std::string uuid;
    PhysicalDeviceType type = PhysicalDeviceType::UNKNOWN;
    std::string pciAddress;

    bool operator==(const PhysicalDevice&) const = default;
};

enum class DeviceSelection : uint8_t {
    Automatic,
    Saved,
    Missing,
    Override,
};

struct Config {
    std::optional<U64> deviceId;
    std::string deviceUuid;
#ifdef JST_DEBUG_MODE
    bool validationEnabled = true;
#else
    bool validationEnabled = false;
#endif
    U64 multisampling = 4;
    bool headless = false;
    std::string pythonRuntimePath;
    std::string dependencyPolicy;
};

}  // namespace Jetstream::Backend

template <> struct jst::fmt::formatter<Jetstream::Backend::PhysicalDeviceType> : ostream_formatter {};

#endif
