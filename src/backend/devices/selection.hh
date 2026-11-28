#ifndef JETSTREAM_BACKEND_DEVICES_SELECTION_HH
#define JETSTREAM_BACKEND_DEVICES_SELECTION_HH

#include <optional>
#include <string>
#include <vector>

#include "jetstream/backend/config.hh"

namespace Jetstream::Backend {

struct DeviceChoice {
    U64 index = 0;
    DeviceSelection selection = DeviceSelection::Automatic;

    bool operator==(const DeviceChoice&) const = default;
};

inline std::string FormatDeviceUuid(const U8* bytes) {
    std::string uuid;
    for (U64 i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            uuid += '-';
        }
        uuid += jst::fmt::format("{:02x}", bytes[i]);
    }
    return uuid;
}

inline std::optional<DeviceChoice> ChooseDevice(const std::vector<PhysicalDevice>& devices,
                                                const std::optional<U64>& deviceId,
                                                const std::string& deviceUuid) {
    if (devices.empty()) {
        return std::nullopt;
    }
    if (deviceId.has_value()) {
        if (deviceId.value() >= devices.size()) {
            return std::nullopt;
        }
        return DeviceChoice{.index = deviceId.value(), .selection = DeviceSelection::Override};
    }
    if (!deviceUuid.empty()) {
        for (U64 i = 0; i < devices.size(); ++i) {
            if (devices[i].uuid == deviceUuid) {
                return DeviceChoice{.index = i, .selection = DeviceSelection::Saved};
            }
        }
        return DeviceChoice{.index = 0, .selection = DeviceSelection::Missing};
    }
    return DeviceChoice{.index = 0, .selection = DeviceSelection::Automatic};
}

inline bool IsDiscreteDevice(PhysicalDeviceType type) {
    return type == PhysicalDeviceType::DISCRETE || type == PhysicalDeviceType::EXTERNAL;
}

inline std::optional<U64> UnusedDiscreteDevice(const std::vector<PhysicalDevice>& devices, const DeviceChoice& choice) {
    if (choice.selection == DeviceSelection::Saved || choice.selection == DeviceSelection::Override) {
        return std::nullopt;
    }
    if (choice.index >= devices.size() || IsDiscreteDevice(devices[choice.index].type)) {
        return std::nullopt;
    }
    for (U64 i = 0; i < devices.size(); ++i) {
        if (IsDiscreteDevice(devices[i].type) && !devices[i].uuid.empty()) {
            return i;
        }
    }
    return std::nullopt;
}

}  // namespace Jetstream::Backend

#endif  // JETSTREAM_BACKEND_DEVICES_SELECTION_HH
