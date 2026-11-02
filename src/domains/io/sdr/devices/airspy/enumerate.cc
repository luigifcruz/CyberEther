#include "base.hh"

#include <airspy.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

std::vector<Device::Info> Airspy::Enumerate() {
    std::vector<Device::Info> devices;
    const int count = airspy_list_devices(nullptr, 0);
    if (count <= 0) {
        return devices;
    }
    std::vector<uint64_t> serials(static_cast<std::size_t>(count));
    const int listed = airspy_list_devices(serials.data(), count);
    for (int i = 0; i < listed; ++i) {
        const auto serial = jst::fmt::format("{:016X}", serials[static_cast<std::size_t>(i)]);
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("Airspy ({})", serial),
        });
    }
    return devices;
}

}  // namespace Jetstream::Radio
