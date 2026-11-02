#include "base.hh"

#include <birriasdr.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

std::vector<Device::Info> BirriaSdr::Enumerate() {
    std::vector<Device::Info> devices;
    const int count = birriasdr_get_device_count();
    for (int i = 0; i < count; ++i) {
        birriasdr_device_info info = {};
        if (birriasdr_get_device_info(static_cast<U32>(i), &info) != BIRRIASDR_OK ||
            !(info.capabilities & BIRRIASDR_CAP_RX)) {
            continue;
        }
        const auto serial = jst::fmt::format("{}-{}", info.bus, info.address);
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("{} ({})", info.name, serial),
            .biasTee = false,
        });
    }
    return devices;
}

}  // namespace Jetstream::Radio
