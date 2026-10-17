#include "base.hh"

#include <string>

#if defined(JST_OS_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include <libbladeRF.h>
#if defined(JST_OS_WINDOWS)
#undef ERROR
#undef FATAL
#endif

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

std::vector<Device::Info> BladeRf::Enumerate() {
    std::vector<Device::Info> devices;
    bladerf_devinfo* list = nullptr;
    const int count = bladerf_get_device_list(&list);
    for (int i = 0; i < count; ++i) {
        const std::string serial = list[i].serial;
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("bladeRF ({})", serial.substr(0, 8)),
            .biasTee = std::string(list[i].product).find("2.0") != std::string::npos,
        });
    }
    if (list) {
        bladerf_free_device_list(list);
    }
    return devices;
}

}  // namespace Jetstream::Radio
