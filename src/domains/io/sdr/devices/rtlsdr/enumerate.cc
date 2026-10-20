#include "base.hh"

#include <rtl-sdr.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

std::vector<Device::Info> RtlSdr::Enumerate() {
    std::vector<Device::Info> devices;
    const U32 count = rtlsdr_get_device_count();
    for (U32 i = 0; i < count; ++i) {
        char manufacturer[256] = {};
        char product[256] = {};
        char serial[256] = {};
        if (rtlsdr_get_device_usb_strings(i, manufacturer, product, serial) < 0) {
            continue;
        }
        const char* name = rtlsdr_get_device_name(i);
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("{} ({})", name ? name : "RTL-SDR", serial),
        });
    }
    return devices;
}

}  // namespace Jetstream::Radio
