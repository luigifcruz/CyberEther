#include "base.hh"

#include <rtl-sdr.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

std::vector<Device::Info> RtlSdr::Enumerate() {
    std::vector<Device::Info> devices;
    const U32 count = rtlsdr_get_device_count();
    for (U32 i = 0; i < count; ++i) {
        char serial[256] = {};
#ifndef JST_OS_BROWSER
        char manufacturer[256] = {};
        char product[256] = {};
        if (rtlsdr_get_device_usb_strings(i, manufacturer, product, serial) < 0) {
            continue;
        }
#endif
        const char* name = rtlsdr_get_device_name(i);
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("{} ({}) [USB index {}]", name ? name : "RTL-SDR", serial, i),
            .index = i,
        });
    }
    return devices;
}

}  // namespace Jetstream::Radio
