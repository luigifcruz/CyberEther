#include "base.hh"

#include <mutex>
#include <string>

#include <hackrf.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

bool InitLibrary() {
    static std::once_flag flag;
    static int status = HACKRF_ERROR_OTHER;
    std::call_once(flag, [] {
        status = hackrf_init();
    });
    return status == HACKRF_SUCCESS;
}

std::string ShortSerial(const std::string& serial) {
    return serial.size() > 16 ? serial.substr(serial.size() - 16) : serial;
}

}  // namespace

std::vector<Device::Info> HackRf::Enumerate() {
    std::vector<Device::Info> devices;
    if (!InitLibrary()) {
        return devices;
    }
    auto* list = hackrf_device_list();
    if (!list) {
        return devices;
    }
    for (int i = 0; i < list->devicecount; ++i) {
        const char* serial = list->serial_numbers[i];
        if (!serial) {
            continue;
        }
        const auto board = hackrf_usb_board_id_name(list->usb_board_ids[i]);
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("{} ({})", board ? board : "HackRF", ShortSerial(serial)),
        });
    }
    hackrf_device_list_free(list);
    return devices;
}

}  // namespace Jetstream::Radio
