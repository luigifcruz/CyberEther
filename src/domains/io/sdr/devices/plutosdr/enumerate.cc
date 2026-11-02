#include "base.hh"

#include <cerrno>
#include <string>

#include <iio.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr const char* ScanFilter = "usb=0456:b673";

struct Board {
    std::string uri;
    std::string serial;
};

std::string ErrorText(const long long error) {
    char text[256] = {};
    iio_strerror(static_cast<int>(error < 0 ? -error : error), text, sizeof(text));
    return text;
}

std::vector<Board> Scan() {
    std::vector<Board> boards;
    auto* scan = iio_create_scan_context(ScanFilter, 0);
    if (!scan) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to create the USB scan context: {}", ErrorText(errno));
        return boards;
    }
    iio_context_info** list = nullptr;
    const auto count = iio_scan_context_get_info_list(scan, &list);
    if (count < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to enumerate devices: {}", ErrorText(count));
    }
    for (ssize_t i = 0; i < count; ++i) {
        const std::string description = iio_context_info_get_description(list[i]);
        const auto marker = description.rfind("serial=");
        boards.push_back({
            .uri = iio_context_info_get_uri(list[i]),
            .serial = marker == std::string::npos ? "" : description.substr(marker + 7),
        });
    }
    if (list) {
        iio_context_info_list_free(list);
    }
    iio_scan_context_destroy(scan);
    return boards;
}

}  // namespace

std::vector<Device::Info> PlutoSdr::Enumerate() {
    std::vector<Device::Info> devices;
    for (const auto& board : Scan()) {
        devices.push_back({
            .serial = board.serial,
            .label = jst::fmt::format("PlutoSDR ({})", board.serial),
            .biasTee = false,
        });
    }
    return devices;
}

}  // namespace Jetstream::Radio
