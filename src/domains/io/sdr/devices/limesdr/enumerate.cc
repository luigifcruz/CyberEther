#include "base.hh"
#include "tools/library.hh"

#include <string_view>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr std::string_view FtdiAddressPrefix = "0403:";

}  // namespace

std::vector<Device::Info> LimeSdr::Enumerate() {
    std::vector<Device::Info> devices;
    for (const auto& board : LimeSdrLibrary::Boards()) {
        if (board.serial.empty()) {
            JST_DEBUG("[SDR_LIMESDR] Skipping '{}', which could not be opened to read its serial. "
                      "Check the USB permissions.", board.name.empty() ? board.addr : board.name);
            continue;
        }
        devices.push_back({
            .serial = board.serial,
            .label = jst::fmt::format("{} ({})", board.name.empty() ? "LimeSDR" : board.name,
                                      board.serial),
            .biasTee = false,
            .receiveChannels = board.addr.starts_with(FtdiAddressPrefix) ? 1u : 2u,
        });
    }
    return devices;
}

}  // namespace Jetstream::Radio
