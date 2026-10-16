#include "base.hh"
#include "tools/library.hh"

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

std::vector<Device::Info> LimeSdr::Enumerate() {
    std::vector<Device::Info> devices;
    for (const auto& board : LimeSdrLibrary::Boards()) {
        devices.push_back({
            .serial = board.serial,
            .label = jst::fmt::format("{} ({})", board.name.empty() ? "LimeSDR" : board.name,
                                      board.serial),
            .biasTee = false,
        });
    }
    return devices;
}

}  // namespace Jetstream::Radio
