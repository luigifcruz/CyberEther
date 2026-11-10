#include "base.hh"
#include "tools/library.hh"

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

std::vector<Device::Info> EnumerateBoards(LmsApi& api) {
    std::vector<Device::Info> devices;
    for (const auto& board : api.boards()) {
        devices.push_back({
            .serial = board.serial,
            .label = jst::fmt::format("{} ({})", board.name.empty() ? "LimeSDR" : board.name,
                                      board.serial),
            .biasTee = false,
        });
    }
    return devices;
}

}  // namespace

std::vector<Device::Info> LimeSdr::Enumerate() {
    return EnumerateBoards(LmsApi::Bundled());
}

std::vector<Device::Info> LimeSuiteNg::Enumerate() {
    auto* api = LmsApi::Ng();
    return api ? EnumerateBoards(*api) : std::vector<Device::Info>{};
}

}  // namespace Jetstream::Radio
