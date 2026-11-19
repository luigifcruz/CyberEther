#include "base.hh"
#include "tools/catalog.hh"

namespace Jetstream::Radio {

std::vector<Device::Info> SoapySdr::Enumerate() {
    const auto library = SoapyRuntime::Acquire();
    if (!library) {
        return {};
    }
    SoapyKwargsList devices;
    if (library->enumerate({}, devices) != Result::SUCCESS) {
        return {};
    }
    std::vector<Device::Info> infos;
    for (auto& entry : SoapyCatalog::List(devices)) {
        infos.push_back(std::move(entry.info));
    }
    return infos;
}

}  // namespace Jetstream::Radio
