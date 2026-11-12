#include "base.hh"
#include "tools/catalog.hh"

#include "../stream.hh"

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

std::vector<Device::Info> SoapySdr::Enumerate() {
    const auto library = Modules::SoapyRuntime::Acquire();
    if (!library) {
        return {};
    }
    Modules::SoapyKwargsList devices;
    if (library->enumerate({}, devices) != Result::SUCCESS) {
        return {};
    }
    std::vector<Device::Info> infos;
    for (auto& entry : SoapyCatalog::List(devices)) {
        infos.push_back(std::move(entry.info));
    }
    return infos;
}

Result SoapySdr::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>&) {
    JST_ERROR("[SDR_SOAPY] {} only supports sweeps here. Use the Soapy block to stream it.",
              info.label);
    return Result::ERROR;
}

}  // namespace Jetstream::Radio
