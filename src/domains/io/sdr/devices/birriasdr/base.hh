#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_BIRRIASDR_BASE_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_BIRRIASDR_BASE_HH

#include <memory>
#include <vector>

#include "../device.hh"

namespace Jetstream::Radio {

struct BirriaSdr {
    static std::vector<Device::Info> Enumerate();
    static Result Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream);
    static Result Sweep(const Device::Info& info, std::unique_ptr<Radio::Sweep>& sweep);
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_BIRRIASDR_BASE_HH
