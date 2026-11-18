#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_CATALOG_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_CATALOG_HH

#include <optional>
#include <string>
#include <vector>

#include "domains/io/sdr/devices/device.hh"
#include "domains/io/soapy/library/soapy_runtime.hh"

namespace Jetstream::Radio {

struct JETSTREAM_API SoapyCatalog {
    struct Entry {
        Device::Info info;
        Modules::SoapyKwargs args;
    };

    static bool Skipped(const Modules::SoapyKwargs& args);

    static std::optional<Entry> Describe(const Modules::SoapyKwargs& args);

    static std::vector<Entry> List(const Modules::SoapyKwargsList& devices);

    static F64 SweepRate(const std::vector<Device::Range>& reported, F64 ceiling);
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_CATALOG_HH
