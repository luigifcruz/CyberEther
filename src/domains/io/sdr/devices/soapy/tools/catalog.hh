#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_CATALOG_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_CATALOG_HH

#include <optional>
#include <string>
#include <vector>

#include "domains/io/sdr/devices/device.hh"
#include "domains/io/sdr/devices/soapy/tools/runtime.hh"

namespace Jetstream::Radio {

struct JETSTREAM_API SoapyCatalog {
    struct Entry {
        Device::Info info;
        SoapyKwargs args;
    };

    static bool Skipped(const SoapyKwargs& args);

    static std::optional<Entry> Describe(const SoapyKwargs& args);

    static std::vector<Entry> List(const SoapyKwargsList& devices);

    static F64 SweepRate(const std::vector<Device::Range>& reported, F64 ceiling);
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_CATALOG_HH
