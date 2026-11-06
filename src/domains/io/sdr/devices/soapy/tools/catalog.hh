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

    // Drivers that already have a native backend are left to it.
    static bool Native(const Modules::SoapyKwargs& args);

    // Builds the device entry, or nothing when the device is left to a native backend.
    // The serial is the device serial when the driver reports one that fits in a
    // selector, otherwise a stable identifier derived from the device arguments.
    static std::optional<Entry> Describe(const Modules::SoapyKwargs& args);

    static std::vector<Entry> List(const Modules::SoapyKwargsList& devices);

    // Collapses the reported ranges into one span that reaches the probed rate, since some
    // drivers only report the rates of their current master clock.
    static std::vector<Device::Range> SampleRates(const std::vector<Device::Range>& reported,
                                                  F64 probed);
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_SOAPY_TOOLS_CATALOG_HH
