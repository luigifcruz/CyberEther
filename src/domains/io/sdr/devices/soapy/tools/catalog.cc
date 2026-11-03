#include "catalog.hh"

#include <algorithm>
#include <array>
#include <functional>
#include <string_view>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr std::array NativeDrivers = {
#ifdef JETSTREAM_LOADER_RTLSDR_AVAILABLE
    std::string_view{"rtlsdr"},
#endif
#ifdef JETSTREAM_LOADER_AIRSPY_AVAILABLE
    std::string_view{"airspy"},
#endif
#ifdef JETSTREAM_LOADER_AIRSPYHF_AVAILABLE
    std::string_view{"airspyhf"},
#endif
#ifdef JETSTREAM_LOADER_HACKRF_AVAILABLE
    std::string_view{"hackrf"},
#endif
#ifdef JETSTREAM_LOADER_LIMESUITE_AVAILABLE
    std::string_view{"lime"},
#endif
#ifdef JETSTREAM_LOADER_BLADERF_AVAILABLE
    std::string_view{"bladerf"},
#endif
#ifdef JETSTREAM_LOADER_PLUTOSDR_AVAILABLE
    std::string_view{"plutosdr"},
#endif
    std::string_view{"audio"},
};

std::string Value(const Modules::SoapyKwargs& args, const std::string& key) {
    const auto it = args.find(key);
    return it == args.end() ? std::string{} : it->second;
}

bool Selectable(const std::string& serial) {
    return !serial.empty() && serial.find_first_of(",=") == std::string::npos;
}

std::string Identifier(const Modules::SoapyKwargs& args) {
    std::string flat;
    for (const auto& [key, value] : args) {
        flat += key;
        flat += '=';
        flat += value;
        flat += ';';
    }
    return jst::fmt::format("args-{:016x}", std::hash<std::string>{}(flat));
}

}  // namespace

bool SoapyCatalog::Native(const Modules::SoapyKwargs& args) {
    for (const auto& key : {"driver", "remote:driver"}) {
        const auto driver = Value(args, key);
        if (std::find(NativeDrivers.begin(), NativeDrivers.end(), driver) != NativeDrivers.end()) {
            return true;
        }
    }
    return false;
}

std::optional<SoapyCatalog::Entry> SoapyCatalog::Describe(const Modules::SoapyKwargs& args) {
    if (Native(args)) {
        return std::nullopt;
    }

    const auto serial = Value(args, "serial");
    auto label = Value(args, "label");
    if (label.empty()) {
        label = Value(args, "driver");
    }
    if (label.empty()) {
        label = "Device";
    }

    Entry entry;
    entry.args = args;
    entry.info.serial = Selectable(serial) ? serial : Identifier(args);
    entry.info.label = jst::fmt::format("{} (SoapySDR)", label);
    entry.info.biasTee = false;
    return entry;
}

std::vector<SoapyCatalog::Entry> SoapyCatalog::List(const Modules::SoapyKwargsList& devices) {
    std::vector<Entry> entries;
    for (const auto& args : devices) {
        if (auto entry = Describe(args)) {
            entries.push_back(std::move(*entry));
        }
    }
    return entries;
}

std::vector<Device::Range> SoapyCatalog::SampleRates(const std::vector<Device::Range>& reported,
                                                     const F64 probed) {
    if (reported.empty()) {
        return probed > 0.0 ? std::vector<Device::Range>{{probed, probed}}
                            : std::vector<Device::Range>{};
    }
    Device::Range span = reported.front();
    for (const auto& range : reported) {
        span.minimum = std::min(span.minimum, range.minimum);
        span.maximum = std::max(span.maximum, range.maximum);
    }
    span.maximum = std::max(span.maximum, probed);
    return {span};
}

}  // namespace Jetstream::Radio
