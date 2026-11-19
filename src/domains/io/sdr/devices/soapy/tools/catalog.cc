#include "catalog.hh"

#include <algorithm>
#include <array>
#include <map>
#include <string_view>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr std::array SkippedDrivers = {
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
#ifdef JETSTREAM_LOADER_BIRRIASDR_AVAILABLE
    std::string_view{"birria"},
#endif
    std::string_view{"audio"},
};

std::string Value(const SoapyKwargs& args, const std::string& key) {
    const auto it = args.find(key);
    return it == args.end() ? std::string{} : it->second;
}

bool Selectable(const std::string& serial) {
    return !serial.empty() && serial.find_first_of(",=") == std::string::npos;
}

std::string Identifier(const SoapyKwargs& args) {
    U64 hash = 14695981039346656037ULL;
    const auto mix = [&](const std::string_view text) {
        for (const char c : text) {
            hash = (hash ^ static_cast<U8>(c)) * 1099511628211ULL;
        }
    };
    for (const auto& [key, value] : args) {
        mix(key);
        mix("=");
        mix(value);
        mix(";");
    }
    return jst::fmt::format("args-{:016x}", hash);
}

}  // namespace

bool SoapyCatalog::Skipped(const SoapyKwargs& args) {
    const auto driver = Value(args, "driver");
    if (driver == "plutosdr" && !Value(args, "uri").starts_with("usb:")) {
        return false;
    }
    if (driver == "lime" && !Value(args, "media").empty() &&
        Value(args, "media").find("USB") == std::string::npos) {
        return false;
    }
    return std::find(SkippedDrivers.begin(), SkippedDrivers.end(), driver) != SkippedDrivers.end();
}

std::optional<SoapyCatalog::Entry> SoapyCatalog::Describe(const SoapyKwargs& args) {
    if (Skipped(args)) {
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
    return entry;
}

std::vector<SoapyCatalog::Entry> SoapyCatalog::List(const SoapyKwargsList& devices) {
    std::vector<Entry> entries;
    for (const auto& args : devices) {
        if (auto entry = Describe(args)) {
            entries.push_back(std::move(*entry));
        }
    }
    std::map<std::string, U64> serials;
    for (const auto& entry : entries) {
        ++serials[entry.info.serial];
    }
    for (auto& entry : entries) {
        if (serials.at(entry.info.serial) > 1) {
            entry.info.serial = Identifier(entry.args);
        }
    }
    return entries;
}

F64 SoapyCatalog::SweepRate(const std::vector<Device::Range>& reported, const F64 ceiling) {
    F64 preferred = 0.0;
    F64 lowest = 0.0;
    for (const auto& range : reported) {
        if (range.minimum <= ceiling) {
            preferred = std::max(preferred, std::min(range.maximum, ceiling));
        }
        lowest = lowest > 0.0 ? std::min(lowest, range.minimum) : range.minimum;
    }
    return preferred > 0.0 ? preferred : lowest;
}

}  // namespace Jetstream::Radio
