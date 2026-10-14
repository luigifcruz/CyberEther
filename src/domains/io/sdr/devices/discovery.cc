#include "device.hh"

#ifdef JETSTREAM_LOADER_RTLSDR_AVAILABLE
#include "rtlsdr/rtlsdr.hh"
#endif
#ifdef JETSTREAM_LOADER_AIRSPY_AVAILABLE
#include "airspy/airspy.hh"
#endif
#ifdef JETSTREAM_LOADER_AIRSPYHF_AVAILABLE
#include "airspyhf/airspyhf.hh"
#endif
#ifdef JETSTREAM_LOADER_HACKRF_AVAILABLE
#include "hackrf/hackrf.hh"
#endif
#ifdef JETSTREAM_LOADER_LIMESUITE_AVAILABLE
#include "limesdr/limesdr.hh"
#endif
#ifdef JETSTREAM_LOADER_BLADERF_AVAILABLE
#include "bladerf/bladerf.hh"
#endif

#include <algorithm>
#include <chrono>
#include <map>

#include "jetstream/logger.hh"

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#endif

namespace Jetstream::Radio {

namespace {

using Clock = std::chrono::steady_clock;

constexpr auto DiscoveryLifetime = std::chrono::seconds(1);

struct DiscoveryCache {
    std::mutex mutex;
    std::optional<std::vector<Device::Driver>> overrideDrivers;
    std::vector<Device::Info> devices;
    std::map<std::string, Device::Capabilities> capabilities;
    Clock::time_point expiresAt{};
    bool ready = false;
#ifdef JST_OS_BROWSER
    int usbRevision = -1;
#endif
};

DiscoveryCache& Cache() {
    static DiscoveryCache cache;
    return cache;
}

#ifdef JST_OS_BROWSER
int WebUsbRevision() {
    return MAIN_THREAD_EM_ASM_INT({
        if (!navigator.usb) {
            return -1;
        }
        let state = Module['sdrUsbDiscovery'];
        if (!state || state.usb !== navigator.usb) {
            state = ({
                usb: navigator.usb,
                revision: state ? (state.revision + 1) & 0x7fffffff : 0
            });
            Module['sdrUsbDiscovery'] = state;
            const changed = () => {
                state.revision = (state.revision + 1) & 0x7fffffff;
            };
            state.usb.addEventListener('connect', changed);
            state.usb.addEventListener('disconnect', changed);
        }
        return state.revision;
    });
}
#endif

std::vector<Device::Driver> BuiltinDrivers() {
    std::vector<Device::Driver> drivers;
#ifdef JETSTREAM_LOADER_RTLSDR_AVAILABLE
    drivers.push_back({"rtlsdr", &RtlSdrDevice::Enumerate, &RtlSdrDevice::Create});
#endif
#ifdef JETSTREAM_LOADER_AIRSPY_AVAILABLE
    drivers.push_back({"airspy", &AirspyDevice::Enumerate, &AirspyDevice::Create});
#endif
#ifdef JETSTREAM_LOADER_AIRSPYHF_AVAILABLE
    drivers.push_back({"airspyhf", &AirspyHfDevice::Enumerate, &AirspyHfDevice::Create});
#endif
#ifdef JETSTREAM_LOADER_HACKRF_AVAILABLE
    drivers.push_back({"hackrf", &HackRfDevice::Enumerate, &HackRfDevice::Create});
#endif
#ifdef JETSTREAM_LOADER_LIMESUITE_AVAILABLE
    drivers.push_back({"limesdr", &LimeSdrDevice::Enumerate, &LimeSdrDevice::Create});
#endif
#ifdef JETSTREAM_LOADER_BLADERF_AVAILABLE
    drivers.push_back({"bladerf", &BladeRfDevice::Enumerate, &BladeRfDevice::Create});
#endif
    return drivers;
}

std::vector<Device::Info> Enumerate(const std::vector<Device::Driver>& drivers) {
    std::vector<Device::Info> devices;
    for (const auto& driver : drivers) {
        try {
            for (auto& info : driver.enumerate()) {
                info.driver = driver.name;
                devices.push_back(std::move(info));
            }
        } catch (const std::exception& e) {
            JST_ERROR("[SDR] Failed to enumerate {} devices: {}", driver.name, e.what());
        } catch (...) {
            JST_ERROR("[SDR] Failed to enumerate {} devices.", driver.name);
        }
    }
    return devices;
}

}  // namespace

std::string Device::Info::selector() const {
    return jst::fmt::format("driver={},serial={}", driver, serial);
}

std::optional<Device::Selection> Device::Selection::Parse(const std::string& selector) {
    Selection selection;
    bool hasDriver = false;
    bool hasSerial = false;
    std::size_t begin = 0;
    while (begin <= selector.size()) {
        const auto end = std::min(selector.find(',', begin), selector.size());
        const auto field = selector.substr(begin, end - begin);
        const auto split = field.find('=');
        if (split == std::string::npos) {
            return std::nullopt;
        }
        const auto key = field.substr(0, split);
        const auto value = field.substr(split + 1);
        if (key == "driver" && !hasDriver) {
            selection.driver = value;
            hasDriver = true;
        } else if (key == "serial" && !hasSerial) {
            selection.serial = value;
            hasSerial = true;
        } else {
            return std::nullopt;
        }
        begin = end + 1;
    }
    if (!hasDriver || !hasSerial || selection.driver.empty()) {
        return std::nullopt;
    }
    return selection;
}

bool Device::Selection::matches(const Info& info) const {
    return info.driver == driver && info.serial == serial;
}

std::vector<Device::Driver> Device::Drivers() {
    auto& cache = Cache();
    std::lock_guard lock(cache.mutex);
    return cache.overrideDrivers ? *cache.overrideDrivers : BuiltinDrivers();
}

void Device::Override(std::optional<std::vector<Driver>> drivers) {
    auto& cache = Cache();
    std::lock_guard lock(cache.mutex);
    cache.overrideDrivers = std::move(drivers);
    cache.ready = false;
    cache.devices.clear();
}

void Device::ClearCache() {
    auto& cache = Cache();
    std::lock_guard lock(cache.mutex);
    cache.ready = false;
    cache.devices.clear();
}

std::vector<Device::Info> Device::List(const bool refresh) {
#ifdef JST_OS_BROWSER
    const auto usbRevision = WebUsbRevision();
    if (usbRevision < 0) {
        JST_ERROR("[SDR] Browser not compatible with WebUSB.");
        return {};
    }
#endif

    auto& cache = Cache();
    {
        std::lock_guard lock(cache.mutex);
#ifdef JST_OS_BROWSER
        if (cache.usbRevision != usbRevision) {
            cache.ready = false;
            cache.usbRevision = usbRevision;
        }
#endif
        if (!refresh && cache.ready && Clock::now() < cache.expiresAt) {
            return cache.devices;
        }
    }

    const auto devices = Enumerate(Drivers());

    std::lock_guard lock(cache.mutex);
    cache.devices = devices;
    cache.expiresAt = Clock::now() + DiscoveryLifetime;
    cache.ready = true;
    return devices;
}

Device::Labels Device::Label(const std::vector<Info>& devices) {
    Labels labeled;
    std::map<std::string, U64> counts;
    for (const auto& device : devices) {
        ++counts[device.label];
    }
    std::map<std::string, U64> used;
    for (const auto& device : devices) {
        std::string label = device.label.empty() ? device.driver : device.label;
        if (counts[device.label] > 1 && !device.serial.empty() &&
            label.find(device.serial) == std::string::npos) {
            label += " [" + device.serial + "]";
        }
        const auto base = label;
        U64 suffix = 2;
        while (used.contains(label)) {
            label = base + " #" + std::to_string(suffix++);
        }
        used[label] = 1;
        labeled.emplace_back(std::move(label), device);
    }
    return labeled;
}

std::optional<Device::Capabilities> Device::Cached(const std::string& selector) {
    auto& cache = Cache();
    std::lock_guard lock(cache.mutex);
    const auto entry = cache.capabilities.find(selector);
    if (entry == cache.capabilities.end()) {
        return std::nullopt;
    }
    return entry->second;
}

bool Device::SupportsBiasTee(const std::string& selector) {
    if (selector.empty()) {
        return true;
    }
    if (const auto capabilities = Cached(selector)) {
        return capabilities->biasTee;
    }
    for (const auto& info : List()) {
        if (info.selector() == selector) {
            return info.biasTee;
        }
    }
    return true;
}

Result Device::Open(const std::string& selector, std::unique_ptr<Device>& device) {
    device.reset();

    const auto selection = Selection::Parse(selector);
    if (!selection) {
        JST_ERROR("[SDR] Device selection '{}' is not a valid direct device selection.", selector);
        return Result::ERROR;
    }

    const auto drivers = Drivers();
    const auto driver = std::find_if(drivers.begin(), drivers.end(), [&](const auto& entry) {
        return entry.name == selection->driver;
    });
    if (driver == drivers.end()) {
        JST_ERROR("[SDR] Driver '{}' is not available in this build.", selection->driver);
        return Result::INCOMPLETE;
    }

    std::vector<Info> matches;
    for (const auto& info : List(true)) {
        if (selection->matches(info)) {
            matches.push_back(info);
        }
    }
    if (matches.empty()) {
        JST_ERROR("[SDR] Selected device ({}) is not connected.", selector);
        return Result::INCOMPLETE;
    }
    if (matches.size() != 1) {
        JST_ERROR("[SDR] Device selection ({}) is ambiguous: {} devices share this "
                  "serial number.", selector, matches.size());
        return Result::INCOMPLETE;
    }

    Result result = Result::ERROR;
    try {
        result = driver->create(matches.front(), device);
    } catch (const std::exception& e) {
        JST_ERROR("[SDR] Failed to open device ({}): {}", selector, e.what());
        result = Result::ERROR;
    } catch (...) {
        JST_ERROR("[SDR] Failed to open device ({}).", selector);
        result = Result::ERROR;
    }

    if (result != Result::SUCCESS || !device) {
        device.reset();
        ClearCache();
        return result == Result::SUCCESS ? Result::ERROR : result;
    }

    auto& cache = Cache();
    std::lock_guard lock(cache.mutex);
    cache.capabilities[selector] = device->capabilities();
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
