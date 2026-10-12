#include "device.hh"

#include "../tools/samples.hh"

#ifdef JETSTREAM_LOADER_RTLSDR_AVAILABLE
#include "rtlsdr/rtlsdr.hh"
#endif
#ifdef JETSTREAM_LOADER_AIRSPY_AVAILABLE
#include "airspy/airspy.hh"
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
#include <cmath>
#include <map>
#include <new>

#include "jetstream/logger.hh"

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#endif

namespace Jetstream::Radio {

namespace {

using Clock = std::chrono::steady_clock;

constexpr auto DiscoveryLifetime = std::chrono::seconds(1);
constexpr U64 MaxHopDiscontinuities = 8;
constexpr U64 MaxReadChunk = 16384;
constexpr U64 MaxSweepFailures = 3;

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

bool Device::Range::Contains(const std::vector<Range>& ranges, const F64 value) {
    return std::isfinite(value) &&
           std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) {
               return value >= range.minimum && value <= range.maximum;
           });
}

std::vector<F64> Device::Capabilities::fixedSampleRates() const {
    std::vector<F64> rates;
    for (const auto& range : sampleRateRanges) {
        if (range.minimum != range.maximum) {
            return {};
        }
        rates.push_back(range.minimum);
    }
    std::sort(rates.begin(), rates.end());
    rates.erase(std::unique(rates.begin(), rates.end()), rates.end());
    return rates;
}

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

const Device::Info& Device::info() const {
    return deviceInfo;
}

const Device::Capabilities& Device::capabilities() const {
    return deviceCapabilities;
}

const Device::Applied& Device::applied() const {
    return appliedSettings;
}

U64 Device::generation() const {
    return currentGeneration.load();
}

Device::Statistics Device::statistics() const {
    std::lock_guard lock(statisticsMutex);
    return deviceStatistics;
}

U64 Device::nextGeneration() {
    return ++currentGeneration;
}

U64 Device::guardSamples(const Settings& settings, const F64 sampleRate) const {
    const F64 seconds = settings.settlingTime >= 0.0
        ? settings.settlingTime
        : defaultSettlingTime;
    return GuardSamples(seconds, sampleRate);
}

Result Device::restore(const Settings& previous, const Result failure) {
    stop();
    if (start(previous) == Result::SUCCESS) {
        JST_WARN("[SDR] Restored the previous settings of {} after a failed change.",
                 deviceInfo.label);
        return failure == Result::SUCCESS ? Result::ERROR : failure;
    }
    stop();
    JST_ERROR("[SDR] Failed to restore {} after a failed change. The device is stopped.",
              deviceInfo.label);
    return Result::ERROR;
}

Result Device::apply(const Settings& settings) {
    const auto previous = requestedSettings;
    stop();
    const auto result = start(settings);
    if (result != Result::SUCCESS) {
        return restore(previous, result);
    }
    return Result::SUCCESS;
}

Result Device::tune(const F64 frequency) {
    auto settings = requestedSettings;
    settings.frequency = frequency;
    return apply(settings);
}

Result Device::setAutomaticGain(const bool automaticGain) {
    auto settings = requestedSettings;
    settings.automaticGain = automaticGain;
    return apply(settings);
}

Result Device::restart(const std::function<Result()>& change) {
    const auto previous = requestedSettings;
    const auto applied = appliedSettings;
    stop();
    const auto result = change();
    if (result == Result::SUCCESS) {
        return result;
    }
    const auto restored = restore(previous, result);
    requestedSettings = previous;
    appliedSettings = applied;
    return restored;
}

Result Device::prepareSweep(const SweepRequest& request) {
    if (request.firmware) {
        JST_ERROR("[SDR] Firmware sweeps are not supported by {}.", deviceInfo.label);
        return Result::ERROR;
    }
    sweepRequest = request;
    sweepPrepared = true;
    return Result::SUCCESS;
}

Result Device::captureHop(const U64 hop, std::span<CF32> samples, U64& segments,
                          SweepCapture& capture, const SweepRunning& running) {
    if (!sweepPrepared || hop >= sweepRequest.frequencies.size()) {
        JST_ERROR("[SDR] Sweep hop {} is not part of the prepared plan.", hop);
        return Result::ERROR;
    }
    const U64 length = sweepRequest.captureLength;
    const U64 required = length * sweepRequest.dwell;
    if (samples.size() < required || segments > sweepRequest.dwell) {
        JST_ERROR("[SDR] Sweep capture buffer is too small.");
        return Result::ERROR;
    }

    if (!running()) {
        return Result::YIELD;
    }
    JST_CHECK(tuneHop(hop));

    const U64 generation = this->generation();
    const auto timeout = std::chrono::milliseconds(100);
    U64 filled = segments * length;
    U64 discontinuities = 0;
    auto idleSince = std::chrono::steady_clock::now();
    const F64 idleLimit = std::chrono::duration<F64>(StallTimeout).count() +
        (appliedSettings.sampleRate > 0.0
            ? static_cast<F64>(guardLength) / appliedSettings.sampleRate
            : 0.0);
    while (filled < required) {
        if (!running()) {
            return Result::YIELD;
        }
        const U64 chunk = std::min(required - filled, MaxReadChunk);
        const auto received = read(samples.subspan(filled, chunk), timeout);
        const auto now = std::chrono::steady_clock::now();

        if (received.status == Status::Disconnected || received.status == Status::Failure) {
            JST_ERROR("[SDR] Sweep capture failed on {}: {}", deviceInfo.label, received.error);
            return Result::FATAL;
        }
        if (received.status == Status::Samples && received.generation != generation) {
            filled = segments * length;
        } else if (received.status == Status::Samples && received.count > 0) {
            filled += received.count;
            segments = filled / length;
            idleSince = now;
            continue;
        }
        if (received.status == Status::Discontinuity) {
            filled = segments * length;
            idleSince = now;
            if (++discontinuities > MaxHopDiscontinuities) {
                JST_ERROR("[SDR] Could not capture {} contiguous samples on {}.",
                          length, deviceInfo.label);
                return Result::ERROR;
            }
            continue;
        }

        const std::chrono::duration<F64> idle = now - idleSince;
        if (idle.count() > idleLimit) {
            JST_ERROR("[SDR] Timed out waiting for sweep samples from {}.", deviceInfo.label);
            return Result::TIMEOUT;
        }
    }

    capture.hop = hop;
    capture.frequency = appliedSettings.frequency;
    capture.samples = samples.first(required);
    return Result::SUCCESS;
}

Result Device::fail(U64& failures) {
    {
        std::lock_guard lock(statisticsMutex);
        ++deviceStatistics.failedHops;
    }
    if (++failures < MaxSweepFailures) {
        return Result::SUCCESS;
    }
    JST_ERROR("[SDR] Stopping the sweep on {} after {} failed hops in a row.", deviceInfo.label,
              failures);
    return Result::ERROR;
}

Result Device::tuneHop(const U64 hop) {
    return tune(sweepRequest.frequencies[hop]);
}

Result Device::runSweep(const SweepRunning& running, const SweepSink& sink) {
    if (!sweepPrepared || sweepRequest.firmware) {
        JST_ERROR("[SDR] No sweep is prepared on {}.", deviceInfo.label);
        return Result::ERROR;
    }
    try {
        sweepBuffer.assign(sweepRequest.captureLength * sweepRequest.dwell, CF32{});
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR] Failed to allocate the sweep capture buffer.");
        return Result::ERROR;
    }

    U64 hop = 0;
    U64 segments = 0;
    U64 failures = 0;
    while (running()) {
        SweepCapture capture;
        const auto result = captureHop(hop, sweepBuffer, segments, capture, running);
        if (result == Result::YIELD || !running()) {
            break;
        }
        if (result == Result::FATAL) {
            return Result::ERROR;
        }
        if (result != Result::SUCCESS) {
            JST_CHECK(fail(failures));
            continue;
        }
        failures = 0;
        segments = 0;
        if (!sink(capture)) {
            return Result::SUCCESS;
        }
        hop = (hop + 1) % sweepRequest.frequencies.size();
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
