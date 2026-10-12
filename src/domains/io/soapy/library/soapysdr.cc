#include "soapysdr.hh"

#include <algorithm>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>

#include <jetstream/logger.hh>

namespace Jetstream::Modules {

namespace {

bool SampleRateInRanges(const std::vector<SoapyAbi::Range>& ranges, const F32 rate) {
    if (!std::isfinite(rate) || rate <= 0.0f) {
        return false;
    }

    return std::any_of(ranges.begin(), ranges.end(), [rate](const auto& range) {
        return rate >= static_cast<F32>(range.minimum) &&
               rate <= static_cast<F32>(range.maximum);
    });
}

using DiscoveryClock = SoapyDiscovery::Clock;
constexpr auto DiscoveryLifetime = std::chrono::seconds(1);

struct DiscoveryEntry {
    std::shared_future<SoapyKwargsList> result;
    DiscoveryClock::time_point expiresAt{};
    bool ready = false;
};

struct DiscoveryCache {
    std::mutex mutex;
    std::map<SoapyKwargs, std::shared_ptr<DiscoveryEntry>> entries;
};

DiscoveryCache& GetDiscoveryCache() {
    static DiscoveryCache cache;
    return cache;
}

Result DiscoverDevices(const SoapyLibrary& library, const SoapyKwargs& args, SoapyKwargsList& devices,
                       const bool refresh = false,
                       const std::optional<DiscoveryClock::time_point> now = std::nullopt) {
    auto& cache = GetDiscoveryCache();
    std::shared_ptr<DiscoveryEntry> entry;
    std::optional<std::promise<SoapyKwargsList>> pending;
    {
        std::lock_guard lock(cache.mutex);
        const auto currentTime = now ? *now : DiscoveryClock::now();
        std::erase_if(cache.entries, [&](const auto& item) {
            return item.second->ready && currentTime >= item.second->expiresAt;
        });

        if (const auto it = cache.entries.find(args); it != cache.entries.end()) {
            if (!refresh || !it->second->ready) {
                entry = it->second;
            } else {
                cache.entries.erase(it);
            }
        }
        if (!entry) {
            pending.emplace();
            entry = std::make_shared<DiscoveryEntry>();
            entry->result = pending->get_future().share();
            cache.entries.emplace(args, entry);
        }
    }

    if (pending) {
        SoapyKwargsList found;
        const auto result = library.enumerate(args, found);
        pending->set_value(found);

        std::lock_guard lock(cache.mutex);
        if (result == Result::SUCCESS) {
            entry->expiresAt = (now ? *now : DiscoveryClock::now()) + DiscoveryLifetime;
            entry->ready = true;
        } else if (const auto it = cache.entries.find(args);
                   it != cache.entries.end() && it->second == entry) {
            cache.entries.erase(it);
        }
        if (result != Result::SUCCESS) {
            return result;
        }
    }

    devices = entry->result.get();
    return Result::SUCCESS;
}

}  // namespace

SoapyDiscovery::DeviceList SoapyDiscovery::List(const std::string& filter, const bool refresh,
                                                const std::optional<Clock::time_point> now) {
    const auto library = SoapyRuntime::Acquire();
    if (!library) {
        return {};
    }
    SoapyKwargs args;
    SoapyKwargsList devices;
    if (library->kwargsFromString(filter, args) != Result::SUCCESS ||
        DiscoverDevices(*library, args, devices, refresh, now) != Result::SUCCESS) {
        return {};
    }
    return BuildDeviceList(devices);
}

void SoapyDiscovery::ClearCache() {
    auto& cache = GetDiscoveryCache();
    std::lock_guard lock(cache.mutex);
    cache.entries.clear();
}

Result SoapyDiscovery::LoadDriverLibrary(const std::string& path) {
    if (path.empty()) {
        return Result::SUCCESS;
    }

    const auto library = SoapyRuntime::Acquire();
    if (!library) {
        JST_ERROR("[MODULE_SOAPY] Cannot load module '{}': {}", path,
                  SoapyRuntime::Availability().message);
        return Result::ERROR;
    }
    JST_CHECK(library->loadModule(path));
    ClearCache();
    return Result::SUCCESS;
}

SoapyDiscovery::DeviceList SoapyDiscovery::BuildDeviceList(const SoapyKwargsList& entries) {
    DeviceList devices;
    for (const auto& entry : entries) {
        const auto labelIt = entry.find("label");
        const auto driverIt = entry.find("driver");
        std::string label = "SoapySDR Device";
        if (labelIt != entry.end() && !labelIt->second.empty()) {
            label = labelIt->second;
        } else if (driverIt != entry.end() && !driverIt->second.empty()) {
            label = driverIt->second;
        }

        std::string uniqueLabel = label;
        if (devices.contains(uniqueLabel)) {
            const auto serialIt = entry.find("serial");
            if (serialIt != entry.end() && !serialIt->second.empty() &&
                label.find(serialIt->second) == std::string::npos) {
                uniqueLabel = label + " [" + serialIt->second + "]";
            }

            const std::string uniqueLabelBase = uniqueLabel;
            U64 suffix = 2;
            while (devices.contains(uniqueLabel)) {
                uniqueLabel = uniqueLabelBase + " #" + std::to_string(suffix++);
            }
        }

        devices.emplace(std::move(uniqueLabel), entry);
    }
    return devices;
}

std::string SoapyDiscovery::FormatArguments(const SoapyKwargs& args) {
    const auto library = SoapyRuntime::Acquire();
    return library ? library->kwargsToString(args) : std::string{};
}

SoapyReceiver::~SoapyReceiver() {
    reset();
}

std::string SoapyReceiver::failure() const {
    const auto error = library ? library->lastError() : std::string{};
    return error.empty() ? std::string("Unknown driver error.") : error;
}

Result SoapyReceiver::open(const std::string& deviceString) {
    if (state != State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Device is already open.");
        return Result::ERROR;
    }
    const auto runtime = SoapyRuntime::Acquire();
    if (!runtime) {
        JST_ERROR("[MODULE_SOAPY] {}", SoapyRuntime::Availability().message);
        return Result::INCOMPLETE;
    }
    SoapyKwargs args;
    JST_CHECK(runtime->kwargsFromString(deviceString, args));
    return openArgs(args);
}

Result SoapyReceiver::openArgs(const SoapyKwargs& args) {
    if (state != State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Device is already open.");
        return Result::ERROR;
    }

    library = SoapyRuntime::Acquire();
    if (!library) {
        JST_ERROR("[MODULE_SOAPY] {}", SoapyRuntime::Availability().message);
        return Result::INCOMPLETE;
    }
    const auto& api = library->api();

    SoapyKwargs deviceArgs;
    {
        SoapyKwargsList devices;
        if (DiscoverDevices(*library, args, devices) != Result::SUCCESS) {
            SoapyDiscovery::ClearCache();
            library.reset();
            return Result::ERROR;
        }
        if (devices.empty()) {
            JST_ERROR("[MODULE_SOAPY] No SoapySDR devices found.");
            library.reset();
            return Result::INCOMPLETE;
        }
        if (devices.size() != 1) {
            JST_ERROR("[MODULE_SOAPY] Device selection is ambiguous.");
            library.reset();
            return Result::INCOMPLETE;
        }
        deviceArgs = devices.front();
        for (const auto& [key, value] : args) {
            deviceArgs.insert_or_assign(key, value);
        }
    }

    const SoapyKwargsView view(deviceArgs);
    device = api.make(view.get());
    if (device == nullptr) {
        const auto openError = failure();
        SoapyDiscovery::ClearCache();
        SoapyKwargsList devices;
        if (DiscoverDevices(*library, args, devices, true) == Result::SUCCESS) {
            if (devices.empty()) {
                JST_ERROR("[MODULE_SOAPY] Selected device is no longer available.");
                library.reset();
                return Result::INCOMPLETE;
            }
            if (devices.size() != 1) {
                JST_ERROR("[MODULE_SOAPY] Device selection is ambiguous.");
                library.reset();
                return Result::INCOMPLETE;
            }
        }
        JST_ERROR("[MODULE_SOAPY] Failed to open device: {}", openError);
        library.reset();
        return Result::ERROR;
    }

    state = State::Open;
    return queryCapabilities();
}

Result SoapyReceiver::queryCapabilities() {
    const auto& api = library->api();

    const auto readRanges = [&](auto function, std::vector<SoapyAbi::Range>& ranges) {
        std::size_t length = 0;
        auto* list = function(device, SoapyAbi::RX, 0, &length);
        const bool ok = api.lastStatus() == 0;
        if (ok && list) {
            ranges.assign(list, list + length);
        }
        api.free(list);
        return ok;
    };
    if (!readRanges(api.getSampleRateRange, sampleRateRanges) ||
        !readRanges(api.getFrequencyRange, frequencyRanges)) {
        JST_ERROR("[MODULE_SOAPY] Failed to get device ranges: {}", failure());
        return Result::ERROR;
    }

    std::size_t settingCount = 0;
    auto* settings = api.getSettingInfo(device, &settingCount);
    if (api.lastStatus() == 0) {
        for (std::size_t i = 0; i < settingCount; ++i) {
            biasTeeSupported |= settings[i].key && std::string(settings[i].key) == "biastee";
        }
    } else {
        JST_WARN("[MODULE_SOAPY] Failed to query optional device settings: {}", failure());
    }
    api.argInfoListClear(settings, settingCount);

    std::size_t antennaCount = 0;
    char** names = api.listAntennas(device, SoapyAbi::RX, 0, &antennaCount);
    if (api.lastStatus() == 0) {
        for (std::size_t i = 0; i < antennaCount; ++i) {
            antennas.emplace_back(names[i] ? names[i] : "");
        }
    } else {
        JST_WARN("[MODULE_SOAPY] Failed to query receive antennas: {}", failure());
    }
    api.stringsClear(&names, antennaCount);

    queryGainRange();

    return Result::SUCCESS;
}

void SoapyReceiver::queryGainRange() {
    gainRange.reset();

    const auto range = library->api().getGainRange(device, SoapyAbi::RX, 0);
    if (library->api().lastStatus() != 0) {
        JST_WARN("[MODULE_SOAPY] Failed to query receive gain range: {}", failure());
        return;
    }
    if (std::isfinite(range.minimum) && std::isfinite(range.maximum) &&
        range.minimum <= range.maximum) {
        gainRange = range;
    }
}

Result SoapyReceiver::validateSettings(const F32 sampleRate, const F32 frequency) const {
    if (!SampleRateInRanges(sampleRateRanges, sampleRate)) {
        JST_ERROR("[MODULE_SOAPY] Sample rate ({:.3f} MHz) not supported.", sampleRate / 1e6);
        return Result::ERROR;
    }

    if (!SoapyRangeContains(frequencyRanges, frequency)) {
        JST_ERROR("[MODULE_SOAPY] Frequency ({:.3f} MHz) not supported.", frequency / 1e6);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

const std::vector<std::string>& SoapyReceiver::listAntennas() const {
    return antennas;
}

bool SoapyReceiver::supportsBiasTee() const {
    return biasTeeSupported;
}

const std::optional<SoapyAbi::Range>& SoapyReceiver::getGainRange() const {
    return gainRange;
}

F32 SoapyReceiver::clampGain(const F32 gain) const {
    if (!gainRange) {
        return gain;
    }

    const F32 minimum = static_cast<F32>(gainRange->minimum);
    const F32 maximum = static_cast<F32>(gainRange->maximum);
    const F32 applied = std::clamp(gain, minimum, maximum);
    if (applied != gain) {
        JST_WARN("[MODULE_SOAPY] Gain ({:.1f} dB) outside device range ({:.1f} to {:.1f} dB). Using {:.1f} dB.",
                 gain, minimum, maximum, applied);
    }
    return applied;
}

Result SoapyReceiver::setAntenna(const std::string& antenna) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set antenna without an active device.");
        return Result::ERROR;
    }
    if (antenna.empty()) {
        return Result::SUCCESS;
    }
    if (std::find(antennas.begin(), antennas.end(), antenna) == antennas.end()) {
        JST_ERROR("[MODULE_SOAPY] Receive antenna '{}' is not supported by the selected device.", antenna);
        return Result::ERROR;
    }

    if (library->api().setAntenna(device, SoapyAbi::RX, 0, antenna.c_str()) != 0) {
        JST_ERROR("[MODULE_SOAPY] Failed to set receive antenna '{}': {}", antenna, failure());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setTunerFrequency(const F32 frequency) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set frequency without an active device.");
        return Result::ERROR;
    }

    if (!SoapyRangeContains(frequencyRanges, frequency)) {
        JST_WARN("[MODULE_SOAPY] Frequency ({:.3f} MHz) not supported.", frequency / 1e6);
        return Result::WARNING;
    }

    const SoapyKwargsView args({});
    if (library->api().setFrequency(device, SoapyAbi::RX, 0, frequency, args.get()) != 0) {
        JST_ERROR("[MODULE_SOAPY] Failed to set frequency: {}", failure());
        return Result::ERROR;
    }

    queryGainRange();

    return Result::SUCCESS;
}

Result SoapyReceiver::setSampleRate(const F32 sampleRate) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set sample rate without an active device.");
        return Result::ERROR;
    }

    if (!SampleRateInRanges(sampleRateRanges, sampleRate)) {
        JST_WARN("[MODULE_SOAPY] Sample rate ({:.3f} MHz) not supported.", sampleRate / 1e6);
        return Result::WARNING;
    }

    if (library->api().setSampleRate(device, SoapyAbi::RX, 0, sampleRate) != 0) {
        JST_ERROR("[MODULE_SOAPY] Failed to set sample rate: {}", failure());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setAutomaticGain(const bool automaticGain) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set gain mode without an active device.");
        return Result::ERROR;
    }

    if (library->api().setGainMode(device, SoapyAbi::RX, 0, automaticGain) != 0) {
        JST_ERROR("[MODULE_SOAPY] Failed to set gain mode: {}", failure());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setGain(const F32 gain) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set gain without an active device.");
        return Result::ERROR;
    }

    if (library->api().setGain(device, SoapyAbi::RX, 0, gain) != 0) {
        JST_ERROR("[MODULE_SOAPY] Failed to set gain: {}", failure());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setBiasTee(const bool enabled) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set Bias-T without an active device.");
        return Result::ERROR;
    }

    if (biasTeeSupported) {
        biasTeeNeedsCleanup = true;
        if (library->api().writeSetting(device, "biastee", enabled ? "true" : "false") != 0) {
            JST_ERROR("[MODULE_SOAPY] Failed to set Bias-T: {}", failure());
            return Result::ERROR;
        }
        biasTeeNeedsCleanup = enabled;
    } else if (enabled) {
        JST_WARN("[MODULE_SOAPY] Bias-T is not supported by the selected device.");
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::startStream(const std::string& streamString) {
    if (!library) {
        JST_ERROR("[MODULE_SOAPY] Cannot setup stream without an active device.");
        return Result::ERROR;
    }
    SoapyKwargs args;
    JST_CHECK(library->kwargsFromString(streamString, args));
    return startStreamArgs(args);
}

Result SoapyReceiver::startStreamArgs(const SoapyKwargs& args) {
    if (state != State::Open) {
        JST_ERROR("[MODULE_SOAPY] Cannot setup stream in the current device state.");
        return Result::ERROR;
    }
    const auto& api = library->api();

    const std::size_t channels[] = {0};
    const SoapyKwargsView view(args);
    stream = api.setupStream(device, SoapyAbi::RX, SoapyAbi::FormatCF32, channels, 1, view.get());
    if (stream == nullptr) {
        JST_ERROR("[MODULE_SOAPY] Failed to setup stream: {}", failure());
        return Result::ERROR;
    }
    state = State::StreamReady;

    const int activationResult = api.activateStream(device, stream, 0, 0, 0);
    if (activationResult != 0) {
        JST_ERROR("[MODULE_SOAPY] Failed to configure device: activateStream returned status {}",
                  activationResult);
        return Result::ERROR;
    }
    state = State::Streaming;

    return Result::SUCCESS;
}

SoapyReceiver::ReadResult SoapyReceiver::read(const std::span<CF32> samples) {
    if (state != State::Streaming) {
        return {.error = "Cannot read without an active stream"};
    }
    if (samples.empty()) {
        return {.error = "Receive buffer cannot be empty"};
    }

    void* buffers[] = {samples.data()};
    int flags = 0;
    long long timeNs = 0;
    const int result = library->api().readStream(device, stream, buffers, samples.size(),
                                                 &flags, &timeNs, 100000);

    if (result > 0) {
        if (static_cast<std::size_t>(result) > samples.size()) {
            return {.error = "Device returned more samples than requested"};
        }
        ReadResult received{
            .status = ReadStatus::Samples,
            .sampleCount = static_cast<std::size_t>(result),
        };
        if (flags & SoapyAbi::HasTime) {
            received.timestampNs = timeNs;
        }
        return received;
    }
    if (result == 0 || result == SoapyAbi::Timeout) {
        return {.status = ReadStatus::Timeout};
    }
    if (result == SoapyAbi::Overflow) {
        return {.status = ReadStatus::Overflow};
    }

    auto error = jst::fmt::format("{} ({})", library->errorString(result), result);
    if (const auto detail = library->lastError(); !detail.empty()) {
        error = jst::fmt::format("{}: {}", error, detail);
    }
    return {.error = error};
}

void SoapyReceiver::reset() {
    if (device && stream) {
        const auto& api = library->api();
        if (state == State::Streaming && api.deactivateStream(device, stream, 0, 0) != 0) {
            JST_ERROR("[MODULE_SOAPY] Failed to deactivate stream: {}", failure());
        }
        if (api.closeStream(device, stream) != 0) {
            JST_ERROR("[MODULE_SOAPY] Failed to close stream: {}", failure());
        }
        stream = nullptr;
        state = State::Open;
    }

    if (biasTeeNeedsCleanup) {
        static_cast<void>(setBiasTee(false));
    }

    if (device) {
        if (library->api().unmake(device) != 0) {
            JST_ERROR("[MODULE_SOAPY] Failed to unmake device: {}", failure());
        }
        device = nullptr;
        state = State::Closed;
    }

    sampleRateRanges.clear();
    frequencyRanges.clear();
    antennas.clear();
    gainRange.reset();
    biasTeeSupported = false;
    biasTeeNeedsCleanup = false;
    library.reset();
}

SoapyReceiveStatus::Action SoapyReceiveStatus::handle(const SoapyReceiver::ReadStatus status) {
    using Status = SoapyReceiver::ReadStatus;
    if (status == Status::Samples) {
        return Action::Samples;
    }
    if (status == Status::Timeout) {
        return Action::Retry;
    }
    if (status != Status::Overflow) {
        return Action::Fail;
    }

    ++deviceOverflows;
    return Action::Retry;
}

}  // namespace Jetstream::Modules
