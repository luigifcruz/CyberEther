#include "soapysdr.hh"

#include <algorithm>
#include <exception>
#include <future>
#include <memory>
#include <mutex>

#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Errors.hpp>
#include <SoapySDR/Modules.hpp>
#include <SoapySDR/Registry.hpp>

#include <jetstream/logger.hh>

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#endif

namespace Jetstream::Modules {

namespace {

using DiscoveryClock = SoapyDiscovery::Clock;
constexpr auto DiscoveryLifetime = std::chrono::seconds(1);

struct DiscoveryEntry {
    std::shared_future<SoapySDR::KwargsList> result;
    DiscoveryClock::time_point expiresAt{};
    bool ready = false;
};

struct DiscoveryCache {
    std::mutex mutex;
    std::map<SoapySDR::Kwargs, std::shared_ptr<DiscoveryEntry>> entries;
#ifdef JST_OS_BROWSER
    int usbRevision = -1;
#endif
};

DiscoveryCache& GetDiscoveryCache() {
    static DiscoveryCache cache;
    return cache;
}

#ifdef JST_OS_BROWSER
int WebUsbRevision() {
    return MAIN_THREAD_EM_ASM_INT({
        if (!navigator.usb) {
            return -1;
        }
        let state = Module['soapyUsbDiscovery'];
        if (!state || state.usb !== navigator.usb) {
            state = ({
                usb: navigator.usb,
                revision: state ? (state.revision + 1) & 0x7fffffff : 0
            });
            Module['soapyUsbDiscovery'] = state;
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

SoapySDR::KwargsList DiscoverDevices(const SoapySDR::Kwargs& args, const bool refresh = false,
                                    const std::optional<DiscoveryClock::time_point> now = std::nullopt) {
#ifdef JST_OS_BROWSER
    const auto usbRevision = WebUsbRevision();
    if (usbRevision < 0) {
        JST_ERROR("[MODULE_SOAPY] Browser not compatible with WebUSB.");
        return {};
    }
#endif

    auto& cache = GetDiscoveryCache();
    std::shared_ptr<DiscoveryEntry> entry;
    std::optional<std::promise<SoapySDR::KwargsList>> pending;
    {
        std::lock_guard lock(cache.mutex);
#ifdef JST_OS_BROWSER
        if (cache.usbRevision != usbRevision) {
            cache.entries.clear();
            cache.usbRevision = usbRevision;
        }
#endif
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
        bool success = false;
        try {
            pending->set_value(SoapySDR::Device::enumerate(args));
            success = true;
        } catch (...) {
            pending->set_exception(std::current_exception());
        }

        std::lock_guard lock(cache.mutex);
        if (success) {
            entry->expiresAt = (now ? *now : DiscoveryClock::now()) + DiscoveryLifetime;
            entry->ready = true;
        } else if (const auto it = cache.entries.find(args);
                   it != cache.entries.end() && it->second == entry) {
            cache.entries.erase(it);
        }
    }

    return entry->result.get();
}

}  // namespace

SoapyDiscovery::DeviceList SoapyDiscovery::ListDevices(const std::string& filter, const bool refresh,
                                                     const std::optional<Clock::time_point> now) {
    try {
        return BuildDeviceList(DiscoverDevices(SoapySDR::KwargsFromString(filter), refresh, now));
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to enumerate devices: {}", e.what());
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to enumerate devices.");
    }
    return {};
}

void SoapyDiscovery::ClearDiscoveryCache() {
    auto& cache = GetDiscoveryCache();
    std::lock_guard lock(cache.mutex);
    cache.entries.clear();
}

Result SoapyDiscovery::LoadDriverLibrary(const std::string& path) {
    if (path.empty()) {
        return Result::SUCCESS;
    }

    try {
        const auto error = SoapySDR::loadModule(path);
        if (error.empty()) {
            ClearDiscoveryCache();
            return Result::SUCCESS;
        }
        if (error.ends_with(" already loaded")) {
            return Result::SUCCESS;
        }
        JST_ERROR("[MODULE_SOAPY] Failed to load SoapySDR module '{}': {}", path, error);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to load SoapySDR module '{}': {}", path, e.what());
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to load SoapySDR module '{}'.", path);
    }
    return Result::ERROR;
}

SoapyDiscovery::DeviceList SoapyDiscovery::BuildDeviceList(const SoapySDR::KwargsList& entries) {
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

SoapyReceiver::~SoapyReceiver() {
    reset();
}

Result SoapyReceiver::open(const SoapySDR::Kwargs& args) {
    if (state != State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Device is already open.");
        return Result::ERROR;
    }

    try {
        const auto& findFuncs = SoapySDR::Registry::listFindFunctions();
        JST_DEBUG("[MODULE_SOAPY] Registered SoapySDR drivers ({}):", findFuncs.size());
        for (const auto& [name, _] : findFuncs) {
            JST_DEBUG("[MODULE_SOAPY]   - {}", name);
        }
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to enumerate drivers.");
        return Result::ERROR;
    }

    SoapySDR::Kwargs deviceArgs;
    try {
        const auto devices = DiscoverDevices(args);
        if (devices.empty()) {
            JST_ERROR("[MODULE_SOAPY] No SoapySDR devices found.");
            return Result::INCOMPLETE;
        }
        if (devices.size() != 1) {
            JST_ERROR("[MODULE_SOAPY] Device selection is ambiguous.");
            return Result::INCOMPLETE;
        }
        deviceArgs = devices.front();
        deviceArgs.insert(args.begin(), args.end());
    } catch (const std::exception& e) {
        SoapyDiscovery::ClearDiscoveryCache();
        JST_ERROR("[MODULE_SOAPY] Failed to discover device: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        SoapyDiscovery::ClearDiscoveryCache();
        JST_ERROR("[MODULE_SOAPY] Failed to discover device.");
        return Result::ERROR;
    }

    std::string openError;
    try {
        device = SoapySDR::Device::make(deviceArgs);
    } catch (const std::exception& e) {
        openError = e.what();
    } catch (...) {
        openError = "Unknown driver error.";
    }

    if (device == nullptr) {
        SoapyDiscovery::ClearDiscoveryCache();
        try {
            const auto devices = DiscoverDevices(args, true);
            if (devices.empty()) {
                JST_ERROR("[MODULE_SOAPY] Selected device is no longer available.");
                return Result::INCOMPLETE;
            }
            if (devices.size() != 1) {
                JST_ERROR("[MODULE_SOAPY] Device selection is ambiguous.");
                return Result::INCOMPLETE;
            }
        } catch (const std::exception& e) {
            JST_DEBUG("[MODULE_SOAPY] Failed to refresh discovery after opening failure: {}", e.what());
        } catch (...) {
            JST_DEBUG("[MODULE_SOAPY] Failed to refresh discovery after opening failure.");
        }
        JST_ERROR("[MODULE_SOAPY] Failed to open device: {}",
                  openError.empty() ? "Driver returned no device." : openError);
        return Result::ERROR;
    }

    state = State::Open;
    return queryCapabilities();
}

Result SoapyReceiver::queryCapabilities() {
    try {
        sampleRateRanges = device->getSampleRateRange(SOAPY_SDR_RX, 0);
        frequencyRanges = device->getFrequencyRange(SOAPY_SDR_RX, 0);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to get device ranges: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to get device ranges.");
        return Result::ERROR;
    }

    try {
        const auto settings = device->getSettingInfo();
        biasTeeSupported = std::any_of(settings.begin(), settings.end(),
            [](const auto& setting) { return setting.key == "biastee"; });
    } catch (const std::exception& e) {
        JST_WARN("[MODULE_SOAPY] Failed to query optional device settings: {}", e.what());
    } catch (...) {
        JST_WARN("[MODULE_SOAPY] Failed to query optional device settings.");
    }

    try {
        antennas = device->listAntennas(SOAPY_SDR_RX, 0);
    } catch (const std::exception& e) {
        JST_WARN("[MODULE_SOAPY] Failed to query receive antennas: {}", e.what());
    } catch (...) {
        JST_WARN("[MODULE_SOAPY] Failed to query receive antennas.");
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::validateSettings(const F32 sampleRate,
                                       const F32 frequency) const {
    if (!SoapyRangeContains(sampleRateRanges, sampleRate)) {
        JST_ERROR("[MODULE_SOAPY] Sample rate ({:.2f} MHz) not supported.", sampleRate / 1e6);
        return Result::ERROR;
    }

    if (!SoapyRangeContains(frequencyRanges, frequency)) {
        JST_ERROR("[MODULE_SOAPY] Frequency ({:.2f} MHz) not supported.", frequency / 1e6);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

const std::vector<std::string>& SoapyReceiver::listAntennas() const {
    return antennas;
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

    try {
        device->setAntenna(SOAPY_SDR_RX, 0, antenna);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set receive antenna '{}': {}", antenna, e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set receive antenna '{}'.", antenna);
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
        JST_WARN("[MODULE_SOAPY] Frequency ({:.2f} MHz) not supported.", frequency / 1e6);
        return Result::WARNING;
    }

    try {
        device->setFrequency(SOAPY_SDR_RX, 0, frequency);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set frequency: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set frequency.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setSampleRate(const F32 sampleRate) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set sample rate without an active device.");
        return Result::ERROR;
    }

    if (!SoapyRangeContains(sampleRateRanges, sampleRate)) {
        JST_WARN("[MODULE_SOAPY] Sample rate ({:.2f} MHz) not supported.", sampleRate / 1e6);
        return Result::WARNING;
    }

    try {
        device->setSampleRate(SOAPY_SDR_RX, 0, sampleRate);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set sample rate: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set sample rate.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setAutomaticGain(const bool automaticGain) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set gain mode without an active device.");
        return Result::ERROR;
    }

    try {
        device->setGainMode(SOAPY_SDR_RX, 0, automaticGain);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set gain mode: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set gain mode.");
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
        try {
            device->writeSetting("biastee", enabled);
        } catch (const std::exception& e) {
            JST_ERROR("[MODULE_SOAPY] Failed to set Bias-T: {}", e.what());
            return Result::ERROR;
        } catch (...) {
            JST_ERROR("[MODULE_SOAPY] Failed to set Bias-T.");
            return Result::ERROR;
        }
        biasTeeNeedsCleanup = enabled;
    } else if (enabled) {
        JST_WARN("[MODULE_SOAPY] Bias-T is not supported by the selected device.");
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::startStream(const SoapySDR::Kwargs& args) {
    if (state != State::Open) {
        JST_ERROR("[MODULE_SOAPY] Cannot setup stream in the current device state.");
        return Result::ERROR;
    }

    try {
        stream = device->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0}, args);
        if (stream == nullptr) {
            JST_ERROR("[MODULE_SOAPY] Failed to setup stream.");
            return Result::ERROR;
        }
        state = State::StreamReady;

        const int activationResult = device->activateStream(stream, 0, 0, 0);
        if (activationResult != 0) {
            JST_ERROR("[MODULE_SOAPY] Failed to configure device: activateStream returned status {}",
                      activationResult);
            return Result::ERROR;
        }
        state = State::Streaming;
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to configure device: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to configure device.");
        return Result::ERROR;
    }

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
    int result;
    try {
        result = device->readStream(stream, buffers, samples.size(), flags, timeNs, 100000);
    } catch (const std::exception& e) {
        return {.error = e.what()};
    } catch (...) {
        return {.error = "Unknown driver exception"};
    }

    if (result > 0) {
        if (static_cast<std::size_t>(result) > samples.size()) {
            return {.error = "Device returned more samples than requested"};
        }
        ReadResult received{
            .status = ReadStatus::Samples,
            .sampleCount = static_cast<std::size_t>(result),
        };
        if (flags & SOAPY_SDR_HAS_TIME) {
            received.timestampNs = timeNs;
        }
        return received;
    }
    if (result == 0 || result == SOAPY_SDR_TIMEOUT) {
        return {.status = ReadStatus::Timeout};
    }
    if (result == SOAPY_SDR_OVERFLOW) {
        return {.status = ReadStatus::Overflow};
    }

    return {.error = jst::fmt::format("{} ({})", SoapySDR::errToStr(result), result)};
}

void SoapyReceiver::reset() {
    if (device && stream) {
        try {
            if (state == State::Streaming) {
                device->deactivateStream(stream, 0, 0);
                state = State::StreamReady;
            }
            device->closeStream(stream);
        } catch (const std::exception& e) {
            JST_ERROR("[MODULE_SOAPY] Failed to deactivate/close stream: {}", e.what());
        } catch (...) {
            JST_ERROR("[MODULE_SOAPY] Failed to deactivate/close stream.");
        }
        stream = nullptr;
        state = State::Open;
    }

    if (biasTeeNeedsCleanup) {
        static_cast<void>(setBiasTee(false));
    }

    if (device) {
        try {
            SoapySDR::Device::unmake(device);
        } catch (const std::exception& e) {
            JST_ERROR("[MODULE_SOAPY] Failed to unmake device: {}", e.what());
        } catch (...) {
            JST_ERROR("[MODULE_SOAPY] Failed to unmake device.");
        }
        device = nullptr;
        state = State::Closed;
    }

    sampleRateRanges.clear();
    frequencyRanges.clear();
    antennas.clear();
    biasTeeSupported = false;
    biasTeeNeedsCleanup = false;
}

SoapyReceiveStatus::Action SoapyReceiveStatus::handle(const SoapyReceiver::ReadStatus status,
                                                     const Clock::time_point now) {
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
    if (deviceOverflows == 1 || now >= nextWarning) {
        nextWarning = now + std::chrono::seconds(1);
        return Action::WarnOverflow;
    }
    return Action::Retry;
}

}  // namespace Jetstream::Modules
