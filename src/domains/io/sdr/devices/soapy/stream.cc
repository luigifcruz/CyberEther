#include "base.hh"
#include "tools/catalog.hh"

#include "../stream.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

namespace Abi = SoapyAbi;

using Status = Device::Status;
using Clock = std::chrono::steady_clock;

constexpr F64 SettlingTime = 1.0e-3;
constexpr F64 SweepSampleRate = 20.0e6;
constexpr F64 ReadbackTolerance = 1.0e-9;

class SoapySdrStream final : public Stream {
 public:
    ~SoapySdrStream() override;

    Result connect(const Device::Info& info);
    Device::Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) override;
    Device::Capture readChannels(std::span<std::span<CF32>> samples,
                                 std::chrono::milliseconds timeout) override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result tune(F64 frequency) override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;
    Result setChannelGains(const std::vector<F64>& gains) override;

 private:
    std::string failure() const;
    Result queryCapabilities();
    Result setSampleRate(F64 sampleRate);
    Result setFrequency(F64 frequency);
    Result setBiasTee(bool enabled);
    Result refreshGainRange();
    Result applyGain(F64 gain, bool force);

    std::shared_ptr<SoapyLibrary> library;
    Abi::Device* device = nullptr;
    Abi::Stream* stream = nullptr;
    std::vector<Device::Range> reportedRates;
    bool streaming = false;
    bool biasTeeSetting = false;
    bool biasTeeEnabled = false;
    U64 receiveChannels = 1;
    std::vector<F64> channelGains;
    U64 queueGeneration = 0;
    U64 discard = 0;
    Clock::time_point received{};
    std::vector<void*> buffers;
};

SoapySdrStream::~SoapySdrStream() {
    close();
    if (!device) {
        return;
    }
    if (biasTeeEnabled) {
        static_cast<void>(setBiasTee(false));
    }
    if (library->api().unmake(device) != 0) {
        JST_WARN("[SDR_SOAPY] Failed to release {}: {}", deviceInfo.label, failure());
    }
}

std::string SoapySdrStream::failure() const {
    const auto error = library ? library->lastError() : std::string{};
    return error.empty() ? std::string("Unknown driver error.") : error;
}

Result SoapySdrStream::connect(const Device::Info& info) {
    library = SoapyRuntime::Acquire();
    if (!library) {
        JST_ERROR("[SDR_SOAPY] {}", SoapyRuntime::Availability().message);
        return Result::INCOMPLETE;
    }

    SoapyKwargsList devices;
    JST_CHECK(library->enumerate({}, devices));
    std::vector<SoapyCatalog::Entry> matches;
    for (auto& entry : SoapyCatalog::List(devices)) {
        if (entry.info.serial == info.serial) {
            matches.push_back(std::move(entry));
        }
    }
    if (matches.empty()) {
        JST_ERROR("[SDR_SOAPY] Device with serial '{}' is no longer available.", info.serial);
        return Result::INCOMPLETE;
    }
    if (matches.size() != 1) {
        JST_ERROR("[SDR_SOAPY] Several devices report serial '{}'.", info.serial);
        return Result::INCOMPLETE;
    }

    const SoapyKwargsView view(matches.front().args);
    device = library->api().make(view.get());
    if (!device) {
        JST_ERROR("[SDR_SOAPY] Failed to open device '{}': {}", info.label, failure());
        return Result::ERROR;
    }
    deviceInfo = info;
    JST_CHECK(queryCapabilities());

    JST_DEBUG("[SDR_SOAPY] Opened '{}'.", info.label);
    return Result::SUCCESS;
}

Result SoapySdrStream::queryCapabilities() {
    const auto& api = library->api();
    const auto readRanges = [&](auto function, std::vector<Device::Range>& ranges) {
        std::size_t length = 0;
        auto* list = function(device, Abi::RX, 0, &length);
        if (api.lastStatus() != 0) {
            api.free(list);
            return false;
        }
        for (std::size_t i = 0; i < length; ++i) {
            ranges.push_back({list[i].minimum, list[i].maximum});
        }
        api.free(list);
        return true;
    };

    std::vector<Device::Range> frequencies;
    std::vector<Device::Range> rates;
    if (!readRanges(api.getFrequencyRange, frequencies) ||
        !readRanges(api.getSampleRateRange, rates) || frequencies.empty()) {
        JST_ERROR("[SDR_SOAPY] Failed to query device ranges: {}", failure());
        return Result::ERROR;
    }

    auto& caps = deviceCapabilities;
    caps.frequencyRanges = std::move(frequencies);
    reportedRates = rates;
    caps.sampleRateRanges = {{std::numeric_limits<F64>::min(), std::numeric_limits<F64>::max()}};
    caps.sweepSampleRate = reportedRates.empty()
        ? SweepSampleRate
        : SoapyCatalog::SweepRate(reportedRates, SweepSampleRate);
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 20.0e3;
    defaultSettlingTime = SettlingTime;

    const std::size_t channels = api.getNumChannels(device, Abi::RX);
    caps.receiveChannels = api.lastStatus() == 0 && channels > 0 ? channels : 1;

    std::size_t antennaCount = 0;
    char** names = api.listAntennas(device, Abi::RX, 0, &antennaCount);
    if (api.lastStatus() == 0) {
        for (std::size_t i = 0; i < antennaCount; ++i) {
            if (names[i]) {
                caps.antennas.emplace_back(names[i]);
            }
        }
    }
    api.stringsClear(&names, antennaCount);

    caps.automaticGain = api.hasGainMode(device, Abi::RX, 0) && api.lastStatus() == 0;

    std::size_t settingCount = 0;
    auto* settings = api.getSettingInfo(device, &settingCount);
    if (api.lastStatus() == 0) {
        for (std::size_t i = 0; i < settingCount; ++i) {
            biasTeeSetting = biasTeeSetting ||
                (settings[i].key && std::string(settings[i].key) == "biastee");
        }
    }
    api.argInfoListClear(settings, settingCount);
    caps.biasTee = biasTeeSetting;
    deviceInfo.biasTee = biasTeeSetting;
    biasTeeEnabled = biasTeeSetting;

    F64 fullScale = 0.0;
    api.free(api.getNativeStreamFormat(device, Abi::RX, 0, &fullScale));
    caps.fullScale = fullScale;

    return refreshGainRange();
}

Result SoapySdrStream::apply(const Settings& settings) {
    const auto& api = library->api();
    if (!std::isfinite(settings.sampleRate) || settings.sampleRate <= 0.0) {
        JST_ERROR("[SDR_SOAPY] Sample rate ({:.3f} MHz) is not valid.", settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!reportedRates.empty() && !Device::Range::Contains(reportedRates, settings.sampleRate)) {
        JST_WARN("[SDR_SOAPY] Sample rate ({:.3f} MHz) is not one that {} reports. "
                 "Requesting it anyway.", settings.sampleRate / 1e6, deviceInfo.label);
    }
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_SOAPY] Frequency ({:.3f} MHz) not supported by {}. Use {}.",
                  settings.frequency / 1e6, deviceInfo.label,
                  Device::Range::Describe(deviceCapabilities.frequencyRanges));
        return Result::ERROR;
    }

    receiveChannels = settings.receiveChannels;
    channelGains = settings.channelGains;

    if (!settings.antenna.empty()) {
        const auto& antennas = deviceCapabilities.antennas;
        if (std::find(antennas.begin(), antennas.end(), settings.antenna) == antennas.end()) {
            JST_ERROR("[SDR_SOAPY] Receive antenna '{}' is not supported.", settings.antenna);
            return Result::ERROR;
        }
        for (U64 channel = 0; channel < receiveChannels; ++channel) {
            if (api.setAntenna(device, Abi::RX, channel, settings.antenna.c_str()) != 0) {
                JST_ERROR("[SDR_SOAPY] Failed to select antenna '{}': {}", settings.antenna,
                          failure());
                return Result::ERROR;
            }
        }
    }
    if (settings.biasTee && !biasTeeSetting) {
        JST_WARN("[SDR_SOAPY] Bias-T is not supported by {}.", deviceInfo.label);
    }
    if (biasTeeSetting && (settings.biasTee || biasTeeEnabled)) {
        JST_CHECK(setBiasTee(settings.biasTee));
    }

    JST_CHECK(setSampleRate(settings.sampleRate));
    for (U64 channel = 0; channel < receiveChannels; ++channel) {
        if (api.setBandwidth(device, Abi::RX, channel, appliedSettings.sampleRate) != 0) {
            JST_WARN("[SDR_SOAPY] Failed to set the RF bandwidth: {}", failure());
            break;
        }
    }
    JST_CHECK(setFrequency(settings.frequency));
    requestedSettings = settings;
    JST_CHECK(setAutomaticGain(settings.automaticGain));
    JST_CHECK(applyGain(settings.gain, true));

    appliedSettings.antenna = settings.antenna;
    appliedSettings.biasTee = settings.biasTee && biasTeeSetting;
    return Result::SUCCESS;
}

Result SoapySdrStream::setBiasTee(const bool enabled) {
    if (library->api().writeSetting(device, "biastee", enabled ? "true" : "false") != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to set Bias-T: {}", failure());
        return Result::ERROR;
    }
    biasTeeEnabled = enabled;
    return Result::SUCCESS;
}

Result SoapySdrStream::setSampleRate(const F64 sampleRate) {
    const auto& api = library->api();
    for (U64 channel = 0; channel < receiveChannels; ++channel) {
        if (api.setSampleRate(device, Abi::RX, channel, sampleRate) != 0) {
            JST_ERROR("[SDR_SOAPY] Failed to set the sample rate: {}", failure());
            return Result::ERROR;
        }
    }
    const F64 applied = api.getSampleRate(device, Abi::RX, 0);
    if (!std::isfinite(applied) || applied < 1.0) {
        JST_ERROR("[SDR_SOAPY] Invalid applied sample rate: {}", applied);
        return Result::ERROR;
    }
    const bool requested = std::abs(applied - sampleRate) <= sampleRate * ReadbackTolerance;
    appliedSettings.sampleRate = requested ? sampleRate : applied;
    return Result::SUCCESS;
}

Result SoapySdrStream::setFrequency(const F64 frequency) {
    const auto& api = library->api();
    for (U64 channel = 0; channel < receiveChannels; ++channel) {
        if (api.setFrequency(device, Abi::RX, channel, frequency, nullptr) != 0) {
            JST_ERROR("[SDR_SOAPY] Failed to tune to {:.3f} MHz: {}", frequency / 1e6,
                      failure());
            return Result::ERROR;
        }
    }
    const F64 applied = api.getFrequency(device, Abi::RX, 0);
    appliedSettings.frequency = std::isfinite(applied) && applied > 0.0 ? applied : frequency;
    return refreshGainRange();
}

Result SoapySdrStream::open() {
    const auto& api = library->api();
    SoapyKwargs arguments;
    JST_CHECK(library->kwargsFromString(requestedSettings.driverArguments, arguments));

    std::vector<std::size_t> channels(receiveChannels);
    for (U64 channel = 0; channel < receiveChannels; ++channel) {
        channels[channel] = channel;
    }
    const SoapyKwargsView view(arguments);
    stream = api.setupStream(device, Abi::RX, Abi::FormatCF32, channels.data(),
                             channels.size(), view.get());
    if (!stream) {
        JST_ERROR("[SDR_SOAPY] Failed to set up the receive stream: {}", failure());
        return Result::ERROR;
    }
    if (const int status = api.activateStream(device, stream, 0, 0, 0); status != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to start the receive stream: {}",
                  library->errorString(status));
        close();
        return Result::ERROR;
    }
    streaming = true;
    queueGeneration = nextGeneration();
    discard = guard();
    received = Clock::now();
    return Result::SUCCESS;
}

void SoapySdrStream::close() {
    if (!stream) {
        return;
    }
    const auto& api = library->api();
    if (streaming && api.deactivateStream(device, stream, 0, 0) != 0) {
        JST_WARN("[SDR_SOAPY] Failed to stop the receive stream: {}", failure());
    }
    streaming = false;
    if (api.closeStream(device, stream) != 0) {
        JST_WARN("[SDR_SOAPY] Failed to close the receive stream: {}", failure());
    }
    stream = nullptr;
}

Result SoapySdrStream::tune(const F64 frequency) {
    JST_CHECK(setFrequency(frequency));
    requestedSettings.frequency = frequency;
    if (streaming) {
        queueGeneration = nextGeneration();
        discard = guard();
    }
    JST_CHECK(applyGain(requestedSettings.gain, true));
    return Result::SUCCESS;
}

Device::Capture SoapySdrStream::read(std::span<CF32> samples,
                                     const std::chrono::milliseconds timeout) {
    std::array destinations{samples};
    return readChannels(destinations, timeout);
}

Device::Capture SoapySdrStream::readChannels(std::span<std::span<CF32>> samples,
                                             const std::chrono::milliseconds timeout) {
    const auto finish = [&](Device::Capture capture) {
        capture.generation = queueGeneration;
        capture.frequency = appliedSettings.frequency;
        capture.sampleRate = appliedSettings.sampleRate;
        return capture;
    };

    if (!streaming) {
        return {.status = Status::Failure, .error = "stream is not running"};
    }
    if (samples.size() != receiveChannels || samples.front().empty() ||
        std::any_of(samples.begin(), samples.end(), [&](const auto channel) {
            return channel.size() != samples.front().size();
        })) {
        return {.status = Status::Failure, .error = "invalid receive channel buffers"};
    }
    const U64 size = samples.front().size();

    buffers.clear();
    for (const auto channel : samples) {
        buffers.push_back(channel.data());
    }
    int flags = 0;
    long long timeNs = 0;
    const auto timeoutUs = std::chrono::duration_cast<std::chrono::microseconds>(timeout);
    const int status = library->api().readStream(device, stream, buffers.data(), size, &flags,
                                                 &timeNs, static_cast<long>(timeoutUs.count()));

    if (status > 0 && static_cast<U64>(status) > size) {
        return finish({.status = Status::Failure,
                       .error = jst::fmt::format("{} returned more samples than requested",
                                                 deviceInfo.label)});
    }
    if (status > 0) {
        received = Clock::now();
        const U64 count = static_cast<U64>(status);
        const U64 skip = std::min(discard, count);
        discard -= skip;
        if (skip == count) {
            return finish({.status = Status::Timeout});
        }
        if (skip > 0) {
            for (const auto channel : samples) {
                std::copy(channel.begin() + skip, channel.begin() + count, channel.begin());
            }
        }
        return finish({.status = Status::Samples, .count = count - skip});
    }
    if (status == 0 || status == Abi::Timeout) {
        if (Clock::now() - received > Device::StallTimeout) {
            return finish({.status = Status::Disconnected,
                           .error = jst::fmt::format("{} stopped delivering samples",
                                                     deviceInfo.label)});
        }
        return finish({.status = Status::Timeout});
    }
    if (status == Abi::Overflow) {
        return finish({.status = Status::Discontinuity});
    }
    auto error = jst::fmt::format("{} ({})", library->errorString(status), status);
    if (const auto detail = library->lastError(); !detail.empty()) {
        error = jst::fmt::format("{}: {}", error, detail);
    }
    return finish({.status = Status::Failure, .error = std::move(error)});
}

Result SoapySdrStream::setAutomaticGain(const bool automaticGain) {
    const auto& api = library->api();
    if (deviceCapabilities.automaticGain) {
        for (U64 channel = 0; channel < receiveChannels; ++channel) {
            if (api.setGainMode(device, Abi::RX, channel, automaticGain) != 0) {
                JST_ERROR("[SDR_SOAPY] Failed to set the gain mode: {}", failure());
                return Result::ERROR;
            }
        }
    } else if (automaticGain) {
        JST_WARN("[SDR_SOAPY] Automatic gain is not supported by {}. Using the manual gain.",
                 deviceInfo.label);
    }
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain && deviceCapabilities.automaticGain;
    return appliedSettings.automaticGain ? Result::SUCCESS
                                         : applyGain(requestedSettings.gain, true);
}

Result SoapySdrStream::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

Result SoapySdrStream::setChannelGains(const std::vector<F64>& gains) {
    channelGains = gains;
    JST_CHECK(applyGain(requestedSettings.gain, false));
    requestedSettings.channelGains = gains;
    return Result::SUCCESS;
}

Result SoapySdrStream::refreshGainRange() {
    const auto range = library->api().getGainRange(device, Abi::RX, 0);
    if (library->api().lastStatus() != 0 || !std::isfinite(range.minimum) ||
        !std::isfinite(range.maximum) || range.minimum > range.maximum) {
        appliedSettings.gainRange.reset();
        return Result::SUCCESS;
    }
    appliedSettings.gainRange = Device::Range{range.minimum, range.maximum};
    return Result::SUCCESS;
}

Result SoapySdrStream::applyGain(const F64 gain, const bool force) {
    appliedSettings.channelGains.resize(receiveChannels, std::numeric_limits<F64>::quiet_NaN());
    appliedSettings.gain = resolveGain(gain, appliedSettings.gainRange);
    if (appliedSettings.automaticGain) {
        return Result::SUCCESS;
    }
    for (U64 channel = 0; channel < receiveChannels; ++channel) {
        const F64 requested = channel < channelGains.size() ? channelGains[channel] : gain;
        const F64 target = resolveGain(requested, appliedSettings.gainRange);
        if (!force && target == appliedSettings.channelGains[channel]) {
            continue;
        }
        if (library->api().setGain(device, Abi::RX, channel, target) != 0) {
            JST_ERROR("[SDR_SOAPY] Failed to set the RX{} gain to {:.1f} dB: {}", channel + 1,
                      target, failure());
            appliedSettings.channelGains[channel] = std::numeric_limits<F64>::quiet_NaN();
            return Result::ERROR;
        }
        appliedSettings.channelGains[channel] = target;
    }
    return Result::SUCCESS;
}

}  // namespace

Result SoapySdr::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<SoapySdrStream>();
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
