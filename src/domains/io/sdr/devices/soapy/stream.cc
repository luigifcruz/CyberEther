#include "base.hh"
#include "tools/catalog.hh"

#include "../stream.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
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
constexpr U64 DrainSamples = 16384;
constexpr long DrainTimeoutUs = 1000;
constexpr auto DrainLimit = std::chrono::milliseconds(100);
constexpr long long TimedStartNs = 100000000;

struct SoapyFormat {
    DataType type;
    const char* name;
    F64 fullScale;
};

constexpr std::array<SoapyFormat, 5> SoapyFormats = {{
    {DataType::CF32, Abi::FormatCF32, 1.0},
    {DataType::CF64, Abi::FormatCF64, 1.0},
    {DataType::CI16, Abi::FormatCS16, 32768.0},
    {DataType::CI8, Abi::FormatCS8, 128.0},
    {DataType::CU8, Abi::FormatCU8, 128.0},
}};

const char* SoapyFormatName(const DataType type) {
    for (const auto& format : SoapyFormats) {
        if (format.type == type) {
            return format.name;
        }
    }
    return Abi::FormatCF32;
}

class SoapySdrStream final : public Stream {
 public:
    ~SoapySdrStream() override;

    Result connect(const Device::Info& info);
    Device::Capture read(std::span<std::byte> samples, std::chrono::milliseconds timeout) override;
    Device::Capture readChannels(std::span<std::span<std::byte>> samples,
                                 std::chrono::milliseconds timeout) override;
    F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const override;

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
    std::optional<U64> timestampOf(int flags, long long timeNs, U64 count);
    void restartTimestamps();
    void markCutoff();
    void drain();
    int activate();
    U64 staleSamples(long long timeNs, U64 count) const;

    std::shared_ptr<SoapyLibrary> library;
    std::vector<std::byte> drainBuffer;
    Abi::Device* device = nullptr;
    Abi::Stream* stream = nullptr;
    std::vector<Device::Range> reportedRates;
    bool streaming = false;
    bool biasTeeSetting = false;
    bool biasTeeEnabled = false;
    bool manualBandwidthSet = false;
    F64 gainStep = 0.0;
    std::string biasTeeKey;
    bool hardwareTime = false;
    bool hardwareTimed = false;
    U64 receiveChannels = 1;
    std::vector<F64> channelGains;
    std::vector<std::vector<std::string>> channelAntennaLists;
    std::vector<std::string> openedAntennas;
    std::vector<bool> antennasOverridden;
    F64 driverRate = 0.0;

    U64 queueGeneration = 0;
    U64 discard = 0;
    std::optional<long long> cutoffNs;
    std::optional<long long> lastTimeNs;
    std::optional<F64> expectedNs;
    bool timestamped = false;
    Clock::time_point cutoffSince{};
    Clock::time_point received{};
    std::vector<void*> buffers;

    std::optional<U64> nextTimestamp;
    U64 anchorTimestamp = 0;
    long long anchorTimeNs = 0;
    U64 timestampEpoch = 0;
    bool abruptEnd = false;
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

    std::vector<Device::Range> bandwidths;
    if (!readRanges(api.getBandwidthRange, bandwidths)) {
        bandwidths.clear();
    }
    std::erase_if(bandwidths, [](const auto& range) {
        return !(range.maximum > 0.0) || range.minimum > range.maximum;
    });

    auto& caps = deviceCapabilities;
    caps.frequencyRanges = std::move(frequencies);
    caps.bandwidthRanges = std::move(bandwidths);
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
    if (caps.receiveChannels > 1) {
        std::vector<std::size_t> indices(caps.receiveChannels);
        for (U64 channel = 0; channel < caps.receiveChannels; ++channel) {
            indices[channel] = channel;
        }
        const SoapyKwargs none;
        const SoapyKwargsView view(none);
        auto* probe = api.setupStream(device, Abi::RX, Abi::FormatCF32, indices.data(),
                                      indices.size(), view.get());
        if (probe) {
            api.closeStream(device, probe);
        } else {
            caps.receiveChannels = 1;
        }
    }

    channelAntennaLists.assign(caps.receiveChannels, {});
    openedAntennas.assign(caps.receiveChannels, {});
    antennasOverridden.assign(caps.receiveChannels, false);
    for (U64 channel = 0; channel < caps.receiveChannels; ++channel) {
        char* opened = api.getAntenna(device, Abi::RX, channel);
        if (api.lastStatus() == 0 && opened) {
            openedAntennas[channel] = opened;
        }
        api.free(opened);

        std::size_t antennaCount = 0;
        char** names = api.listAntennas(device, Abi::RX, channel, &antennaCount);
        if (api.lastStatus() == 0) {
            for (std::size_t i = 0; i < antennaCount; ++i) {
                if (!names[i]) {
                    continue;
                }
                channelAntennaLists[channel].emplace_back(names[i]);
            }
        }
        api.stringsClear(&names, antennaCount);
    }
    caps.antennas = channelAntennaLists.front();

    caps.automaticGain = api.hasGainMode(device, Abi::RX, 0) && api.lastStatus() == 0;

    std::size_t settingCount = 0;
    auto* settings = api.getSettingInfo(device, &settingCount);
    if (api.lastStatus() == 0) {
        for (std::size_t i = 0; i < settingCount; ++i) {
            const std::string key = settings[i].key ? settings[i].key : "";
            if (key == "biastee" || (key == "biastee_rx" && biasTeeKey.empty())) {
                biasTeeKey = key;
            }
        }
    }
    api.argInfoListClear(settings, settingCount);
    if (!biasTeeKey.empty() && api.writeSetting(device, biasTeeKey.c_str(), "false") != 0) {
        biasTeeKey.clear();
    }
    biasTeeSetting = !biasTeeKey.empty();
    caps.biasTee = biasTeeSetting;
    deviceInfo.biasTee = biasTeeSetting;
    biasTeeEnabled = false;
    hardwareTime = api.hasHardwareTime(device, "");

    F64 fullScale = 0.0;
    char* native = api.getNativeStreamFormat(device, Abi::RX, 0, &fullScale);
    const std::string nativeFormat = native ? native : "";
    api.free(native);
    caps.fullScale = fullScale;

    std::size_t formatCount = 0;
    char** formats = api.getStreamFormats(device, Abi::RX, 0, &formatCount);
    caps.formats = {{}};
    if (api.lastStatus() == 0 && formats) {
        for (const auto& format : SoapyFormats) {
            const bool listed = std::any_of(formats, formats + formatCount, [&](const char* name) {
                return name && std::string(name) == format.name;
            });
            if (format.type == DataType::CF32 || !listed) {
                continue;
            }
            const bool scaled = format.name == nativeFormat && fullScale > 0.0;
            caps.formats.push_back({format.type, scaled ? fullScale : format.fullScale});
        }
    }
    if (formats) {
        api.stringsClear(&formats, formatCount);
    }

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
    F64 frequency = settings.frequency;
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, frequency)) {
        // SDR block settings are F32. An exact hardware range endpoint may
        // round just outside the F64 range when represented in that field.
        // Only snap a value that is itself exactly representable as F32 and
        // matches the F32 representation of an endpoint.
        bool matchedEndpoint = false;
        if (std::isfinite(frequency) && static_cast<F64>(static_cast<F32>(frequency)) == frequency) {
            for (const auto& range : deviceCapabilities.frequencyRanges) {
                for (const auto endpoint : {range.minimum, range.maximum}) {
                    if (std::isfinite(endpoint) && static_cast<F64>(static_cast<F32>(endpoint)) == frequency) {
                        frequency = endpoint;
                        matchedEndpoint = true;
                        break;
                    }
                }
                if (matchedEndpoint) break;
            }
        }
        if (!matchedEndpoint) {
            JST_ERROR("[SDR_SOAPY] Frequency ({:.3f} MHz) not supported by {}. Use {}.",
                      settings.frequency / 1e6, deviceInfo.label,
                      Device::Range::Describe(deviceCapabilities.frequencyRanges));
            return Result::ERROR;
        }
    }

    receiveChannels = settings.receiveChannels;
    channelGains = settings.channelGains;

    for (U64 channel = 0; channel < receiveChannels; ++channel) {
        const auto& selected = channel < settings.channelAntennas.size()
            ? settings.channelAntennas[channel]
            : settings.antenna;
        if (selected.empty() && !antennasOverridden[channel]) {
            continue;
        }
        const auto& antenna = selected.empty() ? openedAntennas[channel] : selected;
        const auto& antennas = channelAntennaLists[channel];
        const bool listed = std::find(antennas.begin(), antennas.end(), antenna) != antennas.end();
        if (antenna.empty() || (selected.empty() && !listed)) {
            continue;
        }
        if (!listed) {
            JST_ERROR("[SDR_SOAPY] Receive antenna '{}' is not supported on RX{}.", antenna,
                      channel + 1);
            return Result::ERROR;
        }
        if (api.setAntenna(device, Abi::RX, channel, antenna.c_str()) != 0) {
            JST_ERROR("[SDR_SOAPY] Failed to select antenna '{}': {}", antenna, failure());
            return Result::ERROR;
        }
        antennasOverridden[channel] = !selected.empty();
    }
    if (settings.biasTee && !biasTeeSetting) {
        JST_WARN("[SDR_SOAPY] Bias-T is not supported by {}.", deviceInfo.label);
    }
    if (biasTeeSetting && (settings.biasTee || biasTeeEnabled)) {
        JST_CHECK(setBiasTee(settings.biasTee));
    }

    JST_CHECK(setSampleRate(settings.sampleRate));
    const bool manualBandwidth = settings.bandwidth > 0.0 &&
                                 !deviceCapabilities.bandwidthRanges.empty();
    const F64 bandwidth = manualBandwidth ? settings.bandwidth : appliedSettings.sampleRate;
    for (U64 channel = 0; channel < receiveChannels; ++channel) {
        if (api.setBandwidth(device, Abi::RX, channel, bandwidth) != 0) {
            if (manualBandwidth || manualBandwidthSet) {
                JST_ERROR("[SDR_SOAPY] Failed to set the filter bandwidth to {:.3f} MHz: {}",
                          bandwidth / 1e6, failure());
                return Result::ERROR;
            }
            JST_WARN("[SDR_SOAPY] Failed to set the RF bandwidth: {}", failure());
            break;
        }
    }
    manualBandwidthSet = manualBandwidth;
    JST_CHECK(setFrequency(frequency));
    requestedSettings = settings;
    JST_CHECK(setAutomaticGain(settings.automaticGain));
    JST_CHECK(applyGain(settings.gain, true));

    appliedSettings.antenna = settings.antenna;
    appliedSettings.biasTee = settings.biasTee && biasTeeSetting;
    return Result::SUCCESS;
}

Result SoapySdrStream::setBiasTee(const bool enabled) {
    if (library->api().writeSetting(device, biasTeeKey.c_str(), enabled ? "true" : "false") != 0) {
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
    driverRate = applied;
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
    hardwareTimed = hardwareTime;
    timestamped = false;
    try {
        drainBuffer.resize(DrainSamples * receiveChannels * sampleBytes());
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR_SOAPY] Failed to allocate the drain buffer.");
        return Result::ERROR;
    }
    SoapyKwargs arguments;
    JST_CHECK(library->kwargsFromString(requestedSettings.driverArguments, arguments));

    std::vector<std::size_t> channels(receiveChannels);
    for (U64 channel = 0; channel < receiveChannels; ++channel) {
        channels[channel] = channel;
    }
    const SoapyKwargsView view(arguments);
    stream = api.setupStream(device, Abi::RX, SoapyFormatName(requestedSettings.format),
                             channels.data(), channels.size(), view.get());
    if (!stream) {
        JST_ERROR("[SDR_SOAPY] Failed to set up the receive stream: {}", failure());
        return Result::ERROR;
    }
    if (const int status = activate(); status != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to start the receive stream: {}",
                  library->errorString(status));
        close();
        return Result::ERROR;
    }
    streaming = true;
    queueGeneration = nextGeneration();
    discard = guard();
    restartTimestamps();
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

void SoapySdrStream::restartTimestamps() {
    nextTimestamp.reset();
    cutoffNs.reset();
    lastTimeNs.reset();
    expectedNs.reset();
    abruptEnd = false;
    ++timestampEpoch;
}

int SoapySdrStream::activate() {
    const auto& api = library->api();
    const int status = api.activateStream(device, stream, 0, 0, 0);
    if (status == 0 || receiveChannels < 2 || !hardwareTime) {
        return status;
    }
    const long long now = api.getHardwareTime(device, "");
    if (api.lastStatus() != 0) {
        return status;
    }
    return api.activateStream(device, stream, Abi::HasTime, now + TimedStartNs, 0);
}

void SoapySdrStream::drain() {
    std::vector<void*> targets(receiveChannels);
    for (U64 channel = 0; channel < receiveChannels; ++channel) {
        targets[channel] = drainBuffer.data() + channel * DrainSamples * sampleBytes();
    }
    const auto deadline = Clock::now() + DrainLimit;
    while (Clock::now() < deadline) {
        int flags = 0;
        long long timeNs = 0;
        const int status = library->api().readStream(device, stream, targets.data(), DrainSamples,
                                                     &flags, &timeNs, DrainTimeoutUs);
        if (status == Abi::Timeout || (status <= 0 && status != Abi::Overflow)) {
            return;
        }
    }
}

void SoapySdrStream::markCutoff() {
    cutoffNs.reset();
    if (!hardwareTimed) {
        return;
    }
    const long long now = library->api().getHardwareTime(device, "");
    if (library->api().lastStatus() == 0) {
        cutoffNs = now;
        cutoffSince = Clock::now();
    }
}

U64 SoapySdrStream::staleSamples(const long long timeNs, const U64 count) const {
    if (!cutoffNs || timeNs >= *cutoffNs || !(driverRate > 0.0)) {
        return 0;
    }
    const auto distance = static_cast<unsigned long long>(*cutoffNs) -
                          static_cast<unsigned long long>(timeNs);
    const F64 samples = std::ceil(static_cast<F64>(distance) * 1.0e-9 * driverRate);
    return samples >= static_cast<F64>(count) ? count : static_cast<U64>(samples);
}

std::optional<U64> SoapySdrStream::timestampOf(const int flags, const long long timeNs,
                                               const U64 count) {
    const F64 rate = driverRate;
    const bool timed = (flags & Abi::HasTime) != 0;
    if (!(rate > 0.0) || (!nextTimestamp && !timed)) {
        nextTimestamp.reset();
        return std::nullopt;
    }
    if (!nextTimestamp) {
        nextTimestamp = 0;
        anchorTimestamp = 0;
        anchorTimeNs = timeNs;
    } else if (timed) {
        const F64 expectedNs = static_cast<F64>(*nextTimestamp - anchorTimestamp) * 1.0e9 / rate;
        const F64 drift = (static_cast<F64>(timeNs - anchorTimeNs) - expectedNs) * rate * 1.0e-9;
        if (drift >= 0.5) {
            *nextTimestamp += static_cast<U64>(std::llround(drift));
        }
        anchorTimestamp = *nextTimestamp;
        anchorTimeNs = timeNs;
    }
    const U64 timestamp = *nextTimestamp;
    *nextTimestamp += count;
    return timestamp;
}

Result SoapySdrStream::tune(const F64 frequency) {
    const auto& api = library->api();
    const bool restart = streaming;
    if (restart) {
        if (api.deactivateStream(device, stream, 0, 0) != 0) {
            JST_WARN("[SDR_SOAPY] Failed to stop the receive stream: {}", failure());
        }
        drain();
    }
    streaming = false;
    JST_CHECK(setFrequency(frequency));
    if (restart) {
        if (const int status = activate(); status != 0) {
            JST_ERROR("[SDR_SOAPY] Failed to restart the receive stream: {}",
                      library->errorString(status));
            return Result::ERROR;
        }
        streaming = true;
        received = Clock::now();
    }
    requestedSettings.frequency = frequency;
    if (streaming) {
        queueGeneration = nextGeneration();
        discard = guard();
        restartTimestamps();
        markCutoff();
    }
    JST_CHECK(applyGain(requestedSettings.gain, true));
    return Result::SUCCESS;
}

Device::Capture SoapySdrStream::read(std::span<std::byte> samples,
                                     const std::chrono::milliseconds timeout) {
    std::array destinations{samples};
    return readChannels(destinations, timeout);
}

Device::Capture SoapySdrStream::readChannels(std::span<std::span<std::byte>> samples,
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
    if (samples.size() != receiveChannels || samples.front().size() < sampleBytes() ||
        std::any_of(samples.begin(), samples.end(), [&](const auto channel) {
            return channel.size() != samples.front().size();
        })) {
        return {.status = Status::Failure, .error = "invalid receive channel buffers"};
    }
    if (abruptEnd) {
        abruptEnd = false;
        nextTimestamp.reset();
        return finish({.status = Status::Discontinuity});
    }
    const U64 size = samples.front().size() / sampleBytes();

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
        const auto timestamp = timestampOf(flags, timeNs, count);
        abruptEnd = (flags & Abi::EndAbrupt) != 0;
        const bool timed = (flags & Abi::HasTime) != 0;
        if (timed && cutoffNs && lastTimeNs && timeNs <= *lastTimeNs) {
            cutoffNs.reset();
            hardwareTimed = false;
            JST_WARN("[SDR_SOAPY] Stream timestamps of {} do not advance. "
                     "Settling by sample count instead.", deviceInfo.label);
        }
        if (timed) {
            lastTimeNs = timeNs;
            timestamped = true;
        }
        std::optional<long long> readNs;
        if (timed) {
            readNs = timeNs;
        } else if (expectedNs) {
            readNs = std::llround(*expectedNs);
        }
        if (readNs && driverRate > 0.0) {
            expectedNs = static_cast<F64>(*readNs) + static_cast<F64>(count) * 1.0e9 / driverRate;
        }
        const U64 stale = !cutoffNs || (!readNs && !timestamped) ? 0
            : readNs ? staleSamples(*readNs, count) : count;
        if (stale < count) {
            cutoffNs.reset();
        } else if (Clock::now() - cutoffSince > Device::StallTimeout) {
            cutoffNs.reset();
            hardwareTimed = false;
            JST_WARN("[SDR_SOAPY] Stream timestamps of {} do not follow its hardware time. "
                     "Settling by sample count instead.", deviceInfo.label);
        }
        const U64 settled = std::min(discard, count - stale);
        discard -= settled;
        const U64 skip = stale + settled;
        if (skip == count) {
            return finish({.status = Status::Timeout});
        }
        if (skip > 0) {
            for (const auto channel : samples) {
                std::memmove(channel.data(), channel.data() + skip * sampleBytes(),
                             (count - skip) * sampleBytes());
            }
        }
        return finish({
            .status = Status::Samples,
            .count = count - skip,
            .timestamp = timestamp ? std::optional<U64>(*timestamp + skip) : std::nullopt,
            .epoch = timestampEpoch,
        });
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
        nextTimestamp.reset();
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
    gainStep = std::isfinite(range.step) && range.step > 0.0 ? range.step : 0.0;
    return Result::SUCCESS;
}

F64 SoapySdrStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    const F64 clamped = Stream::resolveGain(gain, range);
    if (!range || !(gainStep > 0.0)) {
        return clamped;
    }
    const F64 steps = std::round((clamped - range->minimum) / gainStep);
    return std::min(range->maximum, range->minimum + steps * gainStep);
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
