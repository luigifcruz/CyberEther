#include "base.hh"
#include "tools/catalog.hh"

#include "../sweep.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

class SoapySdrSweep final : public Sweep::Retune {
 public:
    ~SoapySdrSweep() override;

    Result connect(const Device::Info& info);
    F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result tune(U64 hop) override;
    Device::Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;

 private:
    std::string failure() const;
    Result queryCapabilities();
    Result setSampleRate(F64 sampleRate);
    Result setFrequency(F64 frequency);
    Result refreshGainRange();
    Result applyGainMode(bool automatic, F64 gain);
    Result applyGain(F64 gain, bool force);
    std::optional<U64> timestampOf(int flags, long long timeNs, U64 count);
    void restartTimestamps();
    void markCutoff();
    void drain();
    U64 staleSamples(long long timeNs, U64 count) const;

    std::shared_ptr<SoapyLibrary> library;
    std::vector<CF32> drainBuffer;
    Abi::Device* device = nullptr;
    Abi::Stream* stream = nullptr;
    std::vector<Device::Range> reportedRates;
    bool streaming = false;
    bool hardwareTime = false;
    bool hardwareTimed = false;
    bool biasTeeEnabled = false;
    F64 gainStep = 0.0;
    std::string biasTeeKey;

    U64 queueGeneration = 0;
    U64 discard = 0;
    std::optional<long long> cutoffNs;
    std::optional<long long> lastTimeNs;
    std::optional<F64> expectedNs;
    bool timestamped = false;
    Clock::time_point cutoffSince{};
    Clock::time_point received{};

    std::optional<U64> nextTimestamp;
    U64 anchorTimestamp = 0;
    long long anchorTimeNs = 0;
    bool abruptEnd = false;
};

SoapySdrSweep::~SoapySdrSweep() {
    close();
    if (device) {
        if (biasTeeEnabled) {
            library->api().writeSetting(device, biasTeeKey.c_str(), "false");
        }
        if (library->api().unmake(device) != 0) {
            JST_WARN("[SDR_SOAPY] Failed to release {}: {}", deviceInfo.label, failure());
        }
    }
}

std::string SoapySdrSweep::failure() const {
    const auto error = library ? library->lastError() : std::string{};
    return error.empty() ? std::string("Unknown driver error.") : error;
}

Result SoapySdrSweep::connect(const Device::Info& info) {
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

Result SoapySdrSweep::queryCapabilities() {
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
        return !ranges.empty();
    };

    std::vector<Device::Range> frequencies;
    std::vector<Device::Range> rates;
    if (!readRanges(api.getFrequencyRange, frequencies) ||
        !readRanges(api.getSampleRateRange, rates)) {
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
    const bool biasTee = !biasTeeKey.empty();

    auto& caps = deviceCapabilities;
    reportedRates = rates;
    caps.frequencyRanges = std::move(frequencies);
    caps.bandwidthRanges = std::move(bandwidths);
    caps.sampleRateRanges = {{std::numeric_limits<F64>::min(), std::numeric_limits<F64>::max()}};
    caps.automaticGain = api.hasGainMode(device, Abi::RX, 0) && api.lastStatus() == 0;
    const std::size_t channels = api.getNumChannels(device, Abi::RX);
    caps.receiveChannels = api.lastStatus() == 0 && channels > 0 ? channels : 1;
    caps.biasTee = biasTee;
    caps.sweepSampleRate = SoapyCatalog::SweepRate(reportedRates, SweepSampleRate);
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 20.0e3;
    defaultSettlingTime = SettlingTime;
    deviceInfo.biasTee = biasTee;
    biasTeeEnabled = false;
    hardwareTime = api.hasHardwareTime(device, "");

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

    return refreshGainRange();
}

Result SoapySdrSweep::apply(const Settings& settings) {
    const auto& api = library->api();
    if (!std::isfinite(settings.sampleRate) || settings.sampleRate <= 0.0) {
        JST_ERROR("[SDR_SOAPY] Sample rate ({:.3f} MHz) is not valid.", settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Device::Range::Contains(reportedRates, settings.sampleRate)) {
        JST_WARN("[SDR_SOAPY] Sample rate ({:.3f} MHz) is not one that {} reports. "
                 "Requesting it anyway.", settings.sampleRate / 1e6, deviceInfo.label);
    }
    if (!settings.antenna.empty()) {
        const auto& antennas = deviceCapabilities.antennas;
        if (std::find(antennas.begin(), antennas.end(), settings.antenna) == antennas.end()) {
            JST_ERROR("[SDR_SOAPY] Receive antenna '{}' is not supported.", settings.antenna);
            return Result::ERROR;
        }
        if (api.setAntenna(device, Abi::RX, 0, settings.antenna.c_str()) != 0) {
            JST_ERROR("[SDR_SOAPY] Failed to select antenna '{}': {}", settings.antenna,
                      failure());
            return Result::ERROR;
        }
    }
    if (settings.biasTee && !deviceCapabilities.biasTee) {
        JST_WARN("[SDR_SOAPY] Bias-T is not supported by {}.", deviceInfo.label);
    }

    JST_CHECK(setSampleRate(settings.sampleRate));
    if (api.setBandwidth(device, Abi::RX, 0, appliedSettings.sampleRate) != 0) {
        JST_WARN("[SDR_SOAPY] Failed to set the RF bandwidth: {}", failure());
    }
    JST_CHECK(refreshGainRange());
    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));

    if (deviceCapabilities.biasTee && (settings.biasTee || biasTeeEnabled)) {
        const char* value = settings.biasTee ? "true" : "false";
        if (api.writeSetting(device, biasTeeKey.c_str(), value) != 0) {
            JST_ERROR("[SDR_SOAPY] Failed to set Bias-T: {}", failure());
            return Result::ERROR;
        }
        biasTeeEnabled = settings.biasTee;
    }

    requestedSettings = settings;
    appliedSettings.antenna = settings.antenna;
    appliedSettings.biasTee = settings.biasTee && deviceCapabilities.biasTee;
    return Result::SUCCESS;
}

Result SoapySdrSweep::setSampleRate(const F64 sampleRate) {
    const auto& api = library->api();
    if (api.setSampleRate(device, Abi::RX, 0, sampleRate) != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to set the sample rate: {}", failure());
        return Result::ERROR;
    }
    const F64 applied = api.getSampleRate(device, Abi::RX, 0);
    if (!std::isfinite(applied) || applied <= 0.0) {
        JST_ERROR("[SDR_SOAPY] Invalid applied sample rate: {}", applied);
        return Result::ERROR;
    }
    const bool requested = std::abs(applied - sampleRate) <= sampleRate * ReadbackTolerance;
    appliedSettings.sampleRate = requested ? sampleRate : applied;
    return Result::SUCCESS;
}

Result SoapySdrSweep::setFrequency(const F64 frequency) {
    const auto& api = library->api();
    if (api.setFrequency(device, Abi::RX, 0, frequency, nullptr) != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to tune to {:.3f} MHz: {}", frequency / 1e6, failure());
        return Result::ERROR;
    }
    const F64 applied = api.getFrequency(device, Abi::RX, 0);
    tunedFrequency = std::isfinite(applied) && applied > 0.0 ? applied : frequency;
    return refreshGainRange();
}

Result SoapySdrSweep::open() {
    if (appliedSettings.sampleRate <= 0.0) {
        JST_ERROR("[SDR_SOAPY] Invalid sweep sample rate.");
        return Result::ERROR;
    }
    hardwareTimed = hardwareTime;
    timestamped = false;
    try {
        drainBuffer.resize(DrainSamples);
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR_SOAPY] Failed to allocate the drain buffer.");
        return Result::ERROR;
    }
    const auto& api = library->api();
    const std::size_t channels[] = {0};
    const SoapyKwargs streamArgs;
    const SoapyKwargsView view(streamArgs);
    stream = api.setupStream(device, Abi::RX, Abi::FormatCF32, channels, 1, view.get());
    if (!stream) {
        JST_ERROR("[SDR_SOAPY] Failed to set up the receive stream: {}", failure());
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

void SoapySdrSweep::close() {
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

Result SoapySdrSweep::tune(const U64 hop) {
    if (!stream) {
        JST_ERROR("[SDR_SOAPY] Cannot retune a stopped sweep.");
        return Result::ERROR;
    }
    const auto& api = library->api();
    if (streaming) {
        if (api.deactivateStream(device, stream, 0, 0) != 0) {
            JST_WARN("[SDR_SOAPY] Failed to stop the receive stream: {}", failure());
        }
        drain();
    }
    streaming = false;
    const auto previousRange = appliedSettings.gainRange;
    JST_CHECK(setFrequency(sweepRequest.frequencies[hop]));
    const auto& range = appliedSettings.gainRange;
    const bool rangeChanged = range.has_value() != previousRange.has_value() ||
        (range && (range->minimum != previousRange->minimum ||
                   range->maximum != previousRange->maximum));
    JST_CHECK(applyGain(requestedSettings.gain, rangeChanged));
    if (const int status = api.activateStream(device, stream, 0, 0, 0); status != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to start the receive stream: {}",
                  library->errorString(status));
        return Result::ERROR;
    }
    streaming = true;
    received = Clock::now();
    queueGeneration = nextGeneration();
    discard = guard();
    restartTimestamps();
    markCutoff();
    return Result::SUCCESS;
}

void SoapySdrSweep::restartTimestamps() {
    nextTimestamp.reset();
    cutoffNs.reset();
    lastTimeNs.reset();
    expectedNs.reset();
    abruptEnd = false;
}

void SoapySdrSweep::drain() {
    void* buffers[] = {drainBuffer.data()};
    const auto deadline = Clock::now() + DrainLimit;
    while (Clock::now() < deadline) {
        int flags = 0;
        long long timeNs = 0;
        const int status = library->api().readStream(device, stream, buffers, drainBuffer.size(),
                                                     &flags, &timeNs, DrainTimeoutUs);
        if (status == Abi::Timeout || (status <= 0 && status != Abi::Overflow)) {
            return;
        }
    }
}

void SoapySdrSweep::markCutoff() {
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

U64 SoapySdrSweep::staleSamples(const long long timeNs, const U64 count) const {
    const F64 rate = appliedSettings.sampleRate;
    if (!cutoffNs || timeNs >= *cutoffNs || !(rate > 0.0)) {
        return 0;
    }
    const auto distance = static_cast<unsigned long long>(*cutoffNs) -
                          static_cast<unsigned long long>(timeNs);
    const F64 samples = std::ceil(static_cast<F64>(distance) * 1.0e-9 * rate);
    return samples >= static_cast<F64>(count) ? count : static_cast<U64>(samples);
}

std::optional<U64> SoapySdrSweep::timestampOf(const int flags, const long long timeNs,
                                              const U64 count) {
    const F64 rate = appliedSettings.sampleRate;
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

Device::Capture SoapySdrSweep::read(std::span<CF32> samples,
                                    const std::chrono::milliseconds timeout) {
    const auto finish = [&](Device::Capture capture) {
        capture.generation = queueGeneration;
        capture.frequency = tunedFrequency;
        capture.sampleRate = appliedSettings.sampleRate;
        return capture;
    };

    if (!streaming) {
        return {.status = Status::Failure, .error = "stream is not running"};
    }
    if (abruptEnd) {
        abruptEnd = false;
        nextTimestamp.reset();
        return finish({.status = Status::Discontinuity});
    }

    void* buffers[] = {samples.data()};
    int flags = 0;
    long long timeNs = 0;
    const auto timeoutUs = std::chrono::duration_cast<std::chrono::microseconds>(timeout);
    const int status = library->api().readStream(device, stream, buffers, samples.size(), &flags,
                                                 &timeNs, static_cast<long>(timeoutUs.count()));

    if (status > 0 && static_cast<U64>(status) > samples.size()) {
        return finish({.status = Status::Failure,
                       .error = jst::fmt::format("{} returned more samples than requested",
                                                 deviceInfo.label)});
    }
    if (status > 0) {
        received = Clock::now();
        const U64 count = static_cast<U64>(status);
        const auto expected = nextTimestamp;
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
        if (hardwareTimed && expected && timestamp && *timestamp != *expected) {
            return finish({.status = Status::Discontinuity, .lost = *timestamp - *expected});
        }
        std::optional<long long> readNs;
        if (timed) {
            readNs = timeNs;
        } else if (expectedNs) {
            readNs = std::llround(*expectedNs);
        }
        if (readNs && appliedSettings.sampleRate > 0.0) {
            expectedNs = static_cast<F64>(*readNs) + static_cast<F64>(count) * 1.0e9 / appliedSettings.sampleRate;
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
            std::copy(samples.begin() + static_cast<std::ptrdiff_t>(skip),
                      samples.begin() + static_cast<std::ptrdiff_t>(count), samples.begin());
        }
        return finish({
            .status = Status::Samples,
            .count = count - skip,
            .timestamp = timestamp ? std::optional<U64>(*timestamp + skip) : std::nullopt,
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

Result SoapySdrSweep::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(applyGainMode(automaticGain, requestedSettings.gain));
    requestedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result SoapySdrSweep::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

Result SoapySdrSweep::refreshGainRange() {
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

F64 SoapySdrSweep::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    const F64 clamped = Sweep::resolveGain(gain, range);
    if (!range || !(gainStep > 0.0)) {
        return clamped;
    }
    const F64 steps = std::round((clamped - range->minimum) / gainStep);
    return std::min(range->maximum, range->minimum + steps * gainStep);
}

Result SoapySdrSweep::applyGainMode(const bool automatic, const F64 gain) {
    if (deviceCapabilities.automaticGain &&
        library->api().setGainMode(device, Abi::RX, 0, automatic) != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to set the gain mode: {}", failure());
        return Result::ERROR;
    }
    if (automatic && !deviceCapabilities.automaticGain) {
        JST_WARN("[SDR_SOAPY] Automatic gain is not supported by {}.", deviceInfo.label);
    }
    appliedSettings.automaticGain = automatic && deviceCapabilities.automaticGain;
    return applyGain(gain, true);
}

Result SoapySdrSweep::applyGain(const F64 gain, const bool force) {
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    if (appliedSettings.automaticGain || (!force && target == appliedSettings.gain)) {
        appliedSettings.gain = target;
        return Result::SUCCESS;
    }
    if (library->api().setGain(device, Abi::RX, 0, target) != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to set the gain to {:.1f} dB: {}", target, failure());
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

}  // namespace

Result SoapySdr::Sweep(const Device::Info& info, std::unique_ptr<Radio::Sweep>& sweep) {
    auto created = std::make_unique<SoapySdrSweep>();
    JST_CHECK(created->connect(info));
    sweep = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
