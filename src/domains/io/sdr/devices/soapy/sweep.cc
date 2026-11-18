#include "base.hh"
#include "tools/catalog.hh"

#include "../sweep.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
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

using Clock = std::chrono::steady_clock;

constexpr F64 SettlingTime = 1.0e-3;
constexpr F64 SweepSampleRate = 20.0e6;
constexpr long ReadTimeoutUs = 100000;
constexpr U64 MaxReadChunk = 65536;
constexpr U64 MaxDrainReads = 4096;
constexpr F64 ReadbackTolerance = 1.0e-9;

class SoapySdrSweep final : public Sweep {
 public:
    ~SoapySdrSweep() override;

    Result connect(const Device::Info& info);
    Result run(const Running& running, const Sink& sink) override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;

 private:
    enum class Mode {
        Burst,
        Continuous,
    };

    std::string failure() const;
    Result queryCapabilities();
    Result setSampleRate(F64 sampleRate);
    Result retune(F64 frequency);
    Result refreshGainRange();
    Result applyGainMode(bool automatic, F64 gain);
    Result applyGain(F64 gain, bool force);
    Result activate(U64 samples);
    void deactivate();
    void drain();
    Result checkBurstEnd();
    Result capture(std::span<CF32> samples, const Running& running);

    std::shared_ptr<SoapyLibrary> library;
    Abi::Device* device = nullptr;
    Abi::Stream* stream = nullptr;
    std::vector<Device::Range> reportedRates;
    std::vector<CF32> discard;
    Mode mode = Mode::Burst;
    bool active = false;
    bool silentBurstEnd = false;
    bool biasTeeEnabled = false;
    F64 tunedFrequency = 0.0;
};

SoapySdrSweep::~SoapySdrSweep() {
    close();
    if (device) {
        if (biasTeeEnabled) {
            library->api().writeSetting(device, "biastee", "false");
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

    std::size_t settingCount = 0;
    bool biasTee = false;
    auto* settings = api.getSettingInfo(device, &settingCount);
    if (api.lastStatus() == 0) {
        for (std::size_t i = 0; i < settingCount; ++i) {
            biasTee = biasTee || (settings[i].key && std::string(settings[i].key) == "biastee");
        }
    }
    api.argInfoListClear(settings, settingCount);

    auto& caps = deviceCapabilities;
    reportedRates = rates;
    caps.frequencyRanges = std::move(frequencies);
    caps.sampleRateRanges = {{std::numeric_limits<F64>::min(), std::numeric_limits<F64>::max()}};
    caps.automaticGain = api.hasGainMode(device, Abi::RX, 0) && api.lastStatus() == 0;
    caps.biasTee = biasTee;
    caps.sweepSampleRate = SoapyCatalog::SweepRate(reportedRates, SweepSampleRate);
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 20.0e3;
    defaultSettlingTime = SettlingTime;
    deviceInfo.biasTee = biasTee;
    biasTeeEnabled = biasTee;

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

    if (deviceCapabilities.biasTee) {
        if (api.writeSetting(device, "biastee", settings.biasTee ? "true" : "false") != 0) {
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

Result SoapySdrSweep::retune(const F64 frequency) {
    const auto& api = library->api();
    if (api.setFrequency(device, Abi::RX, 0, frequency, nullptr) != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to tune to {:.3f} MHz: {}", frequency / 1e6, failure());
        return Result::ERROR;
    }
    const F64 applied = api.getFrequency(device, Abi::RX, 0);
    tunedFrequency = std::isfinite(applied) && applied > 0.0 ? applied : frequency;
    return Result::SUCCESS;
}

Result SoapySdrSweep::open() {
    if (appliedSettings.sampleRate <= 0.0) {
        JST_ERROR("[SDR_SOAPY] Invalid sweep sample rate.");
        return Result::ERROR;
    }
    try {
        discard.resize(MaxReadChunk);
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
    return retune(sweepRequest.frequencies.front());
}

void SoapySdrSweep::close() {
    if (!stream) {
        return;
    }
    deactivate();
    if (library->api().closeStream(device, stream) != 0) {
        JST_WARN("[SDR_SOAPY] Failed to close the receive stream: {}", failure());
    }
    stream = nullptr;
}

Result SoapySdrSweep::activate(const U64 samples) {
    const auto& api = library->api();
    if (mode == Mode::Burst) {
        const int status = api.activateStream(device, stream, Abi::EndBurst, 0, samples);
        if (status == 0) {
            active = true;
            return Result::SUCCESS;
        }
        if (status != Abi::NotSupported) {
            JST_ERROR("[SDR_SOAPY] Failed to request a {}-sample burst: {}", samples,
                      library->errorString(status));
            return Result::ERROR;
        }
        JST_INFO("[SDR_SOAPY] {} does not support bursts. Restarting the stream at each hop.",
                 deviceInfo.label);
        mode = Mode::Continuous;
    }
    if (const int status = api.activateStream(device, stream, 0, 0, 0); status != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to start the receive stream: {}",
                  library->errorString(status));
        return Result::ERROR;
    }
    active = true;
    return Result::SUCCESS;
}

void SoapySdrSweep::deactivate() {
    if (!active) {
        return;
    }
    active = false;
    if (library->api().deactivateStream(device, stream, 0, 0) != 0) {
        JST_WARN("[SDR_SOAPY] Failed to stop the receive stream: {}", failure());
    }
    drain();
}

void SoapySdrSweep::drain() {
    void* buffers[] = {discard.data()};
    for (U64 i = 0; i < MaxDrainReads; ++i) {
        int flags = 0;
        long long timeNs = 0;
        const int status = library->api().readStream(device, stream, buffers, discard.size(),
                                                     &flags, &timeNs, 1000);
        if (status == Abi::Timeout || (status <= 0 && status != Abi::Overflow)) {
            return;
        }
    }
}

Result SoapySdrSweep::checkBurstEnd() {
    void* buffers[] = {discard.data()};
    int flags = 0;
    long long timeNs = 0;
    const int status = library->api().readStream(device, stream, buffers, discard.size(), &flags,
                                                 &timeNs, ReadTimeoutUs);
    if (status > 0 && static_cast<U64>(status) > discard.size()) {
        JST_ERROR("[SDR_SOAPY] {} returned more samples than requested.", deviceInfo.label);
        return Result::FATAL;
    }
    if (status == 0 || status == Abi::Timeout) {
        silentBurstEnd = true;
        return Result::SUCCESS;
    }
    JST_INFO("[SDR_SOAPY] {} does not end bursts. Restarting the stream at each hop.",
             deviceInfo.label);
    mode = Mode::Continuous;
    return Result::SUCCESS;
}

Result SoapySdrSweep::capture(std::span<CF32> samples, const Running& running) {
    const auto& api = library->api();
    JST_CHECK(activate(samples.size()));

    U64 filled = 0;
    bool ended = false;
    auto idleSince = Clock::now();
    while (filled < samples.size() && !ended) {
        if (!running()) {
            deactivate();
            return Result::YIELD;
        }
        void* buffers[] = {samples.data() + filled};
        int flags = 0;
        long long timeNs = 0;
        const auto chunk = std::min<U64>(samples.size() - filled, MaxReadChunk);
        const int status = api.readStream(device, stream, buffers, chunk, &flags, &timeNs,
                                          ReadTimeoutUs);
        if (status > 0 && static_cast<U64>(status) > chunk) {
            deactivate();
            JST_ERROR("[SDR_SOAPY] {} returned more samples than requested.", deviceInfo.label);
            return Result::FATAL;
        }
        if (status > 0) {
            filled += static_cast<U64>(status);
            ended = (flags & Abi::EndBurst) != 0;
            idleSince = Clock::now();
            continue;
        }
        if (status == 0 || status == Abi::Timeout) {
            if (Clock::now() - idleSince > Device::StallTimeout) {
                deactivate();
                JST_ERROR("[SDR_SOAPY] Timed out waiting for sweep samples from {}.",
                          deviceInfo.label);
                return Result::TIMEOUT;
            }
            continue;
        }
        deactivate();
        if (status == Abi::Overflow) {
            return Result::INCOMPLETE;
        }
        JST_ERROR("[SDR_SOAPY] Sweep reception failed on {}: {}", deviceInfo.label,
                  library->errorString(status));
        return Result::FATAL;
    }

    if (mode == Mode::Burst && !ended && !silentBurstEnd) {
        if (const auto result = checkBurstEnd(); result != Result::SUCCESS) {
            deactivate();
            return result;
        }
    }
    if (mode == Mode::Burst) {
        active = false;
    } else {
        deactivate();
    }
    return filled == samples.size() ? Result::SUCCESS : Result::INCOMPLETE;
}

Result SoapySdrSweep::run(const Running& running, const Sink& sink) {
    if (!stream) {
        JST_ERROR("[SDR_SOAPY] No sweep is started on {}.", deviceInfo.label);
        return Result::ERROR;
    }
    const auto& request = sweepRequest;
    const U64 required = request.captureLength * request.dwell;
    const U64 settle = guard();
    std::vector<CF32> samples;
    try {
        samples.resize(settle + required);
    } catch (const std::exception&) {
        JST_ERROR("[SDR_SOAPY] Failed to allocate the sweep capture buffer.");
        return Result::ERROR;
    }

    U64 hop = 0;
    U64 failures = 0;
    while (running()) {
        if (retune(request.frequencies[hop]) != Result::SUCCESS) {
            JST_CHECK(fail(failures));
            continue;
        }
        const auto status = capture(samples, running);
        if (status == Result::YIELD || !running()) {
            break;
        }
        if (status == Result::FATAL) {
            return Result::ERROR;
        }
        if (status != Result::SUCCESS) {
            JST_CHECK(fail(failures));
            continue;
        }
        failures = 0;
        const std::span<const CF32> captured(samples.data() + settle, required);
        if (!sink({.hop = hop, .frequency = tunedFrequency, .samples = captured})) {
            break;
        }
        hop = (hop + 1) % request.frequencies.size();
    }
    return Result::SUCCESS;
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
    return Result::SUCCESS;
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
