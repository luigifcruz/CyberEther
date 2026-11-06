#include "base.hh"
#include "tools/catalog.hh"

#include "../sweep.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

namespace Abi = Modules::SoapyAbi;

using Clock = std::chrono::steady_clock;

constexpr F64 SettlingTime = 1.0e-3;
constexpr F64 UsableBandwidth = 0.8;
constexpr F64 DcExclusion = 20.0e3;
constexpr long ReadTimeoutUs = 100000;
constexpr U64 MaxReadChunk = 65536;
constexpr U64 MaxDrainReads = 4096;
constexpr F64 RateTolerance = 1.0e-3;

// Rates above the reported ranges worth probing. Some drivers, such as UHD on the B200
// family, only report the rates of the current master clock but retune it on demand.
constexpr std::array ProbeRates = {61.44e6, 56.0e6, 50.0e6, 40.0e6, 30.72e6, 20.0e6};

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
    F64 probeSampleRate(F64 reported);
    Result setSampleRate(F64 sampleRate);
    Result retune(F64 frequency);
    Result refreshGainRange();
    Result applyGain(F64 gain, bool force);
    Result activate(U64 samples);
    void deactivate();
    void drain();
    Result capture(std::span<CF32> samples, const Running& running);

    std::shared_ptr<Modules::SoapyLibrary> library;
    Abi::Device* device = nullptr;
    Abi::Stream* stream = nullptr;
    Mode mode = Mode::Burst;
    bool active = false;
    bool biasTeeSetting = false;
    F64 tunedFrequency = 0.0;
};

SoapySdrSweep::~SoapySdrSweep() {
    close();
    if (device && library->api().unmake(device) != 0) {
        JST_WARN("[SDR_SOAPY] Failed to release {}: {}", deviceInfo.label, failure());
    }
}

std::string SoapySdrSweep::failure() const {
    const auto error = library ? library->lastError() : std::string{};
    return error.empty() ? std::string("Unknown driver error.") : error;
}

Result SoapySdrSweep::connect(const Device::Info& info) {
    library = Modules::SoapyRuntime::Acquire();
    if (!library) {
        JST_ERROR("[SDR_SOAPY] {}", Modules::SoapyRuntime::Availability().message);
        return Result::INCOMPLETE;
    }

    Modules::SoapyKwargsList devices;
    JST_CHECK(library->enumerate({}, devices));
    std::vector<SoapyCatalog::Entry> matches;
    for (auto& entry : SoapyCatalog::List(devices)) {
        if (entry.info.serial == info.serial) {
            matches.push_back(std::move(entry));
        }
    }
    if (matches.empty()) {
        JST_ERROR("[SDR_SOAPY] Device '{}' is no longer available.", info.label);
        return Result::INCOMPLETE;
    }
    if (matches.size() != 1) {
        JST_ERROR("[SDR_SOAPY] Several devices report serial '{}'.", info.serial);
        return Result::INCOMPLETE;
    }

    const Modules::SoapyKwargsView view(matches.front().args);
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
        JST_ERROR("[SDR_SOAPY] Failed to query the ranges of {}: {}", deviceInfo.label,
                  failure());
        return Result::ERROR;
    }

    auto& caps = deviceCapabilities;
    caps.frequencyRanges = std::move(frequencies);
    const F64 reported = SoapyCatalog::SampleRates(rates, 0.0).front().maximum;
    caps.sampleRateRanges = SoapyCatalog::SampleRates(rates, probeSampleRate(reported));
    caps.sweepSampleRate = caps.sampleRateRanges.front().maximum;
    caps.usableBandwidth = UsableBandwidth;
    caps.dcExclusion = DcExclusion;
    defaultSettlingTime = SettlingTime;

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

    return refreshGainRange();
}

F64 SoapySdrSweep::probeSampleRate(const F64 reported) {
    const auto& api = library->api();
    for (const auto rate : ProbeRates) {
        if (rate <= reported) {
            break;
        }
        if (api.setSampleRate(device, Abi::RX, 0, rate) != 0) {
            continue;
        }
        const F64 applied = api.getSampleRate(device, Abi::RX, 0);
        if (std::abs(applied - rate) <= rate * RateTolerance) {
            JST_DEBUG("[SDR_SOAPY] {} accepts {:.3f} MS/s above its reported rates.",
                      deviceInfo.label, rate / 1e6);
            return rate;
        }
    }
    return 0.0;
}

Result SoapySdrSweep::apply(const Settings& settings) {
    const auto& api = library->api();
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_SOAPY] Sample rate ({:.3f} MHz) not supported by {}. Use {}.",
                  settings.sampleRate / 1e6, deviceInfo.label,
                  Device::Range::Describe(deviceCapabilities.sampleRateRanges));
        return Result::ERROR;
    }
    if (!settings.antenna.empty()) {
        const auto& antennas = deviceCapabilities.antennas;
        if (std::find(antennas.begin(), antennas.end(), settings.antenna) == antennas.end()) {
            JST_ERROR("[SDR_SOAPY] Antenna '{}' is not available on {}.", settings.antenna,
                      deviceInfo.label);
            return Result::ERROR;
        }
        if (api.setAntenna(device, Abi::RX, 0, settings.antenna.c_str()) != 0) {
            JST_ERROR("[SDR_SOAPY] Failed to select antenna '{}': {}", settings.antenna,
                      failure());
            return Result::ERROR;
        }
    }
    if (settings.biasTee && !biasTeeSetting) {
        JST_WARN("[SDR_SOAPY] Bias-T is not supported by {}.", deviceInfo.label);
    }
    if (biasTeeSetting &&
        api.writeSetting(device, "biastee", settings.biasTee ? "true" : "false") != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to set the bias-T: {}", failure());
        return Result::ERROR;
    }

    JST_CHECK(setSampleRate(settings.sampleRate));
    JST_CHECK(refreshGainRange());
    JST_CHECK(setAutomaticGain(settings.automaticGain));
    JST_CHECK(applyGain(settings.gain, true));

    requestedSettings = settings;
    appliedSettings.antenna = settings.antenna;
    appliedSettings.biasTee = settings.biasTee && biasTeeSetting;
    return Result::SUCCESS;
}

Result SoapySdrSweep::setSampleRate(const F64 sampleRate) {
    const auto& api = library->api();
    if (api.setSampleRate(device, Abi::RX, 0, sampleRate) != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to set the sample rate to {:.3f} MS/s: {}",
                  sampleRate / 1e6, failure());
        return Result::ERROR;
    }
    const F64 applied = api.getSampleRate(device, Abi::RX, 0);
    if (!std::isfinite(applied) || applied <= 0.0) {
        JST_ERROR("[SDR_SOAPY] Invalid applied sample rate: {}", applied);
        return Result::ERROR;
    }
    // Drivers read back coerced rates with rounding noise; keep the requested value then.
    const bool requested = std::abs(applied - sampleRate) <= sampleRate * 1.0e-9;
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
    const auto& api = library->api();
    const std::size_t channels[] = {0};
    const Modules::SoapyKwargs streamArgs;
    const Modules::SoapyKwargsView view(streamArgs);
    stream = api.setupStream(device, Abi::RX, Abi::FormatCF32, channels, 1, view.get());
    if (!stream) {
        JST_ERROR("[SDR_SOAPY] Failed to set up the receive stream: {}", failure());
        return Result::ERROR;
    }
    mode = Mode::Burst;
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
    // Samples left in the host buffers predate the next retune.
    std::vector<CF32> scratch(MaxReadChunk);
    void* buffers[] = {scratch.data()};
    for (U64 i = 0; i < MaxDrainReads; ++i) {
        int flags = 0;
        long long timeNs = 0;
        const int status = library->api().readStream(device, stream, buffers, scratch.size(),
                                                     &flags, &timeNs, 1000);
        if (status == Abi::Timeout || (status <= 0 && status != Abi::Overflow)) {
            return;
        }
    }
}

Result SoapySdrSweep::capture(std::span<CF32> samples, const Running& running) {
    const auto& api = library->api();
    JST_CHECK(activate(samples.size()));

    U64 filled = 0;
    auto idleSince = Clock::now();
    while (filled < samples.size()) {
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
        if (status > 0) {
            filled += static_cast<U64>(status);
            idleSince = Clock::now();
            if ((flags & Abi::EndBurst) && filled < samples.size()) {
                break;
            }
            continue;
        }
        if (status == Abi::Timeout) {
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

    if (mode == Mode::Continuous || filled < samples.size()) {
        deactivate();
    } else {
        active = false;
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
    const auto& api = library->api();
    if (deviceCapabilities.automaticGain &&
        api.setGainMode(device, Abi::RX, 0, automaticGain) != 0) {
        JST_ERROR("[SDR_SOAPY] Failed to set the gain mode: {}", failure());
        return Result::ERROR;
    }
    if (automaticGain && !deviceCapabilities.automaticGain) {
        JST_WARN("[SDR_SOAPY] Automatic gain is not supported by {}.", deviceInfo.label);
    }
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain && deviceCapabilities.automaticGain;
    return appliedSettings.automaticGain ? Result::SUCCESS
                                         : applyGain(requestedSettings.gain, true);
}

Result SoapySdrSweep::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

Result SoapySdrSweep::refreshGainRange() {
    const auto range = library->api().getGainRange(device, Abi::RX, 0);
    if (library->api().lastStatus() != 0 || range.minimum > range.maximum) {
        appliedSettings.gainRange.reset();
        return Result::SUCCESS;
    }
    appliedSettings.gainRange = Device::Range{range.minimum, range.maximum};
    return Result::SUCCESS;
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
