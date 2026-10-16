#include "base.hh"

#include "../stream.hh"
#include "../../tools/capture_queue.hh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>

#include <airspy.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr F64 SettlingTime = 10.0e-3;
constexpr std::array<U8, 3> MaxStageGains = {14, 15, 15};
constexpr U64 InFlightTransfers = 24;

std::array<U8, 3> StageGains(U64 total) {
    std::array<U8, 3> stages = {0, 0, 0};
    while (total > 0) {
        U64 next = stages.size();
        for (U64 i = 0; i < stages.size(); ++i) {
            if (stages[i] < MaxStageGains[i] &&
                (next == stages.size() || stages[i] < stages[next])) {
                next = i;
            }
        }
        if (next == stages.size()) {
            break;
        }
        ++stages[next];
        --total;
    }
    return stages;
}

class AirspyStream final : public Stream {
 public:
    ~AirspyStream() override;

    Result connect(const Device::Info& info);
    F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const override;
    Device::Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result tune(F64 frequency) override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;

 private:
    static int Callback(airspy_transfer* transfer);

    Result applyGainMode(bool automatic, F64 gain);
    Result applyGain(F64 gain, bool force);

    airspy_device* handle = nullptr;
    bool biasTeeEnabled = false;
    bool streaming = false;
    CaptureQueue queue;
    std::atomic<U64> queueGeneration{0};
};

AirspyStream::~AirspyStream() {
    close();
    if (handle) {
        if (biasTeeEnabled) {
            airspy_set_rf_bias(handle, 0);
        }
        airspy_close(handle);
    }
}

Result AirspyStream::connect(const Device::Info& info) {
    U64 serial = 0;
    try {
        serial = std::stoull(info.serial, nullptr, 16);
    } catch (...) {
        JST_ERROR("[SDR_AIRSPY] Invalid serial number '{}'.", info.serial);
        return Result::ERROR;
    }
    if (airspy_open_sn(&handle, serial) != AIRSPY_SUCCESS) {
        handle = nullptr;
        JST_ERROR("[SDR_AIRSPY] Failed to open device '{}'.", info.label);
        return Result::ERROR;
    }
    deviceInfo = info;

    if (airspy_set_sample_type(handle, AIRSPY_SAMPLE_FLOAT32_IQ) != AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to select IQ samples.");
        return Result::ERROR;
    }
    U32 count = 0;
    airspy_get_samplerates(handle, &count, 0);
    std::vector<U32> rates(count);
    if (count == 0 || airspy_get_samplerates(handle, rates.data(), count) != AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to query sample rates.");
        return Result::ERROR;
    }

    auto& caps = deviceCapabilities;
    caps.frequencyRanges = {{24.0e6, 1800.0e6}};
    for (const auto rate : rates) {
        caps.sampleRateRanges.push_back({static_cast<F64>(rate), static_cast<F64>(rate)});
        caps.sweepSampleRate = std::max(caps.sweepSampleRate, static_cast<F64>(rate));
    }
    caps.automaticGain = true;
    caps.biasTee = true;
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 0.0;
    defaultSettlingTime = SettlingTime;
    appliedSettings.gainRange = Device::Range{
        0.0, static_cast<F64>(MaxStageGains[0] + MaxStageGains[1] + MaxStageGains[2])};
    return Result::SUCCESS;
}

Result AirspyStream::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_AIRSPY] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_AIRSPY] Frequency ({:.3f} MHz) not supported.",
                  settings.frequency / 1e6);
        return Result::ERROR;
    }
    if (!settings.antenna.empty()) {
        JST_ERROR("[SDR_AIRSPY] Antenna selection is not supported.");
        return Result::ERROR;
    }

    if (airspy_set_samplerate(handle, static_cast<U32>(std::lround(settings.sampleRate))) !=
        AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to set the sample rate.");
        return Result::ERROR;
    }
    if (airspy_set_freq(handle, static_cast<U32>(std::llround(settings.frequency))) !=
        AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to tune to {:.3f} MHz.", settings.frequency / 1e6);
        return Result::ERROR;
    }

    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));

    if (settings.biasTee || biasTeeEnabled) {
        if (airspy_set_rf_bias(handle, settings.biasTee ? 1 : 0) != AIRSPY_SUCCESS) {
            JST_ERROR("[SDR_AIRSPY] Failed to set Bias-T.");
            return Result::ERROR;
        }
        biasTeeEnabled = settings.biasTee;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = settings.sampleRate;
    appliedSettings.frequency = std::round(settings.frequency);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = settings.biasTee;
    return Result::SUCCESS;
}

int AirspyStream::Callback(airspy_transfer* transfer) {
    auto* stream = static_cast<AirspyStream*>(transfer->ctx);
    const auto* data = static_cast<const F32*>(transfer->samples);
    stream->queue.push(stream->queueGeneration, static_cast<U64>(transfer->sample_count),
                       [data](CF32* output, const U64 offset, const U64 count) {
                           for (U64 i = 0; i < count; ++i) {
                               output[i] = {data[2 * (offset + i)], data[2 * (offset + i) + 1]};
                           }
                       },
                       transfer->dropped_samples);
    return 0;
}

Result AirspyStream::open() {
    const F64 sampleRate = appliedSettings.sampleRate;
    JST_CHECK(queue.resize(std::max<U64>(static_cast<U64>(sampleRate / 2.0), 1 << 18)));
    queueGeneration = nextGeneration();
    queue.reset(queueGeneration, guard(), appliedSettings.frequency, sampleRate);

    if (airspy_start_rx(handle, &AirspyStream::Callback, this) != AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to start reception.");
        airspy_stop_rx(handle);
        queue.close();
        return Result::ERROR;
    }
    streaming = true;
    return Result::SUCCESS;
}

void AirspyStream::close() {
    if (!streaming) {
        return;
    }
    airspy_stop_rx(handle);
    streaming = false;
    queue.close();
}

Result AirspyStream::tune(const F64 frequency) {
    if (airspy_set_freq(handle, static_cast<U32>(std::llround(frequency))) != AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = std::round(frequency);
    if (!streaming) {
        return Result::SUCCESS;
    }
    queueGeneration = nextGeneration();
    queue.retune(queueGeneration, InFlightTransfers, guard(), appliedSettings.frequency);
    return Result::SUCCESS;
}

Device::Capture AirspyStream::read(std::span<CF32> samples,
                                   const std::chrono::milliseconds timeout) {
    auto capture = queue.pop(samples, timeout);
    if (capture.status == Device::Status::Timeout && streaming &&
        airspy_is_streaming(handle) != AIRSPY_TRUE) {
        capture.status = Device::Status::Disconnected;
        capture.error = "Airspy stopped streaming";
    }
    return capture;
}

Result AirspyStream::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(applyGainMode(automaticGain, requestedSettings.gain));
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result AirspyStream::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 AirspyStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Stream::resolveGain(gain, range));
}

Result AirspyStream::applyGainMode(const bool automatic, const F64 gain) {
    const U8 agc = automatic ? 1 : 0;
    if (airspy_set_lna_agc(handle, agc) != AIRSPY_SUCCESS ||
        airspy_set_mixer_agc(handle, agc) != AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to set the gain mode.");
        return Result::ERROR;
    }
    appliedSettings.automaticGain = automatic;
    return applyGain(gain, true);
}

Result AirspyStream::applyGain(const F64 gain, const bool force) {
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    if (appliedSettings.automaticGain || (!force && target == appliedSettings.gain)) {
        appliedSettings.gain = target;
        return Result::SUCCESS;
    }
    const auto stages = StageGains(static_cast<U64>(target));
    if (airspy_set_lna_gain(handle, stages[0]) != AIRSPY_SUCCESS ||
        airspy_set_mixer_gain(handle, stages[1]) != AIRSPY_SUCCESS ||
        airspy_set_vga_gain(handle, stages[2]) != AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to set the gain to {:.0f}.", target);
        appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

}  // namespace

Result Airspy::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<AirspyStream>();
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
