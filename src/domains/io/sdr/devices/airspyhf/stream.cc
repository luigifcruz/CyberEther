#include "base.hh"

#include "../stream.hh"
#include "../../tools/capture_queue.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

#include <airspyhf.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr F64 SettlingTime = 5.0e-3;
constexpr F64 AttenuationStep = 6.0;
constexpr U8 MaxAttenuation = 8;
constexpr U64 InFlightTransfers = 24;

class AirspyHfStream final : public Stream {
 public:
    ~AirspyHfStream() override;

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
    static int Callback(airspyhf_transfer_t* transfer);

    Result applyGainMode(bool automatic, F64 gain);
    Result applyGain(F64 gain, bool force);

    airspyhf_device_t* handle = nullptr;
    bool streaming = false;
    CaptureQueue queue;
    std::atomic<U64> streamGeneration{0};
};

AirspyHfStream::~AirspyHfStream() {
    close();
    if (handle) {
        airspyhf_close(handle);
    }
}

Result AirspyHfStream::connect(const Device::Info& info) {
    U64 serial = 0;
    try {
        serial = std::stoull(info.serial, nullptr, 16);
    } catch (...) {
        JST_ERROR("[SDR_AIRSPYHF] Invalid serial number '{}'.", info.serial);
        return Result::ERROR;
    }
    if (airspyhf_open_sn(&handle, serial) != AIRSPYHF_SUCCESS) {
        handle = nullptr;
        JST_ERROR("[SDR_AIRSPYHF] Failed to open device '{}'.", info.label);
        return Result::ERROR;
    }
    deviceInfo = info;

    U32 count = 0;
    airspyhf_get_samplerates(handle, &count, 0);
    std::vector<U32> rates(count);
    if (count == 0 || airspyhf_get_samplerates(handle, rates.data(), count) != AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to query sample rates.");
        return Result::ERROR;
    }
    if (airspyhf_set_lib_dsp(handle, 1) != AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to enable IQ correction.");
        return Result::ERROR;
    }

    auto& caps = deviceCapabilities;
    caps.frequencyRanges = {{9.0e3, 31.0e6}, {60.0e6, 260.0e6}};
    for (const auto rate : rates) {
        caps.sampleRateRanges.push_back({static_cast<F64>(rate), static_cast<F64>(rate)});
        caps.sweepSampleRate = std::max(caps.sweepSampleRate, static_cast<F64>(rate));
    }
    caps.automaticGain = true;
    caps.biasTee = false;
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 0.0;
    defaultSettlingTime = SettlingTime;
    appliedSettings.gainRange = Device::Range{0.0, AttenuationStep * MaxAttenuation};
    return Result::SUCCESS;
}

Result AirspyHfStream::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_AIRSPYHF] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_AIRSPYHF] Frequency ({:.3f} MHz) not supported.",
                  settings.frequency / 1e6);
        return Result::ERROR;
    }
    if (!settings.antenna.empty()) {
        JST_ERROR("[SDR_AIRSPYHF] Antenna selection is not supported.");
        return Result::ERROR;
    }
    if (settings.biasTee) {
        JST_WARN("[SDR_AIRSPYHF] Bias-T is not supported by the Airspy HF+.");
    }

    if (airspyhf_set_samplerate(handle, static_cast<U32>(std::lround(settings.sampleRate))) !=
        AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to set the sample rate.");
        return Result::ERROR;
    }
    if (airspyhf_set_freq(handle, static_cast<U32>(std::llround(settings.frequency))) !=
        AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to tune to {:.3f} MHz.", settings.frequency / 1e6);
        return Result::ERROR;
    }

    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));

    requestedSettings = settings;
    appliedSettings.sampleRate = settings.sampleRate;
    appliedSettings.frequency = std::round(settings.frequency);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = false;
    return Result::SUCCESS;
}

int AirspyHfStream::Callback(airspyhf_transfer_t* transfer) {
    auto* stream = static_cast<AirspyHfStream*>(transfer->ctx);
    const auto* data = transfer->samples;
    stream->queue.push(stream->streamGeneration, static_cast<U64>(transfer->sample_count),
                       [data](CF32* output, const U64 offset, const U64 count) {
                           for (U64 i = 0; i < count; ++i) {
                               output[i] = {data[offset + i].re, data[offset + i].im};
                           }
                       },
                       transfer->dropped_samples);
    return 0;
}

Result AirspyHfStream::open() {
    const F64 sampleRate = appliedSettings.sampleRate;
    JST_CHECK(queue.resize(std::max<U64>(static_cast<U64>(sampleRate / 2.0), 1 << 16)));
    streamGeneration = nextGeneration();
    queue.reset(streamGeneration, guard(), appliedSettings.frequency, sampleRate);

    if (airspyhf_start(handle, &AirspyHfStream::Callback, this) != AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to start reception.");
        airspyhf_stop(handle);
        queue.close();
        return Result::ERROR;
    }
    streaming = true;
    return Result::SUCCESS;
}

void AirspyHfStream::close() {
    if (!streaming) {
        return;
    }
    airspyhf_stop(handle);
    streaming = false;
    queue.close();
}

Result AirspyHfStream::tune(const F64 frequency) {
    if (airspyhf_set_freq(handle, static_cast<U32>(std::llround(frequency))) != AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = std::round(frequency);
    if (!streaming) {
        return Result::SUCCESS;
    }
    streamGeneration = nextGeneration();
    queue.retune(streamGeneration, InFlightTransfers, guard(), appliedSettings.frequency);
    return Result::SUCCESS;
}

Device::Capture AirspyHfStream::read(std::span<CF32> samples,
                                   const std::chrono::milliseconds timeout) {
    auto capture = queue.pop(samples, timeout);
    if (capture.status == Device::Status::Timeout && streaming &&
        airspyhf_is_streaming(handle) == 0) {
        capture.status = Device::Status::Disconnected;
        capture.error = "Airspy HF+ stopped streaming";
    }
    return capture;
}

Result AirspyHfStream::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(applyGainMode(automaticGain, requestedSettings.gain));
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result AirspyHfStream::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 AirspyHfStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Stream::resolveGain(gain, range) / AttenuationStep) * AttenuationStep;
}

Result AirspyHfStream::applyGainMode(const bool automatic, const F64 gain) {
    if (airspyhf_set_hf_agc(handle, automatic ? 1 : 0) != AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to set the gain mode.");
        return Result::ERROR;
    }
    appliedSettings.automaticGain = automatic;
    return applyGain(gain, true);
}

Result AirspyHfStream::applyGain(const F64 gain, const bool force) {
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    if (appliedSettings.automaticGain || (!force && target == appliedSettings.gain)) {
        appliedSettings.gain = target;
        return Result::SUCCESS;
    }
    const auto attenuation = static_cast<U8>(
        MaxAttenuation - static_cast<U8>(std::lround(target / AttenuationStep)));
    if (airspyhf_set_hf_att(handle, attenuation) != AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to set the gain to {:.0f} dB.", target);
        appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

}  // namespace

Result AirspyHf::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<AirspyHfStream>();
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
