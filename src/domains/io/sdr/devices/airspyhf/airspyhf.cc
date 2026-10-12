#include "airspyhf.hh"

#include "../../tools/capture_queue.hh"

#include <algorithm>
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

std::string SerialString(const U64 serial) {
    return jst::fmt::format("{:016X}", serial);
}

}  // namespace

struct AirspyHfDevice::Impl {
    explicit Impl(AirspyHfDevice& owner) : device(owner) {}

    static int Callback(airspyhf_transfer_t* transfer);

    Result open(const Info& info);
    Result applyGain(bool automatic, F64 gain, bool setMode);
    Result startStream();
    void stopStream();

    AirspyHfDevice& device;
    airspyhf_device_t* handle = nullptr;
    bool streaming = false;
    CaptureQueue queue;
    std::atomic<U64> streamGeneration{0};
};

int AirspyHfDevice::Impl::Callback(airspyhf_transfer_t* transfer) {
    auto* impl = static_cast<Impl*>(transfer->ctx);
    const auto* data = transfer->samples;
    impl->queue.push(impl->streamGeneration, static_cast<U64>(transfer->sample_count),
                     [data](CF32* output, const U64 offset, const U64 count) {
                         for (U64 i = 0; i < count; ++i) {
                             output[i] = {data[offset + i].re, data[offset + i].im};
                         }
                     },
                     transfer->dropped_samples);
    return 0;
}

AirspyHfDevice::AirspyHfDevice() : pimpl(std::make_unique<Impl>(*this)) {}

AirspyHfDevice::~AirspyHfDevice() {
    stop();
    if (pimpl->handle) {
        airspyhf_close(pimpl->handle);
    }
}

std::vector<Device::Info> AirspyHfDevice::Enumerate() {
    std::vector<Info> devices;
    const int count = airspyhf_list_devices(nullptr, 0);
    if (count <= 0) {
        return devices;
    }
    std::vector<uint64_t> serials(static_cast<std::size_t>(count));
    const int listed = airspyhf_list_devices(serials.data(), count);
    for (int i = 0; i < listed; ++i) {
        const auto serial = SerialString(serials[static_cast<std::size_t>(i)]);
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("Airspy HF+ ({})", serial),
        });
    }
    return devices;
}

Result AirspyHfDevice::Create(const Info& info, std::unique_ptr<Device>& device) {
    auto airspyhf = std::make_unique<AirspyHfDevice>();
    JST_CHECK(airspyhf->pimpl->open(info));
    device = std::move(airspyhf);
    return Result::SUCCESS;
}

Result AirspyHfDevice::Impl::open(const Info& info) {
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
    device.deviceInfo = info;

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

    auto& caps = device.deviceCapabilities;
    caps.frequencyRanges = {{9.0e3, 31.0e6}, {60.0e6, 260.0e6}};
    for (const auto rate : rates) {
        caps.sampleRateRanges.push_back({static_cast<F64>(rate), static_cast<F64>(rate)});
        caps.sweepSampleRate = std::max(caps.sweepSampleRate, static_cast<F64>(rate));
    }
    caps.automaticGain = true;
    caps.biasTee = false;
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 0.0;
    device.defaultSettlingTime = SettlingTime;
    device.appliedSettings.gainRange = Range{0.0, AttenuationStep * MaxAttenuation};
    return Result::SUCCESS;
}

Result AirspyHfDevice::start(const Settings& settings) {
    pimpl->stopStream();

    auto* handle = pimpl->handle;
    if (!Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_AIRSPYHF] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
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

    JST_CHECK(pimpl->applyGain(settings.automaticGain, settings.gain, true));

    requestedSettings = settings;
    appliedSettings.sampleRate = settings.sampleRate;
    appliedSettings.frequency = std::round(settings.frequency);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = false;
    return pimpl->startStream();
}

void AirspyHfDevice::stop() {
    pimpl->stopStream();
}

Result AirspyHfDevice::setAutomaticGain(const bool automaticGain) {
    if (pimpl->applyGain(automaticGain, requestedSettings.gain, true) != Result::SUCCESS) {
        return Device::setAutomaticGain(automaticGain);
    }
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result AirspyHfDevice::setGain(const F64 gain) {
    if (pimpl->applyGain(appliedSettings.automaticGain, gain, false) != Result::SUCCESS) {
        return Device::setGain(gain);
    }
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 AirspyHfDevice::resolveGain(const F64 gain, const std::optional<Range>& range) const {
    return std::round(Device::resolveGain(gain, range) / AttenuationStep) * AttenuationStep;
}

Result AirspyHfDevice::Impl::applyGain(const bool automatic, const F64 gain, const bool setMode) {
    const F64 target = device.resolveGain(gain, device.appliedSettings.gainRange);
    if (setMode && airspyhf_set_hf_agc(handle, automatic ? 1 : 0) != AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to set the gain mode.");
        return Result::ERROR;
    }
    const bool unchanged = !setMode && target == device.appliedSettings.gain;
    if (!automatic && !unchanged) {
        const auto attenuation = static_cast<U8>(
            MaxAttenuation - static_cast<U8>(std::lround(target / AttenuationStep)));
        if (airspyhf_set_hf_att(handle, attenuation) != AIRSPYHF_SUCCESS) {
            JST_ERROR("[SDR_AIRSPYHF] Failed to set the gain to {:.0f} dB.", target);
            return Result::ERROR;
        }
    }
    device.appliedSettings.gain = target;
    return Result::SUCCESS;
}

Result AirspyHfDevice::tune(const F64 frequency) {
    if (!Range::Contains(deviceCapabilities.frequencyRanges, frequency)) {
        JST_ERROR("[SDR_AIRSPYHF] Frequency ({:.3f} MHz) not supported.", frequency / 1e6);
        return Result::ERROR;
    }
    const auto set = [&] {
        if (airspyhf_set_freq(pimpl->handle, static_cast<U32>(std::llround(frequency))) !=
            AIRSPYHF_SUCCESS) {
            JST_ERROR("[SDR_AIRSPYHF] Failed to tune to {:.3f} MHz.", frequency / 1e6);
            return Result::ERROR;
        }
        requestedSettings.frequency = frequency;
        appliedSettings.frequency = std::round(frequency);
        return Result::SUCCESS;
    };
    const auto change = [&] {
        JST_CHECK(set());
        return pimpl->startStream();
    };
    if (pimpl->streaming && set() == Result::SUCCESS) {
        pimpl->queue.relabel(appliedSettings.frequency);
        return Result::SUCCESS;
    }
    return restart(change);
}

Device::Capture AirspyHfDevice::read(std::span<CF32> samples, const std::chrono::milliseconds timeout) {
    auto capture = pimpl->queue.pop(samples, timeout);
    if (capture.status == Status::Timeout && pimpl->streaming &&
        airspyhf_is_streaming(pimpl->handle) == 0) {
        capture.status = Status::Disconnected;
        capture.error = "Airspy HF+ stopped streaming";
    }
    return capture;
}

Result AirspyHfDevice::tuneHop(const U64 hop) {
    const F64 frequency = sweepRequest.frequencies[hop];
    if (!pimpl->streaming) {
        JST_ERROR("[SDR_AIRSPYHF] Cannot retune a stopped stream.");
        return Result::ERROR;
    }
    if (airspyhf_set_freq(pimpl->handle, static_cast<U32>(std::llround(frequency))) !=
        AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = std::round(frequency);
    pimpl->streamGeneration = nextGeneration();
    pimpl->queue.retune(pimpl->streamGeneration, InFlightTransfers, guardLength,
                        appliedSettings.frequency);
    return Result::SUCCESS;
}

Result AirspyHfDevice::Impl::startStream() {
    const F64 sampleRate = device.appliedSettings.sampleRate;
    JST_CHECK(queue.resize(std::max<U64>(static_cast<U64>(sampleRate / 2.0), 1 << 16)));
    device.guardLength = device.guardSamples(device.requestedSettings, sampleRate);
    if (device.guardLength == std::numeric_limits<U64>::max()) {
        JST_ERROR("[SDR_AIRSPYHF] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    streamGeneration = device.nextGeneration();
    queue.reset(streamGeneration, device.guardLength, device.appliedSettings.frequency,
                sampleRate);

    if (airspyhf_start(handle, &Impl::Callback, this) != AIRSPYHF_SUCCESS) {
        JST_ERROR("[SDR_AIRSPYHF] Failed to start reception.");
        airspyhf_stop(handle);
        queue.close();
        return Result::ERROR;
    }
    streaming = true;
    return Result::SUCCESS;
}

void AirspyHfDevice::Impl::stopStream() {
    if (!streaming) {
        return;
    }
    airspyhf_stop(handle);
    streaming = false;
    queue.close();
}

}  // namespace Jetstream::Radio
