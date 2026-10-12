#include "airspy.hh"

#include "../../tools/capture_queue.hh"
#include "../../tools/samples.hh"

#include <cmath>
#include <limits>

#include <airspy.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr F64 SettlingTime = 10.0e-3;
constexpr U8 FixedLinearityGain = 10;
constexpr U64 InFlightTransfers = 24;

std::string SerialString(const U64 serial) {
    return jst::fmt::format("{:016X}", serial);
}

}  // namespace

struct AirspyDevice::Impl {
    explicit Impl(AirspyDevice& owner) : device(owner) {}

    static int Callback(airspy_transfer* transfer);

    Result open(const Info& info);
    Result applyGainMode(bool automatic);
    Result startStream();
    void stopStream();

    AirspyDevice& device;
    airspy_device* handle = nullptr;
    bool biasTeeEnabled = false;
    bool streaming = false;
    CaptureQueue queue;
    std::atomic<U64> streamGeneration{0};
};

int AirspyDevice::Impl::Callback(airspy_transfer* transfer) {
    auto* impl = static_cast<Impl*>(transfer->ctx);
    const auto* data = static_cast<const F32*>(transfer->samples);
    impl->queue.push(impl->streamGeneration, static_cast<U64>(transfer->sample_count),
                     [data](CF32* output, const U64 offset, const U64 count) {
                         for (U64 i = 0; i < count; ++i) {
                             output[i] = {data[2 * (offset + i)], data[2 * (offset + i) + 1]};
                         }
                     },
                     transfer->dropped_samples);
    return 0;
}

AirspyDevice::AirspyDevice() : pimpl(std::make_unique<Impl>(*this)) {}

AirspyDevice::~AirspyDevice() {
    stop();
    if (pimpl->handle) {
        if (pimpl->biasTeeEnabled) {
            airspy_set_rf_bias(pimpl->handle, 0);
        }
        airspy_close(pimpl->handle);
    }
}

std::vector<Device::Info> AirspyDevice::Enumerate() {
    std::vector<Info> devices;
    const int count = airspy_list_devices(nullptr, 0);
    if (count <= 0) {
        return devices;
    }
    std::vector<uint64_t> serials(static_cast<std::size_t>(count));
    const int listed = airspy_list_devices(serials.data(), count);
    for (int i = 0; i < listed; ++i) {
        const auto serial = SerialString(serials[static_cast<std::size_t>(i)]);
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("Airspy ({})", serial),
        });
    }
    return devices;
}

Result AirspyDevice::Create(const Info& info, std::unique_ptr<Device>& device) {
    auto airspy = std::make_unique<AirspyDevice>();
    JST_CHECK(airspy->pimpl->open(info));
    device = std::move(airspy);
    return Result::SUCCESS;
}

Result AirspyDevice::Impl::open(const Info& info) {
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
    device.deviceInfo = info;

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

    auto& caps = device.deviceCapabilities;
    caps.frequencyRanges = {{24.0e6, 1800.0e6}};
    for (const auto rate : rates) {
        caps.sampleRateRanges.push_back({static_cast<F64>(rate), static_cast<F64>(rate)});
        caps.sweepSampleRate = std::max(caps.sweepSampleRate, static_cast<F64>(rate));
    }
    caps.automaticGain = true;
    caps.biasTee = true;
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 0.0;
    device.defaultSettlingTime = SettlingTime;
    return Result::SUCCESS;
}

Result AirspyDevice::start(const Settings& settings) {
    pimpl->stopStream();

    auto* handle = pimpl->handle;
    if (!Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_AIRSPY] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
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

    JST_CHECK(pimpl->applyGainMode(settings.automaticGain));

    if (settings.biasTee || pimpl->biasTeeEnabled) {
        if (airspy_set_rf_bias(handle, settings.biasTee ? 1 : 0) != AIRSPY_SUCCESS) {
            JST_ERROR("[SDR_AIRSPY] Failed to set Bias-T.");
            return Result::ERROR;
        }
        pimpl->biasTeeEnabled = settings.biasTee;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = settings.sampleRate;
    appliedSettings.frequency = std::round(settings.frequency);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = settings.biasTee;
    return pimpl->startStream();
}

void AirspyDevice::stop() {
    pimpl->stopStream();
}

Result AirspyDevice::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(pimpl->applyGainMode(automaticGain));
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result AirspyDevice::Impl::applyGainMode(const bool automatic) {
    const U8 agc = automatic ? 1 : 0;
    if (airspy_set_lna_agc(handle, agc) != AIRSPY_SUCCESS ||
        airspy_set_mixer_agc(handle, agc) != AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to set the gain mode.");
        return Result::ERROR;
    }
    if (!automatic && airspy_set_linearity_gain(handle, FixedLinearityGain) != AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to set a fixed gain.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result AirspyDevice::tune(const F64 frequency) {
    if (!Range::Contains(deviceCapabilities.frequencyRanges, frequency)) {
        JST_ERROR("[SDR_AIRSPY] Frequency ({:.3f} MHz) not supported.", frequency / 1e6);
        return Result::ERROR;
    }
    const auto set = [&] {
        if (airspy_set_freq(pimpl->handle, static_cast<U32>(std::llround(frequency))) !=
            AIRSPY_SUCCESS) {
            JST_ERROR("[SDR_AIRSPY] Failed to tune to {:.3f} MHz.", frequency / 1e6);
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
        return Result::SUCCESS;
    }
    return restart(change);
}

Device::Capture AirspyDevice::read(std::span<CF32> samples, const std::chrono::milliseconds timeout) {
    auto capture = pimpl->queue.pop(samples, timeout);
    if (capture.status == Status::Timeout && pimpl->streaming &&
        airspy_is_streaming(pimpl->handle) != AIRSPY_TRUE) {
        capture.status = Status::Disconnected;
        capture.error = "Airspy stopped streaming";
    }
    return capture;
}

Result AirspyDevice::tuneHop(const U64 hop) {
    const F64 frequency = sweepRequest.frequencies[hop];
    if (!pimpl->streaming) {
        JST_ERROR("[SDR_AIRSPY] Cannot retune a stopped stream.");
        return Result::ERROR;
    }
    if (airspy_set_freq(pimpl->handle, static_cast<U32>(std::llround(frequency))) !=
        AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = std::round(frequency);
    pimpl->streamGeneration = nextGeneration();
    pimpl->queue.retune(pimpl->streamGeneration, InFlightTransfers, guardLength,
                        appliedSettings.frequency);
    return Result::SUCCESS;
}

Result AirspyDevice::Impl::startStream() {
    const F64 sampleRate = device.appliedSettings.sampleRate;
    JST_CHECK(queue.resize(std::max<U64>(static_cast<U64>(sampleRate / 2.0), 1 << 18)));
    device.guardLength = device.guardSamples(device.requestedSettings, sampleRate);
    if (device.guardLength == std::numeric_limits<U64>::max()) {
        JST_ERROR("[SDR_AIRSPY] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    streamGeneration = device.nextGeneration();
    queue.reset(streamGeneration, device.guardLength, device.appliedSettings.frequency,
                sampleRate);

    if (airspy_start_rx(handle, &Impl::Callback, this) != AIRSPY_SUCCESS) {
        JST_ERROR("[SDR_AIRSPY] Failed to start reception.");
        airspy_stop_rx(handle);
        queue.close();
        return Result::ERROR;
    }
    streaming = true;
    return Result::SUCCESS;
}

void AirspyDevice::Impl::stopStream() {
    if (!streaming) {
        return;
    }
    airspy_stop_rx(handle);
    streaming = false;
    queue.close();
}

}  // namespace Jetstream::Radio
