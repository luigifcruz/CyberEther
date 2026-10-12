#include "rtlsdr.hh"

#include "../../tools/capture_queue.hh"
#include "../../tools/samples.hh"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <thread>

#include <rtl-sdr.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr F64 SettlingTime = 5.0e-3;
constexpr U32 TransferCount = 12;
constexpr F64 TransferDuration = 10.0e-3;
constexpr U32 MinTransferBytes = 16384;

const char* TunerName(const rtlsdr_tuner tuner) {
    switch (tuner) {
        case RTLSDR_TUNER_E4000:  return "E4000";
        case RTLSDR_TUNER_FC0012: return "FC0012";
        case RTLSDR_TUNER_FC0013: return "FC0013";
        case RTLSDR_TUNER_FC2580: return "FC2580";
        case RTLSDR_TUNER_R820T:  return "R820T";
        case RTLSDR_TUNER_R828D:  return "R828D";
        default:                  return "Unknown";
    }
}

std::vector<Device::Range> TunerRanges(const rtlsdr_tuner tuner, const bool upconverter) {
    switch (tuner) {
        case RTLSDR_TUNER_E4000:
            return {{52.0e6, 1100.0e6}, {1250.0e6, 2200.0e6}};
        case RTLSDR_TUNER_FC0012:
            return {{22.0e6, 948.6e6}};
        case RTLSDR_TUNER_FC0013:
            return {{22.0e6, 1100.0e6}};
        case RTLSDR_TUNER_FC2580:
            return {{146.0e6, 308.0e6}, {438.0e6, 924.0e6}};
        case RTLSDR_TUNER_R820T:
        case RTLSDR_TUNER_R828D:
            return {{upconverter ? 0.5e6 : 24.0e6, 1766.0e6}};
        default:
            return {};
    }
}

U32 TransferBytes(const F64 sampleRate, const bool lowLatency) {
    if (lowLatency) {
        return MinTransferBytes;
    }
    const auto bytes = static_cast<U64>(sampleRate * 2.0 * TransferDuration);
    const U64 rounded = std::clamp<U64>((bytes / MinTransferBytes) * MinTransferBytes,
                                        MinTransferBytes, 262144);
    return static_cast<U32>(rounded);
}

}  // namespace

struct RtlSdrDevice::Impl {
    explicit Impl(RtlSdrDevice& owner) : device(owner) {}

    static void Callback(unsigned char* buffer, uint32_t length, void* context);

    Result open(const Info& info);
    Result applyGain(bool automatic, F64 gain, bool setMode);
    Result startStream();
    void stopStream();

    RtlSdrDevice& device;
    rtlsdr_dev_t* handle = nullptr;
    std::vector<int> gains;
    bool biasTeeEnabled = false;
    CaptureQueue queue;
    std::thread worker;
    std::atomic<bool> stopping{false};
    std::atomic<U64> streamGeneration{0};
    bool lowLatency = false;
    std::mutex finishedMutex;
    std::condition_variable finishedCondition;
    bool finished = true;
};

void RtlSdrDevice::Impl::Callback(unsigned char* buffer, const uint32_t length, void* context) {
    auto* impl = static_cast<Impl*>(context);
    if (impl->stopping) {
        return;
    }
    const U8* data = buffer;
    impl->queue.push(impl->streamGeneration, length / 2,
                     [data](CF32* output, const U64 offset, const U64 count) {
                         ConvertU8(data + offset * 2, output, count);
                     });
}

RtlSdrDevice::RtlSdrDevice() : pimpl(std::make_unique<Impl>(*this)) {}

RtlSdrDevice::~RtlSdrDevice() {
    stop();
    if (pimpl->handle) {
        if (pimpl->biasTeeEnabled) {
            rtlsdr_set_bias_tee(pimpl->handle, 0);
        }
        rtlsdr_close(pimpl->handle);
    }
}

std::vector<Device::Info> RtlSdrDevice::Enumerate() {
    std::vector<Info> devices;
    const U32 count = rtlsdr_get_device_count();
    for (U32 i = 0; i < count; ++i) {
        char manufacturer[256] = {};
        char product[256] = {};
        char serial[256] = {};
        if (rtlsdr_get_device_usb_strings(i, manufacturer, product, serial) < 0) {
            continue;
        }
        const char* name = rtlsdr_get_device_name(i);
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("{} ({})", name ? name : "RTL-SDR", serial),
        });
    }
    return devices;
}

Result RtlSdrDevice::Create(const Info& info, std::unique_ptr<Device>& device) {
    auto rtlsdr = std::make_unique<RtlSdrDevice>();
    JST_CHECK(rtlsdr->pimpl->open(info));
    device = std::move(rtlsdr);
    return Result::SUCCESS;
}

Result RtlSdrDevice::Impl::open(const Info& info) {
    const auto index = rtlsdr_get_index_by_serial(info.serial.c_str());
    if (index < 0) {
        JST_ERROR("[SDR_RTLSDR] Device with serial '{}' is no longer available.", info.serial);
        return Result::INCOMPLETE;
    }
    if (const int status = rtlsdr_open(&handle, static_cast<U32>(index)); status < 0) {
        handle = nullptr;
        JST_ERROR("[SDR_RTLSDR] Failed to open device '{}' ({}).", info.label, status);
        return Result::ERROR;
    }

    device.deviceInfo = info;
    const auto tuner = rtlsdr_get_tuner_type(handle);
    char manufacturer[256] = {};
    char product[256] = {};
    char serial[256] = {};
    rtlsdr_get_usb_strings(handle, manufacturer, product, serial);
    const bool upconverter = std::string(manufacturer) == "RTLSDRBlog" &&
                             std::string(product).starts_with("Blog V4");

    auto& caps = device.deviceCapabilities;
    caps.frequencyRanges = TunerRanges(tuner, upconverter);
    if (caps.frequencyRanges.empty()) {
        JST_ERROR("[SDR_RTLSDR] Unsupported tuner on '{}'.", info.label);
        return Result::ERROR;
    }
    caps.sampleRateRanges = {{225001.0, 300000.0}, {900001.0, 3200000.0}};
    caps.automaticGain = true;
    caps.biasTee = true;
    caps.sweepSampleRate = 2.4e6;
    caps.usableBandwidth = 0.75;
    caps.dcExclusion = 5.0e3;
    device.defaultSettlingTime = SettlingTime;

    const int count = rtlsdr_get_tuner_gains(handle, nullptr);
    if (count > 0) {
        gains.resize(static_cast<std::size_t>(count));
        rtlsdr_get_tuner_gains(handle, gains.data());
    }
    if (!gains.empty()) {
        const auto [minimum, maximum] = std::minmax_element(gains.begin(), gains.end());
        device.appliedSettings.gainRange = Range{*minimum / 10.0, *maximum / 10.0};
    }

    JST_DEBUG("[SDR_RTLSDR] Opened '{}' with a {} tuner.", info.label, TunerName(tuner));
    return Result::SUCCESS;
}

Result RtlSdrDevice::start(const Settings& settings) {
    pimpl->stopStream();

    auto* handle = pimpl->handle;
    if (!Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_RTLSDR] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_RTLSDR] Frequency ({:.3f} MHz) not supported.",
                  settings.frequency / 1e6);
        return Result::ERROR;
    }
    if (!settings.antenna.empty()) {
        JST_ERROR("[SDR_RTLSDR] Antenna selection is not supported.");
        return Result::ERROR;
    }

    if (rtlsdr_set_sample_rate(handle, static_cast<U32>(std::lround(settings.sampleRate))) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to set the sample rate.");
        return Result::ERROR;
    }
    if (rtlsdr_set_center_freq(handle, static_cast<U32>(std::llround(settings.frequency))) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to tune to {:.3f} MHz.", settings.frequency / 1e6);
        return Result::ERROR;
    }

    JST_CHECK(pimpl->applyGain(settings.automaticGain, settings.gain, true));

    if (settings.biasTee || pimpl->biasTeeEnabled) {
        if (rtlsdr_set_bias_tee(handle, settings.biasTee ? 1 : 0) < 0) {
            JST_ERROR("[SDR_RTLSDR] Failed to set Bias-T.");
            return Result::ERROR;
        }
        pimpl->biasTeeEnabled = settings.biasTee;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = rtlsdr_get_sample_rate(handle);
    appliedSettings.frequency = rtlsdr_get_center_freq(handle);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = settings.biasTee;

    return pimpl->startStream();
}

void RtlSdrDevice::stop() {
    pimpl->stopStream();
}

Result RtlSdrDevice::setAutomaticGain(const bool automaticGain) {
    if (pimpl->applyGain(automaticGain, requestedSettings.gain, true) != Result::SUCCESS) {
        return Device::setAutomaticGain(automaticGain);
    }
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result RtlSdrDevice::setGain(const F64 gain) {
    if (pimpl->applyGain(appliedSettings.automaticGain, gain, false) != Result::SUCCESS) {
        return Device::setGain(gain);
    }
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 RtlSdrDevice::resolveGain(const F64 gain, const std::optional<Range>&) const {
    const auto& gains = pimpl->gains;
    if (gains.empty()) {
        return gain;
    }
    const int target = static_cast<int>(std::lround(gain * 10.0));
    int nearest = gains.front();
    for (const auto step : gains) {
        if (std::abs(step - target) < std::abs(nearest - target)) {
            nearest = step;
        }
    }
    return nearest / 10.0;
}

Result RtlSdrDevice::Impl::applyGain(const bool automatic, const F64 gain, const bool setMode) {
    const F64 target = device.resolveGain(gain, device.appliedSettings.gainRange);
    if (setMode && rtlsdr_set_tuner_gain_mode(handle, automatic ? 0 : 1) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to set the gain mode.");
        return Result::ERROR;
    }
    const bool unchanged = !setMode && target == device.appliedSettings.gain;
    if (!automatic && !unchanged &&
        rtlsdr_set_tuner_gain(handle, static_cast<int>(std::lround(target * 10.0))) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to set the gain to {:.1f} dB.", target);
        return Result::ERROR;
    }
    device.appliedSettings.gain = target;
    return Result::SUCCESS;
}

Result RtlSdrDevice::tune(const F64 frequency) {
    if (!Range::Contains(deviceCapabilities.frequencyRanges, frequency)) {
        JST_ERROR("[SDR_RTLSDR] Frequency ({:.3f} MHz) not supported.", frequency / 1e6);
        return Result::ERROR;
    }
    const auto set = [&] {
        if (rtlsdr_set_center_freq(pimpl->handle, static_cast<U32>(std::llround(frequency))) < 0) {
            JST_ERROR("[SDR_RTLSDR] Failed to tune to {:.3f} MHz.", frequency / 1e6);
            return Result::ERROR;
        }
        requestedSettings.frequency = frequency;
        appliedSettings.frequency = rtlsdr_get_center_freq(pimpl->handle);
        return Result::SUCCESS;
    };
    const auto change = [&] {
        JST_CHECK(set());
        return pimpl->startStream();
    };
    if (pimpl->worker.joinable() && set() == Result::SUCCESS) {
        pimpl->queue.relabel(appliedSettings.frequency);
        return Result::SUCCESS;
    }
    return restart(change);
}

Device::Capture RtlSdrDevice::read(std::span<CF32> samples, const std::chrono::milliseconds timeout) {
    return pimpl->queue.pop(samples, timeout);
}

Result RtlSdrDevice::prepareSweep(const SweepRequest& request) {
    JST_CHECK(Device::prepareSweep(request));
    if (pimpl->lowLatency) {
        return Result::SUCCESS;
    }
    pimpl->lowLatency = true;
    if (!pimpl->worker.joinable()) {
        return Result::SUCCESS;
    }
    pimpl->stopStream();
    return pimpl->startStream();
}

Result RtlSdrDevice::tuneHop(const U64 hop) {
    const F64 frequency = sweepRequest.frequencies[hop];
    if (!pimpl->worker.joinable()) {
        JST_ERROR("[SDR_RTLSDR] Cannot retune a stopped stream.");
        return Result::ERROR;
    }
    if (rtlsdr_set_center_freq(pimpl->handle, static_cast<U32>(std::llround(frequency))) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = rtlsdr_get_center_freq(pimpl->handle);
    pimpl->streamGeneration = nextGeneration();
    pimpl->queue.retune(pimpl->streamGeneration, TransferCount, guardLength,
                        appliedSettings.frequency);
    return Result::SUCCESS;
}

Result RtlSdrDevice::Impl::startStream() {
    if (rtlsdr_reset_buffer(handle) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to reset the endpoint FIFO.");
        return Result::ERROR;
    }

    const F64 sampleRate = device.appliedSettings.sampleRate;
    const U32 transferBytes = TransferBytes(sampleRate, lowLatency);
    const U64 capacity = std::max<U64>(static_cast<U64>(sampleRate / 2.0),
                                       static_cast<U64>(transferBytes) * TransferCount);
    JST_CHECK(queue.resize(capacity));

    device.guardLength = device.guardSamples(device.requestedSettings, sampleRate);
    if (device.guardLength == std::numeric_limits<U64>::max()) {
        JST_ERROR("[SDR_RTLSDR] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    streamGeneration = device.nextGeneration();
    queue.reset(streamGeneration, device.guardLength, device.appliedSettings.frequency,
                sampleRate);

    finished = false;
    try {
        worker = std::thread([impl = this, transferBytes] {
            const int status = rtlsdr_read_async(impl->handle, &Impl::Callback, impl,
                                                 TransferCount, transferBytes);
            if (status < 0 && !impl->stopping) {
                impl->queue.fail(Status::Disconnected,
                                 jst::fmt::format("reception stopped ({})", status));
            }
            {
                std::lock_guard lock(impl->finishedMutex);
                impl->finished = true;
            }
            impl->finishedCondition.notify_all();
        });
    } catch (const std::exception& e) {
        JST_ERROR("[SDR_RTLSDR] Failed to start the receive thread: {}", e.what());
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

void RtlSdrDevice::Impl::stopStream() {
    if (!worker.joinable()) {
        return;
    }
    stopping = true;
    std::unique_lock lock(finishedMutex);
    while (!finished) {
        rtlsdr_cancel_async(handle);
        finishedCondition.wait_for(lock, std::chrono::milliseconds(10));
    }
    lock.unlock();
    worker.join();
    stopping = false;
    queue.close();
}

}  // namespace Jetstream::Radio
