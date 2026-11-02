#include "base.hh"

#include "../stream.hh"
#include "../../tools/capture_queue.hh"
#include "../../tools/samples.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <thread>

#include <rtl-sdr.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr F64 SettlingTime = 5.0e-3;
constexpr U32 InFlightTransfers = 12;
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

U32 TransferBytes(const F64 sampleRate) {
    const auto bytes = static_cast<U64>(sampleRate * 2.0 * TransferDuration);
    const U64 rounded = std::clamp<U64>((bytes / MinTransferBytes) * MinTransferBytes,
                                        MinTransferBytes, 262144);
    return static_cast<U32>(rounded);
}

class RtlSdrStream final : public Stream {
 public:
    ~RtlSdrStream() override;

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
    static void Callback(unsigned char* buffer, uint32_t length, void* context);

    Result setFrequency(F64 frequency);
    Result applyGainMode(bool automatic, F64 gain);
    Result applyGain(F64 gain, bool force);

    rtlsdr_dev_t* handle = nullptr;
    std::vector<int> gains;
    bool biasTeeEnabled = false;
    CaptureQueue queue;
    std::thread worker;
    std::atomic<bool> stopping{false};
    std::atomic<U64> queueGeneration{0};
    std::mutex finishedMutex;
    std::condition_variable finishedCondition;
    bool finished = true;
};

RtlSdrStream::~RtlSdrStream() {
    close();
    if (handle) {
        if (biasTeeEnabled) {
            rtlsdr_set_bias_tee(handle, 0);
        }
        rtlsdr_close(handle);
    }
}

Result RtlSdrStream::connect(const Device::Info& info) {
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

    deviceInfo = info;
    const auto tuner = rtlsdr_get_tuner_type(handle);
    char manufacturer[256] = {};
    char product[256] = {};
    char serial[256] = {};
    rtlsdr_get_usb_strings(handle, manufacturer, product, serial);
    const bool upconverter = std::string(manufacturer) == "RTLSDRBlog" &&
                             std::string(product).starts_with("Blog V4");

    auto& caps = deviceCapabilities;
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
    defaultSettlingTime = SettlingTime;

    const int count = rtlsdr_get_tuner_gains(handle, nullptr);
    if (count > 0) {
        gains.resize(static_cast<std::size_t>(count));
        rtlsdr_get_tuner_gains(handle, gains.data());
    }
    if (!gains.empty()) {
        const auto [minimum, maximum] = std::minmax_element(gains.begin(), gains.end());
        appliedSettings.gainRange = Device::Range{*minimum / 10.0, *maximum / 10.0};
    }

    JST_DEBUG("[SDR_RTLSDR] Opened '{}' with a {} tuner.", info.label, TunerName(tuner));
    return Result::SUCCESS;
}

Result RtlSdrStream::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_RTLSDR] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_RTLSDR] Frequency ({:.3f} MHz) not supported. Use {}.",
                  settings.frequency / 1e6,
                  Device::Range::Describe(deviceCapabilities.frequencyRanges));
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
    JST_CHECK(setFrequency(settings.frequency));

    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));

    if (settings.biasTee || biasTeeEnabled) {
        if (rtlsdr_set_bias_tee(handle, settings.biasTee ? 1 : 0) < 0) {
            JST_ERROR("[SDR_RTLSDR] Failed to set Bias-T.");
            return Result::ERROR;
        }
        biasTeeEnabled = settings.biasTee;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = rtlsdr_get_sample_rate(handle);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = settings.biasTee;
    return Result::SUCCESS;
}

void RtlSdrStream::Callback(unsigned char* buffer, const uint32_t length, void* context) {
    auto* stream = static_cast<RtlSdrStream*>(context);
    if (stream->stopping) {
        return;
    }
    const U8* data = buffer;
    stream->queue.push(stream->queueGeneration, length / 2,
                       [data](CF32* output, const U64 offset, const U64 count) {
                           ConvertU8(data + offset * 2, output, count);
                       });
}

Result RtlSdrStream::open() {
    if (rtlsdr_reset_buffer(handle) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to reset the endpoint FIFO.");
        return Result::ERROR;
    }

    const F64 sampleRate = appliedSettings.sampleRate;
    const U32 transferBytes = TransferBytes(sampleRate);
    const U64 capacity = std::max<U64>(static_cast<U64>(sampleRate / 2.0),
                                       static_cast<U64>(transferBytes) * InFlightTransfers);
    JST_CHECK(queue.resize(capacity));
    queueGeneration = nextGeneration();
    queue.reset(queueGeneration, guard(), appliedSettings.frequency, sampleRate);

    finished = false;
    try {
        worker = std::thread([this, transferBytes] {
            const int status = rtlsdr_read_async(handle, &RtlSdrStream::Callback, this,
                                                 InFlightTransfers, transferBytes);
            if (status < 0 && !stopping) {
                queue.fail(Device::Status::Disconnected,
                           jst::fmt::format("reception stopped ({})", status));
            }
            {
                std::lock_guard lock(finishedMutex);
                finished = true;
            }
            finishedCondition.notify_all();
        });
    } catch (const std::exception& e) {
        JST_ERROR("[SDR_RTLSDR] Failed to start the receive thread: {}", e.what());
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

void RtlSdrStream::close() {
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

Result RtlSdrStream::setFrequency(const F64 frequency) {
    if (rtlsdr_set_center_freq(handle, static_cast<U32>(std::llround(frequency))) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    const U32 actual = rtlsdr_get_center_freq(handle);
    if (actual == 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to read back the tuned frequency.");
        return Result::ERROR;
    }
    appliedSettings.frequency = actual;
    return Result::SUCCESS;
}

Result RtlSdrStream::tune(const F64 frequency) {
    JST_CHECK(setFrequency(frequency));
    requestedSettings.frequency = frequency;
    if (!worker.joinable()) {
        return Result::SUCCESS;
    }
    queueGeneration = nextGeneration();
    queue.retune(queueGeneration, InFlightTransfers, guard(), appliedSettings.frequency);
    return Result::SUCCESS;
}

Device::Capture RtlSdrStream::read(std::span<CF32> samples,
                                   const std::chrono::milliseconds timeout) {
    return queue.pop(samples, timeout);
}

Result RtlSdrStream::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(applyGainMode(automaticGain, requestedSettings.gain));
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result RtlSdrStream::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 RtlSdrStream::resolveGain(const F64 gain, const std::optional<Device::Range>&) const {
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

Result RtlSdrStream::applyGainMode(const bool automatic, const F64 gain) {
    if (rtlsdr_set_tuner_gain_mode(handle, automatic ? 0 : 1) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to set the gain mode.");
        return Result::ERROR;
    }
    appliedSettings.automaticGain = automatic;
    return applyGain(gain, true);
}

Result RtlSdrStream::applyGain(const F64 gain, const bool force) {
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    if (appliedSettings.automaticGain || (!force && target == appliedSettings.gain)) {
        appliedSettings.gain = target;
        return Result::SUCCESS;
    }
    if (rtlsdr_set_tuner_gain(handle, static_cast<int>(std::lround(target * 10.0))) < 0) {
        JST_ERROR("[SDR_RTLSDR] Failed to set the gain to {:.1f} dB.", target);
        appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

}  // namespace

Result RtlSdr::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<RtlSdrStream>();
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
