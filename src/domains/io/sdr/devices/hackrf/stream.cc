#include "base.hh"
#include "tools/firmware.hh"

#include "../stream.hh"
#include "../../tools/capture_queue.hh"
#include "../../tools/samples.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>

#include <hackrf.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr F64 SettlingTime = 2.0e-3;
constexpr U32 MaxLnaGain = 40;
constexpr U32 LnaGainStep = 8;
constexpr U32 MaxVgaGain = 62;
constexpr U32 VgaGainStep = 2;
constexpr U16 SweepApiVersion = 0x0104;
constexpr U64 InFlightTransfers = 4;

bool InitLibrary() {
    static std::once_flag flag;
    static int status = HACKRF_ERROR_OTHER;
    std::call_once(flag, [] {
        status = hackrf_init();
    });
    return status == HACKRF_SUCCESS;
}

std::pair<U32, U32> StageGains(const F64 gain) {
    const auto total = static_cast<U32>(std::lround(std::max(gain, 0.0)));
    const U32 lna = std::min(MaxLnaGain, total / LnaGainStep * LnaGainStep);
    const U32 vga = std::min(MaxVgaGain, (total - lna) / VgaGainStep * VgaGainStep);
    return {lna, vga};
}

class HackRfStream final : public Stream {
 public:
    ~HackRfStream() override;

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
    static int Callback(hackrf_transfer* transfer);

    Result setSampleRate(F64 sampleRate);
    Result applyGain(F64 gain, bool force);

    hackrf_device* handle = nullptr;
    bool biasTeeEnabled = false;
    bool streaming = false;
    CaptureQueue queue;
    std::atomic<U64> streamGeneration{0};
};

HackRfStream::~HackRfStream() {
    close();
    if (handle) {
        if (biasTeeEnabled) {
            hackrf_set_antenna_enable(handle, 0);
        }
        hackrf_close(handle);
    }
}

Result HackRfStream::connect(const Device::Info& info) {
    if (!InitLibrary()) {
        JST_ERROR("[SDR_HACKRF] Failed to initialize libhackrf.");
        return Result::ERROR;
    }
    if (hackrf_open_by_serial(info.serial.c_str(), &handle) != HACKRF_SUCCESS) {
        handle = nullptr;
        JST_ERROR("[SDR_HACKRF] Failed to open device '{}'.", info.label);
        return Result::ERROR;
    }
    deviceInfo = info;

    U16 version = 0;
    hackrf_usb_api_version_read(handle, &version);

    auto& caps = deviceCapabilities;
    caps.frequencyRanges = {{1.0e6, 6000.0e6}};
    caps.sampleRateRanges = {{2.0e6, 20.0e6}};
    caps.automaticGain = false;
    caps.biasTee = true;
    caps.sweepSampleRate = 20.0e6;
    caps.usableBandwidth = 0.75;
    caps.dcExclusion = 50.0e3;
    defaultSettlingTime = SettlingTime;
    appliedSettings.gainRange = Device::Range{0.0, static_cast<F64>(MaxLnaGain + MaxVgaGain)};
    if (version >= SweepApiVersion) {
        caps.firmwareSweep = HackRfFirmware::Capability();
    }
    return Result::SUCCESS;
}

Result HackRfStream::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_HACKRF] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_HACKRF] Frequency ({:.3f} MHz) not supported.",
                  settings.frequency / 1e6);
        return Result::ERROR;
    }
    if (!settings.antenna.empty()) {
        JST_ERROR("[SDR_HACKRF] Antenna selection is not supported.");
        return Result::ERROR;
    }
    if (settings.automaticGain) {
        JST_WARN("[SDR_HACKRF] HackRF has no automatic gain control. Using the manual gain.");
    }

    JST_CHECK(setSampleRate(settings.sampleRate));
    JST_CHECK(applyGain(settings.gain, true));
    if (hackrf_set_freq(handle, static_cast<U64>(std::llround(settings.frequency))) !=
        HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to tune to {:.3f} MHz.", settings.frequency / 1e6);
        return Result::ERROR;
    }
    if (settings.biasTee || biasTeeEnabled) {
        if (hackrf_set_antenna_enable(handle, settings.biasTee ? 1 : 0) != HACKRF_SUCCESS) {
            JST_ERROR("[SDR_HACKRF] Failed to set Bias-T.");
            return Result::ERROR;
        }
        biasTeeEnabled = settings.biasTee;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = settings.sampleRate;
    appliedSettings.frequency = std::round(settings.frequency);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = false;
    appliedSettings.biasTee = settings.biasTee;
    return Result::SUCCESS;
}

int HackRfStream::Callback(hackrf_transfer* transfer) {
    auto* stream = static_cast<HackRfStream*>(transfer->rx_ctx);
    const auto* data = reinterpret_cast<const I8*>(transfer->buffer);
    stream->queue.push(stream->streamGeneration, static_cast<U64>(transfer->valid_length) / 2,
                       [data](CF32* output, const U64 offset, const U64 count) {
                           ConvertI8(data + offset * 2, output, count);
                       });
    return 0;
}

Result HackRfStream::open() {
    const F64 sampleRate = appliedSettings.sampleRate;
    JST_CHECK(queue.resize(std::max<U64>(static_cast<U64>(sampleRate / 2.0), 1 << 18)));
    streamGeneration = nextGeneration();
    queue.reset(streamGeneration, guard(), appliedSettings.frequency, sampleRate);
    if (hackrf_start_rx(handle, &HackRfStream::Callback, this) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to start reception.");
        hackrf_stop_rx(handle);
        queue.close();
        return Result::ERROR;
    }
    streaming = true;
    return Result::SUCCESS;
}

void HackRfStream::close() {
    if (!streaming) {
        return;
    }
    hackrf_stop_rx(handle);
    streaming = false;
    queue.close();
}

Result HackRfStream::tune(const F64 frequency) {
    if (hackrf_set_freq(handle, static_cast<U64>(std::llround(frequency))) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to tune to {:.3f} MHz.", frequency / 1e6);
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

Device::Capture HackRfStream::read(std::span<CF32> samples,
                                   const std::chrono::milliseconds timeout) {
    auto capture = queue.pop(samples, timeout);
    if (capture.status == Device::Status::Timeout && streaming &&
        hackrf_is_streaming(handle) != HACKRF_TRUE) {
        capture.status = Device::Status::Disconnected;
        capture.error = "HackRF stopped streaming";
    }
    return capture;
}

Result HackRfStream::setAutomaticGain(const bool automaticGain) {
    if (automaticGain) {
        JST_WARN("[SDR_HACKRF] HackRF has no automatic gain control. Using the manual gain.");
    }
    requestedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result HackRfStream::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 HackRfStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    const auto [lna, vga] = StageGains(Stream::resolveGain(gain, range));
    return static_cast<F64>(lna + vga);
}

Result HackRfStream::setSampleRate(const F64 sampleRate) {
    const auto filter = hackrf_compute_baseband_filter_bw(static_cast<U32>(sampleRate * 0.75));
    if (hackrf_set_sample_rate(handle, sampleRate) != HACKRF_SUCCESS ||
        hackrf_set_baseband_filter_bandwidth(handle, filter) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to set the sample rate.");
        return Result::ERROR;
    }
    if (hackrf_set_amp_enable(handle, 0) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to disable the RF amplifier.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result HackRfStream::applyGain(const F64 gain, const bool force) {
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    if (!force && target == appliedSettings.gain) {
        return Result::SUCCESS;
    }
    const auto [lna, vga] = StageGains(target);
    if (hackrf_set_lna_gain(handle, lna) != HACKRF_SUCCESS ||
        hackrf_set_vga_gain(handle, vga) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to set the gain to {:.0f} dB.", target);
        appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

}  // namespace

Result HackRf::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<HackRfStream>();
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
