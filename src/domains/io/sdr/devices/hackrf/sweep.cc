#include "base.hh"
#include "tools/firmware.hh"
#include "tools/sweep_decoder.hh"
#include "tools/transfer_queue.hh"

#include "../sweep.hh"
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
constexpr U64 BlockQueueDepth = 64;
constexpr U64 InFlightTransfers = 4;

static_assert(HackRfFirmware::BlockBytes == BYTES_PER_BLOCK);

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

class HackRfSweep final : public Sweep::Retune {
 public:
    ~HackRfSweep() override;

    Result connect(const Device::Info& info);
    F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const override;
    Result run(const Running& running, const Sink& sink) override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result tune(U64 hop) override;
    Device::Capture read(std::span<CF32> samples, std::chrono::milliseconds timeout) override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;

 private:
    static int Callback(hackrf_transfer* transfer);
    static int FirmwareCallback(hackrf_transfer* transfer);

    Result setFrequency(F64 frequency);
    Result setSampleRate(F64 sampleRate);
    Result prepareFirmware();
    Result runFirmware(const Running& running, const Sink& sink);
    Result applyGain(F64 gain, bool force);

    hackrf_device* handle = nullptr;
    bool biasTeeEnabled = false;
    bool streaming = false;
    CaptureQueue queue;
    std::atomic<U64> queueGeneration{0};
    HackRfFirmware::Range range;
    std::vector<U64> headers;
    HackRfTransferQueue transfers{BlockQueueDepth};
};

HackRfSweep::~HackRfSweep() {
    close();
    if (handle) {
        if (biasTeeEnabled) {
            hackrf_set_antenna_enable(handle, 0);
        }
        hackrf_close(handle);
    }
}

Result HackRfSweep::connect(const Device::Info& info) {
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

Result HackRfSweep::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_HACKRF] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
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
    if (settings.biasTee || biasTeeEnabled) {
        if (hackrf_set_antenna_enable(handle, settings.biasTee ? 1 : 0) != HACKRF_SUCCESS) {
            JST_ERROR("[SDR_HACKRF] Failed to set Bias-T.");
            return Result::ERROR;
        }
        biasTeeEnabled = settings.biasTee;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = settings.sampleRate;
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = false;
    appliedSettings.biasTee = settings.biasTee;
    return Result::SUCCESS;
}

int HackRfSweep::Callback(hackrf_transfer* transfer) {
    auto* sweep = static_cast<HackRfSweep*>(transfer->rx_ctx);
    const auto* data = reinterpret_cast<const I8*>(transfer->buffer);
    sweep->queue.push(sweep->queueGeneration, static_cast<U64>(transfer->valid_length) / 2,
                       [data](CF32* output, const U64 offset, const U64 count) {
                           ConvertI8(data + offset * 2, output, count);
                       });
    return 0;
}

Result HackRfSweep::open() {
    if (sweepRequest.firmware) {
        return prepareFirmware();
    }
    JST_CHECK(setFrequency(sweepRequest.frequencies.front()));
    const F64 sampleRate = appliedSettings.sampleRate;
    JST_CHECK(queue.resize(std::max<U64>(static_cast<U64>(sampleRate / 2.0), 1 << 18)));
    queueGeneration = nextGeneration();
    queue.reset(queueGeneration, guard(), tunedFrequency, sampleRate);
    if (hackrf_start_rx(handle, &HackRfSweep::Callback, this) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to start reception.");
        hackrf_stop_rx(handle);
        queue.close();
        return Result::ERROR;
    }
    streaming = true;
    return Result::SUCCESS;
}

void HackRfSweep::close() {
    if (!streaming) {
        return;
    }
    hackrf_stop_rx(handle);
    streaming = false;
    queue.close();
}

Result HackRfSweep::run(const Running& running, const Sink& sink) {
    if (sweepRequest.firmware) {
        return runFirmware(running, sink);
    }
    return Retune::run(running, sink);
}

Result HackRfSweep::setFrequency(const F64 frequency) {
    if (hackrf_set_freq(handle, static_cast<U64>(std::llround(frequency))) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    tunedFrequency = std::round(frequency);
    return Result::SUCCESS;
}

Result HackRfSweep::tune(const U64 hop) {
    if (!streaming) {
        JST_ERROR("[SDR_HACKRF] Cannot retune a stopped stream.");
        return Result::ERROR;
    }
    JST_CHECK(setFrequency(sweepRequest.frequencies[hop]));
    queueGeneration = nextGeneration();
    queue.retune(queueGeneration, InFlightTransfers, guard(), tunedFrequency);
    return Result::SUCCESS;
}

Device::Capture HackRfSweep::read(std::span<CF32> samples,
                                   const std::chrono::milliseconds timeout) {
    auto capture = queue.pop(samples, timeout);
    if (capture.status == Device::Status::Timeout && streaming &&
        hackrf_is_streaming(handle) != HACKRF_TRUE) {
        capture.status = Device::Status::Disconnected;
        capture.error = "HackRF stopped streaming";
    }
    return capture;
}

Result HackRfSweep::setAutomaticGain(const bool automaticGain) {
    if (automaticGain) {
        JST_WARN("[SDR_HACKRF] HackRF has no automatic gain control. Using the manual gain.");
    }
    requestedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result HackRfSweep::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 HackRfSweep::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    const auto [lna, vga] = StageGains(Sweep::resolveGain(gain, range));
    return static_cast<F64>(lna + vga);
}

Result HackRfSweep::setSampleRate(const F64 sampleRate) {
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

Result HackRfSweep::applyGain(const F64 gain, const bool force) {
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

Result HackRfSweep::prepareFirmware() {
    const auto& request = sweepRequest;
    if (requestedSettings.settlingTime > 0.0) {
        JST_ERROR("[SDR_HACKRF] Firmware sweeps have fixed settling timing and "
                  "cannot honor a Settling Time override.");
        return Result::ERROR;
    }
    const auto snapped = HackRfFirmware::Snap(request.startFrequency, request.endFrequency);
    Device::Layout layout;
    if (!snapped ||
        !HackRfFirmware::Layout(request.startFrequency, request.endFrequency,
                                request.captureLength, layout) ||
        layout.hops.size() != request.frequencies.size() ||
        request.sampleRate != HackRfFirmware::SampleRate ||
        request.captureLength > HackRfFirmware::MaxCaptureLength ||
        request.dwell > HackRfFirmware::MaxDwell) {
        JST_ERROR("[SDR_HACKRF] Firmware sweep request violates the firmware constraints.");
        return Result::ERROR;
    }
    for (U64 i = 0; i < layout.hops.size(); ++i) {
        if (layout.hops[i].frequency != request.frequencies[i]) {
            JST_ERROR("[SDR_HACKRF] Firmware sweep hop {} does not match the firmware "
                      "sequence.", i);
            return Result::ERROR;
        }
    }
    range = *snapped;
    headers = HackRfFirmware::Headers(range);
    return Result::SUCCESS;
}

int HackRfSweep::FirmwareCallback(hackrf_transfer* transfer) {
    auto* sweep = static_cast<HackRfSweep*>(transfer->rx_ctx);
    sweep->transfers.push(transfer->buffer, static_cast<U64>(transfer->valid_length));
    return 0;
}

Result HackRfSweep::runFirmware(const Running& running, const Sink& sink) {
    const auto& request = sweepRequest;

    JST_CHECK(setSampleRate(HackRfFirmware::SampleRate));
    JST_CHECK(applyGain(requestedSettings.gain, true));

    const uint16_t ranges[2] = {static_cast<uint16_t>(range.start),
                                static_cast<uint16_t>(range.end)};
    if (hackrf_init_sweep(handle, ranges, 1,
                          static_cast<U32>(request.dwell * HackRfFirmware::BlockBytes),
                          static_cast<U32>(HackRfFirmware::StepWidth),
                          static_cast<U32>(HackRfFirmware::Offset),
                          INTERLEAVED) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to initialize the firmware sweep.");
        return Result::ERROR;
    }

    HackRfSweepDecoder decoder;
    decoder.configure(headers, request.dwell, request.captureLength, HackRfFirmware::BlockBytes,
                      HackRfFirmware::Offset);
    transfers.clear();

    if (hackrf_start_rx_sweep(handle, &HackRfSweep::FirmwareCallback, this) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to start the firmware sweep.");
        hackrf_stop_rx(handle);
        return Result::ERROR;
    }

    Result result = Result::SUCCESS;
    U64 failures = 0;
    U64 losses = 0;
    U64 blocks = 0;
    auto progress = std::chrono::steady_clock::now();
    const auto deliver = [&](const Capture& capture) {
        failures = 0;
        return sink(capture);
    };
    while (running() && result == Result::SUCCESS) {
        HackRfTransferQueue::Transfer transfer;
        if (transfers.pop(transfer, std::chrono::milliseconds(100))) {
            if (!decoder.decode(transfer, deliver)) {
                break;
            }
        } else if (hackrf_is_streaming(handle) != HACKRF_TRUE) {
            JST_ERROR("[SDR_HACKRF] Firmware sweep stopped streaming.");
            result = Result::ERROR;
            break;
        }
        for (; losses < decoder.losses() && result == Result::SUCCESS; ++losses) {
            result = fail(failures);
        }
        if (decoder.blocks() != blocks) {
            blocks = decoder.blocks();
            progress = std::chrono::steady_clock::now();
        } else if (result == Result::SUCCESS &&
                   std::chrono::steady_clock::now() - progress > Device::StallTimeout) {
            JST_ERROR("[SDR_HACKRF] Timed out waiting for sweep samples from {}.",
                      deviceInfo.label);
            progress = std::chrono::steady_clock::now();
            result = fail(failures);
        }
    }
    hackrf_stop_rx(handle);
    return result;
}

}  // namespace

Result HackRf::Sweep(const Device::Info& info, std::unique_ptr<Radio::Sweep>& sweep) {
    auto created = std::make_unique<HackRfSweep>();
    JST_CHECK(created->connect(info));
    sweep = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
