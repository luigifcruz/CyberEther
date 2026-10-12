#include "hackrf.hh"

#include "../../tools/capture_queue.hh"
#include "../../tools/samples.hh"
#include "sweep.hh"
#include "sweep_decoder.hh"
#include "transfer_queue.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>

#include <hackrf.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr F64 SettlingTime = 2.0e-3;
constexpr U32 FixedLnaGain = 16;
constexpr U32 FixedVgaGain = 20;
constexpr U16 SweepApiVersion = 0x0104;
constexpr U64 BlockQueueDepth = 64;
constexpr U64 InFlightTransfers = 4;

static_assert(HackRfSweep::BlockBytes == BYTES_PER_BLOCK);

std::once_flag& InitFlag() {
    static std::once_flag flag;
    return flag;
}

bool InitLibrary() {
    static int status = HACKRF_ERROR_OTHER;
    std::call_once(InitFlag(), [] {
        status = hackrf_init();
    });
    return status == HACKRF_SUCCESS;
}

std::string ShortSerial(const std::string& serial) {
    return serial.size() > 16 ? serial.substr(serial.size() - 16) : serial;
}

}  // namespace

struct HackRfDevice::Impl {
    explicit Impl(HackRfDevice& owner) : device(owner) {}

    static int Callback(hackrf_transfer* transfer);
    static int SweepCallback(hackrf_transfer* transfer);

    Result open(const Info& info);
    Result configure(F64 sampleRate);
    Result startStream();
    void stopStream();

    HackRfDevice& device;
    hackrf_device* handle = nullptr;
    bool biasTeeEnabled = false;
    bool streaming = false;
    bool sweeping = false;
    CaptureQueue queue;
    std::atomic<U64> streamGeneration{0};
    std::vector<U64> sweepHeaders;
    HackRfSweep::Range sweepRange;
    TransferQueue transfers{BlockQueueDepth};
};

int HackRfDevice::Impl::Callback(hackrf_transfer* transfer) {
    auto* impl = static_cast<Impl*>(transfer->rx_ctx);
    const auto* data = reinterpret_cast<const I8*>(transfer->buffer);
    impl->queue.push(impl->streamGeneration, static_cast<U64>(transfer->valid_length) / 2,
                     [data](CF32* output, const U64 offset, const U64 count) {
                         ConvertI8(data + offset * 2, output, count);
                     });
    return 0;
}

int HackRfDevice::Impl::SweepCallback(hackrf_transfer* transfer) {
    auto* impl = static_cast<Impl*>(transfer->rx_ctx);
    impl->transfers.push(transfer->buffer, static_cast<U64>(transfer->valid_length));
    return 0;
}

HackRfDevice::HackRfDevice() : pimpl(std::make_unique<Impl>(*this)) {}

HackRfDevice::~HackRfDevice() {
    stop();
    if (pimpl->handle) {
        if (pimpl->biasTeeEnabled) {
            hackrf_set_antenna_enable(pimpl->handle, 0);
        }
        hackrf_close(pimpl->handle);
    }
}

std::vector<Device::Info> HackRfDevice::Enumerate() {
    std::vector<Info> devices;
    if (!InitLibrary()) {
        return devices;
    }
    auto* list = hackrf_device_list();
    if (!list) {
        return devices;
    }
    for (int i = 0; i < list->devicecount; ++i) {
        const char* serial = list->serial_numbers[i];
        if (!serial) {
            continue;
        }
        const auto board = hackrf_usb_board_id_name(list->usb_board_ids[i]);
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("{} ({})", board ? board : "HackRF", ShortSerial(serial)),
        });
    }
    hackrf_device_list_free(list);
    return devices;
}

Result HackRfDevice::Create(const Info& info, std::unique_ptr<Device>& device) {
    if (!InitLibrary()) {
        JST_ERROR("[SDR_HACKRF] Failed to initialize libhackrf.");
        return Result::ERROR;
    }
    auto hackrf = std::make_unique<HackRfDevice>();
    JST_CHECK(hackrf->pimpl->open(info));
    device = std::move(hackrf);
    return Result::SUCCESS;
}

Result HackRfDevice::Impl::open(const Info& info) {
    if (hackrf_open_by_serial(info.serial.c_str(), &handle) != HACKRF_SUCCESS) {
        handle = nullptr;
        JST_ERROR("[SDR_HACKRF] Failed to open device '{}'.", info.label);
        return Result::ERROR;
    }
    device.deviceInfo = info;

    U16 version = 0;
    hackrf_usb_api_version_read(handle, &version);

    auto& caps = device.deviceCapabilities;
    caps.frequencyRanges = {{1.0e6, 6000.0e6}};
    caps.sampleRateRanges = {{2.0e6, 20.0e6}};
    caps.automaticGain = false;
    caps.biasTee = true;
    caps.sweepSampleRate = 20.0e6;
    caps.usableBandwidth = 0.75;
    caps.dcExclusion = 50.0e3;
    device.defaultSettlingTime = SettlingTime;
    if (version >= SweepApiVersion) {
        caps.firmwareSweep = HackRfSweep::Capability();
    }
    return Result::SUCCESS;
}

Result HackRfDevice::start(const Settings& settings) {
    pimpl->stopStream();

    auto* handle = pimpl->handle;
    if (!Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_HACKRF] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_HACKRF] Frequency ({:.3f} MHz) not supported.",
                  settings.frequency / 1e6);
        return Result::ERROR;
    }
    if (!settings.antenna.empty()) {
        JST_ERROR("[SDR_HACKRF] Antenna selection is not supported.");
        return Result::ERROR;
    }
    if (settings.automaticGain) {
        JST_WARN("[SDR_HACKRF] HackRF has no automatic gain control. Using a fixed LNA {} dB, "
                 "VGA {} dB.", FixedLnaGain, FixedVgaGain);
    }

    JST_CHECK(pimpl->configure(settings.sampleRate));
    if (hackrf_set_freq(handle, static_cast<U64>(std::llround(settings.frequency))) !=
        HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to tune to {:.3f} MHz.", settings.frequency / 1e6);
        return Result::ERROR;
    }
    if (settings.biasTee || pimpl->biasTeeEnabled) {
        if (hackrf_set_antenna_enable(handle, settings.biasTee ? 1 : 0) != HACKRF_SUCCESS) {
            JST_ERROR("[SDR_HACKRF] Failed to set Bias-T.");
            return Result::ERROR;
        }
        pimpl->biasTeeEnabled = settings.biasTee;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = settings.sampleRate;
    appliedSettings.frequency = std::round(settings.frequency);
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = false;
    appliedSettings.biasTee = settings.biasTee;
    return pimpl->startStream();
}

void HackRfDevice::stop() {
    pimpl->stopStream();
}

Result HackRfDevice::tune(const F64 frequency) {
    if (!Range::Contains(deviceCapabilities.frequencyRanges, frequency)) {
        JST_ERROR("[SDR_HACKRF] Frequency ({:.3f} MHz) not supported.", frequency / 1e6);
        return Result::ERROR;
    }
    return restart([&] {
        if (hackrf_set_freq(pimpl->handle, static_cast<U64>(std::llround(frequency))) !=
            HACKRF_SUCCESS) {
            JST_ERROR("[SDR_HACKRF] Failed to tune to {:.3f} MHz.", frequency / 1e6);
            return Result::ERROR;
        }
        requestedSettings.frequency = frequency;
        appliedSettings.frequency = std::round(frequency);
        return pimpl->startStream();
    });
}

Device::Capture HackRfDevice::read(std::span<CF32> samples, const std::chrono::milliseconds timeout) {
    auto capture = pimpl->queue.pop(samples, timeout);
    if (capture.status == Status::Timeout && pimpl->streaming &&
        hackrf_is_streaming(pimpl->handle) != HACKRF_TRUE) {
        capture.status = Status::Disconnected;
        capture.error = "HackRF stopped streaming";
    }
    return capture;
}

Result HackRfDevice::tuneHop(const U64 hop) {
    const F64 frequency = sweepRequest.frequencies[hop];
    if (!pimpl->streaming) {
        JST_ERROR("[SDR_HACKRF] Cannot retune a stopped stream.");
        return Result::ERROR;
    }
    if (hackrf_set_freq(pimpl->handle, static_cast<U64>(std::llround(frequency))) !=
        HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = std::round(frequency);
    pimpl->streamGeneration = nextGeneration();
    pimpl->queue.retune(pimpl->streamGeneration, InFlightTransfers, guardLength,
                        appliedSettings.frequency);
    return Result::SUCCESS;
}

Result HackRfDevice::prepareSweep(const SweepRequest& request) {
    if (!request.firmware) {
        return Device::prepareSweep(request);
    }
    if (requestedSettings.settlingTime > 0.0) {
        JST_ERROR("[SDR_HACKRF] Firmware sweeps have fixed settling timing and "
                  "cannot honor a Settling Time override.");
        return Result::ERROR;
    }
    if (!deviceCapabilities.firmwareSweep) {
        JST_ERROR("[SDR_HACKRF] Firmware sweeps need USB API 0x{:04x} or newer.",
                  SweepApiVersion);
        return Result::ERROR;
    }
    const auto range = HackRfSweep::Snap(request.startFrequency, request.endFrequency);
    SweepLayout layout;
    if (!range ||
        !HackRfSweep::Layout(request.startFrequency, request.endFrequency, request.captureLength,
                             layout) ||
        layout.hops.size() != request.frequencies.size() ||
        request.sampleRate != HackRfSweep::SampleRate ||
        request.captureLength > HackRfSweep::MaxCaptureLength ||
        request.dwell == 0 || request.dwell > HackRfSweep::MaxDwell) {
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
    sweepRequest = request;
    pimpl->sweepHeaders = HackRfSweep::Headers(*range);
    pimpl->sweepRange = *range;
    sweepPrepared = true;
    return Result::SUCCESS;
}

Result HackRfDevice::runSweep(const SweepRunning& running, const SweepSink& sink) {
    if (!sweepPrepared || !sweepRequest.firmware) {
        return Device::runSweep(running, sink);
    }
    auto* handle = pimpl->handle;
    auto& transfers = pimpl->transfers;
    const auto& request = sweepRequest;

    pimpl->stopStream();
    JST_CHECK(pimpl->configure(HackRfSweep::SampleRate));

    const uint16_t ranges[2] = {static_cast<uint16_t>(pimpl->sweepRange.start),
                                static_cast<uint16_t>(pimpl->sweepRange.end)};
    if (hackrf_init_sweep(handle, ranges, 1,
                          static_cast<U32>(request.dwell * HackRfSweep::BlockBytes),
                          static_cast<U32>(HackRfSweep::StepWidth),
                          static_cast<U32>(HackRfSweep::Offset), INTERLEAVED) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to initialize the firmware sweep.");
        return Result::ERROR;
    }

    SweepDecoder decoder;
    decoder.configure(pimpl->sweepHeaders, request.dwell, request.captureLength,
                      HackRfSweep::BlockBytes, HackRfSweep::Offset);
    transfers.clear();

    if (hackrf_start_rx_sweep(handle, &Impl::SweepCallback, pimpl.get()) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to start the firmware sweep.");
        hackrf_stop_rx(handle);
        return Result::ERROR;
    }
    pimpl->sweeping = true;

    Result result = Result::SUCCESS;
    U64 failures = 0;
    U64 losses = 0;
    U64 blocks = 0;
    auto progress = std::chrono::steady_clock::now();
    const auto deliver = [&](const SweepCapture& capture) {
        failures = 0;
        return sink(capture);
    };
    while (running() && result == Result::SUCCESS) {
        TransferQueue::Transfer transfer;
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
                   std::chrono::steady_clock::now() - progress > StallTimeout) {
            JST_ERROR("[SDR] Timed out waiting for sweep samples from {}.", deviceInfo.label);
            progress = std::chrono::steady_clock::now();
            result = fail(failures);
        }
    }
    hackrf_stop_rx(handle);
    pimpl->sweeping = false;
    return result;
}

Result HackRfDevice::Impl::configure(const F64 sampleRate) {
    const auto filter = hackrf_compute_baseband_filter_bw(static_cast<U32>(sampleRate * 0.75));
    if (hackrf_set_sample_rate(handle, sampleRate) != HACKRF_SUCCESS ||
        hackrf_set_baseband_filter_bandwidth(handle, filter) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to set the sample rate.");
        return Result::ERROR;
    }
    if (hackrf_set_amp_enable(handle, 0) != HACKRF_SUCCESS ||
        hackrf_set_lna_gain(handle, FixedLnaGain) != HACKRF_SUCCESS ||
        hackrf_set_vga_gain(handle, FixedVgaGain) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to set a fixed gain.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result HackRfDevice::Impl::startStream() {
    const F64 sampleRate = device.appliedSettings.sampleRate;
    JST_CHECK(queue.resize(std::max<U64>(static_cast<U64>(sampleRate / 2.0), 1 << 18)));
    device.guardLength = device.guardSamples(device.requestedSettings, sampleRate);
    if (device.guardLength == std::numeric_limits<U64>::max()) {
        JST_ERROR("[SDR_HACKRF] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    streamGeneration = device.nextGeneration();
    queue.reset(streamGeneration, device.guardLength, device.appliedSettings.frequency,
                sampleRate);
    if (hackrf_start_rx(handle, &Impl::Callback, this) != HACKRF_SUCCESS) {
        JST_ERROR("[SDR_HACKRF] Failed to start reception.");
        hackrf_stop_rx(handle);
        queue.close();
        return Result::ERROR;
    }
    streaming = true;
    return Result::SUCCESS;
}

void HackRfDevice::Impl::stopStream() {
    if (streaming || sweeping) {
        hackrf_stop_rx(handle);
    }
    streaming = false;
    sweeping = false;
    queue.close();
}

}  // namespace Jetstream::Radio
