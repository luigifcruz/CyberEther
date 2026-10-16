#include "base.hh"
#include "../stream.hh"
#include "../../tools/capture_queue.hh"
#include "../../tools/samples.hh"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <ad9361.h>
#include <iio.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr const char* ScanFilter = "usb=0456:b673";
constexpr F32 Q11Scale = 2048.0f;

struct Board {
    std::string uri;
    std::string serial;
};

std::string ErrorText(const long long error) {
    char text[256] = {};
    iio_strerror(static_cast<int>(error < 0 ? -error : error), text, sizeof(text));
    return text;
}

std::vector<Board> Scan() {
    std::vector<Board> boards;
    auto* scan = iio_create_scan_context(ScanFilter, 0);
    if (!scan) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to create the USB scan context: {}", ErrorText(errno));
        return boards;
    }
    iio_context_info** list = nullptr;
    const auto count = iio_scan_context_get_info_list(scan, &list);
    if (count < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to enumerate devices: {}", ErrorText(count));
    }
    for (ssize_t i = 0; i < count; ++i) {
        const std::string description = iio_context_info_get_description(list[i]);
        const auto marker = description.rfind("serial=");
        boards.push_back({
            .uri = iio_context_info_get_uri(list[i]),
            .serial = marker == std::string::npos ? "" : description.substr(marker + 7),
        });
    }
    if (list) {
        iio_context_info_list_free(list);
    }
    iio_scan_context_destroy(scan);
    return boards;
}

constexpr F64 SettlingTime = 1.0e-3;
constexpr F64 SweepSampleRate = 20.0e6;
constexpr F64 MinBandwidth = 200.0e3;
constexpr F64 MaxBandwidth = 56.0e6;
constexpr unsigned long RangeSampleRate = 3000000;
constexpr F64 FirDecimation = 4.0;
constexpr unsigned ContextTimeout = 2000;

U64 GainBand(const F64 frequency) {
    return frequency <= 1.3e9 ? 0 : frequency <= 4.0e9 ? 1 : 2;
}

std::optional<Device::Range> ReadRange(const iio_channel* channel, const char* attribute) {
    char text[128] = {};
    if (iio_channel_attr_read(channel, attribute, text, sizeof(text)) < 0) {
        return std::nullopt;
    }
    F64 minimum = 0.0;
    F64 step = 0.0;
    F64 maximum = 0.0;
    if (std::sscanf(text, "[%lf %lf %lf]", &minimum, &step, &maximum) != 3 || minimum > maximum) {
        return std::nullopt;
    }
    return Device::Range{minimum, maximum};
}

constexpr unsigned KernelBuffers = 4;
constexpr F64 TransferDuration = 10.0e-3;
constexpr U64 TransferAlignment = 4096;
constexpr U64 MinTransferSamples = 16384;
constexpr U64 MaxTransferSamples = 1 << 20;

U64 TransferSamples(const F64 sampleRate) {
    const auto samples = static_cast<U64>(sampleRate * TransferDuration);
    return std::clamp<U64>((samples / TransferAlignment) * TransferAlignment,
                           MinTransferSamples, MaxTransferSamples);
}

class PlutoSdrStream final : public Stream {
 public:
    ~PlutoSdrStream() override;

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
    Result write(const char* attribute, long long value, const char* what);
    Result configureStreamRate();
    Result setOscillator(F64 frequency);
    Result retune(F64 frequency);
    Result refreshGainRange();
    Result applyGainMode(bool automatic, F64 gain);
    Result applyGain(F64 gain, bool force);

    iio_context* context = nullptr;
    iio_device* phy = nullptr;
    iio_device* receiver = nullptr;
    iio_channel* control = nullptr;
    iio_channel* oscillator = nullptr;
    iio_channel* inPhase = nullptr;
    iio_buffer* buffer = nullptr;
    CaptureQueue queue;
    std::thread worker;
    std::atomic<bool> stopping{false};
    std::atomic<U64> streamGeneration{0};
};

PlutoSdrStream::~PlutoSdrStream() {
    close();
    if (context) {
        iio_context_destroy(context);
    }
}

Result PlutoSdrStream::connect(const Device::Info& info) {
    std::vector<std::string> matches;
    for (const auto& board : Scan()) {
        if (board.serial == info.serial) {
            matches.push_back(board.uri);
        }
    }
    if (matches.empty()) {
        JST_ERROR("[SDR_PLUTOSDR] Device with serial '{}' is no longer available.", info.serial);
        return Result::INCOMPLETE;
    }
    if (matches.size() != 1) {
        JST_ERROR("[SDR_PLUTOSDR] Several boards report serial '{}'.", info.serial);
        return Result::INCOMPLETE;
    }

    context = iio_create_context_from_uri(matches.front().c_str());
    if (!context) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to open device '{}': {}", info.label, ErrorText(errno));
        return Result::ERROR;
    }
    if (const int status = iio_context_set_timeout(context, ContextTimeout); status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to set the device timeout: {}", ErrorText(status));
        return Result::ERROR;
    }

    phy = iio_context_find_device(context, "ad9361-phy");
    receiver = iio_context_find_device(context, "cf-ad9361-lpc");
    control = phy ? iio_device_find_channel(phy, "voltage0", false) : nullptr;
    oscillator = phy ? iio_device_find_channel(phy, "altvoltage0", true) : nullptr;
    inPhase = receiver ? iio_device_find_channel(receiver, "voltage0", false) : nullptr;
    auto* quadrature = receiver ? iio_device_find_channel(receiver, "voltage1", false) : nullptr;
    if (!control || !oscillator || !inPhase || !quadrature) {
        JST_ERROR("[SDR_PLUTOSDR] Device '{}' does not expose an AD9361 receiver.", info.label);
        return Result::ERROR;
    }
    iio_channel_enable(inPhase);
    iio_channel_enable(quadrature);
    deviceInfo = info;

    if (iio_channel_find_attr(oscillator, "powerdown")) {
        if (const int status = iio_channel_attr_write_bool(oscillator, "powerdown", false);
            status < 0) {
            JST_ERROR("[SDR_PLUTOSDR] Failed to enable the receive oscillator: {}",
                      ErrorText(status));
            return Result::ERROR;
        }
    }

    if (const int status = ad9361_set_bb_rate(phy, RangeSampleRate); status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to prepare the sample rate range: {}", ErrorText(status));
        return Result::ERROR;
    }
    if (const int status = ad9361_set_trx_fir_enable(phy, false); status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to bypass the FIR: {}", ErrorText(status));
        return Result::ERROR;
    }

    const auto frequency = ReadRange(oscillator, "frequency_available");
    const auto rate = ReadRange(control, "sampling_frequency_available");
    if (!frequency || !rate) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to query device ranges.");
        return Result::ERROR;
    }

    auto& caps = deviceCapabilities;
    caps.frequencyRanges = {*frequency};
    caps.sampleRateRanges = {{std::ceil(rate->minimum / FirDecimation), rate->maximum}};
    caps.automaticGain = true;
    caps.biasTee = false;
    caps.sweepSampleRate = std::min(rate->maximum, SweepSampleRate);
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 20.0e3;
    defaultSettlingTime = SettlingTime;
    JST_CHECK(refreshGainRange());

    JST_DEBUG("[SDR_PLUTOSDR] Opened '{}' at {}.", info.label, matches.front());
    return Result::SUCCESS;
}

Result PlutoSdrStream::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_PLUTOSDR] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_PLUTOSDR] Frequency ({:.3f} MHz) not supported.",
                  settings.frequency / 1e6);
        return Result::ERROR;
    }
    if (!settings.antenna.empty()) {
        JST_ERROR("[SDR_PLUTOSDR] Antenna selection is not supported.");
        return Result::ERROR;
    }
    if (settings.biasTee) {
        JST_WARN("[SDR_PLUTOSDR] Bias-T is not supported by PlutoSDR.");
    }

    const auto rate = static_cast<unsigned long>(std::lround(settings.sampleRate));
    if (const int status = ad9361_set_bb_rate(phy, rate); status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to set the sample rate: {}", ErrorText(status));
        return Result::ERROR;
    }
    const F64 bandwidth = std::clamp(settings.sampleRate, MinBandwidth, MaxBandwidth);
    JST_CHECK(write("rf_bandwidth", std::llround(bandwidth), "the RF bandwidth"));
    JST_CHECK(setOscillator(settings.frequency));
    JST_CHECK(refreshGainRange());
    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));

    JST_CHECK(configureStreamRate());

    requestedSettings = settings;
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = false;
    return Result::SUCCESS;
}

Result PlutoSdrStream::setOscillator(const F64 frequency) {
    const auto target = std::llround(frequency);
    if (const int status = iio_channel_attr_write_longlong(oscillator, "frequency", target);
        status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to tune to {:.3f} MHz: {}", frequency / 1e6,
                  ErrorText(status));
        return Result::ERROR;
    }
    long long applied = target;
    if (const int status = iio_channel_attr_read_longlong(oscillator, "frequency", &applied);
        status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to read back the tuned frequency: {}", ErrorText(status));
        return Result::ERROR;
    }
    appliedSettings.frequency = static_cast<F64>(applied);
    return Result::SUCCESS;
}

Result PlutoSdrStream::retune(const F64 frequency) {
    if (GainBand(frequency) != GainBand(appliedSettings.frequency)) {
        appliedSettings.gainRange.reset();
    }
    JST_CHECK(setOscillator(frequency));
    if (!appliedSettings.gainRange) {
        JST_CHECK(refreshGainRange());
        if (const auto result = applyGain(requestedSettings.gain, true);
            result != Result::SUCCESS) {
            appliedSettings.gainRange.reset();
            return result;
        }
    }
    requestedSettings.frequency = frequency;
    return Result::SUCCESS;
}

Result PlutoSdrStream::open() {
    const F64 sampleRate = appliedSettings.sampleRate;
    const U64 transferSamples = TransferSamples(sampleRate);
    const U64 capacity = std::max<U64>(static_cast<U64>(sampleRate / 2.0),
                                       transferSamples * (KernelBuffers + 1));
    JST_CHECK(queue.resize(capacity));

    if (const int status = iio_device_set_kernel_buffers_count(receiver, KernelBuffers);
        status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to set the kernel buffer count: {}",
                  ErrorText(status));
        return Result::ERROR;
    }
    buffer = iio_device_create_buffer(receiver, transferSamples, false);
    if (!buffer) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to create the receive buffer: {}",
                  ErrorText(errno));
        return Result::ERROR;
    }
    if (iio_buffer_step(buffer) != 2 * sizeof(I16)) {
        JST_ERROR("[SDR_PLUTOSDR] Unexpected receive sample format.");
        iio_buffer_destroy(buffer);
        buffer = nullptr;
        return Result::ERROR;
    }

    streamGeneration = nextGeneration();
    queue.reset(streamGeneration, guard(), appliedSettings.frequency, sampleRate);

    try {
        worker = std::thread([this] {
            while (!stopping) {
                const auto bytes = iio_buffer_refill(buffer);
                if (bytes < 0) {
                    if (!stopping) {
                        queue.fail(Device::Status::Disconnected, jst::fmt::format(
                            "reception stopped ({})", ErrorText(bytes)));
                    }
                    return;
                }
                const auto* data = static_cast<const I16*>(iio_buffer_start(buffer));
                const U64 samples = static_cast<U64>(bytes) / (2 * sizeof(I16));
                queue.push(streamGeneration, samples,
                           [data](CF32* output, const U64 offset, const U64 count) {
                               ConvertI16(data + offset * 2, output, count,
                                          Q11Scale);
                           });
            }
        });
    } catch (const std::exception& e) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to start the receive thread: {}", e.what());
        iio_buffer_destroy(buffer);
        buffer = nullptr;
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

void PlutoSdrStream::close() {
    if (!buffer) {
        return;
    }
    stopping = true;
    iio_buffer_cancel(buffer);
    if (worker.joinable()) {
        worker.join();
    }
    iio_buffer_destroy(buffer);
    buffer = nullptr;
    stopping = false;
    queue.close();
}

Result PlutoSdrStream::tune(const F64 frequency) {
    JST_CHECK(retune(frequency));
    queue.relabel(appliedSettings.frequency);
    return Result::SUCCESS;
}

Device::Capture PlutoSdrStream::read(std::span<CF32> samples,
                                     const std::chrono::milliseconds timeout) {
    return queue.pop(samples, timeout);
}

Result PlutoSdrStream::write(const char* attribute, const long long value, const char* what) {
    if (const int status = iio_channel_attr_write_longlong(control, attribute, value); status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to set {}: {}", what, ErrorText(status));
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result PlutoSdrStream::configureStreamRate() {
    long long rate = 0;
    if (const int status = iio_channel_attr_read_longlong(control, "sampling_frequency", &rate);
        status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to read back the sample rate: {}", ErrorText(status));
        return Result::ERROR;
    }
    if (rate <= 0) {
        JST_ERROR("[SDR_PLUTOSDR] Invalid applied sample rate: {}", rate);
        return Result::ERROR;
    }

    const bool decimation = iio_channel_find_attr(inPhase, "sampling_frequency_available");
    if (decimation) {
        if (const int status = iio_channel_attr_write_longlong(inPhase, "sampling_frequency", rate);
            status < 0) {
            JST_ERROR("[SDR_PLUTOSDR] Failed to bypass FPGA decimation: {}", ErrorText(status));
            return Result::ERROR;
        }
    }
    if (decimation || iio_channel_find_attr(inPhase, "sampling_frequency")) {
        long long streamRate = 0;
        if (const int status = iio_channel_attr_read_longlong(inPhase, "sampling_frequency",
                                                              &streamRate); status < 0) {
            JST_ERROR("[SDR_PLUTOSDR] Failed to read back the receive sample rate: {}",
                      ErrorText(status));
            return Result::ERROR;
        }
        if (streamRate != rate) {
            JST_ERROR("[SDR_PLUTOSDR] Receive sample rate ({}) differs from the PHY rate ({}).",
                      streamRate, rate);
            return Result::ERROR;
        }
    }
    appliedSettings.sampleRate = static_cast<F64>(rate);
    return Result::SUCCESS;
}

Result PlutoSdrStream::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(applyGainMode(automaticGain, requestedSettings.gain));
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result PlutoSdrStream::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 PlutoSdrStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Stream::resolveGain(gain, range));
}

Result PlutoSdrStream::refreshGainRange() {
    const auto range = ReadRange(control, "hardwaregain_available");
    if (!range) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to query the gain range.");
        return Result::ERROR;
    }
    appliedSettings.gainRange = range;
    return Result::SUCCESS;
}

Result PlutoSdrStream::applyGainMode(const bool automatic, const F64 gain) {
    const char* mode = automatic ? "slow_attack" : "manual";
    if (const auto status = iio_channel_attr_write(control, "gain_control_mode", mode);
        status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to set the gain mode: {}", ErrorText(status));
        return Result::ERROR;
    }
    appliedSettings.automaticGain = automatic;
    return applyGain(gain, true);
}

Result PlutoSdrStream::applyGain(const F64 gain, const bool force) {
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    if (appliedSettings.automaticGain || (!force && target == appliedSettings.gain)) {
        appliedSettings.gain = target;
        return Result::SUCCESS;
    }
    if (const int status = iio_channel_attr_write_double(control, "hardwaregain", target);
        status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to set the gain to {:.0f} dB: {}", target,
                  ErrorText(status));
        appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

}  // namespace

Result PlutoSdr::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<PlutoSdrStream>();
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
