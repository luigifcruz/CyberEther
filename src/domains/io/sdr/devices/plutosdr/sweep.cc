#include "base.hh"
#include "tools/sweep_layout.hh"

#include "../sweep.hh"
#include "../../tools/samples.hh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <ad9361.h>
#include <iio.h>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* ScanFilter = "usb=0456:b673";
constexpr F32 Q11Scale = 2048.0f;
constexpr F64 SettlingTime = 1.0e-3;
constexpr F64 SweepSampleRate = 20.0e6;
constexpr F64 MinBandwidth = 200.0e3;
constexpr F64 MaxBandwidth = 56.0e6;
constexpr unsigned long RangeSampleRate = 3000000;
constexpr F64 FirDecimation = 4.0;
constexpr unsigned ContextTimeout = 2000;
constexpr auto SettlePollInterval = std::chrono::duration<F64>(0.01);
constexpr unsigned KernelBuffers = 1;

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

class PlutoSdrSweep final : public Sweep {
 public:
    ~PlutoSdrSweep() override;

    Result connect(const Device::Info& info);
    F64 resolveGain(F64 gain, const std::optional<Device::Range>& range) const override;
    Result run(const Running& running, const Sink& sink) override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;

 private:
    Result write(const char* attribute, long long value, const char* what);
    Result configureStreamRate();
    Result tuneOscillator(F64 frequency);
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
    std::optional<PlutoSdrSweepLayout> layout;
    F64 tunedFrequency = 0.0;
};

PlutoSdrSweep::~PlutoSdrSweep() {
    close();
    if (context) {
        iio_context_destroy(context);
    }
}

Result PlutoSdrSweep::connect(const Device::Info& info) {
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

Result PlutoSdrSweep::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_PLUTOSDR] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
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
    JST_CHECK(refreshGainRange());
    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));

    JST_CHECK(configureStreamRate());

    requestedSettings = settings;
    appliedSettings.antenna.clear();
    appliedSettings.automaticGain = settings.automaticGain;
    appliedSettings.biasTee = false;
    return Result::SUCCESS;
}

Result PlutoSdrSweep::tuneOscillator(const F64 frequency) {
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
    tunedFrequency = static_cast<F64>(applied);
    return Result::SUCCESS;
}

Result PlutoSdrSweep::retune(const F64 frequency) {
    if (GainBand(frequency) != GainBand(tunedFrequency)) {
        appliedSettings.gainRange.reset();
    }
    JST_CHECK(tuneOscillator(frequency));
    if (!appliedSettings.gainRange) {
        JST_CHECK(refreshGainRange());
        if (const auto result = applyGain(requestedSettings.gain, true);
            result != Result::SUCCESS) {
            appliedSettings.gainRange.reset();
            return result;
        }
    }
    return Result::SUCCESS;
}

Result PlutoSdrSweep::open() {
    layout = PlutoSdrSweepLayout::Create(sweepRequest.captureLength, sweepRequest.dwell);
    if (!layout || appliedSettings.sampleRate <= 0.0) {
        JST_ERROR("[SDR_PLUTOSDR] Invalid sweep capture size or sample rate.");
        return Result::ERROR;
    }
    appliedSettings.gainRange.reset();
    return retune(sweepRequest.frequencies.front());
}

void PlutoSdrSweep::close() {
    layout.reset();
}

Result PlutoSdrSweep::run(const Running& running, const Sink& sink) {
    if (!layout) {
        JST_ERROR("[SDR_PLUTOSDR] No sweep is started on {}.", deviceInfo.label);
        return Result::ERROR;
    }
    const auto& request = sweepRequest;
    const U64 required = layout->captureSamples;
    std::vector<CF32> samples;
    try {
        samples.resize(required);
    } catch (const std::exception&) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to allocate the sweep capture buffer.");
        return Result::ERROR;
    }

    if (const int status = iio_device_set_kernel_buffers_count(receiver, KernelBuffers);
        status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to set the sweep buffer count: {}", ErrorText(status));
        return Result::ERROR;
    }
    std::unique_ptr<iio_buffer, decltype(&iio_buffer_destroy)> buffer(
        iio_device_create_buffer(receiver, layout->transferSamples, false),
        &iio_buffer_destroy);
    if (!buffer) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to create the sweep buffer: {}", ErrorText(errno));
        return Result::ERROR;
    }
    if (iio_buffer_step(buffer.get()) != 2 * sizeof(I16)) {
        JST_ERROR("[SDR_PLUTOSDR] Unexpected sweep sample format.");
        return Result::ERROR;
    }

    U64 hop = 0;
    U64 failures = 0;
    while (running()) {
        if (retune(request.frequencies[hop]) != Result::SUCCESS) {
            JST_CHECK(fail(failures));
            continue;
        }
        const auto settle = std::chrono::duration<F64>(
            static_cast<F64>(guard()) / appliedSettings.sampleRate);
        const auto settled = Clock::now() + settle;
        while (running() && Clock::now() < settled) {
            const auto remaining = std::chrono::duration<F64>(settled - Clock::now());
            std::this_thread::sleep_for(std::min(remaining, SettlePollInterval));
        }
        U64 filled = 0;
        unsigned discard = KernelBuffers + 1;
        while (running() && filled < required) {
            const auto bytes = iio_buffer_refill(buffer.get());
            if (bytes < 0) {
                JST_ERROR("[SDR_PLUTOSDR] Sweep reception stopped: {}", ErrorText(bytes));
                return Result::ERROR;
            }
            if (discard > 0) {
                --discard;
                continue;
            }
            const U64 count = std::min(layout->batchSamples, required - filled);
            if (static_cast<U64>(bytes) / (2 * sizeof(I16)) < count) {
                break;
            }
            const auto* data = static_cast<const I16*>(iio_buffer_start(buffer.get()));
            ConvertI16(data, samples.data() + filled, count, Q11Scale);
            filled += count;
        }
        if (!running()) {
            break;
        }
        if (filled != required) {
            JST_CHECK(fail(failures));
            continue;
        }
        failures = 0;
        if (!sink({.hop = hop, .frequency = tunedFrequency, .samples = samples})) {
            break;
        }
        hop = (hop + 1) % request.frequencies.size();
    }
    return Result::SUCCESS;
}

Result PlutoSdrSweep::write(const char* attribute, const long long value, const char* what) {
    if (const int status = iio_channel_attr_write_longlong(control, attribute, value); status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to set {}: {}", what, ErrorText(status));
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result PlutoSdrSweep::configureStreamRate() {
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

Result PlutoSdrSweep::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(applyGainMode(automaticGain, requestedSettings.gain));
    requestedSettings.automaticGain = automaticGain;
    appliedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result PlutoSdrSweep::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 PlutoSdrSweep::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Sweep::resolveGain(gain, range));
}

Result PlutoSdrSweep::refreshGainRange() {
    const auto range = ReadRange(control, "hardwaregain_available");
    if (!range) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to query the gain range.");
        return Result::ERROR;
    }
    appliedSettings.gainRange = range;
    return Result::SUCCESS;
}

Result PlutoSdrSweep::applyGainMode(const bool automatic, const F64 gain) {
    const char* mode = automatic ? "slow_attack" : "manual";
    if (const auto status = iio_channel_attr_write(control, "gain_control_mode", mode);
        status < 0) {
        JST_ERROR("[SDR_PLUTOSDR] Failed to set the gain mode: {}", ErrorText(status));
        return Result::ERROR;
    }
    appliedSettings.automaticGain = automatic;
    return applyGain(gain, true);
}

Result PlutoSdrSweep::applyGain(const F64 gain, const bool force) {
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

Result PlutoSdr::Sweep(const Device::Info& info, std::unique_ptr<Radio::Sweep>& sweep) {
    auto created = std::make_unique<PlutoSdrSweep>();
    JST_CHECK(created->connect(info));
    sweep = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
