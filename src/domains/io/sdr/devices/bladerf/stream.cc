#include "base.hh"
#include "tools/quick_tune_budget.hh"

#include "../stream.hh"
#include "../../tools/samples.hh"
#include "../../tools/timestamp_filter.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#if defined(JST_OS_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include <libbladeRF.h>
#if defined(JST_OS_WINDOWS)
#undef ERROR
#undef FATAL
#endif

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

using Status = Device::Status;

constexpr F64 SettlingTime = 1.0e-3;
constexpr unsigned SyncBuffers = 16;
constexpr unsigned SyncBufferSize = 8192;
constexpr unsigned SyncTransfers = 8;
constexpr U64 DrainSamples = 65536;
constexpr U64 MessageSamples = 508;
constexpr F32 Q11Scale = 2048.0f;
const bladerf_channel Channel = BLADERF_CHANNEL_RX(0);

using Clock = std::chrono::steady_clock;

U64 GainBand(const F64 frequency) {
    return frequency <= 1.3e9 ? 0 : frequency <= 4.0e9 ? 1 : 2;
}

std::vector<Device::Range> ToRanges(const bladerf_range* range) {
    if (!range) {
        return {};
    }
    return {{static_cast<F64>(range->min) * range->scale,
             static_cast<F64>(range->max) * range->scale}};
}

std::chrono::milliseconds Remaining(const Clock::time_point deadline) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
    return std::max(left, std::chrono::milliseconds(1));
}

Device::Capture Failed(const int status) {
    switch (status) {
        case BLADERF_ERR_TIMEOUT:
            return {.status = Status::Timeout};
        case BLADERF_ERR_NODEV:
            return {.status = Status::Disconnected, .error = "device disconnected"};
        default:
            return {.status = Status::Failure, .error = bladerf_strerror(status)};
    }
}

class BladeRfStream final : public Stream {
 public:
    ~BladeRfStream() override;

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
    Result retune(F64 frequency);
    Result resynchronize(U64 settle);
    Result applyGainMode(bool requested, F64 gain);
    Result applyGain(F64 gain, bool force);

    bladerf* handle = nullptr;
    std::string board;
    bool biasTeeEnabled = false;
    bool streaming = false;
    bool synchronized = false;
    U64 epoch = 0;
    U64 queueGeneration = 0;
    TimestampFilter timestamps;
    std::vector<I16> raw;
    std::vector<I16> drain;
};

BladeRfStream::~BladeRfStream() {
    close();
    if (handle) {
        if (biasTeeEnabled) {
            bladerf_set_bias_tee(handle, Channel, false);
        }
        bladerf_close(handle);
    }
}

Result BladeRfStream::connect(const Device::Info& info) {
    bladerf_devinfo devinfo;
    bladerf_init_devinfo(&devinfo);
    std::strncpy(devinfo.serial, info.serial.c_str(), BLADERF_SERIAL_LENGTH - 1);
    if (const int status = bladerf_open_with_devinfo(&handle, &devinfo); status != 0) {
        handle = nullptr;
        JST_ERROR("[SDR_BLADERF] Failed to open device '{}': {}", info.label,
                  bladerf_strerror(status));
        return Result::ERROR;
    }
    deviceInfo = info;
    board = bladerf_get_board_name(handle);

    const bladerf_range* frequency = nullptr;
    const bladerf_range* rate = nullptr;
    if (bladerf_get_frequency_range(handle, Channel, &frequency) != 0 ||
        bladerf_get_sample_rate_range(handle, Channel, &rate) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to query device ranges.");
        return Result::ERROR;
    }

    const bool second = board == "bladerf2";
    auto& caps = deviceCapabilities;
    caps.frequencyRanges = ToRanges(frequency);
    caps.sampleRateRanges = ToRanges(rate);
    caps.automaticGain = second;
    caps.biasTee = second;
    caps.sweepSampleRate = second ? 30.72e6 : 20.0e6;
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 20.0e3;
    defaultSettlingTime = SettlingTime;
    return Result::SUCCESS;
}

Result BladeRfStream::apply(const Settings& settings) {
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_BLADERF] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_BLADERF] Frequency ({:.3f} MHz) not supported.",
                  settings.frequency / 1e6);
        return Result::ERROR;
    }
    if (!settings.antenna.empty()) {
        JST_ERROR("[SDR_BLADERF] Antenna selection is not supported.");
        return Result::ERROR;
    }
    if (settings.biasTee && !deviceCapabilities.biasTee) {
        JST_WARN("[SDR_BLADERF] Bias-T is not supported by this bladeRF.");
    }

    bladerf_sample_rate actualRate = 0;
    bladerf_bandwidth actualBandwidth = 0;
    const auto rate = static_cast<bladerf_sample_rate>(std::lround(settings.sampleRate));
    if (bladerf_set_sample_rate(handle, Channel, rate, &actualRate) != 0 ||
        bladerf_set_bandwidth(handle, Channel, static_cast<bladerf_bandwidth>(rate * 0.8),
                              &actualBandwidth) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to set the sample rate.");
        return Result::ERROR;
    }

    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));

    if (deviceCapabilities.biasTee && (settings.biasTee || biasTeeEnabled)) {
        if (bladerf_set_bias_tee(handle, Channel, settings.biasTee) != 0) {
            JST_ERROR("[SDR_BLADERF] Failed to set Bias-T.");
            return Result::ERROR;
        }
        biasTeeEnabled = settings.biasTee;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = actualRate;
    appliedSettings.antenna.clear();
    appliedSettings.biasTee = settings.biasTee && deviceCapabilities.biasTee;
    return retune(settings.frequency);
}

Result BladeRfStream::retune(const F64 frequency) {
    const auto target = static_cast<bladerf_frequency>(std::llround(frequency));
    if (bladerf_set_frequency(handle, Channel, target) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    bladerf_frequency actual = 0;
    if (bladerf_get_frequency(handle, Channel, &actual) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to read back the tuned frequency.");
        return Result::ERROR;
    }
    const bool changedBand = board == "bladerf2" &&
        GainBand(appliedSettings.frequency) != GainBand(static_cast<F64>(actual));
    requestedSettings.frequency = frequency;
    appliedSettings.frequency = static_cast<F64>(actual);
    return applyGain(requestedSettings.gain, changedBand);
}

Result BladeRfStream::tune(const F64 frequency) {
    JST_CHECK(retune(frequency));
    if (!streaming) {
        return Result::SUCCESS;
    }
    queueGeneration = nextGeneration();
    return resynchronize(guard());
}

Result BladeRfStream::open() {
    try {
        drain.resize(DrainSamples * 2);
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR_BLADERF] Failed to allocate the drain buffer.");
        return Result::ERROR;
    }
    if (bladerf_sync_config(handle, BLADERF_RX_X1, BLADERF_FORMAT_SC16_Q11_META,
                            SyncBuffers, SyncBufferSize, SyncTransfers, 3500) != 0 ||
        bladerf_enable_module(handle, Channel, true) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to start the stream.");
        return Result::ERROR;
    }
    streaming = true;
    ++epoch;
    queueGeneration = nextGeneration();
    return resynchronize(guard());
}

void BladeRfStream::close() {
    if (streaming) {
        bladerf_enable_module(handle, Channel, false);
    }
    streaming = false;
    synchronized = false;
}

Result BladeRfStream::resynchronize(const U64 settle) {
    synchronized = false;
    bladerf_timestamp now = 0;
    if (bladerf_get_timestamp(handle, BLADERF_RX, &now) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to read the hardware timestamp.");
        return Result::ERROR;
    }
    if (settle > std::numeric_limits<U64>::max() - now) {
        JST_ERROR("[SDR_BLADERF] Settling guard exceeds the hardware timestamp range.");
        return Result::ERROR;
    }
    timestamps.retune(now + settle);
    synchronized = true;
    return Result::SUCCESS;
}

Device::Capture BladeRfStream::read(std::span<CF32> samples,
                                              const std::chrono::milliseconds timeout) {
    const auto finish = [&](Device::Capture capture) {
        capture.generation = queueGeneration;
        capture.frequency = appliedSettings.frequency;
        capture.sampleRate = appliedSettings.sampleRate;
        capture.epoch = epoch;
        return capture;
    };
    const auto receive = [&](I16* data, const U64 count, bladerf_metadata& meta,
                             const std::chrono::milliseconds wait) {
        meta.flags = BLADERF_META_FLAG_RX_NOW;
        const int status = bladerf_sync_rx(handle, data, static_cast<unsigned>(count),
                                           &meta, static_cast<unsigned>(wait.count()));
        if (status == 0 && (meta.status & BLADERF_META_STATUS_OVERRUN) != 0 &&
            meta.actual_count >= count) {
            const U64 latest = std::numeric_limits<U64>::max() - MessageSamples -
                               meta.actual_count;
            timestamps.lose(meta.timestamp > latest
                ? std::numeric_limits<U64>::max()
                : meta.timestamp + meta.actual_count + MessageSamples);
        }
        return status;
    };

    if (!streaming) {
        return finish({.status = Status::Failure, .error = "stream is not running"});
    }
    if (!synchronized && resynchronize(guard()) != Result::SUCCESS) {
        return finish({.status = Status::Failure,
                       .error = "failed to read the hardware timestamp"});
    }

    raw.resize(samples.size() * 2);
    const auto deadline = Clock::now() + timeout;
    U64 limit = DrainSamples;
    while (const U64 pending = timestamps.pending()) {
        bladerf_metadata meta = {};
        const U64 count = std::min(pending, limit);
        if (const int status = receive(drain.data(), count, meta, Remaining(deadline));
            status != 0) {
            return finish(Failed(status));
        }
        if (meta.actual_count == 0) {
            return finish({.status = Status::Timeout});
        }
        const auto decision = timestamps.accept(meta.timestamp, meta.actual_count);
        if (decision.status == Status::Failure || decision.status == Status::Discontinuity) {
            return finish({.status = decision.status, .lost = decision.lost,
                           .error = decision.error});
        }
        if (Clock::now() >= deadline) {
            return finish({.status = Status::Timeout});
        }
        limit = meta.actual_count < count ? std::min<U64>(samples.size(), DrainSamples)
                                          : DrainSamples;
    }

    bladerf_metadata meta = {};
    if (const int status = receive(raw.data(), samples.size(), meta, Remaining(deadline));
        status != 0) {
        return finish(Failed(status));
    }
    const U64 count = std::min<U64>(meta.actual_count, samples.size());
    if (count == 0) {
        return finish({.status = Status::Timeout});
    }

    const auto decision = timestamps.accept(meta.timestamp, count);
    Device::Capture capture{.status = decision.status, .lost = decision.lost,
                            .error = decision.error};
    if (decision.status != Status::Samples) {
        return finish(capture);
    }
    ConvertI16(raw.data() + decision.skip * 2, samples.data(), count - decision.skip, Q11Scale);
    capture.count = count - decision.skip;
    capture.timestamp = meta.timestamp + decision.skip;
    return finish(capture);
}

Result BladeRfStream::setAutomaticGain(const bool automaticGain) {
    JST_CHECK(applyGainMode(automaticGain, requestedSettings.gain));
    requestedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result BladeRfStream::setGain(const F64 gain) {
    JST_CHECK(applyGain(gain, false));
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 BladeRfStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Stream::resolveGain(gain, range));
}

Result BladeRfStream::applyGainMode(const bool requested, const F64 gain) {
    const bool automatic = requested && deviceCapabilities.automaticGain;
    if (requested && !automatic) {
        JST_WARN("[SDR_BLADERF] Automatic gain is not supported by this bladeRF. "
                 "Using the manual gain.");
    }
    if (bladerf_set_gain_mode(handle, Channel,
                              automatic ? BLADERF_GAIN_AUTOMATIC : BLADERF_GAIN_MGC) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to set the gain mode.");
        return Result::ERROR;
    }
    appliedSettings.automaticGain = automatic;
    return applyGain(gain, true);
}

Result BladeRfStream::applyGain(const F64 gain, const bool force) {
    const bladerf_range* range = nullptr;
    if (bladerf_get_gain_range(handle, Channel, &range) != 0 || !range) {
        JST_ERROR("[SDR_BLADERF] Failed to query the gain range.");
        appliedSettings.gainRange.reset();
        return Result::ERROR;
    }
    appliedSettings.gainRange = Device::Range{static_cast<F64>(range->min) * range->scale,
                                             static_cast<F64>(range->max) * range->scale};
    const F64 target = resolveGain(gain, appliedSettings.gainRange);
    const bool unchanged = !force && target == appliedSettings.gain;
    if (!appliedSettings.automaticGain && !unchanged &&
        bladerf_set_gain(handle, Channel, static_cast<bladerf_gain>(target)) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to set the gain to {:.0f} dB.", target);
        appliedSettings.gain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedSettings.gain = target;
    return Result::SUCCESS;
}

}  // namespace

Result BladeRf::Stream(const Device::Info& info, std::unique_ptr<Radio::Stream>& stream) {
    auto created = std::make_unique<BladeRfStream>();
    JST_CHECK(created->connect(info));
    stream = std::move(created);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
