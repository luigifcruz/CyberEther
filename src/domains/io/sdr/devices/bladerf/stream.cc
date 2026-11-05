#include "base.hh"
#include "tools/ownership.hh"

#include "../stream.hh"
#include "../../tools/samples.hh"
#include "../../tools/timestamp_filter.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <tuple>
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
using Clock = std::chrono::steady_clock;

constexpr F64 SettlingTime = 1.0e-3;
constexpr unsigned SyncBuffers = 16;
constexpr unsigned SyncBufferSize = 8192;
constexpr unsigned SyncTransfers = 8;
constexpr unsigned SyncTimeout = 3500;
constexpr U64 DrainSamples = 65536;
// Conservative upper bound across USB speeds and metadata message versions.
constexpr U64 MessageSamples = 2048;
constexpr F32 Q11Scale = 2048.0f;
constexpr bladerf_channel Channel = BLADERF_CHANNEL_RX(0);

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
    const auto left =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
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
    Device::Capture readChannels(std::span<std::span<CF32>> samples,
                                 std::chrono::milliseconds timeout) override;

 protected:
    Result apply(const Settings& settings) override;
    Result open() override;
    void close() override;
    Result tune(F64 frequency) override;
    Result setAutomaticGain(bool automaticGain) override;
    Result setGain(F64 gain) override;
    Result setChannelGain(U64 channel, bool automaticGain, F64 gain) override;

   private:
    Result setFrequency(F64 frequency);
    Result resynchronize(U64 settle);
    Result applyGainMode(bool requested, F64 gain, U64 index = 0);
    Result applyGain(F64 gain, bool force, U64 index = 0);

    BladeRfOwnership ownership;
    bladerf* handle = nullptr;
    std::string board;
    bool biasTeeEnabled = false;
    bool streaming = false;
    U64 enabledChannels = 0;
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
    JST_CHECK(ownership.acquire(info.serial));
    bladerf_devinfo devinfo;
    bladerf_init_devinfo(&devinfo);
    std::strncpy(devinfo.serial, info.serial.c_str(), BLADERF_SERIAL_LENGTH - 1);
    if (const int status = bladerf_open_with_devinfo(&handle, &devinfo); status != 0) {
        handle = nullptr;
        JST_ERROR("[SDR_BLADERF] Failed to open device '{}' ({}).", info.label,
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
    caps.receiveChannels = bladerf_get_channel_count(handle, BLADERF_RX);
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
    if (settings.receiveChannels > 2) {
        JST_ERROR("[SDR_BLADERF] This backend supports at most two RX channels.");
        return Result::ERROR;
    }
    if (settings.receiveChannels == 2) {
        struct bladerf_version version = {};
        if (bladerf_fpga_version(handle, &version) != 0) {
            JST_ERROR(
                "[SDR_BLADERF] Failed to query the FPGA version for multi-channel RX.");
            return Result::ERROR;
        }
        // Earlier images advance the RX timestamp by 255 for 254 sample pairs,
        // breaking capture continuity (Nuand/bladeRF issue #855).
        if (std::tuple{version.major, version.minor, version.patch} <
            std::tuple{0, 15, 3}) {
            JST_ERROR(
                "[SDR_BLADERF] Multiple RX channels require FPGA v0.15.3 or newer "
                "for correct timestamps; the device has v{}.{}.{}. Load a compatible "
                "FPGA image before enabling multiple channels.",
                version.major, version.minor, version.patch);
            return Result::ERROR;
        }
    }
    if (!Device::Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_BLADERF] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
        JST_ERROR("[SDR_BLADERF] Frequency ({:.3f} MHz) not supported. Use {}.",
                  settings.frequency / 1e6,
                  Device::Range::Describe(deviceCapabilities.frequencyRanges));
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

    appliedSettings.additionalGains.resize(settings.receiveChannels - 1);
    JST_CHECK(applyGainMode(settings.automaticGain, settings.gain));
    for (U64 i = 0; i < settings.additionalGains.size(); ++i) {
        const auto& gain = settings.additionalGains[i];
        JST_CHECK(applyGainMode(gain.automatic, gain.gain, i + 1));
    }

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
    return setFrequency(settings.frequency);
}

Result BladeRfStream::open() {
    try {
        drain.resize(DrainSamples * 2 * requestedSettings.receiveChannels);
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR_BLADERF] Failed to allocate the drain buffer.");
        return Result::ERROR;
    }
    const auto layout =
        requestedSettings.receiveChannels == 2 ? BLADERF_RX_X2 : BLADERF_RX_X1;
    if (bladerf_sync_config(handle, layout, BLADERF_FORMAT_SC16_Q11_META, SyncBuffers,
                            SyncBufferSize, SyncTransfers, SyncTimeout) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to configure reception.");
        return Result::ERROR;
    }
    for (U64 i = 0; i < requestedSettings.receiveChannels; ++i) {
        // Also disable the attempted channel if enable fails partway through.
        enabledChannels = i + 1;
        if (bladerf_enable_module(handle, BLADERF_CHANNEL_RX(i), true) != 0) {
            JST_ERROR("[SDR_BLADERF] Failed to enable RX{}.", i + 1);
            close();
            return Result::ERROR;
        }
    }
    streaming = true;
    ++epoch;
    queueGeneration = nextGeneration();
    return resynchronize(guard());
}

void BladeRfStream::close() {
    while (enabledChannels > 0) {
        bladerf_enable_module(handle, BLADERF_CHANNEL_RX(--enabledChannels), false);
    }
    streaming = false;
    synchronized = false;
}

Result BladeRfStream::setFrequency(const F64 frequency) {
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
    JST_CHECK(applyGain(requestedSettings.gain, changedBand));
    for (U64 i = 0; i < requestedSettings.additionalGains.size(); ++i) {
        JST_CHECK(
            applyGain(requestedSettings.additionalGains[i].gain, changedBand, i + 1));
    }
    return Result::SUCCESS;
}

Result BladeRfStream::tune(const F64 frequency) {
    JST_CHECK(setFrequency(frequency));
    if (!streaming) {
        return Result::SUCCESS;
    }
    queueGeneration = nextGeneration();
    return resynchronize(guard());
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
    std::array destinations{samples};
    return readChannels(destinations, timeout);
}

Device::Capture BladeRfStream::readChannels(std::span<std::span<CF32>> samples,
                                           const std::chrono::milliseconds timeout) {
    const U64 channels = requestedSettings.receiveChannels;
    if (samples.size() != channels || samples.empty() || samples.front().empty() ||
        samples.front().size() > std::numeric_limits<unsigned>::max() / channels ||
        std::any_of(samples.begin(), samples.end(), [&](const auto channel) {
            return channel.size() != samples.front().size();
        })) {
        return {.status = Status::Failure, .error = "invalid receive channel buffers"};
    }
    const U64 size = samples.front().size();
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
        const int status =
            bladerf_sync_rx(handle, data, static_cast<unsigned>(count * channels),
                            &meta, static_cast<unsigned>(wait.count()));
        if (status == 0 && meta.actual_count % channels != 0) {
            synchronized = false;
            return BLADERF_ERR_UNEXPECTED;
        }
        meta.actual_count /= channels;
        if (status == 0 && (meta.status & BLADERF_META_STATUS_OVERRUN) != 0 &&
            meta.actual_count >= count) {
            const U64 latest = std::numeric_limits<U64>::max() -
                               MessageSamples / channels - meta.actual_count;
            timestamps.lose(meta.timestamp > latest
                                ? std::numeric_limits<U64>::max()
                                : meta.timestamp + meta.actual_count +
                                      MessageSamples / channels);
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

    raw.resize(size * 2 * channels);
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
        limit = meta.actual_count < count ? std::min<U64>(size, DrainSamples)
                                          : DrainSamples;
    }

    bladerf_metadata meta = {};
    if (const int status = receive(raw.data(), size, meta, Remaining(deadline));
        status != 0) {
        return finish(Failed(status));
    }
    const U64 count = std::min<U64>(meta.actual_count, size);
    if (count == 0) {
        return finish({.status = Status::Timeout});
    }

    const auto decision = timestamps.accept(meta.timestamp, count);
    Device::Capture capture{.status = decision.status, .lost = decision.lost,
                            .error = decision.error};
    if (decision.status != Status::Samples) {
        return finish(capture);
    }
    const auto* accepted = raw.data() + decision.skip * 2 * channels;
    if (channels == 2) {
        ConvertI16Channels(accepted, samples[0].data(), samples[1].data(),
                           count - decision.skip, Q11Scale);
    } else {
        ConvertI16(accepted, samples[0].data(), count - decision.skip, Q11Scale);
    }
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

Result BladeRfStream::setChannelGain(const U64 channel, const bool automaticGain,
                                     const F64 gain) {
    JST_CHECK(applyGainMode(automaticGain, gain, channel));
    requestedSettings.additionalGains[channel - 1] = {automaticGain, gain};
    return Result::SUCCESS;
}

F64 BladeRfStream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return std::round(Stream::resolveGain(gain, range));
}

Result BladeRfStream::applyGainMode(const bool requested, const F64 gain,
                                    const U64 index) {
    const bool automatic = requested && deviceCapabilities.automaticGain;
    if (requested && !automatic) {
        JST_WARN("[SDR_BLADERF] Automatic gain is not supported by this bladeRF. "
                 "Using the manual gain.");
    }
    if (bladerf_set_gain_mode(handle, BLADERF_CHANNEL_RX(index),
                              automatic ? BLADERF_GAIN_AUTOMATIC : BLADERF_GAIN_MGC) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to set the gain mode.");
        return Result::ERROR;
    }
    (index == 0 ? appliedSettings.automaticGain
                : appliedSettings.additionalGains[index - 1].automatic) = automatic;
    return applyGain(gain, true, index);
}

Result BladeRfStream::applyGain(const F64 gain, const bool force, const U64 index) {
    auto& gainRange = index == 0 ? appliedSettings.gainRange
                                 : appliedSettings.additionalGains[index - 1].range;
    auto& appliedGain = index == 0 ? appliedSettings.gain
                                   : appliedSettings.additionalGains[index - 1].gain;
    const bool automatic = index == 0
                               ? appliedSettings.automaticGain
                               : appliedSettings.additionalGains[index - 1].automatic;
    const bladerf_range* range = nullptr;
    if (bladerf_get_gain_range(handle, BLADERF_CHANNEL_RX(index), &range) != 0 ||
        !range) {
        JST_ERROR("[SDR_BLADERF] Failed to query the gain range.");
        gainRange.reset();
        return Result::ERROR;
    }
    gainRange = Device::Range{static_cast<F64>(range->min) * range->scale,
                                             static_cast<F64>(range->max) * range->scale};
    const F64 target = resolveGain(gain, gainRange);
    const bool unchanged = !force && target == appliedGain;
    if (!automatic && !unchanged &&
        bladerf_set_gain(handle, BLADERF_CHANNEL_RX(index),
                         static_cast<bladerf_gain>(target)) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to set the gain to {:.0f} dB.", target);
        appliedGain = std::numeric_limits<F64>::quiet_NaN();
        return Result::ERROR;
    }
    appliedGain = target;
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
