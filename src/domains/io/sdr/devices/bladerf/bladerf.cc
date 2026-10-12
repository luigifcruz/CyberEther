#include "bladerf.hh"

#include "quick_tune_budget.hh"
#include "../../tools/samples.hh"
#include "../../tools/timestamp_filter.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

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

constexpr F64 SettlingTime = 1.0e-3;
constexpr U64 MaxProfiles = 256;
constexpr unsigned SyncBuffers = 16;
constexpr unsigned SyncBufferSize = 8192;
constexpr unsigned SyncTransfers = 8;
constexpr F32 Q11Scale = 2048.0f;
const bladerf_channel Channel = BLADERF_CHANNEL_RX(0);

using Clock = std::chrono::steady_clock;

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
            return {.status = Device::Status::Timeout};
        case BLADERF_ERR_NODEV:
            return {.status = Device::Status::Disconnected, .error = "device disconnected"};
        default:
            return {.status = Device::Status::Failure, .error = bladerf_strerror(status)};
    }
}

}  // namespace

struct BladeRfDevice::Impl {
    explicit Impl(BladeRfDevice& owner) : device(owner) {}

    Result open(const Info& info);
    Result retune(F64 frequency, bool settle = true);
    Result applyGainMode(bool requested, F64 gain);
    Result applyGain(F64 gain, bool force);
    Result resynchronize(U64 guard);

    BladeRfDevice& device;
    bladerf* handle = nullptr;
    std::string board;
    bool biasTeeEnabled = false;
    bool streaming = false;
    U64 epoch = 0;
    bool synchronized = false;
    TimestampFilter timestamps;
    std::vector<I16> raw;

    bool quickTune = false;
    QuickTuneBudget budget;
    std::vector<bladerf_quick_tune> profiles;
    std::vector<F64> actualFrequencies;
};

BladeRfDevice::BladeRfDevice() : pimpl(std::make_unique<Impl>(*this)) {}

BladeRfDevice::~BladeRfDevice() {
    stop();
    if (pimpl->handle) {
        if (pimpl->biasTeeEnabled) {
            bladerf_set_bias_tee(pimpl->handle, Channel, false);
        }
        bladerf_close(pimpl->handle);
    }
}

std::vector<Device::Info> BladeRfDevice::Enumerate() {
    std::vector<Info> devices;
    bladerf_devinfo* list = nullptr;
    const int count = bladerf_get_device_list(&list);
    for (int i = 0; i < count; ++i) {
        const std::string serial = list[i].serial;
        devices.push_back({
            .serial = serial,
            .label = jst::fmt::format("bladeRF ({})", serial.substr(0, 8)),
        });
    }
    if (list) {
        bladerf_free_device_list(list);
    }
    return devices;
}

Result BladeRfDevice::Create(const Info& info, std::unique_ptr<Device>& device) {
    auto bladerf = std::make_unique<BladeRfDevice>();
    JST_CHECK(bladerf->pimpl->open(info));
    device = std::move(bladerf);
    return Result::SUCCESS;
}

Result BladeRfDevice::Impl::open(const Info& info) {
    bladerf_devinfo devinfo;
    bladerf_init_devinfo(&devinfo);
    std::strncpy(devinfo.serial, info.serial.c_str(), BLADERF_SERIAL_LENGTH - 1);
    if (const int status = bladerf_open_with_devinfo(&handle, &devinfo); status != 0) {
        handle = nullptr;
        JST_ERROR("[SDR_BLADERF] Failed to open device '{}': {}", info.label,
                  bladerf_strerror(status));
        return Result::ERROR;
    }
    device.deviceInfo = info;
    board = bladerf_get_board_name(handle);

    const bladerf_range* frequency = nullptr;
    const bladerf_range* rate = nullptr;
    if (bladerf_get_frequency_range(handle, Channel, &frequency) != 0 ||
        bladerf_get_sample_rate_range(handle, Channel, &rate) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to query device ranges.");
        return Result::ERROR;
    }

    const bool second = board == "bladerf2";
    budget = QuickTuneBudget(second ? std::optional<U64>(MaxProfiles) : std::nullopt);
    auto& caps = device.deviceCapabilities;
    caps.frequencyRanges = ToRanges(frequency);
    caps.sampleRateRanges = ToRanges(rate);
    caps.automaticGain = second;
    caps.gainFollowsFrequency = second;
    caps.biasTee = second;
    caps.sweepSampleRate = second ? 30.72e6 : 20.0e6;
    caps.usableBandwidth = 0.8;
    caps.dcExclusion = 20.0e3;
    device.defaultSettlingTime = SettlingTime;
    return Result::SUCCESS;
}

Result BladeRfDevice::start(const Settings& settings) {
    auto* handle = pimpl->handle;
    if (!Range::Contains(deviceCapabilities.sampleRateRanges, settings.sampleRate)) {
        JST_ERROR("[SDR_BLADERF] Sample rate ({:.3f} MHz) not supported.",
                  settings.sampleRate / 1e6);
        return Result::ERROR;
    }
    if (!Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
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

    JST_CHECK(pimpl->applyGainMode(settings.automaticGain, settings.gain));

    if (deviceCapabilities.biasTee && (settings.biasTee || pimpl->biasTeeEnabled)) {
        if (bladerf_set_bias_tee(handle, Channel, settings.biasTee) != 0) {
            JST_ERROR("[SDR_BLADERF] Failed to set Bias-T.");
            return Result::ERROR;
        }
        pimpl->biasTeeEnabled = settings.biasTee;
    }

    if (guardSamples(settings, actualRate) == std::numeric_limits<U64>::max()) {
        JST_ERROR("[SDR_BLADERF] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }

    if (!pimpl->streaming) {
        if (bladerf_sync_config(handle, BLADERF_RX_X1, BLADERF_FORMAT_SC16_Q11_META,
                                SyncBuffers, SyncBufferSize, SyncTransfers, 3500) != 0 ||
            bladerf_enable_module(handle, Channel, true) != 0) {
            JST_ERROR("[SDR_BLADERF] Failed to start the stream.");
            return Result::ERROR;
        }
        pimpl->streaming = true;
        ++pimpl->epoch;
    }

    requestedSettings = settings;
    appliedSettings.sampleRate = actualRate;
    appliedSettings.antenna.clear();
    appliedSettings.biasTee = settings.biasTee && deviceCapabilities.biasTee;
    return pimpl->retune(settings.frequency);
}

void BladeRfDevice::stop() {
    if (pimpl->streaming) {
        bladerf_enable_module(pimpl->handle, Channel, false);
    }
    pimpl->streaming = false;
    pimpl->synchronized = false;
}

Result BladeRfDevice::apply(const Settings& settings) {
    const auto previous = requestedSettings;
    const auto result = start(settings);
    if (result != Result::SUCCESS) {
        return restore(previous, result);
    }
    return Result::SUCCESS;
}

Result BladeRfDevice::setAutomaticGain(const bool automaticGain) {
    if (pimpl->applyGainMode(automaticGain, requestedSettings.gain) != Result::SUCCESS) {
        return Device::setAutomaticGain(automaticGain);
    }
    requestedSettings.automaticGain = automaticGain;
    return Result::SUCCESS;
}

Result BladeRfDevice::setGain(const F64 gain) {
    if (pimpl->applyGain(gain, false) != Result::SUCCESS) {
        return Device::setGain(gain);
    }
    requestedSettings.gain = gain;
    return Result::SUCCESS;
}

F64 BladeRfDevice::resolveGain(const F64 gain, const std::optional<Range>& range) const {
    return std::round(Device::resolveGain(gain, range));
}

Result BladeRfDevice::Impl::applyGainMode(const bool requested, const F64 gain) {
    const bool automatic = requested && device.deviceCapabilities.automaticGain;
    if (requested && !automatic) {
        JST_WARN("[SDR_BLADERF] Automatic gain is not supported by this bladeRF. "
                 "Using the manual gain.");
    }
    if (bladerf_set_gain_mode(handle, Channel,
                              automatic ? BLADERF_GAIN_AUTOMATIC : BLADERF_GAIN_MGC) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to set the gain mode.");
        return Result::ERROR;
    }
    device.appliedSettings.automaticGain = automatic;
    return applyGain(gain, true);
}

Result BladeRfDevice::Impl::applyGain(const F64 gain, const bool force) {
    const bladerf_range* range = nullptr;
    if (bladerf_get_gain_range(handle, Channel, &range) == 0 && range) {
        device.appliedSettings.gainRange = Range{static_cast<F64>(range->min) * range->scale,
                                                 static_cast<F64>(range->max) * range->scale};
    }
    const F64 target = device.resolveGain(gain, device.appliedSettings.gainRange);
    const bool unchanged = !force && target == device.appliedSettings.gain;
    if (!device.appliedSettings.automaticGain && !unchanged &&
        bladerf_set_gain(handle, Channel, static_cast<bladerf_gain>(target)) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to set the gain to {:.0f} dB.", target);
        return Result::ERROR;
    }
    device.appliedSettings.gain = target;
    return Result::SUCCESS;
}

Result BladeRfDevice::tune(const F64 frequency) {
    if (!Range::Contains(deviceCapabilities.frequencyRanges, frequency)) {
        JST_ERROR("[SDR_BLADERF] Frequency ({:.3f} MHz) not supported.", frequency / 1e6);
        return Result::ERROR;
    }
    const auto previous = requestedSettings.frequency;
    if (pimpl->retune(frequency, false) != Result::SUCCESS) {
        pimpl->retune(previous, false);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Device::Capture BladeRfDevice::read(std::span<CF32> samples, const std::chrono::milliseconds timeout) {
    auto& timestamps = pimpl->timestamps;
    auto& raw = pimpl->raw;
    const auto finish = [&](Capture capture) {
        capture.generation = generation();
        capture.frequency = appliedSettings.frequency;
        capture.sampleRate = appliedSettings.sampleRate;
        capture.epoch = pimpl->epoch;
        return capture;
    };
    const auto receive = [&](const U64 count, bladerf_metadata& meta,
                             const std::chrono::milliseconds wait) {
        meta.flags = BLADERF_META_FLAG_RX_NOW;
        return bladerf_sync_rx(pimpl->handle, raw.data(), static_cast<unsigned>(count), &meta,
                               static_cast<unsigned>(wait.count()));
    };

    if (!pimpl->streaming) {
        return finish({.status = Status::Failure, .error = "stream is not running"});
    }
    if (!pimpl->synchronized && pimpl->resynchronize(0) != Result::SUCCESS) {
        return finish({.status = Status::Failure,
                       .error = "failed to read the hardware timestamp"});
    }

    raw.resize(samples.size() * 2);
    const auto deadline = Clock::now() + timeout;
    while (const U64 pending = timestamps.pending()) {
        bladerf_metadata meta = {};
        if (const int status = receive(std::min<U64>(pending, samples.size()), meta,
                                       Remaining(deadline)); status != 0) {
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
    }

    bladerf_metadata meta = {};
    if (const int status = receive(samples.size(), meta, Remaining(deadline)); status != 0) {
        return finish(Failed(status));
    }
    const U64 count = std::min<U64>(meta.actual_count, samples.size());
    if (count == 0) {
        return finish({.status = Status::Timeout});
    }

    const auto decision = timestamps.accept(meta.timestamp, count);
    Capture capture{.status = decision.status, .lost = decision.lost, .error = decision.error};
    if (decision.status != Status::Samples) {
        return finish(capture);
    }
    ConvertI16(raw.data() + decision.skip * 2, samples.data(), count - decision.skip, Q11Scale);
    capture.count = count - decision.skip;
    capture.timestamp = meta.timestamp + decision.skip;
    return finish(capture);
}

Result BladeRfDevice::prepareSweep(const SweepRequest& request) {
    auto* handle = pimpl->handle;
    auto& budget = pimpl->budget;
    pimpl->quickTune = false;
    JST_CHECK(Device::prepareSweep(request));

    switch (budget.plan(request.frequencies)) {
        case QuickTuneBudget::Decision::Reuse:
            pimpl->quickTune = true;
            return Result::SUCCESS;
        case QuickTuneBudget::Decision::Fallback:
            JST_WARN("[SDR_BLADERF] The sweep needs {} quick-tune profiles but only {} remain "
                     "on this handle. Using normal tuning.", request.frequencies.size(),
                     budget.remaining().value_or(0));
            return Result::SUCCESS;
        case QuickTuneBudget::Decision::Build:
            break;
    }

    pimpl->profiles.assign(request.frequencies.size(), {});
    pimpl->actualFrequencies.clear();
    for (U64 i = 0; i < request.frequencies.size(); ++i) {
        const auto frequency = static_cast<bladerf_frequency>(std::llround(request.frequencies[i]));
        const bool tuned = bladerf_set_frequency(handle, Channel, frequency) == 0;
        if (!tuned || bladerf_get_quick_tune(handle, Channel, &pimpl->profiles[i]) != 0) {
            budget.failed(tuned ? i + 1 : i);
            JST_WARN("[SDR_BLADERF] Failed to create quick-tune profile {}. Using normal "
                     "tuning.", i);
            return tune(requestedSettings.frequency);
        }
        bladerf_frequency actual = frequency;
        bladerf_get_frequency(handle, Channel, &actual);
        pimpl->actualFrequencies.push_back(static_cast<F64>(actual));
    }
    budget.built(request.frequencies);
    pimpl->quickTune = true;
    return tune(requestedSettings.frequency);
}

Result BladeRfDevice::tuneHop(const U64 hop) {
    if (!pimpl->quickTune) {
        return pimpl->retune(sweepRequest.frequencies[hop]);
    }
    const int status = bladerf_schedule_retune(pimpl->handle, Channel, BLADERF_RETUNE_NOW, 0,
                                               &pimpl->profiles[hop]);
    if (status != 0) {
        JST_WARN("[SDR_BLADERF] Quick tune to hop {} failed ({}). Using normal tuning.", hop,
                 bladerf_strerror(status));
        pimpl->quickTune = false;
        return pimpl->retune(sweepRequest.frequencies[hop]);
    }
    requestedSettings.frequency = sweepRequest.frequencies[hop];
    appliedSettings.frequency = pimpl->actualFrequencies[hop];
    nextGeneration();
    return pimpl->resynchronize(guardLength);
}

Result BladeRfDevice::Impl::retune(const F64 frequency, const bool settle) {
    const auto target = static_cast<bladerf_frequency>(std::llround(frequency));
    if (bladerf_set_frequency(handle, Channel, target) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to tune to {:.3f} MHz.", frequency / 1e6);
        return Result::ERROR;
    }
    bladerf_frequency actual = target;
    bladerf_get_frequency(handle, Channel, &actual);
    device.requestedSettings.frequency = frequency;
    device.appliedSettings.frequency = static_cast<F64>(actual);
    JST_CHECK(applyGain(device.requestedSettings.gain, false));
    if (!settle) {
        return Result::SUCCESS;
    }
    device.guardLength = device.guardSamples(device.requestedSettings,
                                             device.appliedSettings.sampleRate);
    device.nextGeneration();
    return resynchronize(device.guardLength);
}

Result BladeRfDevice::Impl::resynchronize(const U64 guard) {
    synchronized = false;
    bladerf_timestamp now = 0;
    if (bladerf_get_timestamp(handle, BLADERF_RX, &now) != 0) {
        JST_ERROR("[SDR_BLADERF] Failed to read the hardware timestamp.");
        return Result::ERROR;
    }
    if (guard > std::numeric_limits<U64>::max() - now) {
        JST_ERROR("[SDR_BLADERF] Settling guard exceeds the hardware timestamp range.");
        return Result::ERROR;
    }
    timestamps.retune(now + guard);
    synchronized = true;
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
