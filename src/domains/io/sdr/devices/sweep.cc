#include "sweep.hh"

#include "../tools/samples.hh"

#include <algorithm>
#include <chrono>
#include <limits>
#include <new>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr U64 MaxHopDiscontinuities = 8;
constexpr U64 MaxReadChunk = 16384;
constexpr U64 MaxSweepFailures = 3;

}  // namespace

const Device::Info& Sweep::info() const {
    return deviceInfo;
}

const Device::Capabilities& Sweep::capabilities() const {
    return deviceCapabilities;
}

const Sweep::Settings& Sweep::settings() const {
    return requestedSettings;
}

const Sweep::Applied& Sweep::applied() const {
    return appliedSettings;
}

const Sweep::Request& Sweep::request() const {
    return sweepRequest;
}

Sweep::Statistics Sweep::statistics() const {
    return {.failedHops = failedHops.load()};
}

F64 Sweep::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return range ? std::clamp(gain, range->minimum, range->maximum) : gain;
}

U64 Sweep::guard() const {
    const F64 seconds = requestedSettings.settlingTime >= 0.0
        ? requestedSettings.settlingTime
        : defaultSettlingTime;
    return GuardSamples(seconds, appliedSettings.sampleRate);
}

Result Sweep::configure(const Settings& settings) {
    close();
    JST_CHECK(apply(settings));
    if (guard() == std::numeric_limits<U64>::max()) {
        JST_ERROR("[SDR] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result Sweep::start(const Request& request) {
    if (request.frequencies.empty() || request.captureLength == 0 || request.dwell == 0) {
        JST_ERROR("[SDR] Sweep request on {} has no hops or no capture length.",
                  deviceInfo.label);
        return Result::ERROR;
    }
    if (request.firmware && !deviceCapabilities.firmwareSweep) {
        JST_ERROR("[SDR] Firmware sweeps are not supported by {}.", deviceInfo.label);
        return Result::ERROR;
    }
    close();
    sweepRequest = request;
    return open();
}

Result Sweep::setSettlingTime(const F64 settlingTime) {
    const auto previous = requestedSettings.settlingTime;
    requestedSettings.settlingTime = settlingTime;
    if (guard() == std::numeric_limits<U64>::max()) {
        requestedSettings.settlingTime = previous;
        JST_ERROR("[SDR] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result Sweep::update(const Settings& settings) {
    const auto previous = requestedSettings;
    const auto change = [&](const Settings& target) {
        if (target.automaticGain != requestedSettings.automaticGain) {
            JST_CHECK(setAutomaticGain(target.automaticGain));
        }
        if (target.gain != requestedSettings.gain ||
            resolveGain(target.gain, appliedSettings.gainRange) != appliedSettings.gain) {
            JST_CHECK(setGain(target.gain));
        }
        if (target.settlingTime != requestedSettings.settlingTime) {
            JST_CHECK(setSettlingTime(target.settlingTime));
        }
        return Result::SUCCESS;
    };
    const auto result = change(settings);
    if (result == Result::SUCCESS) {
        return result;
    }
    const auto restored = [&] {
        JST_CHECK(setAutomaticGain(previous.automaticGain));
        JST_CHECK(setGain(previous.gain));
        JST_CHECK(setSettlingTime(previous.settlingTime));
        return Result::SUCCESS;
    }();
    if (restored != Result::SUCCESS) {
        close();
        JST_ERROR("[SDR] Failed to restore the previous sweep settings of {} after a failed change. "
                  "The sweep is stopped.", deviceInfo.label);
        return Result::FATAL;
    }
    JST_WARN("[SDR] Restored the previous sweep settings of {} after a failed change.",
             deviceInfo.label);
    return result;
}

Result Sweep::fail(U64& failures) {
    ++failedHops;
    if (++failures < MaxSweepFailures) {
        return Result::SUCCESS;
    }
    JST_ERROR("[SDR] Stopping the sweep on {} after {} failed hops in a row.",
              deviceInfo.label, failures);
    return Result::ERROR;
}

U64 Sweep::Retune::generation() const {
    return currentGeneration.load();
}

U64 Sweep::Retune::nextGeneration() {
    return ++currentGeneration;
}

Result Sweep::Retune::capture(const U64 hop, std::span<CF32> samples, U64& segments,
                                      Capture& capture, const Running& running) {
    if (hop >= sweepRequest.frequencies.size()) {
        JST_ERROR("[SDR] Sweep hop {} is not part of the prepared plan.", hop);
        return Result::ERROR;
    }
    const U64 length = sweepRequest.captureLength;
    const U64 required = length * sweepRequest.dwell;
    if (samples.size() < required || segments > sweepRequest.dwell) {
        JST_ERROR("[SDR] Sweep capture buffer is too small.");
        return Result::ERROR;
    }

    if (!running()) {
        return Result::YIELD;
    }
    JST_CHECK(tune(hop));

    const U64 generation = this->generation();
    const auto timeout = std::chrono::milliseconds(100);
    const F64 sampleRate = appliedSettings.sampleRate;
    U64 filled = segments * length;
    U64 discontinuities = 0;
    auto idleSince = std::chrono::steady_clock::now();
    const F64 idleLimit = std::chrono::duration<F64>(Device::StallTimeout).count() +
        (sampleRate > 0.0 ? static_cast<F64>(guard()) / sampleRate : 0.0);
    while (filled < required) {
        if (!running()) {
            return Result::YIELD;
        }
        const U64 chunk = std::min(required - filled, MaxReadChunk);
        const auto received = read(samples.subspan(filled, chunk), timeout);
        const auto now = std::chrono::steady_clock::now();

        if (received.status == Device::Status::Disconnected ||
            received.status == Device::Status::Failure) {
            JST_ERROR("[SDR] Sweep capture failed on {}: {}", deviceInfo.label,
                      received.error);
            return Result::FATAL;
        }
        if (received.status == Device::Status::Samples && received.generation != generation) {
            filled = segments * length;
        } else if (received.status == Device::Status::Samples && received.count > 0) {
            filled += received.count;
            segments = filled / length;
            idleSince = now;
            continue;
        }
        if (received.status == Device::Status::Discontinuity) {
            filled = segments * length;
            idleSince = now;
            if (++discontinuities > MaxHopDiscontinuities) {
                JST_ERROR("[SDR] Could not capture {} contiguous samples on {}.",
                          length, deviceInfo.label);
                return Result::ERROR;
            }
            continue;
        }

        const std::chrono::duration<F64> idle = now - idleSince;
        if (idle.count() > idleLimit) {
            JST_ERROR("[SDR] Timed out waiting for sweep samples from {}.", deviceInfo.label);
            return Result::TIMEOUT;
        }
    }

    capture.hop = hop;
    capture.frequency = tunedFrequency;
    capture.samples = samples.first(required);
    return Result::SUCCESS;
}

Result Sweep::Retune::run(const Running& running, const Sink& sink) {
    try {
        buffer.assign(sweepRequest.captureLength * sweepRequest.dwell, CF32{});
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR] Failed to allocate the sweep capture buffer.");
        return Result::ERROR;
    }

    U64 hop = 0;
    U64 segments = 0;
    U64 failures = 0;
    while (running()) {
        Capture result;
        const auto status = capture(hop, buffer, segments, result, running);
        if (status == Result::YIELD || !running()) {
            break;
        }
        if (status == Result::FATAL) {
            return Result::ERROR;
        }
        if (status != Result::SUCCESS) {
            JST_CHECK(fail(failures));
            continue;
        }
        failures = 0;
        segments = 0;
        if (!sink(result)) {
            return Result::SUCCESS;
        }
        hop = (hop + 1) % sweepRequest.frequencies.size();
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
