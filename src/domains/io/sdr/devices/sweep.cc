#include "device.hh"

#include <algorithm>
#include <chrono>
#include <new>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

namespace {

constexpr U64 MaxHopDiscontinuities = 8;
constexpr U64 MaxReadChunk = 16384;
constexpr U64 MaxSweepFailures = 3;

}  // namespace

Result Device::prepareSweep(const SweepRequest& request) {
    if (request.firmware) {
        JST_ERROR("[SDR] Firmware sweeps are not supported by {}.", deviceInfo.label);
        return Result::ERROR;
    }
    sweepRequest = request;
    sweepPrepared = true;
    return Result::SUCCESS;
}

Result Device::captureHop(const U64 hop, std::span<CF32> samples, U64& segments,
                          SweepCapture& capture, const SweepRunning& running) {
    if (!sweepPrepared || hop >= sweepRequest.frequencies.size()) {
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
    JST_CHECK(tuneHop(hop));

    const U64 generation = this->generation();
    const auto timeout = std::chrono::milliseconds(100);
    U64 filled = segments * length;
    U64 discontinuities = 0;
    auto idleSince = std::chrono::steady_clock::now();
    const F64 idleLimit = std::chrono::duration<F64>(StallTimeout).count() +
        (appliedSettings.sampleRate > 0.0
            ? static_cast<F64>(guardLength) / appliedSettings.sampleRate
            : 0.0);
    while (filled < required) {
        if (!running()) {
            return Result::YIELD;
        }
        const U64 chunk = std::min(required - filled, MaxReadChunk);
        const auto received = read(samples.subspan(filled, chunk), timeout);
        const auto now = std::chrono::steady_clock::now();

        if (received.status == Status::Disconnected || received.status == Status::Failure) {
            JST_ERROR("[SDR] Sweep capture failed on {}: {}", deviceInfo.label, received.error);
            return Result::FATAL;
        }
        if (received.status == Status::Samples && received.generation != generation) {
            filled = segments * length;
        } else if (received.status == Status::Samples && received.count > 0) {
            filled += received.count;
            segments = filled / length;
            idleSince = now;
            continue;
        }
        if (received.status == Status::Discontinuity) {
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
    capture.frequency = appliedSettings.frequency;
    capture.samples = samples.first(required);
    return Result::SUCCESS;
}

Result Device::fail(U64& failures) {
    {
        std::lock_guard lock(statisticsMutex);
        ++deviceStatistics.failedHops;
    }
    if (++failures < MaxSweepFailures) {
        return Result::SUCCESS;
    }
    JST_ERROR("[SDR] Stopping the sweep on {} after {} failed hops in a row.", deviceInfo.label,
              failures);
    return Result::ERROR;
}

Result Device::tuneHop(const U64 hop) {
    return tune(sweepRequest.frequencies[hop]);
}

Result Device::runSweep(const SweepRunning& running, const SweepSink& sink) {
    if (!sweepPrepared || sweepRequest.firmware) {
        JST_ERROR("[SDR] No sweep is prepared on {}.", deviceInfo.label);
        return Result::ERROR;
    }
    try {
        sweepBuffer.assign(sweepRequest.captureLength * sweepRequest.dwell, CF32{});
    } catch (const std::bad_alloc&) {
        JST_ERROR("[SDR] Failed to allocate the sweep capture buffer.");
        return Result::ERROR;
    }

    U64 hop = 0;
    U64 segments = 0;
    U64 failures = 0;
    while (running()) {
        SweepCapture capture;
        const auto result = captureHop(hop, sweepBuffer, segments, capture, running);
        if (result == Result::YIELD || !running()) {
            break;
        }
        if (result == Result::FATAL) {
            return Result::ERROR;
        }
        if (result != Result::SUCCESS) {
            JST_CHECK(fail(failures));
            continue;
        }
        failures = 0;
        segments = 0;
        if (!sink(capture)) {
            return Result::SUCCESS;
        }
        hop = (hop + 1) % sweepRequest.frequencies.size();
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream::Radio
