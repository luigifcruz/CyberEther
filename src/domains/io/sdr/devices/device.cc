#include "device.hh"

#include "../tools/samples.hh"

#include <algorithm>
#include <cmath>
#include <limits>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

bool Device::Range::Contains(const std::vector<Range>& ranges, const F64 value) {
    return std::isfinite(value) &&
           std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) {
               return value >= range.minimum && value <= range.maximum;
           });
}

std::vector<F64> Device::Capabilities::fixedSampleRates() const {
    std::vector<F64> rates;
    for (const auto& range : sampleRateRanges) {
        if (range.minimum != range.maximum) {
            return {};
        }
        rates.push_back(range.minimum);
    }
    std::sort(rates.begin(), rates.end());
    rates.erase(std::unique(rates.begin(), rates.end()), rates.end());
    return rates;
}

const Device::Info& Device::info() const {
    return deviceInfo;
}

const Device::Capabilities& Device::capabilities() const {
    return deviceCapabilities;
}

const Device::Applied& Device::applied() const {
    return appliedSettings;
}

const Device::Settings& Device::settings() const {
    return requestedSettings;
}

U64 Device::generation() const {
    return currentGeneration.load();
}

Device::Statistics Device::statistics() const {
    std::lock_guard lock(statisticsMutex);
    return deviceStatistics;
}

U64 Device::nextGeneration() {
    return ++currentGeneration;
}

U64 Device::guardSamples(const Settings& settings, const F64 sampleRate) const {
    const F64 seconds = settings.settlingTime >= 0.0
        ? settings.settlingTime
        : defaultSettlingTime;
    return GuardSamples(seconds, sampleRate);
}

Result Device::restore(const Settings& previous, const Result failure) {
    stop();
    if (start(previous) == Result::SUCCESS) {
        JST_WARN("[SDR] Restored the previous settings of {} after a failed change.",
                 deviceInfo.label);
        return failure == Result::SUCCESS ? Result::ERROR : failure;
    }
    stop();
    JST_ERROR("[SDR] Failed to restore {} after a failed change. The device is stopped.",
              deviceInfo.label);
    return Result::ERROR;
}

Result Device::apply(const Settings& settings) {
    const auto previous = requestedSettings;
    stop();
    const auto result = start(settings);
    if (result != Result::SUCCESS) {
        return restore(previous, result);
    }
    return Result::SUCCESS;
}

Result Device::tune(const F64 frequency) {
    auto settings = requestedSettings;
    settings.frequency = frequency;
    return apply(settings);
}

Result Device::update(const Settings& settings) {
    const auto previous = requestedSettings;
    if (settings.sampleRate != previous.sampleRate || settings.antenna != previous.antenna ||
        settings.biasTee != previous.biasTee || settings.lowLatency != previous.lowLatency) {
        return apply(settings);
    }

    const auto result = [&] {
        if (settings.frequency != previous.frequency) {
            JST_CHECK(tune(settings.frequency));
        }
        if (settings.automaticGain != previous.automaticGain) {
            JST_CHECK(setAutomaticGain(settings.automaticGain));
        }
        if (settings.gain != previous.gain ||
            resolveGain(settings.gain, appliedSettings.gainRange) != appliedSettings.gain) {
            JST_CHECK(setGain(settings.gain));
        }
        if (settings.settlingTime != previous.settlingTime) {
            JST_CHECK(setSettlingTime(settings.settlingTime));
        }
        return Result::SUCCESS;
    }();
    return result == Result::SUCCESS ? result : restore(previous, result);
}

Result Device::setAutomaticGain(const bool automaticGain) {
    auto settings = requestedSettings;
    settings.automaticGain = automaticGain;
    return apply(settings);
}

Result Device::setGain(const F64 gain) {
    auto settings = requestedSettings;
    settings.gain = gain;
    return apply(settings);
}

F64 Device::resolveGain(const F64 gain, const std::optional<Range>& range) const {
    return range ? std::clamp(gain, range->minimum, range->maximum) : gain;
}

Result Device::setSettlingTime(const F64 settlingTime) {
    auto settings = requestedSettings;
    settings.settlingTime = settlingTime;
    const U64 guard = guardSamples(settings, appliedSettings.sampleRate);
    if (guard == std::numeric_limits<U64>::max()) {
        JST_ERROR("[SDR] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    requestedSettings.settlingTime = settlingTime;
    guardLength = guard;
    return Result::SUCCESS;
}

Result Device::restart(const std::function<Result()>& change) {
    const auto previous = requestedSettings;
    const auto applied = appliedSettings;
    stop();
    const auto result = change();
    if (result == Result::SUCCESS) {
        return result;
    }
    const auto restored = restore(previous, result);
    requestedSettings = previous;
    appliedSettings = applied;
    return restored;
}

}  // namespace Jetstream::Radio
