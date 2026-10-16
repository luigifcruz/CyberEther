#include "stream.hh"

#include "../tools/samples.hh"

#include <algorithm>
#include <limits>

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

const Device::Info& Stream::info() const {
    return deviceInfo;
}

const Device::Capabilities& Stream::capabilities() const {
    return deviceCapabilities;
}

const Stream::Settings& Stream::settings() const {
    return requestedSettings;
}

const Stream::Applied& Stream::applied() const {
    return appliedSettings;
}

U64 Stream::generation() const {
    return currentGeneration.load();
}

U64 Stream::nextGeneration() {
    return ++currentGeneration;
}

F64 Stream::resolveGain(const F64 gain, const std::optional<Device::Range>& range) const {
    return range ? std::clamp(gain, range->minimum, range->maximum) : gain;
}

U64 Stream::guard() const {
    const F64 seconds = requestedSettings.settlingTime >= 0.0
        ? requestedSettings.settlingTime
        : defaultSettlingTime;
    return GuardSamples(seconds, appliedSettings.sampleRate);
}

Result Stream::start(const Settings& settings) {
    close();
    JST_CHECK(configure(settings));
    return open();
}

Result Stream::configure(const Settings& settings) {
    JST_CHECK(apply(settings));
    if (guard() == std::numeric_limits<U64>::max()) {
        JST_ERROR("[SDR] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result Stream::setSettlingTime(const F64 settlingTime) {
    const auto previous = requestedSettings.settlingTime;
    requestedSettings.settlingTime = settlingTime;
    if (guard() == std::numeric_limits<U64>::max()) {
        requestedSettings.settlingTime = previous;
        JST_ERROR("[SDR] Settling guard exceeds the sample count range.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result Stream::update(const Settings& settings) {
    const auto previous = requestedSettings;
    if (settings.sampleRate != previous.sampleRate || settings.antenna != previous.antenna ||
        settings.biasTee != previous.biasTee) {
        return restart(settings, previous);
    }

    const auto result = [&] {
        if (settings.frequency != previous.frequency) {
            if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, settings.frequency)) {
                JST_ERROR("[SDR] Frequency ({:.3f} MHz) not supported by {}.",
                          settings.frequency / 1e6, deviceInfo.label);
                return Result::ERROR;
            }
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

Result Stream::restart(const Settings& settings, const Settings& previous) {
    close();
    auto result = configure(settings);
    if (result == Result::SUCCESS) {
        result = open();
    }
    return result == Result::SUCCESS ? result : restore(previous, result);
}

Result Stream::restore(const Settings& previous, const Result failure) {
    close();
    if (configure(previous) == Result::SUCCESS && open() == Result::SUCCESS) {
        JST_WARN("[SDR] Restored the previous settings of {} after a failed change.",
                 deviceInfo.label);
        return failure == Result::SUCCESS ? Result::ERROR : failure;
    }
    close();
    JST_ERROR("[SDR] Failed to restore {} after a failed change. The stream is stopped.",
              deviceInfo.label);
    return Result::ERROR;
}

}  // namespace Jetstream::Radio
