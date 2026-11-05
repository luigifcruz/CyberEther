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

Device::Capture Stream::readChannels(std::span<std::span<CF32>> samples,
                                    const std::chrono::milliseconds timeout) {
    if (samples.size() != 1) {
        return {.status = Device::Status::Failure,
                .error = "this radio supports only single-channel reads"};
    }
    return read(samples.front(), timeout);
}

Result Stream::setChannelGain(const U64 channel, bool, F64) {
    JST_ERROR("[SDR] {} does not support gain control on RX{}.", deviceInfo.label,
              channel + 1);
    return Result::ERROR;
}

Result Stream::start(const Settings& settings) {
    JST_CHECK(configure(settings));
    const auto result = open();
    if (result != Result::SUCCESS) {
        close();
    }
    return result;
}

Result Stream::configure(const Settings& settings) {
    if (settings.receiveChannels == 0 ||
        settings.receiveChannels > deviceCapabilities.receiveChannels) {
        JST_ERROR("[SDR] {} does not support {} receive channels.",
                  deviceInfo.label, settings.receiveChannels);
        return Result::ERROR;
    }
    if (!settings.additionalGains.empty() &&
        settings.additionalGains.size() != settings.receiveChannels - 1) {
        JST_ERROR("[SDR] Gain settings must match the receive channel count.");
        return Result::ERROR;
    }
    Settings normalized = settings;
    normalized.additionalGains.resize(settings.receiveChannels - 1);
    close();
    JST_CHECK(apply(normalized));
    if (appliedSettings.additionalGains.size() != normalized.additionalGains.size()) {
        JST_ERROR("[SDR] {} did not apply all receive channel gains.",
                  deviceInfo.label);
        return Result::ERROR;
    }
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
    if (settings.receiveChannels != previous.receiveChannels ||
        settings.sampleRate != previous.sampleRate ||
        settings.antenna != previous.antenna || settings.biasTee != previous.biasTee) {
        return restart(settings, previous);
    }

    if (!settings.additionalGains.empty() &&
        settings.additionalGains.size() != settings.receiveChannels - 1) {
        JST_ERROR("[SDR] Gain settings must match the receive channel count.");
        return Result::ERROR;
    }
    Settings normalized = settings;
    normalized.additionalGains.resize(settings.receiveChannels - 1);
    const auto change = [&](const Settings& target) {
        if (target.frequency != previous.frequency) {
            if (!Device::Range::Contains(deviceCapabilities.frequencyRanges, target.frequency)) {
                JST_ERROR("[SDR] Frequency ({:.3f} MHz) not supported by {}. Use {}.",
                          target.frequency / 1e6, deviceInfo.label,
                          Device::Range::Describe(deviceCapabilities.frequencyRanges));
                return Result::ERROR;
            }
            JST_CHECK(tune(target.frequency));
        }
        if (target.automaticGain != previous.automaticGain) {
            JST_CHECK(setAutomaticGain(target.automaticGain));
        }
        if (target.gain != previous.gain ||
            resolveGain(target.gain, appliedSettings.gainRange) != appliedSettings.gain) {
            JST_CHECK(setGain(target.gain));
        }
        for (U64 i = 0; i < target.additionalGains.size(); ++i) {
            const auto& gain = target.additionalGains[i];
            const auto& old = previous.additionalGains[i];
            const auto& applied = appliedSettings.additionalGains[i];
            if (gain.automatic != old.automatic || gain.gain != old.gain ||
                resolveGain(gain.gain, applied.range) != applied.gain) {
                JST_CHECK(setChannelGain(i + 1, gain.automatic, gain.gain));
            }
        }
        if (target.settlingTime != previous.settlingTime) {
            JST_CHECK(setSettlingTime(target.settlingTime));
        }
        return Result::SUCCESS;
    };
    const auto result = change(normalized);
    return result == Result::SUCCESS ? result : restore(previous, result);
}

Result Stream::restart(const Settings& settings, const Settings& previous) {
    auto result = configure(settings);
    if (result == Result::SUCCESS) {
        result = open();
    }
    return result == Result::SUCCESS ? result : restore(previous, result);
}

Result Stream::restore(const Settings& previous, const Result failure) {
    if (configure(previous) == Result::SUCCESS && open() == Result::SUCCESS) {
        JST_WARN("[SDR] Restored the previous stream settings of {} after a failed change.",
                 deviceInfo.label);
        return failure == Result::SUCCESS ? Result::ERROR : failure;
    }
    close();
    JST_ERROR("[SDR] Failed to restore the previous stream settings of {} after a failed change. "
              "The stream is stopped.", deviceInfo.label);
    return Result::FATAL;
}

}  // namespace Jetstream::Radio
