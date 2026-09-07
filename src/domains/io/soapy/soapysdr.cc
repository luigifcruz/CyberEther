#include "soapysdr.hh"

#include <algorithm>
#include <exception>

#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Errors.hpp>
#include <SoapySDR/Registry.hpp>

#include <jetstream/logger.hh>

namespace Jetstream::Modules {

SoapyReceiver::~SoapyReceiver() {
    reset();
}

Result SoapyReceiver::open(const SoapySDR::Kwargs& args) {
    if (state != State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Device is already open.");
        return Result::ERROR;
    }

    try {
        const auto& findFuncs = SoapySDR::Registry::listFindFunctions();
        JST_DEBUG("[MODULE_SOAPY] Registered SoapySDR drivers ({}):", findFuncs.size());
        for (const auto& [name, _] : findFuncs) {
            JST_DEBUG("[MODULE_SOAPY]   - {}", name);
        }
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to enumerate drivers.");
        return Result::ERROR;
    }

    try {
        const auto devices = SoapySDR::Device::enumerate(args);
        if (devices.empty()) {
            JST_ERROR("[MODULE_SOAPY] No SoapySDR devices found.");
            return Result::INCOMPLETE;
        }
        device = SoapySDR::Device::make(devices.at(0));
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to open device: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to open device.");
        return Result::ERROR;
    }

    if (device == nullptr) {
        JST_ERROR("[MODULE_SOAPY] Can't open SoapySDR device.");
        return Result::ERROR;
    }

    state = State::Open;
    return queryCapabilities();
}

Result SoapyReceiver::queryCapabilities() {
    try {
        sampleRateRanges = device->getSampleRateRange(SOAPY_SDR_RX, 0);
        frequencyRanges = device->getFrequencyRange(SOAPY_SDR_RX, 0);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to get device ranges: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to get device ranges.");
        return Result::ERROR;
    }

    try {
        const auto settings = device->getSettingInfo();
        biasTeeSupported = std::any_of(settings.begin(), settings.end(),
            [](const auto& setting) { return setting.key == "biastee"; });
    } catch (const std::exception& e) {
        JST_WARN("[MODULE_SOAPY] Failed to query optional device settings: {}", e.what());
    } catch (...) {
        JST_WARN("[MODULE_SOAPY] Failed to query optional device settings.");
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::validateSettings(const F32 sampleRate,
                                       const F32 frequency) const {
    if (!SoapyRangeContains(sampleRateRanges, sampleRate)) {
        JST_ERROR("[MODULE_SOAPY] Sample rate ({:.2f} MHz) not supported.", sampleRate / 1e6);
        return Result::ERROR;
    }

    if (!SoapyRangeContains(frequencyRanges, frequency)) {
        JST_ERROR("[MODULE_SOAPY] Frequency ({:.2f} MHz) not supported.", frequency / 1e6);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setTunerFrequency(const F32 frequency) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set frequency without an active device.");
        return Result::ERROR;
    }

    if (!SoapyRangeContains(frequencyRanges, frequency)) {
        JST_WARN("[MODULE_SOAPY] Frequency ({:.2f} MHz) not supported.", frequency / 1e6);
        return Result::WARNING;
    }

    try {
        device->setFrequency(SOAPY_SDR_RX, 0, frequency);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set frequency: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set frequency.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setSampleRate(const F32 sampleRate) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set sample rate without an active device.");
        return Result::ERROR;
    }

    if (!SoapyRangeContains(sampleRateRanges, sampleRate)) {
        JST_WARN("[MODULE_SOAPY] Sample rate ({:.2f} MHz) not supported.", sampleRate / 1e6);
        return Result::WARNING;
    }

    try {
        device->setSampleRate(SOAPY_SDR_RX, 0, sampleRate);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set sample rate: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set sample rate.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setAutomaticGain(const bool automaticGain) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set gain mode without an active device.");
        return Result::ERROR;
    }

    try {
        device->setGainMode(SOAPY_SDR_RX, 0, automaticGain);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set gain mode: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set gain mode.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::setBiasTee(const bool enabled) {
    if (state == State::Closed) {
        JST_ERROR("[MODULE_SOAPY] Cannot set Bias-T without an active device.");
        return Result::ERROR;
    }

    if (biasTeeSupported) {
        biasTeeNeedsCleanup = true;
        try {
            device->writeSetting("biastee", enabled);
        } catch (const std::exception& e) {
            JST_ERROR("[MODULE_SOAPY] Failed to set Bias-T: {}", e.what());
            return Result::ERROR;
        } catch (...) {
            JST_ERROR("[MODULE_SOAPY] Failed to set Bias-T.");
            return Result::ERROR;
        }
        biasTeeNeedsCleanup = enabled;
    } else if (enabled) {
        JST_WARN("[MODULE_SOAPY] Bias-T is not supported by the selected device.");
    }

    return Result::SUCCESS;
}

Result SoapyReceiver::startStream(const SoapySDR::Kwargs& args) {
    if (state != State::Open) {
        JST_ERROR("[MODULE_SOAPY] Cannot setup stream in the current device state.");
        return Result::ERROR;
    }

    try {
        stream = device->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0}, args);
        if (stream == nullptr) {
            JST_ERROR("[MODULE_SOAPY] Failed to setup stream.");
            return Result::ERROR;
        }
        state = State::StreamReady;

        const int activationResult = device->activateStream(stream, 0, 0, 0);
        if (activationResult != 0) {
            JST_ERROR("[MODULE_SOAPY] Failed to configure device: activateStream returned status {}",
                      activationResult);
            return Result::ERROR;
        }
        state = State::Streaming;
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to configure device: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to configure device.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

SoapyReceiver::ReadResult SoapyReceiver::read(const std::span<CF32> samples) {
    if (state != State::Streaming) {
        return {.error = "Cannot read without an active stream"};
    }
    if (samples.empty()) {
        return {.error = "Receive buffer cannot be empty"};
    }

    void* buffers[] = {samples.data()};
    int flags = 0;
    long long timeNs = 0;
    int result;
    try {
        result = device->readStream(stream, buffers, samples.size(), flags, timeNs, 100000);
    } catch (const std::exception& e) {
        return {.error = e.what()};
    } catch (...) {
        return {.error = "Unknown driver exception"};
    }

    if (result > 0) {
        if (static_cast<std::size_t>(result) > samples.size()) {
            return {.error = "Device returned more samples than requested"};
        }
        ReadResult received{
            .status = ReadStatus::Samples,
            .sampleCount = static_cast<std::size_t>(result),
        };
        if (flags & SOAPY_SDR_HAS_TIME) {
            received.timestampNs = timeNs;
        }
        return received;
    }
    if (result == 0 || result == SOAPY_SDR_TIMEOUT) {
        return {.status = ReadStatus::Timeout};
    }
    if (result == SOAPY_SDR_OVERFLOW) {
        return {.status = ReadStatus::Overflow};
    }

    return {.error = jst::fmt::format("{} ({})", SoapySDR::errToStr(result), result)};
}

void SoapyReceiver::reset() {
    if (device && stream) {
        try {
            if (state == State::Streaming) {
                device->deactivateStream(stream, 0, 0);
                state = State::StreamReady;
            }
            device->closeStream(stream);
        } catch (const std::exception& e) {
            JST_ERROR("[MODULE_SOAPY] Failed to deactivate/close stream: {}", e.what());
        } catch (...) {
            JST_ERROR("[MODULE_SOAPY] Failed to deactivate/close stream.");
        }
        stream = nullptr;
        state = State::Open;
    }

    if (biasTeeNeedsCleanup) {
        static_cast<void>(setBiasTee(false));
    }

    if (device) {
        try {
            SoapySDR::Device::unmake(device);
        } catch (const std::exception& e) {
            JST_ERROR("[MODULE_SOAPY] Failed to unmake device: {}", e.what());
        } catch (...) {
            JST_ERROR("[MODULE_SOAPY] Failed to unmake device.");
        }
        device = nullptr;
        state = State::Closed;
    }

    sampleRateRanges.clear();
    frequencyRanges.clear();
    biasTeeSupported = false;
    biasTeeNeedsCleanup = false;
}

SoapyReceiveStatus::Action SoapyReceiveStatus::handle(const SoapyReceiver::ReadStatus status,
                                                     const Clock::time_point now) {
    using Status = SoapyReceiver::ReadStatus;
    if (status == Status::Samples) {
        return Action::Samples;
    }
    if (status == Status::Timeout) {
        return Action::Retry;
    }
    if (status != Status::Overflow) {
        return Action::Fail;
    }

    ++deviceOverflows;
    if (deviceOverflows == 1 || now >= nextWarning) {
        nextWarning = now + std::chrono::seconds(1);
        return Action::WarnOverflow;
    }
    return Action::Retry;
}

}  // namespace Jetstream::Modules
