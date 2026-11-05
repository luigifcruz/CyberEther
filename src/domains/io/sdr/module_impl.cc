#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <new>

#include <jetstream/memory/axis.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

constexpr auto ReadTimeout = std::chrono::milliseconds(50);
constexpr U64 ReadChunk = 16384;

}  // namespace

SdrImpl::~SdrImpl() {
    stopReceiver();
}

Result SdrImpl::validate() {
    const auto& config = *candidate();
    validatedOutputSizeBytes = 0;
    validatedInternalElements = 0;
    validatedInternalSizeBytes = 0;

    if (!std::isfinite(config.frequency)) {
        JST_ERROR("[MODULE_SDR] Frequency must be finite.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.sampleRate) || config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_SDR] Sample rate must be finite and positive.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.manualGain) ||
        std::ranges::any_of(config.manualGains,
                            [](F32 gain) { return !std::isfinite(gain); })) {
        JST_ERROR("[MODULE_SDR] Manual gain must be finite.");
        return Result::ERROR;
    }
    if (config.channels == 0 ||
        std::ranges::any_of(config.automaticGains, [](U64 mode) { return mode > 1; })) {
        JST_ERROR("[MODULE_SDR] Channels must be positive; automatic gain entries "
                  "must be 0 (manual) or 1 (automatic).");
        return Result::ERROR;
    }
    if (config.numberOfBatches == 0) {
        JST_ERROR("[MODULE_SDR] Number of batches cannot be zero.");
        return Result::ERROR;
    }
    if (config.numberOfTimeSamples == 0) {
        JST_ERROR("[MODULE_SDR] Number of time samples cannot be zero.");
        return Result::ERROR;
    }
    if (config.bufferMultiplier == 0) {
        JST_ERROR("[MODULE_SDR] Buffer multiplier cannot be zero.");
        return Result::ERROR;
    }

    U64 outputElements = 0;
    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(config.numberOfBatches, config.numberOfTimeSamples,
                                 outputElements) ||
        !detail::CheckedMultiply(outputElements, static_cast<U64>(sizeof(CF32)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_SDR] Output buffer dimensions are too large.");
        return Result::ERROR;
    }

    U64 internalElements = 0;
    U64 internalSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElements, config.bufferMultiplier, internalElements) ||
        !detail::CheckedMultiply(internalElements, static_cast<U64>(sizeof(CF32)),
                                 internalSizeBytes)) {
        JST_ERROR("[MODULE_SDR] Internal buffer layout is too large.");
        return Result::ERROR;
    }

    if (!detail::CheckedMultiply(outputSizeBytes, config.channels, outputSizeBytes) ||
        !detail::CheckedMultiply(internalSizeBytes, config.channels,
                                 internalSizeBytes)) {
        JST_ERROR("[MODULE_SDR] Multi-channel buffers are too large.");
        return Result::ERROR;
    }
    validatedOutputSizeBytes = outputSizeBytes;
    validatedInternalElements = internalElements;
    validatedInternalSizeBytes = internalSizeBytes;
    return Result::SUCCESS;
}

Result SdrImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));
    return Result::SUCCESS;
}

Result SdrImpl::create() {
    if (deviceString.empty()) {
        JST_ERROR("[MODULE_SDR] No device selected.");
        return Result::INCOMPLETE;
    }

    errored = false;
    streaming = false;
    bufferFill.publish(0.0f);
    throughput.publish({0.0f, 0.0f});
    deviceGaps.publish(0);
    deviceLostSamples.publish(0);

    JST_CHECK(Radio::Stream::Open(deviceString, radio));
    const auto& caps = radio->capabilities();
    antennas = caps.antennas;
    if (channels > caps.receiveChannels) {
        JST_ERROR(
            "[MODULE_SDR] Selected radio supports {} receive channels; requested {}.",
            caps.receiveChannels, channels);
        radio.reset();
        return Result::ERROR;
    }
    if (const auto rates = caps.fixedSampleRates();
        !rates.empty() && !Radio::Device::Range::Contains(caps.sampleRateRanges, sampleRate)) {
        JST_INFO("[MODULE_SDR] {:.3f} MHz is not a rate of {}. Using {:.3f} MHz.",
                 sampleRate / 1e6, radio->info().label, rates.front() / 1e6);
        sampleRate = static_cast<F32>(rates.front());
        candidate()->sampleRate = sampleRate;
    }

    try {
        if (!automaticGains.empty()) {
            automaticGains.resize(channels, automaticGain ? 1 : 0);
        }
        if (!manualGains.empty()) {
            manualGains.resize(channels, manualGain);
        }
        Shape shape{numberOfBatches, numberOfTimeSamples};
        if (channels > 1) {
            shape.insert(shape.begin(), channels);
        }
        JST_CHECK(buffer.create(device(), DataType::CF32, shape));
        JST_CHECK(SetSignalAxes(
            buffer,
            {
                .sample = Index{shape.size() - 1},
                .batch = Index{shape.size() - 2},
                .channel = channels > 1 ? std::optional<Index>{0} : std::nullopt,
            }));
        outputChannels.resize(channels);
        const U64 channelElements = numberOfBatches * numberOfTimeSamples;
        for (U64 i = 0; i < channels; ++i) {
            outputChannels[i] = buffer.data<CF32>() + i * channelElements;
        }
        JST_CHECK(stream.resize(validatedInternalElements, channels));
    } catch (const std::bad_alloc&) {
        JST_ERROR("[MODULE_SDR] Failed to allocate the internal buffer.");
        radio.reset();
        return Result::ERROR;
    }

    if (const auto result = radio->start(settingsOf(*this)); result != Result::SUCCESS) {
        radio.reset();
        return result;
    }
    resetStream();
    if (channels == 1 && manualGains.empty() && radio->applied().gainRange) {
        manualGain = static_cast<F32>(radio->applied().gain);
    }

    if (!manualGains.empty()) {
        manualGains[0] = static_cast<F32>(radio->applied().gain);
        for (U64 i = 1; i < channels; ++i) {
            manualGains[i] =
                static_cast<F32>(radio->applied().additionalGains[i - 1].gain);
        }
    }
    outputs()["signal"].produced(name(), "signal", buffer);
    publishedMetadata = {};
    JST_CHECK(publishMetadata(stream.metadata()));

    if (const auto result = startReceiver(); result != Result::SUCCESS) {
        radio.reset();
        return result;
    }
    return Result::SUCCESS;
}

Result SdrImpl::destroy() {
    stopReceiver();
    radio.reset();
    bufferFill.publish(0.0f);
    throughput.publish({0.0f, 0.0f});
    return Result::SUCCESS;
}

Result SdrImpl::reconfigure() {
    const auto& config = *candidate();

    if (config.channels != channels || config.deviceString != deviceString ||
        config.antenna != antenna || config.biasTee != biasTee ||
        config.numberOfBatches != numberOfBatches ||
        config.numberOfTimeSamples != numberOfTimeSamples ||
        config.bufferMultiplier != bufferMultiplier) {
        return Result::RECREATE;
    }
    if (!radio || errored) {
        return Result::RECREATE;
    }

    const auto& caps = radio->capabilities();
    if (!Radio::Device::Range::Contains(caps.frequencyRanges, config.frequency)) {
        JST_WARN("[MODULE_SDR] Frequency ({:.3f} MHz) not supported. Use {}.",
                 config.frequency / 1e6, Radio::Device::Range::Describe(caps.frequencyRanges));
        return Result::WARNING;
    }
    if (!Radio::Device::Range::Contains(caps.sampleRateRanges, config.sampleRate)) {
        JST_WARN("[MODULE_SDR] Sample rate ({:.3f} MHz) not supported.", config.sampleRate / 1e6);
        return Result::WARNING;
    }

    return access.change([&] {
        const auto previous = radio->applied();
        const U64 generation = radio->generation();
        const auto result = radio->update(settingsOf(config));
        if (result == Result::FATAL) {
            errored = true;
            return Result::ERROR;
        }
        const auto& applied = radio->applied();
        if (radio->generation() != generation || applied.frequency != previous.frequency ||
            applied.sampleRate != previous.sampleRate) {
            resetStream();
        } else {
            published.publish(applied);
        }
        JST_CHECK(result);

        frequency = config.frequency;
        sampleRate = config.sampleRate;
        automaticGain = config.automaticGain;
        automaticGains = config.automaticGains;
        if (!automaticGains.empty()) {
            automaticGains.resize(channels, config.automaticGain ? 1 : 0);
        }
        manualGains = config.manualGains;
        if (!manualGains.empty()) {
            manualGains.resize(channels, config.manualGain);
            manualGains[0] = static_cast<F32>(applied.gain);
            for (U64 i = 1; i < channels; ++i) {
                manualGains[i] = static_cast<F32>(applied.additionalGains[i - 1].gain);
            }
        }
        manualGain = channels == 1 && manualGains.empty() && applied.gainRange
                         ? static_cast<F32>(applied.gain)
                         : config.manualGain;
        return Result::SUCCESS;
    });
}

Radio::Stream::Settings SdrImpl::settingsOf(const Sdr& config) const {
    Radio::Stream::Settings settings{
        .frequency = config.frequency,
        .sampleRate = config.sampleRate,
        .antenna = config.channels > 1 ? std::string{} : config.antenna,
        .automaticGain = config.automaticGains.empty() ? config.automaticGain
                                                       : config.automaticGains[0] != 0,
        .biasTee = config.biasTee,
        .gain = config.manualGains.empty() ? config.manualGain : config.manualGains[0],
        .receiveChannels = config.channels,
    };
    for (U64 i = 1; i < config.channels; ++i) {
        settings.additionalGains.push_back({
            .automatic = i >= config.automaticGains.size()
                             ? config.automaticGain
                             : config.automaticGains[i] != 0,
            .gain = i >= config.manualGains.size() ? config.manualGain
                                                   : config.manualGains[i],
        });
    }
    return settings;
}

void SdrImpl::resetStream() {
    const auto& current = radio->applied();
    stream.reset({
        .frequency = current.frequency,
        .sampleRate = current.sampleRate,
        .generation = radio->generation(),
    });
    published.publish(current);
}

Result SdrImpl::publishMetadata(const Radio::StreamBuffer::Metadata& metadata) {
    if (metadata == publishedMetadata) {
        return Result::SUCCESS;
    }
    JST_CHECK(buffer.setAttribute("frequency", static_cast<F32>(metadata.frequency)));
    JST_CHECK(buffer.setAttribute("sampleRate", static_cast<F32>(metadata.sampleRate)));
    publishedMetadata = metadata;
    return Result::SUCCESS;
}

Result SdrImpl::emitFrame() {
    Radio::StreamBuffer::Metadata metadata;
    const auto result =
        stream.pop(outputChannels, numberOfBatches * numberOfTimeSamples, metadata);
    if (result == Result::INCOMPLETE) {
        return Result::YIELD;
    }
    JST_CHECK(result);
    return publishMetadata(metadata);
}

Result SdrImpl::startReceiver() {
    streaming = true;
    try {
        receiver = std::thread([this] {
            Result result = Result::ERROR;
            try {
                result = receiveLoop();
            } catch (...) {
                result = Result::ERROR;
            }
            if (result != Result::SUCCESS) {
                errored = true;
            }
            Scheduler::Context::NotifyPendingCompute();
        });
    } catch (const std::exception& e) {
        streaming = false;
        JST_ERROR("[MODULE_SDR] Failed to start the receive thread: {}", e.what());
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

void SdrImpl::stopReceiver() {
    streaming = false;
    if (receiver.joinable()) {
        receiver.join();
    }
}

Result SdrImpl::receiveLoop() {
    const U64 chunk = std::min<U64>(ReadChunk, std::max<U64>(stream.capacity(), 1));
    std::vector<std::vector<CF32>> scratch(channels, std::vector<CF32>(chunk));
    std::vector<std::span<CF32>> destinations;
    std::vector<const CF32*> pointers;
    for (auto& channel : scratch) {
        destinations.emplace_back(channel);
        pointers.push_back(channel.data());
    }
    U64 gaps = 0;
    U64 lost = 0;

    while (streaming) {
        const auto lock = access.lock();
        if (!streaming) {
            break;
        }

        if (errored) {
            return Result::ERROR;
        }

        const auto capture = radio->readChannels(destinations, ReadTimeout);
        if (!streaming) {
            break;
        }

        switch (capture.status) {
            case Radio::Device::Status::Samples: {
                if (capture.count > chunk) {
                    JST_ERROR(
                        "[MODULE_SDR] Radio returned more samples than requested.");
                    return Result::ERROR;
                }
                stream.push(capture.generation, pointers, capture.count,
                            capture.timestamp, capture.epoch);
                Scheduler::Context::NotifyPendingCompute();
                const U64 capacity = stream.capacity();
                if (capacity > 0) {
                    const F32 level = static_cast<F32>(stream.size()) / static_cast<F32>(capacity);
                    bufferFill.publish(bufferFill.get() * 0.99f + level * 0.01f);
                }
                const auto current = stream.metadata();
                throughput.publish({
                    static_cast<F32>(stream.throughput() * sizeof(CF32) * channels /
                                     1e6),
                    static_cast<F32>(current.sampleRate * sizeof(CF32) * channels /
                                     1e6),
                });
                break;
            }
            case Radio::Device::Status::Timeout:
                break;
            case Radio::Device::Status::Discontinuity:
                stream.split(capture.generation);
                deviceGaps.publish(++gaps);
                lost += capture.lost;
                deviceLostSamples.publish(lost);
                break;
            case Radio::Device::Status::Disconnected:
            case Radio::Device::Status::Failure:
                JST_ERROR("[MODULE_SDR] Reception stopped on '{}': {}", name(), capture.error);
                return Result::ERROR;
        }
    }
    return Result::SUCCESS;
}

F32 SdrImpl::getBufferFill() const {
    return bufferFill.get();
}

SdrImpl::LossSnapshot SdrImpl::getLoss() const {
    const auto stats = stream.statistics();
    return {
        .bufferLoss = stats.pushedSamples > 0
            ? static_cast<F64>(stats.overwrittenSamples) / static_cast<F64>(stats.pushedSamples)
            : 0.0,
        .deviceGaps = deviceGaps.get(),
        .deviceLostSamples = deviceLostSamples.get(),
    };
}

std::pair<F32, F32> SdrImpl::getThroughput() const {
    return throughput.get();
}

std::vector<std::string> SdrImpl::listAntennas() const {
    return antennas;
}

Radio::Stream::Applied SdrImpl::getApplied() const {
    return published.get();
}

std::optional<Radio::Device::Range> SdrImpl::getGainRange(const U64 channel) const {
    const auto applied = published.get();
    if (channel == 0) {
        return applied.gainRange;
    }
    return channel <= applied.additionalGains.size()
               ? applied.additionalGains[channel - 1].range
               : std::nullopt;
}

}  // namespace Jetstream::Modules
