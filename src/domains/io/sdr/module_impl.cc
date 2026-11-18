#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <new>

#include <jetstream/memory/attributes.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

constexpr auto ReadTimeout = std::chrono::milliseconds(50);
constexpr U64 ReadChunk = 16384;
constexpr F64 DefaultClipLevel = 0.999;

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
    if (!std::isfinite(config.manualGain)) {
        JST_ERROR("[MODULE_SDR] Manual gain must be finite.");
        return Result::ERROR;
    }
    if (std::ranges::any_of(config.manualGains,
                            [](const F32 gain) { return !std::isfinite(gain); })) {
        JST_ERROR("[MODULE_SDR] Manual gains must be finite.");
        return Result::ERROR;
    }
    if (config.channels == 0) {
        JST_ERROR("[MODULE_SDR] Channels cannot be zero.");
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
        !detail::CheckedMultiply(internalSizeBytes, config.channels, internalSizeBytes)) {
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
    sampleCounts.publish({});

    JST_CHECK(Radio::Stream::Open(deviceString, radio));
    const auto& caps = radio->capabilities();
    antennas = caps.antennas;
    if (channels > caps.receiveChannels) {
        JST_ERROR("[MODULE_SDR] Channels ({}) is more than {} supports ({}).",
                  channels, radio->info().label, caps.receiveChannels);
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
        Shape shape{numberOfBatches, numberOfTimeSamples};
        if (channels > 1) {
            shape.insert(shape.begin(), channels);
        }
        JST_CHECK(buffer.create(device(), DataType::CF32, shape));
        JST_CHECK(SetSignalAxes(buffer, {
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
    if (radio->applied().gainRange) {
        manualGain = static_cast<F32>(radio->applied().gain);
    }

    outputs()["signal"].produced(name(), "signal", buffer);
    publishedMetadata = {};
    publishedClippedSamples.reset();
    publishedReceivedSamples.reset();
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

    if (config.deviceString != deviceString ||
        config.driverArguments != driverArguments ||
        config.antenna != antenna ||
        config.channels != channels ||
        config.biasTee != biasTee ||
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
        manualGain = applied.gainRange ? static_cast<F32>(applied.gain) : config.manualGain;
        manualGains = config.manualGains;
        return Result::SUCCESS;
    });
}

Radio::Stream::Settings SdrImpl::settingsOf(const Sdr& config) const {
    Radio::Stream::Settings settings{
        .frequency = config.frequency,
        .sampleRate = config.sampleRate,
        .antenna = config.channels > 1 ? std::string{} : config.antenna,
        .automaticGain = config.automaticGain,
        .biasTee = config.biasTee,
        .gain = config.manualGain,
        .receiveChannels = config.channels,
        .driverArguments = config.driverArguments,
    };
    if (config.channels > 1) {
        for (U64 i = 0; i < config.channels; ++i) {
            settings.channelGains.push_back(i < config.manualGains.size() ? config.manualGains[i]
                                                                           : config.manualGain);
        }
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
    const auto result = stream.pop(outputChannels, numberOfBatches * numberOfTimeSamples, metadata);
    if (result == Result::INCOMPLETE) {
        return Result::YIELD;
    }
    JST_CHECK(result);
    JST_CHECK(publishMetadata(metadata));

    const SampleCounts counts = sampleCounts.get();
    if (publishedClippedSamples != counts.clipped) {
        JST_CHECK(buffer.setAttribute(std::string(ClippedSamplesAttribute), counts.clipped));
        publishedClippedSamples = counts.clipped;
    }

    if (publishedReceivedSamples != counts.received) {
        JST_CHECK(buffer.setAttribute(std::string(ReceivedSamplesAttribute), counts.received));
        publishedReceivedSamples = counts.received;
    }

    return Result::SUCCESS;
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

    const F64 fullScale = radio->capabilities().fullScale;
    const F32 clipLevel = static_cast<F32>(fullScale >= 2.0
        ? std::min(DefaultClipLevel, (fullScale - 1.0) / fullScale)
        : DefaultClipLevel);
    SampleCounts counts;

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
                    JST_ERROR("[MODULE_SDR] Radio returned more samples than requested.");
                    return Result::ERROR;
                }
                for (const CF32* channel : pointers) {
                    for (U64 i = 0; i < capture.count; ++i) {
                        counts.clipped += std::abs(channel[i].real()) >= clipLevel ||
                                          std::abs(channel[i].imag()) >= clipLevel;
                    }
                }
                counts.received += capture.count * channels;
                sampleCounts.publish(counts);
                stream.push(capture.generation, pointers, capture.count, capture.timestamp,
                            capture.epoch);
                Scheduler::Context::NotifyPendingCompute();
                const U64 capacity = stream.capacity();
                if (capacity > 0) {
                    const F32 level = static_cast<F32>(stream.size()) / static_cast<F32>(capacity);
                    bufferFill.publish(bufferFill.get() * 0.99f + level * 0.01f);
                }
                const auto current = stream.metadata();
                throughput.publish({
                    static_cast<F32>(stream.throughput() * sizeof(CF32) * channels / 1e6),
                    static_cast<F32>(current.sampleRate * sizeof(CF32) * channels / 1e6),
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

std::optional<Radio::Device::Range> SdrImpl::getGainRange() const {
    return published.get().gainRange;
}

}  // namespace Jetstream::Modules
