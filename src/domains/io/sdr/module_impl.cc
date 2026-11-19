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

bool IsSampleFormat(const DataType type) {
    return type == DataType::CF32 || type == DataType::CF64 || type == DataType::CI16 ||
           type == DataType::CI8 || type == DataType::CU8;
}

bool IsIntegerFormat(const DataType type) {
    return type == DataType::CI16 || type == DataType::CI8 || type == DataType::CU8;
}

struct ClipCounter {
    F64 offset = 0.0;
    F64 scale = 1.0;
    F64 lower = -DefaultClipLevel;
    F64 upper = DefaultClipLevel;

    template <typename T>
    U64 count(const void* data, const U64 samples) const {
        const auto* values = static_cast<const T*>(data);
        U64 clipped = 0;
        for (U64 i = 0; i < 2 * samples; i += 2) {
            const F64 real = (static_cast<F64>(values[i]) - offset) * scale;
            const F64 imag = (static_cast<F64>(values[i + 1]) - offset) * scale;
            clipped += real >= upper || real <= lower || imag >= upper || imag <= lower;
        }
        return clipped;
    }

    U64 operator()(const DataType type, const void* data, const U64 samples) const {
        switch (type) {
            case DataType::CF64: return count<F64>(data, samples);
            case DataType::CI16: return count<I16>(data, samples);
            case DataType::CI8:  return count<I8>(data, samples);
            case DataType::CU8:  return count<U8>(data, samples);
            default:             return count<F32>(data, samples);
        }
    }
};

}  // namespace

SdrImpl::~SdrImpl() {
    stopReceiver();
}

Result SdrImpl::validate() {
    const auto& config = *candidate();
    validatedFormat = DataType::CF32;
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
    if (!std::isfinite(config.bandwidth) || config.bandwidth < 0.0f) {
        JST_ERROR("[MODULE_SDR] Filter bandwidth must be zero or positive.");
        return Result::ERROR;
    }
    const DataType candidateFormat = NameToDataType(config.sampleFormat);
    if (!IsSampleFormat(candidateFormat)) {
        JST_ERROR("[MODULE_SDR] Sample format '{}' is not one of CF32, CF64, CI16, CI8 or CU8.",
                  config.sampleFormat);
        return Result::ERROR;
    }
    const U64 sampleBytes = DataTypeSize(candidateFormat);
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
        !detail::CheckedMultiply(outputElements, sampleBytes, outputSizeBytes)) {
        JST_ERROR("[MODULE_SDR] Output buffer dimensions are too large.");
        return Result::ERROR;
    }

    U64 internalElements = 0;
    U64 internalSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElements, config.bufferMultiplier, internalElements) ||
        !detail::CheckedMultiply(internalElements, sampleBytes, internalSizeBytes)) {
        JST_ERROR("[MODULE_SDR] Internal buffer layout is too large.");
        return Result::ERROR;
    }

    if (!detail::CheckedMultiply(outputSizeBytes, config.channels, outputSizeBytes) ||
        !detail::CheckedMultiply(internalSizeBytes, config.channels, internalSizeBytes)) {
        JST_ERROR("[MODULE_SDR] Multi-channel buffers are too large.");
        return Result::ERROR;
    }

    validatedFormat = candidateFormat;
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
    availableAntennas = caps.antennas;
    if (channels > caps.receiveChannels) {
        JST_ERROR("[MODULE_SDR] Channels ({}) is more than {} supports ({}).",
                  channels, radio->info().label, caps.receiveChannels);
        radio.reset();
        return Result::ERROR;
    }
    format = validatedFormat;
    const auto selected = caps.format(format);
    if (!selected) {
        std::vector<std::string> supported;
        for (const auto& candidate : caps.formats) {
            supported.emplace_back(DataTypeToName(candidate.type));
        }
        JST_ERROR("[MODULE_SDR] Sample format {} is not supported by {}. Use {}.", format,
                  radio->info().label, jst::fmt::join(supported, ", "));
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
        JST_CHECK(buffer.create(device(), format, shape));
        JST_CHECK(SetSignalAxes(buffer, {
            .sample = Index{shape.size() - 1},
            .batch = Index{shape.size() - 2},
            .channel = channels > 1 ? std::optional<Index>{0} : std::nullopt,
        }));
        if (IsIntegerFormat(format)) {
            JST_CHECK(buffer.setAttribute("sampleScale",
                                          static_cast<F32>(1.0 / selected->fullScale)));
        }
        outputChannels.resize(channels);
        const U64 channelBytes = numberOfBatches * numberOfTimeSamples * DataTypeSize(format);
        for (U64 i = 0; i < channels; ++i) {
            outputChannels[i] = static_cast<std::byte*>(buffer.data()) + i * channelBytes;
        }
        JST_CHECK(stream.resize(validatedInternalElements, channels, DataTypeSize(format)));
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
        config.antenna != antenna ||
        config.channels != channels ||
        config.sampleFormat != sampleFormat ||
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
        bandwidth = config.bandwidth;
        automaticGain = config.automaticGain;
        manualGain = applied.gainRange ? static_cast<F32>(applied.gain) : config.manualGain;
        manualGains = config.manualGains;
        antennas = config.antennas;
        driverArguments = config.driverArguments;
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
        .bandwidth = config.bandwidth,
        .format = NameToDataType(config.sampleFormat),
    };
    if (config.channels > 1) {
        for (U64 i = 0; i < config.channels; ++i) {
            settings.channelGains.push_back(i < config.manualGains.size() ? config.manualGains[i]
                                                                           : config.manualGain);
            if (!availableAntennas.empty()) {
                settings.channelAntennas.push_back(i < config.antennas.size() ? config.antennas[i]
                                                                              : config.antenna);
            }
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
    const U64 sampleBytes = DataTypeSize(format);
    std::vector<std::vector<std::byte>> scratch(channels,
                                                std::vector<std::byte>(chunk * sampleBytes));
    std::vector<std::span<std::byte>> destinations;
    std::vector<const void*> pointers;
    for (auto& channel : scratch) {
        destinations.emplace_back(channel);
        pointers.push_back(channel.data());
    }
    U64 gaps = 0;
    U64 lost = 0;

    const auto& caps = radio->capabilities();
    const bool integer = IsIntegerFormat(format);
    const F64 formatScale = caps.format(format)->fullScale;
    F64 fullScale = caps.fullScale >= 2.0 ? caps.fullScale : 0.0;
    if (integer) {
        fullScale = fullScale >= 2.0 ? std::min(fullScale, formatScale) : formatScale;
    }
    ClipCounter clipped;
    if (integer) {
        clipped.offset = format == DataType::CU8 ? 128.0 : 0.0;
        clipped.scale = 1.0 / formatScale;
    }
    if (fullScale >= 2.0) {
        clipped.upper = (fullScale - 1.0) / fullScale;
        clipped.lower = -1.0;
    }
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
                for (const void* channel : pointers) {
                    counts.clipped += clipped(format, channel, capture.count);
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
                    static_cast<F32>(stream.throughput() * sampleBytes * channels / 1e6),
                    static_cast<F32>(current.sampleRate * sampleBytes * channels / 1e6),
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
    return availableAntennas;
}

Radio::Stream::Applied SdrImpl::getApplied() const {
    return published.get();
}

std::optional<Radio::Device::Range> SdrImpl::getGainRange() const {
    return published.get().gainRange;
}

}  // namespace Jetstream::Modules
