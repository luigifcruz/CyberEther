#include "module_impl.hh"
#include "library/soapysdr.hh"

#include <algorithm>
#include <exception>
#include <new>

#include <jetstream/memory/axis.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

SoapyImpl::~SoapyImpl() {
    stopReceiver();
}

Result SoapyImpl::validate() {
    const auto& config = *candidate();
    validatedOutputSizeBytes = 0;
    validatedInternalElements = 0;
    validatedInternalSizeBytes = 0;

    validatedDataType = NameToDataType(config.dataType);
    if (!FindSoapySampleFormat(validatedDataType)) {
        JST_ERROR("[MODULE_SOAPY] Unsupported data type '{}'.", config.dataType);
        return Result::ERROR;
    }
    const U64 sampleBytes = DataTypeSize(validatedDataType);

    if (!std::isfinite(config.frequency)) {
        JST_ERROR("[MODULE_SOAPY] Frequency must be finite.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.sampleRate) || config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_SOAPY] Sample rate must be finite and positive.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.manualGain)) {
        JST_ERROR("[MODULE_SOAPY] Manual gain must be finite.");
        return Result::ERROR;
    }
    if ((config.gainMode != "total" && config.gainMode != "stages") ||
        !std::isfinite(config.bandwidth) || config.bandwidth < 0.0) {
        JST_ERROR("[MODULE_SOAPY] Invalid gain mode or bandwidth.");
        return Result::ERROR;
    }
    for (const auto& [name, value] : config.gains) {
        F64 gain = 0.0;
        if (Parser::Deserialize(config.gains, name, gain) != Result::SUCCESS ||
            !std::isfinite(gain)) {
            JST_ERROR("[MODULE_SOAPY] Invalid gain for stage '{}'.", name);
            return Result::ERROR;
        }
    }

    if (config.numberOfBatches == 0) {
        JST_ERROR("[MODULE_SOAPY] Number of batches cannot be zero.");
        return Result::ERROR;
    }

    if (config.numberOfTimeSamples == 0) {
        JST_ERROR("[MODULE_SOAPY] Number of time samples cannot be zero.");
        return Result::ERROR;
    }

    if (config.bufferMultiplier == 0) {
        JST_ERROR("[MODULE_SOAPY] Buffer multiplier cannot be zero.");
        return Result::ERROR;
    }

    U64 outputElements = 0;
    if (!detail::CheckedMultiply(config.numberOfBatches,
                                 config.numberOfTimeSamples,
                                 outputElements)) {
        JST_ERROR("[MODULE_SOAPY] Output buffer dimensions are too large.");
        return Result::ERROR;
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElements,
                                 sampleBytes,
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_SOAPY] Output buffer layout is too large.");
        return Result::ERROR;
    }

    U64 internalElements = 0;
    U64 internalSizeBytes = 0;
    if (!detail::CheckedMultiply(outputElements,
                                 config.bufferMultiplier,
                                 internalElements) ||
        !detail::CheckedMultiply(internalElements,
                                 sampleBytes,
                                 internalSizeBytes)) {
        JST_ERROR("[MODULE_SOAPY] Internal buffer layout is too large.");
        return Result::ERROR;
    }

    validatedOutputSizeBytes = outputSizeBytes;
    validatedInternalElements = internalElements;
    validatedInternalSizeBytes = internalSizeBytes;

    return Result::SUCCESS;
}

Result SoapyImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result SoapyImpl::create() {
    if (deviceString.empty()) {
        JST_ERROR("[MODULE_SOAPY] No device selected.");
        return Result::INCOMPLETE;
    }

    if (!SoapyRuntime::Acquire()) {
        JST_ERROR("[MODULE_SOAPY] {}", SoapyRuntime::Availability().message);
        return Result::INCOMPLETE;
    }
    JST_CHECK(SoapyDiscovery::LoadDriverLibrary(modulePath));

    errored = false;
    streaming = false;
    bufferFill.publish(0.0f);
    deviceOverflows.publish(0);
    throughput.publish({0.0f, 0.0f});

    JST_CHECK(receiverDevice.open(deviceString));
    JST_CHECK(receiverDevice.writeSettings(settings));
    JST_CHECK(receiverDevice.validateDataType(validatedDataType));
    JST_CHECK(allocateBuffers());
    JST_CHECK(configureDevice());

    outputs()["signal"].produced(name(), "signal", buffer);
    JST_CHECK(buffer.setAttribute("sampleScale", receiverDevice.sampleScale()));

    generation = 0;
    resetStream(*receiverDevice.info().state.frequency.value,
                *receiverDevice.info().state.sampleRate.value);
    publishedMetadata = {};
    JST_CHECK(publishMetadata(stream.metadata()));

    JST_CHECK(startReceiver());

    return Result::SUCCESS;
}

Result SoapyImpl::allocateBuffers() {
    try {
        JST_CHECK(buffer.create(device(), validatedDataType,
                                 {numberOfBatches, numberOfTimeSamples}));
        JST_CHECK(SetSignalAxes(buffer, {
            .sample = Index{1},
            .batch = Index{0},
        }));
        JST_CHECK(stream.resize(validatedInternalElements,
                                 DataTypeSize(buffer.dtype())));
    } catch (const std::bad_array_new_length&) {
        JST_ERROR("[MODULE_SOAPY] Internal buffer dimensions are too large.");
        return Result::ERROR;
    } catch (const std::bad_alloc&) {
        JST_ERROR("[MODULE_SOAPY] Failed to allocate the internal buffer.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyImpl::configureDevice() {
    const Tuning desired{
        .frequency = frequency,
        .sampleRate = sampleRate,
        .automaticGain = automaticGain,
        .gain = manualGain,
        .gainMode = gainMode,
        .gains = gains,
        .bandwidth = bandwidth,
    };
    JST_CHECK(receiverDevice.setAntenna(antenna));
    JST_CHECK(applyTuning(desired, true));
    JST_CHECK(receiverDevice.setBiasTee(biasTee));
    SoapyKwargs args;
    JST_CHECK(receiverDevice.streamArguments(streamString, streamArgs, args));
    JST_CHECK(receiverDevice.startStreamArgs(args, buffer.dtype()));
    JST_CHECK(updateDeviceInfo(true));
    return Result::SUCCESS;
}

Result SoapyImpl::startReceiver() {
    streaming = true;
    try {
        producer = std::thread([this] {
            Result result;
            try {
                result = soapyThreadLoop();
            } catch (...) {
                result = Result::ERROR;
            }
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                errored = true;
                JST_FATAL("[MODULE_SOAPY] Device thread crashed.");
            }
            Scheduler::Context::NotifyPendingCompute();
        });
    } catch (const std::exception& e) {
        streaming = false;
        JST_ERROR("[MODULE_SOAPY] Failed to start device thread: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        streaming = false;
        JST_ERROR("[MODULE_SOAPY] Failed to start device thread.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

void SoapyImpl::stopReceiver() {
    streaming = false;

    if (producer.joinable()) {
        producer.join();
    }
}

Result SoapyImpl::destroy() {
    stopReceiver();
    receiverDevice.reset();
    deviceInfo.publish({});

    bufferFill.publish(0.0f);
    throughput.publish({0.0f, 0.0f});

    return Result::SUCCESS;
}

Result SoapyImpl::reconfigure() {
    const auto& newConfig = *candidate();

    if (errored || newConfig.modulePath != modulePath ||
        newConfig.deviceString != deviceString ||
        newConfig.streamString != streamString ||
        newConfig.streamArgs != streamArgs || newConfig.settings != settings ||
        (bandwidth != 0.0 && newConfig.bandwidth == 0.0) ||
        newConfig.dataType != dataType ||
        newConfig.antenna != antenna ||
        newConfig.biasTee != biasTee ||
        newConfig.numberOfBatches != numberOfBatches ||
        newConfig.numberOfTimeSamples != numberOfTimeSamples ||
        newConfig.bufferMultiplier != bufferMultiplier) {
        return Result::RECREATE;
    }

    return access.change([&] {
        const Tuning desired{
            .frequency = newConfig.frequency,
            .sampleRate = newConfig.sampleRate,
            .automaticGain = newConfig.automaticGain,
            .gain = newConfig.manualGain,
            .gainMode = newConfig.gainMode,
            .gains = newConfig.gains,
            .bandwidth = newConfig.bandwidth,
        };
        const auto previous = tuned;
        auto restore = previous;
        const auto& state = receiverDevice.info().state;
        if (desired.bandwidth != previous.bandwidth && !state.bandwidth.value) {
            JST_ERROR("[MODULE_SOAPY] Cannot change bandwidth without a readback.");
            return Result::ERROR;
        }
        if (desired.bandwidth != previous.bandwidth && state.bandwidth.value) {
            restore.bandwidth = *state.bandwidth.value;
        }
        if (previous.gainMode == "stages") {
            for (const auto& [name, gain] : state.gains) {
                if (gain.value) {
                    restore.gains[name] = *gain.value;
                }
            }
            for (const auto& [name, value] : desired.gains) {
                if (!restore.gains.contains(name)) {
                    JST_ERROR("[MODULE_SOAPY] Cannot restore gain stage '{}'.", name);
                    return Result::ERROR;
                }
            }
        }
        const auto result = applyTuning(desired);
        if (result != Result::SUCCESS) {
            if (applyTuning(restore, true) != Result::SUCCESS) {
                errored = true;
                JST_ERROR("[MODULE_SOAPY] Failed to restore the previous receiver settings.");
            }
            tuned = previous;
            return result;
        }

        frequency = newConfig.frequency;
        sampleRate = newConfig.sampleRate;
        automaticGain = newConfig.automaticGain;
        manualGain = newConfig.manualGain;
        gainMode = newConfig.gainMode;
        gains = newConfig.gains;
        bandwidth = newConfig.bandwidth;
        return Result::SUCCESS;
    });
}

void SoapyImpl::resetStream(const F64 appliedFrequency, const F64 appliedSampleRate) {
    stream.reset({
        .frequency = appliedFrequency,
        .sampleRate = appliedSampleRate,
        .generation = ++generation,
    });
}

Result SoapyImpl::publishMetadata(const Radio::StreamBuffer::Metadata& metadata) {
    if (metadata == publishedMetadata) {
        return Result::SUCCESS;
    }
    JST_CHECK(buffer.setAttribute("frequency", static_cast<F32>(metadata.frequency)));
    JST_CHECK(buffer.setAttribute("sampleRate", static_cast<F32>(metadata.sampleRate)));
    publishedMetadata = metadata;
    return Result::SUCCESS;
}

Result SoapyImpl::emitFrame() {
    Radio::StreamBuffer::Metadata metadata;
    const auto result = stream.pop(buffer.data(), buffer.size(), metadata);
    if (result == Result::INCOMPLETE) {
        return Result::YIELD;
    }
    JST_CHECK(result);
    return publishMetadata(metadata);
}

Result SoapyImpl::soapyThreadLoop() {
    constexpr std::size_t temporaryBufferSize = 8192;
    const auto readSize = std::min<std::size_t>(temporaryBufferSize, stream.capacity());
    Tensor temporary;
    JST_CHECK(temporary.create(device(), buffer.dtype(), {readSize}));
    const U64 sampleBytes = DataTypeSize(buffer.dtype());
    SoapyReceiveStatus receiveStatus;

    while (streaming) {
        const auto lock = access.lock();
        if (!streaming || errored) {
            break;
        }

        const U64 readGeneration = stream.metadata().generation;
        const auto received = receiverDevice.read(temporary.data(), readSize);
        if (!streaming || errored) {
            break;
        }

        const auto action = receiveStatus.handle(received.status);
        if (received.status == SoapyReceiver::ReadStatus::Overflow) {
            deviceOverflows.publish(receiveStatus.deviceOverflows);
            stream.split(readGeneration);
        }
        if (action == SoapyReceiveStatus::Action::Fail) {
            JST_ERROR("[MODULE_SOAPY] Failed to read stream on '{}': {}. Stopping reception.",
                      name(), received.error);
            errored = true;
            break;
        } else if (action == SoapyReceiveStatus::Action::Samples) {
            stream.push(readGeneration, temporary.data(), received.sampleCount);
            Scheduler::Context::NotifyPendingCompute();
            const U64 capacity = stream.capacity();
            if (capacity > 0) {
                const F32 newFill = static_cast<F32>(stream.size()) /
                                      static_cast<F32>(capacity);
                const F32 smoothedFill = bufferFill.get() * 0.99f + newFill * 0.01f;
                bufferFill.publish(smoothedFill);
            }
            const F32 actualMB =
                static_cast<F32>(stream.throughput() * sampleBytes) / 1e6f;
            const F32 expectedMB =
                static_cast<F32>(stream.metadata().sampleRate * sampleBytes) / 1e6f;
            throughput.publish({actualMB, expectedMB});
        }
    }

    return Result::SUCCESS;
}

F32 SoapyImpl::getBufferFill() const {
    return bufferFill.get();
}

F64 SoapyImpl::getBufferLoss() const {
    const auto stats = stream.statistics();
    return stats.pushedSamples > 0
        ? static_cast<F64>(stats.overwrittenSamples) / static_cast<F64>(stats.pushedSamples)
        : 0.0;
}

U64 SoapyImpl::getDeviceOverflows() const {
    return deviceOverflows.get();
}

std::pair<F32, F32> SoapyImpl::getThroughput() const {
    return throughput.get();
}

SoapyDeviceInfo SoapyImpl::getDeviceInfo() const {
    return deviceInfo.get();
}

Result SoapyImpl::updateDeviceInfo(const bool reset) {
    const auto result = receiverDevice.refresh();
    deviceInfo.publish(receiverDevice.info());
    JST_CHECK(result);
    const auto& current = receiverDevice.info().state;
    const auto actualFrequency = *current.frequency.value;
    const auto actualSampleRate = *current.sampleRate.value;
    if (!std::isfinite(static_cast<F32>(actualFrequency)) ||
        !std::isfinite(static_cast<F32>(actualSampleRate)) ||
        static_cast<F32>(actualSampleRate) <= 0.0f) {
        JST_ERROR("[MODULE_SOAPY] Actual tuning exceeds signal metadata precision.");
        return Result::ERROR;
    }
    const auto previous = stream.metadata();
    if (reset || previous.frequency != actualFrequency ||
        previous.sampleRate != actualSampleRate) {
        resetStream(actualFrequency, actualSampleRate);
    }
    return Result::SUCCESS;
}

Result SoapyImpl::applyTuning(const Tuning& desired, const bool force) {
    const bool rateChanged = force || desired.sampleRate != tuned.sampleRate;
    const bool retuned = force || desired.frequency != tuned.frequency;
    if (rateChanged) {
        JST_CHECK(receiverDevice.setSampleRate(desired.sampleRate));
    }
    if (retuned) {
        JST_CHECK(receiverDevice.setTunerFrequency(desired.frequency));
    }

    const bool modeChanged = force || desired.automaticGain != tuned.automaticGain;
    if (modeChanged || retuned || rateChanged) {
        JST_CHECK(receiverDevice.setAutomaticGain(desired.automaticGain));
    }
    const auto& caps = receiverDevice.info().capabilities;
    const bool manual = caps.automaticGain.value == false ||
                        (!desired.automaticGain && caps.automaticGain.value == true);
    const bool adjustable = caps.gain.value &&
                            caps.gain.value->minimum < caps.gain.value->maximum;
    const bool stages = desired.gainMode == "stages";
    if (manual && stages && (!caps.gains.value || caps.gains.value->empty())) {
        JST_ERROR("[MODULE_SOAPY] Gain stages are unavailable.");
        return Result::ERROR;
    }
    const bool gainsChanged = force || desired.gainMode != tuned.gainMode ||
                              desired.gains != tuned.gains;
    if (manual && !stages && adjustable &&
        (gainsChanged || desired.gain != tuned.gain || retuned || rateChanged || modeChanged)) {
        JST_CHECK(receiverDevice.setGain(receiverDevice.clampGain(desired.gain)));
    }
    if (manual && stages && (gainsChanged || retuned || rateChanged || modeChanged)) {
        for (const auto& [name, value] : desired.gains) {
            F64 gain = 0.0;
            JST_CHECK(Parser::Deserialize(desired.gains, name, gain));
            JST_CHECK(receiverDevice.setGainElement(name, gain));
        }
    }
    const bool bandwidthChanged = desired.bandwidth != tuned.bandwidth;
    if (desired.bandwidth > 0.0 && (force || bandwidthChanged || retuned || rateChanged)) {
        JST_CHECK(receiverDevice.setBandwidth(desired.bandwidth));
    }
    JST_CHECK(updateDeviceInfo(retuned || rateChanged || bandwidthChanged || gainsChanged));
    tuned = desired;
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
