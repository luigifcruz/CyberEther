#include "module_impl.hh"
#include "library/soapysdr.hh"

#include <algorithm>
#include <exception>
#include <new>

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

constexpr auto ControlTimeout = std::chrono::seconds(5);

}  // namespace

SoapyImpl::~SoapyImpl() {
    stopReceiver();
}

Result SoapyImpl::validate() {
    const auto& config = *candidate();
    validatedOutputSizeBytes = 0;
    validatedInternalElements = 0;
    validatedInternalSizeBytes = 0;

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
                                 static_cast<U64>(sizeof(CF32)),
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
                                 static_cast<U64>(sizeof(CF32)),
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
    bufferHealth.publish(0.0f);
    deviceOverflows.publish(0);
    throughput.publish({0.0f, 0.0f});

    JST_CHECK(receiverDevice.open(deviceString));
    JST_CHECK(receiverDevice.validateSettings(sampleRate, frequency));
    JST_CHECK(allocateBuffers());
    JST_CHECK(configureDevice());

    outputs()["signal"].produced(name(), "signal", buffer);

    generation = 0;
    resetStream(frequency, sampleRate);
    publishedMetadata = {};
    JST_CHECK(publishMetadata(stream.metadata()));

    JST_CHECK(startReceiver());

    return Result::SUCCESS;
}

Result SoapyImpl::allocateBuffers() {
    try {
        JST_CHECK(buffer.create(device(), DataType::CF32, {numberOfBatches, numberOfTimeSamples}));
        JST_CHECK(SetSignalAxes(buffer, {
            .sample = Index{1},
            .batch = Index{0},
        }));
        JST_CHECK(stream.resize(validatedInternalElements));
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
    tuned = {
        .frequency = frequency,
        .sampleRate = sampleRate,
        .automaticGain = automaticGain,
        .gain = manualGain,
    };
    JST_CHECK(receiverDevice.setAntenna(antenna));
    JST_CHECK(receiverDevice.setSampleRate(sampleRate));
    JST_CHECK(receiverDevice.setTunerFrequency(frequency));
    gainRange.publish(receiverDevice.getGainRange());
    JST_CHECK(receiverDevice.setAutomaticGain(automaticGain));
    tuned.gain = receiverDevice.clampGain(manualGain);
    if (!automaticGain) {
        JST_CHECK(receiverDevice.setGain(tuned.gain));
    }
    manualGain = tuned.gain;
    JST_CHECK(receiverDevice.setBiasTee(biasTee));
    JST_CHECK(receiverDevice.startStream(streamString));
    return Result::SUCCESS;
}

Result SoapyImpl::startReceiver() {
    streaming = true;
    control.open();
    try {
        producer = std::thread([this] {
            Result result;
            try {
                result = soapyThreadLoop();
            } catch (...) {
                result = Result::ERROR;
            }
            control.close();
            if (result != Result::SUCCESS && result != Result::RELOAD) {
                errored = true;
                JST_FATAL("[MODULE_SOAPY] Device thread crashed.");
            }
        });
    } catch (const std::exception& e) {
        streaming = false;
        control.close();
        JST_ERROR("[MODULE_SOAPY] Failed to start device thread: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        streaming = false;
        control.close();
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
    control.close();
}

Result SoapyImpl::destroy() {
    stopReceiver();
    receiverDevice.reset();

    bufferHealth.publish(0.0f);
    throughput.publish({0.0f, 0.0f});

    return Result::SUCCESS;
}

Result SoapyImpl::reconfigure() {
    const auto& newConfig = *candidate();

    if (newConfig.modulePath != modulePath ||
        newConfig.deviceString != deviceString ||
        newConfig.streamString != streamString ||
        newConfig.antenna != antenna ||
        newConfig.biasTee != biasTee ||
        newConfig.numberOfBatches != numberOfBatches ||
        newConfig.numberOfTimeSamples != numberOfTimeSamples ||
        newConfig.bufferMultiplier != bufferMultiplier) {
        return Result::RECREATE;
    }

    if ((newConfig.frequency != frequency || newConfig.sampleRate != sampleRate) &&
        receiverDevice.validateSettings(newConfig.sampleRate, newConfig.frequency) != Result::SUCCESS) {
        return Result::WARNING;
    }

    Tuning desired{
        .frequency = newConfig.frequency,
        .sampleRate = newConfig.sampleRate,
        .automaticGain = newConfig.automaticGain,
        .gain = newConfig.manualGain,
    };
    if (newConfig.frequency != frequency) {
        JST_CHECK(control.submitAndWait([this, desired] { return applyTuning(desired); }, ControlTimeout));
        desired.gain = tuned.gain;
    } else {
        if (const auto range = gainRange.get()) {
            desired.gain = std::clamp(desired.gain, static_cast<F32>(range->minimum),
                                      static_cast<F32>(range->maximum));
        }
        if (newConfig.sampleRate != sampleRate || newConfig.automaticGain != automaticGain ||
            desired.gain != manualGain) {
            control.submit([this, desired] { return applyTuning(desired); });
        }
    }

    frequency = newConfig.frequency;
    sampleRate = newConfig.sampleRate;
    automaticGain = newConfig.automaticGain;
    manualGain = desired.gain;
    return Result::SUCCESS;
}

void SoapyImpl::resetStream(const F32 appliedFrequency, const F32 appliedSampleRate) {
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
    const auto result = stream.pop(buffer.data<CF32>(), buffer.size(), metadata);
    if (result == Result::INCOMPLETE) {
        return Result::YIELD;
    }
    JST_CHECK(result);
    return publishMetadata(metadata);
}

Result SoapyImpl::soapyThreadLoop() {
    constexpr std::size_t temporaryBufferSize = 8192;
    std::vector<CF32> tmp(temporaryBufferSize);
    const auto readSize = std::min<std::size_t>(temporaryBufferSize, stream.capacity());
    const auto readBuffer = std::span<CF32>{tmp.data(), readSize};
    SoapyReceiveStatus receiveStatus;

    while (streaming) {
        control.service();

        const U64 readGeneration = stream.metadata().generation;
        const auto received = receiverDevice.read(readBuffer);
        if (!streaming || errored) {
            break;
        }

        const auto action = receiveStatus.handle(received.status);
        if (received.status == SoapyReceiver::ReadStatus::Overflow) {
            deviceOverflows.publish(receiveStatus.deviceOverflows);
        }
        if (action == SoapyReceiveStatus::Action::Fail) {
            JST_ERROR("[MODULE_SOAPY] Failed to read stream on '{}': {}. Stopping reception.",
                      name(), received.error);
            errored = true;
            break;
        } else if (action == SoapyReceiveStatus::Action::Samples) {
            stream.push(readGeneration, tmp.data(), received.sampleCount);
            const U64 capacity = stream.capacity();
            if (capacity > 0) {
                const F32 newHealth = static_cast<F32>(stream.size()) /
                                      static_cast<F32>(capacity);
                const F32 smoothedHealth = bufferHealth.get() * 0.99f + newHealth * 0.01f;
                bufferHealth.publish(smoothedHealth);
            }
            const F32 actualMB = static_cast<F32>(stream.throughput() * sizeof(CF32)) / 1e6f;
            const F32 expectedMB = static_cast<F32>(stream.metadata().sampleRate * sizeof(CF32)) / 1e6f;
            throughput.publish({actualMB, expectedMB});
        }
    }

    return Result::SUCCESS;
}

F32 SoapyImpl::getBufferHealth() const {
    return bufferHealth.get();
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

std::vector<std::string> SoapyImpl::listAntennas() const {
    return receiverDevice.listAntennas();
}

bool SoapyImpl::supportsBiasTee() const {
    return receiverDevice.supportsBiasTee();
}

std::optional<SoapyAbi::Range> SoapyImpl::getGainRange() const {
    return gainRange.get();
}

Result SoapyImpl::applyTuning(const Tuning& desired) {
    if (desired.sampleRate != tuned.sampleRate) {
        JST_CHECK(receiverDevice.setSampleRate(desired.sampleRate));
        tuned.sampleRate = desired.sampleRate;
        resetStream(tuned.frequency, tuned.sampleRate);
    }

    const bool retuned = desired.frequency != tuned.frequency;
    if (retuned) {
        JST_CHECK(receiverDevice.setTunerFrequency(desired.frequency));
        tuned.frequency = desired.frequency;
        resetStream(tuned.frequency, tuned.sampleRate);
        gainRange.publish(receiverDevice.getGainRange());
    }

    const bool modeChanged = desired.automaticGain != tuned.automaticGain;
    if (modeChanged) {
        JST_CHECK(receiverDevice.setAutomaticGain(desired.automaticGain));
        tuned.automaticGain = desired.automaticGain;
    }

    const F32 applied = receiverDevice.clampGain(desired.gain);
    if (!tuned.automaticGain && (applied != tuned.gain || retuned || modeChanged)) {
        JST_CHECK(receiverDevice.setGain(applied));
    }
    tuned.gain = applied;

    return Result::SUCCESS;
}

Result SoapyImpl::setBiasTee(const bool enabled) {
    return receiverDevice.setBiasTee(enabled);
}

}  // namespace Jetstream::Modules
