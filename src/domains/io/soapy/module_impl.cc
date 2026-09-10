#include "module_impl.hh"
#include "soapysdr.hh"

#include <SoapySDR/Types.hpp>

#include <algorithm>
#include <exception>
#include <new>

#include <jetstream/memory/axis.hh>
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

    if (!std::isfinite(config.frequency)) {
        JST_ERROR("[MODULE_SOAPY] Frequency must be finite.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.sampleRate) || config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_SOAPY] Sample rate must be finite and positive.");
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

    JST_CHECK(SoapyDiscovery::LoadDriverLibrary(modulePath));

    errored = false;
    streaming = false;
    activeSampleRate = 0.0f;
    bufferHealth.publish(0.0f);
    deviceOverflows.publish(0);
    throughput.publish({0.0f, 0.0f});

    const auto args = SoapySDR::KwargsFromString(deviceString);
    const auto streamArgs = SoapySDR::KwargsFromString(streamString);

    JST_CHECK(receiverDevice.open(args));
    JST_CHECK(receiverDevice.validateSettings(sampleRate, frequency));
    JST_CHECK(allocateBuffers());
    JST_CHECK(configureDevice(streamArgs));

    outputs()["signal"].produced(name(), "signal", buffer);

    buffer.setAttribute("frequency", frequency);
    buffer.setAttribute("sampleRate", sampleRate);
    activeSampleRate = sampleRate;

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
        JST_CHECK(circularBuffer.resize(validatedInternalElements));
    } catch (const std::bad_array_new_length&) {
        JST_ERROR("[MODULE_SOAPY] Internal buffer dimensions are too large.");
        return Result::ERROR;
    } catch (const std::bad_alloc&) {
        JST_ERROR("[MODULE_SOAPY] Failed to allocate the internal buffer.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyImpl::configureDevice(const SoapySDR::Kwargs& streamArgs) {
    try {
        JST_CHECK(receiverDevice.setAntenna(antenna));
        JST_CHECK(setSampleRate(sampleRate));
        JST_CHECK(setTunerFrequency(frequency));
        JST_CHECK(setAutomaticGain(automaticGain));
        JST_CHECK(setBiasTee(biasTee));
        JST_CHECK(receiverDevice.startStream(streamArgs));
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to configure device: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to configure device.");
        return Result::ERROR;
    }

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
    activeSampleRate = 0.0f;

    if (producer.joinable()) {
        producer.join();
    }
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

    if (newConfig.frequency != frequency) {
        JST_CHECK(setTunerFrequency(newConfig.frequency));
    }

    if (newConfig.sampleRate != sampleRate) {
        JST_CHECK(setSampleRate(newConfig.sampleRate));
    }

    if (newConfig.automaticGain != automaticGain) {
        JST_CHECK(setAutomaticGain(newConfig.automaticGain));
    }

    return Result::SUCCESS;
}

Result SoapyImpl::soapyThreadLoop() {
    constexpr std::size_t temporaryBufferSize = 8192;
    CF32 tmp[temporaryBufferSize];
    const auto readSize = std::min<std::size_t>(temporaryBufferSize,
                                                circularBuffer.capacity());
    const auto readBuffer = std::span<CF32>{tmp, readSize};
    SoapyReceiveStatus receiveStatus;

    while (streaming) {
        try {
            const auto received = receiverDevice.read(readBuffer);
            if (!streaming || errored) {
                break;
            }

            const auto action = receiveStatus.handle(received.status);
            if (received.status == SoapyReceiver::ReadStatus::Overflow) {
                deviceOverflows.publish(receiveStatus.deviceOverflows);
            }
            if (action == SoapyReceiveStatus::Action::WarnOverflow) {
                JST_WARN("[MODULE_SOAPY] Device receive overflow on '{}'. Samples were lost "
                         "(total events since stream start: {}).", name(), receiveStatus.deviceOverflows);
            } else if (action == SoapyReceiveStatus::Action::Fail) {
                JST_ERROR("[MODULE_SOAPY] Failed to read stream on '{}': {}. Stopping reception.",
                          name(), received.error);
                errored = true;
                break;
            } else if (action == SoapyReceiveStatus::Action::Samples) {
                JST_CHECK(circularBuffer.push(tmp, received.sampleCount));
                const U64 capacity = circularBuffer.capacity();
                if (capacity > 0) {
                    const F32 newHealth = static_cast<F32>(circularBuffer.size()) /
                                          static_cast<F32>(capacity);
                    const F32 smoothedHealth = bufferHealth.get() * 0.99f + newHealth * 0.01f;
                    bufferHealth.publish(smoothedHealth);
                }
                const F32 actualMB = static_cast<F32>(circularBuffer.throughput() * sizeof(CF32)) / 1e6f;
                const F32 expectedMB = (activeSampleRate.load() * sizeof(CF32)) / 1e6f;
                throughput.publish({actualMB, expectedMB});
            }
        } catch (const std::exception& e) {
            JST_ERROR("[MODULE_SOAPY] Failed to read stream: {}", e.what());
            errored = true;
            break;
        } catch (...) {
            JST_ERROR("[MODULE_SOAPY] Failed to read stream.");
            errored = true;
            break;
        }
    }

    return Result::SUCCESS;
}

F32 SoapyImpl::getBufferHealth() const {
    return bufferHealth.get();
}

F64 SoapyImpl::getBufferLoss() const {
    const auto stats = circularBuffer.statistics();
    return stats.pushedElements > 0
        ? static_cast<F64>(stats.overwrittenElements) / static_cast<F64>(stats.pushedElements)
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

Result SoapyImpl::setTunerFrequency(const F32& freq) {
    JST_CHECK(receiverDevice.setTunerFrequency(freq));

    frequency = freq;
    buffer.setAttribute("frequency", frequency);

    return Result::SUCCESS;
}

Result SoapyImpl::setSampleRate(const F32& rate) {
    JST_CHECK(receiverDevice.setSampleRate(rate));

    sampleRate = rate;
    activeSampleRate = rate;
    buffer.setAttribute("sampleRate", sampleRate);

    return Result::SUCCESS;
}

Result SoapyImpl::setAutomaticGain(const bool& gain) {
    JST_CHECK(receiverDevice.setAutomaticGain(gain));

    automaticGain = gain;

    return Result::SUCCESS;
}

Result SoapyImpl::setBiasTee(const bool enabled) {
    return receiverDevice.setBiasTee(enabled);
}

}  // namespace Jetstream::Modules
