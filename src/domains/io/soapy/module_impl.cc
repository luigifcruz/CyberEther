#include "module_impl.hh"

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Registry.hpp>

#include <algorithm>
#include <new>
#include <stdexcept>

namespace Jetstream::Modules {

Result SoapyImpl::validate() {
    const auto& config = *candidate();
    const bool sameDevice = config.deviceString == deviceString;

    JST_CHECK(ValidateSoapyConfig(config.frequency,
                                  config.sampleRate,
                                  config.numberOfBatches,
                                  config.numberOfTimeSamples,
                                  config.bufferMultiplier,
                                  sameDevice ? &sampleRateRanges : nullptr,
                                  sameDevice ? &frequencyRanges : nullptr));

    return Result::SUCCESS;
}

Result SoapyImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result SoapyImpl::create() {
#ifdef JST_OS_BROWSER
    if (EM_ASM_INT({ return 'usb' in navigator; }) == 0) {
        JST_ERROR("[MODULE_SOAPY] Browser not compatible with WebUSB.");
        return Result::ERROR;
    }
#endif

    errored = false;
    streaming = false;
    activeSampleRate = 0.0f;
    bufferHealth.publish(0.0f);
    throughput.publish({0.0f, 0.0f});

    SoapySDR::Kwargs args = SoapySDR::KwargsFromString(deviceString);
    SoapySDR::Kwargs streamArgs = SoapySDR::KwargsFromString(streamString);

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
        soapyDevice = SoapySDR::Device::make(devices.at(0));
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to open device: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to open device.");
        return Result::ERROR;
    }

    if (soapyDevice == nullptr) {
        JST_ERROR("[MODULE_SOAPY] Can't open SoapySDR device.");
        return Result::ERROR;
    }

    try {
        sampleRateRanges = soapyDevice->getSampleRateRange(SOAPY_SDR_RX, 0);
        frequencyRanges = soapyDevice->getFrequencyRange(SOAPY_SDR_RX, 0);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to get device ranges: {}", e.what());
        SoapySDR::Device::unmake(soapyDevice);
        soapyDevice = nullptr;
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to get device ranges.");
        SoapySDR::Device::unmake(soapyDevice);
        soapyDevice = nullptr;
        return Result::ERROR;
    }

    if (!SoapyRangeContains(sampleRateRanges, sampleRate)) {
        JST_ERROR("[MODULE_SOAPY] Sample rate ({:.2f} MHz) not supported.", sampleRate / 1e6);
        SoapySDR::Device::unmake(soapyDevice);
        soapyDevice = nullptr;
        return Result::ERROR;
    }

    if (!SoapyRangeContains(frequencyRanges, frequency)) {
        JST_ERROR("[MODULE_SOAPY] Frequency ({:.2f} MHz) not supported.", frequency / 1e6);
        SoapySDR::Device::unmake(soapyDevice);
        soapyDevice = nullptr;
        return Result::ERROR;
    }

    try {
        JST_CHECK(buffer.create(device(), DataType::CF32, {numberOfBatches, numberOfTimeSamples}));
        JST_CHECK(circularBuffer.resize(buffer.size() * bufferMultiplier));
    } catch (const std::bad_array_new_length&) {
        JST_ERROR("[MODULE_SOAPY] Internal buffer dimensions are too large.");
        return Result::ERROR;
    } catch (const std::bad_alloc&) {
        JST_ERROR("[MODULE_SOAPY] Failed to allocate the internal buffer.");
        return Result::ERROR;
    }

    try {
        soapyDevice->setSampleRate(SOAPY_SDR_RX, 0, sampleRate);
        soapyDevice->setFrequency(SOAPY_SDR_RX, 0, frequency);
        soapyDevice->setGainMode(SOAPY_SDR_RX, 0, automaticGain);

        soapyStream = soapyDevice->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0}, streamArgs);
        if (soapyStream == nullptr) {
            JST_ERROR("[MODULE_SOAPY] Failed to setup stream.");
            SoapySDR::Device::unmake(soapyDevice);
            soapyDevice = nullptr;
            return Result::ERROR;
        }
        const int activationResult = soapyDevice->activateStream(soapyStream, 0, 0, 0);
        if (activationResult != 0) {
            throw std::runtime_error(jst::fmt::format(
                "activateStream returned status {}", activationResult));
        }

    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to configure device: {}", e.what());
        if (soapyStream) {
            try { soapyDevice->closeStream(soapyStream); } catch (...) {}
            soapyStream = nullptr;
        }
        SoapySDR::Device::unmake(soapyDevice);
        soapyDevice = nullptr;
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to configure device.");
        if (soapyStream) {
            try { soapyDevice->closeStream(soapyStream); } catch (...) {}
            soapyStream = nullptr;
        }
        SoapySDR::Device::unmake(soapyDevice);
        soapyDevice = nullptr;
        return Result::ERROR;
    }

    outputs()["signal"].produced(name(), "signal", buffer);

    buffer.setAttribute("frequency", frequency);
    buffer.setAttribute("sampleRate", sampleRate);
    activeSampleRate = sampleRate;

    streaming = true;
    try {
        producer = std::thread([this] {
            try {
                JST_CHECK_THROW(soapyThreadLoop());
            } catch (...) {
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

Result SoapyImpl::destroy() {
    streaming = false;
    activeSampleRate = 0.0f;

    if (producer.joinable()) {
        producer.join();
    }

    if (soapyDevice && soapyStream) {
        try {
            soapyDevice->deactivateStream(soapyStream, 0, 0);
            soapyDevice->closeStream(soapyStream);
        } catch (const std::exception& e) {
            JST_ERROR("[MODULE_SOAPY] Failed to deactivate/close stream: {}", e.what());
        } catch (...) {
            JST_ERROR("[MODULE_SOAPY] Failed to deactivate/close stream.");
        }
        soapyStream = nullptr;
    }

    if (soapyDevice) {
        try {
            SoapySDR::Device::unmake(soapyDevice);
        } catch (const std::exception& e) {
            JST_ERROR("[MODULE_SOAPY] Failed to unmake device: {}", e.what());
        } catch (...) {
            JST_ERROR("[MODULE_SOAPY] Failed to unmake device.");
        }
        soapyDevice = nullptr;
    }

    sampleRateRanges.clear();
    frequencyRanges.clear();

    bufferHealth.publish(0.0f);
    throughput.publish({0.0f, 0.0f});

    return Result::SUCCESS;
}

Result SoapyImpl::reconfigure() {
    const auto& newConfig = *candidate();

    if (newConfig.deviceString != deviceString ||
        newConfig.streamString != streamString ||
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
    int flags;
    long long timeNs;
    constexpr std::size_t temporaryBufferSize = 8192;
    CF32 tmp[temporaryBufferSize];
    void* tmp_buffers[] = {tmp};
    const auto readSize = std::min<std::size_t>(temporaryBufferSize,
                                                circularBuffer.getCapacity());

    while (streaming) {
        try {
            int ret = soapyDevice->readStream(soapyStream, tmp_buffers, readSize, flags, timeNs, 1e5);
            if (ret > 0 && streaming && !errored) {
                JST_CHECK(circularBuffer.put(tmp, ret));
                const U64 capacity = circularBuffer.getCapacity();
                if (capacity > 0) {
                    const F32 newHealth = static_cast<F32>(circularBuffer.getOccupancy()) /
                                          static_cast<F32>(capacity);
                    const F32 smoothedHealth = bufferHealth.get() * 0.99f + newHealth * 0.01f;
                    bufferHealth.publish(smoothedHealth);
                }
                const F32 actualMB = static_cast<F32>(circularBuffer.getThroughput() * sizeof(CF32)) / 1e6f;
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

SoapyImpl::DeviceList SoapyImpl::ListAvailableDevices(const std::string& filter) {
#ifdef JST_OS_BROWSER
    if (EM_ASM_INT({ return 'usb' in navigator; }) == 0) {
        JST_ERROR("[MODULE_SOAPY] Browser not compatible with WebUSB.");
        return {};
    }
#endif

    const SoapySDR::Kwargs args = SoapySDR::KwargsFromString(filter);

    try {
        return DeviceListFromEntries(SoapySDR::Device::enumerate(args));
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to enumerate devices: {}", e.what());
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to enumerate devices.");
    }

    return {};
}

std::string SoapyImpl::DeviceEntryToString(const DeviceEntry& entry) {
    return SoapySDR::KwargsToString(entry);
}

F32 SoapyImpl::getBufferHealth() const {
    return bufferHealth.get();
}

std::pair<F32, F32> SoapyImpl::getThroughput() const {
    return throughput.get();
}

Result SoapyImpl::setTunerFrequency(const F32& freq) {
    if (!SoapyRangeContains(frequencyRanges, freq)) {
        JST_WARN("[MODULE_SOAPY] Frequency ({:.2f} MHz) not supported.", freq / 1e6);
        return Result::WARNING;
    }

    if (!streaming) {
        return Result::RECREATE;
    }

    try {
        soapyDevice->setFrequency(SOAPY_SDR_RX, 0, freq);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set frequency: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set frequency.");
        return Result::ERROR;
    }

    frequency = freq;
    buffer.setAttribute("frequency", frequency);

    return Result::SUCCESS;
}

Result SoapyImpl::setSampleRate(const F32& rate) {
    if (!SoapyRangeContains(sampleRateRanges, rate)) {
        JST_WARN("[MODULE_SOAPY] Sample rate ({:.2f} MHz) not supported.", rate / 1e6);
        return Result::WARNING;
    }

    if (!streaming) {
        return Result::RECREATE;
    }

    try {
        soapyDevice->setSampleRate(SOAPY_SDR_RX, 0, rate);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set sample rate: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set sample rate.");
        return Result::ERROR;
    }

    sampleRate = rate;
    activeSampleRate = rate;
    buffer.setAttribute("sampleRate", sampleRate);

    return Result::SUCCESS;
}

Result SoapyImpl::setAutomaticGain(const bool& gain) {
    if (!streaming) {
        return Result::RECREATE;
    }

    try {
        soapyDevice->setGainMode(SOAPY_SDR_RX, 0, gain);
    } catch (const std::exception& e) {
        JST_ERROR("[MODULE_SOAPY] Failed to set gain mode: {}", e.what());
        return Result::ERROR;
    } catch (...) {
        JST_ERROR("[MODULE_SOAPY] Failed to set gain mode.");
        return Result::ERROR;
    }

    automaticGain = gain;

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
