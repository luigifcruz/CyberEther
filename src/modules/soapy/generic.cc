#include "jetstream/modules/soapy.hh"

#ifdef JETSTREAM_STATIC
#if __has_include("SoapyAirspy.hpp")
#define ENABLE_STATIC_AIRSPY
#include "SoapyAirspy.hpp"
#endif
#if __has_include("SoapyRTLSDR.hpp")
#define ENABLE_STATIC_RTLSDR
#include "SoapyRTLSDR.hpp"
#endif
#endif

namespace Jetstream {

template<Device D, typename T>
Result Soapy<D, T>::create() {
    JST_DEBUG("Initializing Soapy module.");

    streaming = true;
    deviceName = "None";
    deviceHardwareKey = "None";

    // Initialize output.
    JST_INIT(
        JST_INIT_OUTPUT("buffer", output.buffer, config.outputShape);
    );

    // Initialize circular buffer.
    buffer.resize(config.outputShape.size() * config.bufferMultiplier);

    // Convert requested device and stream strings into arguments.

    SoapySDR::Kwargs args = SoapySDR::KwargsFromString(config.deviceString);
    SoapySDR::Kwargs streamArgs = SoapySDR::KwargsFromString(config.streamString);

    // Try opening device.

    try {
#ifdef JETSTREAM_STATIC
        const std::string device = args["driver"];
#ifdef ENABLE_STATIC_AIRSPY
        if (device.compare("airspy") == 0) {
            soapyDevice = new SoapyAirspy(args);
        }
#endif
#ifdef ENABLE_STATIC_RTLSDR
        // Static SoapyRTLSDR requires serial device number to work.
        // Example: device=rtlsdr,serial=XXXXXXXXXX
        if (device.compare("rtlsdr") == 0) {
            soapyDevice = new SoapyRTLSDR(args);
        }
#endif
#else
        const auto devices = SoapySDR::Device::enumerate(args);
        deviceLabel = devices.at(0).at("label");
        soapyDevice = SoapySDR::Device::make(devices.at(0));
#endif
    } catch(const std::exception& e) {
        JST_ERROR("Failed to open device. Reason: {}", e.what());
        JST_VOID_OUTPUT(output.buffer);
        return Result::ERROR;
    } catch(...) {
        JST_ERROR("Failed to open device.");
        JST_VOID_OUTPUT(output.buffer);
        return Result::ERROR;
    }

    if (soapyDevice == nullptr) {
        JST_ERROR("Can't open SoapySDR device.");
        JST_VOID_OUTPUT(output.buffer);
        return Result::ERROR;
    }

    // Gather device ranges.

    sampleRateRanges = soapyDevice->getSampleRateRange(SOAPY_SDR_RX, 0);
    frequencyRanges = soapyDevice->getFrequencyRange(SOAPY_SDR_RX, 0);

    // Check if requested configuration is supported.

    if (!CheckValidRange(sampleRateRanges, config.sampleRate)) {
        JST_ERROR("Sample rate requested ({}) is not supported by the device.", config.sampleRate);
        JST_VOID_OUTPUT(output.buffer);
        SoapySDR::Device::unmake(soapyDevice);
        return Result::ERROR;
    }

    if (!CheckValidRange(frequencyRanges, config.frequency)) {
        JST_ERROR("Frequency requested ({}) is not supported by the device.", config.frequency);
        JST_VOID_OUTPUT(output.buffer);
        SoapySDR::Device::unmake(soapyDevice);
        return Result::ERROR;
    }

    // Apply requested configuration.

    soapyDevice->setSampleRate(SOAPY_SDR_RX, 0, config.sampleRate);
    soapyDevice->setFrequency(SOAPY_SDR_RX, 0, config.frequency);
    soapyDevice->setGainMode(SOAPY_SDR_RX, 0, config.automaticGain);

    soapyStream = soapyDevice->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0}, streamArgs);
    if (soapyStream == nullptr) {
        JST_ERROR("Failed to setup SoapySDR stream.");
        JST_VOID_OUTPUT(output.buffer);
        SoapySDR::Device::unmake(soapyDevice);
        return Result::ERROR;
    }
    soapyDevice->activateStream(soapyStream, 0, 0, 0);

    deviceName = soapyDevice->getDriverKey();
    deviceHardwareKey = soapyDevice->getHardwareKey();

    // Initialize thread for ingest.
    producer = std::thread([&]{
        try {
            soapyThreadLoop();
        } catch(...) {
            JST_FATAL("[SOAPY] Device thread crashed.");
        }
    });

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::destroy() {
    streaming = false;

    if (producer.joinable()) {
        producer.join();
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
void Soapy<D, T>::soapyThreadLoop() {
    int flags;
    long long timeNs;
    CF32 tmp[8192];
    void *tmp_buffers[] = { tmp };

    // TODO: Replace with zero-copy Soapy API.
    streaming = true;
    while (streaming) {
        int ret = soapyDevice->readStream(soapyStream, tmp_buffers, 8192, flags, timeNs, 1e5);
        if (ret > 0) {
            buffer.put(tmp, ret);
        }
    }

    soapyDevice->deactivateStream(soapyStream, 0, 0);
    soapyDevice->closeStream(soapyStream);

    SoapySDR::Device::unmake(soapyDevice);

    JST_TRACE("SDR Thread Safed");
}

template<Device D, typename T>
void Soapy<D, T>::summary() const {
    JST_INFO("  Device String:      {}", config.deviceString);
    JST_INFO("  Stream String:      {}", config.streamString);
    JST_INFO("  Frequency:          {:.2f} MHz", config.frequency / (1000*1000));
    JST_INFO("  Sample Rate:        {:.2f} MHz", config.sampleRate / (1000*1000));
    JST_INFO("  Output Shape:       {}", config.outputShape);
}

template<Device D, typename T>
Result Soapy<D, T>::setTunerFrequency(F32& frequency) {
    if (!CheckValidRange(frequencyRanges, frequency)) {
        JST_WARN("Frequency requested ({:.2f} MHz) is not supported by the device.", frequency / (1000*1000));
        frequency = config.frequency;
        return Result::WARNING;
    }

    config.frequency = frequency;

    if (!complete()) {
        return Result::RELOAD;
    }

    soapyDevice->setFrequency(SOAPY_SDR_RX, 0, config.frequency);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::setSampleRate(F32& sampleRate) {
    if (!CheckValidRange(sampleRateRanges, sampleRate)) {
        JST_WARN("Sample rate requested ({:.2f} MHz) is not supported by the device.", sampleRate / (1000*1000));
        sampleRate = config.sampleRate;
        return Result::WARNING;
    }

    config.sampleRate = sampleRate;

    if (!complete()) {
        return Result::RELOAD;
    }

    soapyDevice->setSampleRate(SOAPY_SDR_RX, 0, config.sampleRate);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::setAutomaticGain(bool& automaticGain) {
    config.automaticGain = automaticGain;

    if (!complete()) {
        return Result::RELOAD;
    }

    soapyDevice->setGainMode(SOAPY_SDR_RX, 0, config.automaticGain);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create SoapySDR compute core.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::computeReady() {
    if (buffer.getOccupancy() < output.buffer.size()) {
        return buffer.waitBufferOccupancy(output.buffer.size());
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::compute(const RuntimeMetadata&) {
    if (buffer.getOccupancy() < output.buffer.size()) {
        return Result::SKIP;
    }

    buffer.get(output.buffer.data(), output.buffer.size());

    return Result::SUCCESS;
}

template<Device D, typename T>
Soapy<D, T>::DeviceList Soapy<D, T>::ListAvailableDevices(const std::string& filter) {
    DeviceList deviceMap;
    const SoapySDR::Kwargs args = SoapySDR::KwargsFromString(filter);

    for (const auto& device : SoapySDR::Device::enumerate(args)) {
        deviceMap[device.at("label")] = device;
    }

    return deviceMap;
}

template<Device D, typename T>
bool Soapy<D, T>::CheckValidRange(const std::vector<SoapySDR::Range>& ranges, const F32& val) {
    bool isSampleRateSupported = false;

    for (const auto& range : ranges) {
        if (val >= range.minimum() and val <= range.maximum()) {
            isSampleRateSupported = true;
            break;
        }
    }

    return isSampleRateSupported;
}

}  // namespace Jetstream
