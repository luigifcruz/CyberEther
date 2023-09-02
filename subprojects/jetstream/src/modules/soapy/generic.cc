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
    buffer.resize(config.outputShape[0] * config.outputShape[1] * config.bufferMultiplier);

    // Initialize thread for ingest.
    producer = std::thread([&]{ soapyThreadLoop(); });

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::destroy() {
    streaming = false;
    producer.join();

    return Result::SUCCESS;
}

template<Device D, typename T>
void Soapy<D, T>::soapyThreadLoop() {
    SoapySDR::Kwargs args = SoapySDR::KwargsFromString(config.deviceString);
    SoapySDR::Kwargs streamArgs = SoapySDR::KwargsFromString(config.streamString);

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
    soapyDevice = SoapySDR::Device::make(args);
#endif

    if (soapyDevice == nullptr) {
        JST_INFO("Can't open SoapySDR device.");
        JST_CHECK_THROW(Result::ERROR);
    }

    soapyDevice->setSampleRate(SOAPY_SDR_RX, 0, config.sampleRate);
    soapyDevice->setFrequency(SOAPY_SDR_RX, 0, config.frequency);
    soapyDevice->setGainMode(SOAPY_SDR_RX, 0, true);

    soapyStream = soapyDevice->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0}, streamArgs);
    if (soapyStream == nullptr) {
        JST_FATAL("Failed to setup SoapySDR stream.");
        SoapySDR::Device::unmake(soapyDevice);
        JST_CHECK_THROW(Result::FATAL);
    }
    soapyDevice->activateStream(soapyStream, 0, 0, 0);

    deviceName = soapyDevice->getDriverKey();
    deviceHardwareKey = soapyDevice->getHardwareKey();

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
F32 Soapy<D, T>::setTunerFrequency(const F32& frequency) {
    SoapySDR::RangeList freqRange = soapyDevice->getFrequencyRange(SOAPY_SDR_RX, 0);
    float minFreq = freqRange.front().minimum();
    float maxFreq = freqRange.back().maximum();

    config.frequency = frequency;

    if (frequency < minFreq) {
        config.frequency = minFreq;
    }

    if (frequency > maxFreq) {
        config.frequency = maxFreq;
    }

    soapyDevice->setFrequency(SOAPY_SDR_RX, 0, config.frequency);

    return config.frequency;
}

template<Device D, typename T>
Result Soapy<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create SoapySDR compute core.");
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
Result Soapy<D, T>::computeReady() {
    if (buffer.getOccupancy() < output.buffer.size()) {
        return buffer.waitBufferOccupancy(output.buffer.size());
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
