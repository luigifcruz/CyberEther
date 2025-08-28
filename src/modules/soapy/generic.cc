#include "jetstream/modules/soapy.hh"

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Modules.hpp>

namespace Jetstream {

template<Device D, typename T>
std::string Soapy<D, T>::DeviceEntry::toString() const {
    return SoapySDR::KwargsToString(*this);
}

template<Device D, typename T>
struct Soapy<D, T>::Impl {
    SoapySDR::RangeList sampleRateRanges;
    SoapySDR::RangeList frequencyRanges;

    SoapySDR::Device* soapyDevice;
    SoapySDR::Stream* soapyStream;

    std::thread producer;
    bool errored = false;
    bool streaming = false;
    std::string deviceLabel;
    std::string deviceName;
    std::string deviceHardwareKey;
    Memory::CircularBuffer<T> buffer;
    Tensor<Device::CPU, T> hostOutputBuffer;

    Result soapyThreadLoop();

    static bool CheckValidRange(const std::vector<SoapySDR::Range>& ranges, const F32& val);
};

template<Device D, typename T>
Soapy<D, T>::Soapy() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Soapy<D, T>::~Soapy() {
    impl.reset();
}

template<Device D, typename T>
Result Soapy<D, T>::create() {
    JST_DEBUG("Initializing Soapy module.");
    JST_INIT_IO();

    impl->errored = false;
    impl->streaming = false;
    impl->deviceName = "None";
    impl->deviceHardwareKey = "None";

    // If browser, check if WebUSB is supported.

#ifdef JST_OS_BROWSER
    if (EM_ASM_INT({ return 'usb' in navigator; }) == 0) {
        JST_ERROR("This browser is not compatible with WebUSB. "
                  "Try a Chromium based browser like Chrome, Brave, or Opera GX.");
        return Result::ERROR;
    }
#endif

    // Convert requested device and stream strings into arguments.

    SoapySDR::Kwargs args = SoapySDR::KwargsFromString(config.deviceString);
    SoapySDR::Kwargs streamArgs = SoapySDR::KwargsFromString(config.streamString);

    // Try opening device.

    try {
        const auto devices = SoapySDR::Device::enumerate(args);
        impl->deviceLabel = devices.at(0).at("label");
        impl->soapyDevice = SoapySDR::Device::make(devices.at(0));
    } catch(const std::exception& e) {
        JST_ERROR("Failed to open device. Reason: {}", e.what());
        return Result::ERROR;
    } catch(...) {
        JST_ERROR("Failed to open device.");
        return Result::ERROR;
    }

    if (impl->soapyDevice == nullptr) {
        JST_ERROR("Can't open SoapySDR device.");
        return Result::ERROR;
    }

    // Gather device ranges.

    impl->sampleRateRanges = impl->soapyDevice->getSampleRateRange(SOAPY_SDR_RX, 0);
    impl->frequencyRanges = impl->soapyDevice->getFrequencyRange(SOAPY_SDR_RX, 0);

    // Check if requested configuration is supported.

    if (!Impl::CheckValidRange(impl->sampleRateRanges, config.sampleRate)) {
        JST_ERROR("Sample rate requested ({:.2f} MHz) is not supported by the device.", config.sampleRate / JST_MHZ);
        SoapySDR::Device::unmake(impl->soapyDevice);
        return Result::ERROR;
    }

    if (!Impl::CheckValidRange(impl->frequencyRanges, config.frequency)) {
        JST_ERROR("Frequency requested ({:.2f} MHz) is not supported by the device.", config.frequency / JST_MHZ);
        SoapySDR::Device::unmake(impl->soapyDevice);
        return Result::ERROR;
    }

    // Apply requested configuration.

    impl->soapyDevice->setSampleRate(SOAPY_SDR_RX, 0, config.sampleRate);
    impl->soapyDevice->setFrequency(SOAPY_SDR_RX, 0, config.frequency);
    impl->soapyDevice->setGainMode(SOAPY_SDR_RX, 0, config.automaticGain);

    impl->soapyStream = impl->soapyDevice->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, {0}, streamArgs);
    if (impl->soapyStream == nullptr) {
        JST_ERROR("Failed to setup SoapySDR stream.");
        SoapySDR::Device::unmake(impl->soapyDevice);
        return Result::ERROR;
    }
    impl->soapyDevice->activateStream(impl->soapyStream, 0, 0, 0);

    impl->deviceName = impl->soapyDevice->getDriverKey();
    impl->deviceHardwareKey = impl->soapyDevice->getHardwareKey();

    // Calculate shape.

    std::vector<U64> outputShape = { config.numberOfBatches, config.numberOfTimeSamples };

    // Allocate output.

    output.buffer = Tensor<D, T>(outputShape);

    // Allocate circular buffer.

    impl->buffer.resize(output.buffer.size() * config.bufferMultiplier);

    // Initialize thread for ingest.

    impl->producer = std::thread([&]{
        try {
            JST_CHECK_THROW(impl->soapyThreadLoop());
        } catch(...) {
            impl->errored = true;
            JST_FATAL("[SOAPY] Device thread crashed.");
        }
    });

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::destroy() {
    impl->streaming = false;

    if (impl->producer.joinable()) {
        impl->producer.join();
    }

    impl->soapyDevice->deactivateStream(impl->soapyStream, 0, 0);
    impl->soapyDevice->closeStream(impl->soapyStream);

    SoapySDR::Device::unmake(impl->soapyDevice);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::Impl::soapyThreadLoop() {
    int flags;
    long long timeNs;
    CF32 tmp[8192];
    void *tmp_buffers[] = { tmp };

    // TODO: Replace with zero-copy Soapy API.
    streaming = true;
    while (streaming) {
        int ret = soapyDevice->readStream(soapyStream, tmp_buffers, 8192, flags, timeNs, 1e5);
        if (ret > 0 && streaming && !errored) {
            buffer.put(tmp, ret);
        }
    }

    JST_TRACE("SDR Thread Safed");
    return Result::SUCCESS;
}

template<Device D, typename T>
void Soapy<D, T>::info() const {
    JST_DEBUG("  Device String:          {}", config.deviceString);
    JST_DEBUG("  Stream String:          {}", config.streamString);
    JST_DEBUG("  Frequency:              {:.2f} MHz", config.frequency / JST_MHZ);
    JST_DEBUG("  Sample Rate:            {:.2f} MHz", config.sampleRate / JST_MHZ);
    JST_DEBUG("  Automatic Gain:         {}", config.automaticGain ? "YES" : "NO");
    JST_DEBUG("  Number of Batches:      {}", config.numberOfBatches);
    JST_DEBUG("  Number of Time Samples: {}", config.numberOfTimeSamples);
}

template<Device D, typename T>
Result Soapy<D, T>::setTunerFrequency(F32& frequency) {
    if (!Impl::CheckValidRange(impl->frequencyRanges, frequency)) {
        JST_WARN("Frequency requested ({:.2f} MHz) is not supported by the device.", frequency / JST_MHZ);
        frequency = config.frequency;
        return Result::WARNING;
    }

    config.frequency = frequency;

    if (!impl->streaming) {
        return Result::RELOAD;
    }

    impl->soapyDevice->setFrequency(SOAPY_SDR_RX, 0, config.frequency);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::setSampleRate(F32& sampleRate) {
    if (!Impl::CheckValidRange(impl->sampleRateRanges, sampleRate)) {
        JST_WARN("Sample rate requested ({:.2f} MHz) is not supported by the device.", sampleRate / JST_MHZ);
        sampleRate = config.sampleRate;
        return Result::WARNING;
    }

    config.sampleRate = sampleRate;

    if (!impl->streaming) {
        return Result::RELOAD;
    }

    impl->soapyDevice->setSampleRate(SOAPY_SDR_RX, 0, config.sampleRate);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::setAutomaticGain(bool& automaticGain) {
    config.automaticGain = automaticGain;

    if (!impl->streaming) {
        return Result::RELOAD;
    }

    impl->soapyDevice->setGainMode(SOAPY_SDR_RX, 0, config.automaticGain);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::createCompute(const Context&) {
    JST_TRACE("Create SoapySDR compute core.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::computeReady() {
    if (impl->buffer.getOccupancy() < output.buffer.size()) {
        return impl->buffer.waitBufferOccupancy(output.buffer.size());
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::compute(const Context&) {
    if (impl->errored) {
        return Result::ERROR;
    }

    if (impl->buffer.getOccupancy() < output.buffer.size()) {
        return Result::YIELD;
    }

    impl->buffer.get(output.buffer.data(), output.buffer.size());

    return Result::SUCCESS;
}

template<Device D, typename T>
typename Soapy<D, T>::DeviceList Soapy<D, T>::ListAvailableDevices(const std::string& filter) {
#ifdef JST_OS_BROWSER
    if (EM_ASM_INT({ return 'usb' in navigator; }) == 0) {
        JST_ERROR("This browser is not compatible with WebUSB. "
                  "Try a Chromium based browser like Chrome, Brave, or Opera GX.");
        return {};
    }
#endif

    DeviceList deviceMap;
    const SoapySDR::Kwargs args = SoapySDR::KwargsFromString(filter);

    for (const auto& device : SoapySDR::Device::enumerate(args)) {
        deviceMap[device.at("label")] = device;
    }

    return deviceMap;
}

template<Device D, typename T>
bool Soapy<D, T>::Impl::CheckValidRange(const std::vector<SoapySDR::Range>& ranges, const F32& val) {
    bool isSampleRateSupported = false;

    for (const auto& range : ranges) {
        if (val >= range.minimum() and val <= range.maximum()) {
            isSampleRateSupported = true;
            break;
        }
    }

    return isSampleRateSupported;
}
template<Device D, typename T>
Memory::CircularBuffer<T>& Soapy<D, T>::getCircularBuffer() {
    return impl->buffer;
}

template<Device D, typename T>
const std::string& Soapy<D, T>::getDeviceName() const {
    return impl->deviceName;
}

template<Device D, typename T>
const std::string& Soapy<D, T>::getDeviceHardwareKey() const {
    return impl->deviceHardwareKey;
}

template<Device D, typename T>
const std::string& Soapy<D, T>::getDeviceLabel() const {
    return impl->deviceLabel;
}

JST_SOAPY_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
