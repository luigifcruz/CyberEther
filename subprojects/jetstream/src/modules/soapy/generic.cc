#include "jetstream/modules/soapy.hh"

#ifdef JETSTREAM_STATIC
#include "SoapyAirspy.hpp"
#endif

namespace Jetstream { 

template<Device D, typename T>
Soapy<D, T>::Soapy(const Config& config, 
                   const Input& input) 
         : config(config),
           input(input),
           buffer(config.batchSize * config.outputBufferSize * config.bufferMultiplier) {
    JST_DEBUG("Initializing Soapy module.");

    streaming = true;
    deviceName = "None";
    deviceHardwareKey = "None";

    // Initialize output.
    JST_CHECK_THROW(Module::initOutput(this->output.buffer, {config.batchSize, config.outputBufferSize}));

    // Initialize thread for ingest.
    producer = std::thread([&]{ soapyThreadLoop(); });
}

template<Device D, typename T>
Soapy<D, T>::~Soapy() {
    streaming = false;
    producer.join();
}

template<Device D, typename T>
void Soapy<D, T>::soapyThreadLoop() {
    while (streaming) {
        SoapySDR::Kwargs args = SoapySDR::KwargsFromString(config.deviceString);

#ifdef JETSTREAM_STATIC
        const std::string device = args["driver"];
        if (device.compare("airspy") == 0) {
            static auto device = SoapyAirspy(args); 
            soapyDevice = &device;
        }
#else
        soapyDevice = SoapySDR::Device::make(args);
#endif

        if (soapyDevice == nullptr) {
            JST_FATAL("Can't open device.");
            JST_CHECK_THROW(Result::ERROR);
        }

        soapyDevice->setSampleRate(SOAPY_SDR_RX, 0, config.sampleRate);
        soapyDevice->setFrequency(SOAPY_SDR_RX, 0, config.frequency);
        soapyDevice->setGainMode(SOAPY_SDR_RX, 0, true);
        
        soapyStream = soapyDevice->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        if (soapyStream == nullptr) {
            JST_FATAL("Failed to setup stream.");
            SoapySDR::Device::unmake(soapyDevice);
            JST_CHECK_THROW(Result::ERROR);
        }
        soapyDevice->activateStream(soapyStream, 0, 0, 0);

        deviceName = soapyDevice->getDriverKey();
        deviceHardwareKey = soapyDevice->getHardwareKey();

        int flags;
        long long timeNs;
        CF32 tmp[8192];
        void *tmp_buffers[] = { tmp };

        streaming = true;
        while (streaming) {
            int ret = soapyDevice->readStream(soapyStream, tmp_buffers, 8192, flags, timeNs, 1e5);
            if (ret > 0) {
                buffer.put(tmp, ret);
            }
        }

        soapyDevice->deactivateStream(soapyStream, 0, 0);
        soapyDevice->closeStream(soapyStream);

        JST_TRACE("SDR Thread Safed");
    }
}

template<Device D, typename T>
void Soapy<D, T>::summary() const {
    JST_INFO("     Device String:      {}", config.deviceString);
    JST_INFO("     Frequency:          {:.2f} MHz", config.frequency / (1000*1000));
    JST_INFO("     Sample Rate:        {:.2f} MHz", config.sampleRate / (1000*1000));
    JST_INFO("     Buffer Multiplier:  {}", config.bufferMultiplier);
    JST_INFO("     Batch Size:         {}", config.batchSize);
    JST_INFO("     Output Buffer Size: {}", config.outputBufferSize);
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
        JST_DEBUG("Soapy module skipping batch because of lack of samples.");
        return Result::SKIP;
    }

    buffer.get(output.buffer.data(), output.buffer.size());

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Soapy<D, T>::Factory(std::unordered_map<std::string, std::any>&,
                            std::unordered_map<std::string, std::any>&,
                            std::unordered_map<std::string, std::any>& outputMap,
                            std::shared_ptr<Soapy<D, T>>& module) {
    using Module = Soapy<D, T>;

    Module::Config config{};
    Module::Input input{};

    module = std::make_shared<Module>(config, input);

    JST_CHECK(Module::RegisterVariable(outputMap, "output", module->getOutputBuffer()));

    return Result::SUCCESS;
}

}  // namespace Jetstream
