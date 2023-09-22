#ifndef JETSTREAM_MODULES_SOAPY_HH
#define JETSTREAM_MODULES_SOAPY_HH

#include <thread>

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Modules.hpp>

namespace Jetstream {

template<Device D, typename T = CF32>
class Soapy : public Module, public Compute {
 public:
    // Configuration

    struct Config {
        std::string deviceString;
        std::string streamString = "";
        F32 frequency;
        F32 sampleRate;
        VectorShape<2> outputShape;
        U64 bufferMultiplier = 4;

        JST_SERDES(
            JST_SERDES_VAL("deviceString", deviceString);
            JST_SERDES_VAL("streamString", streamString);
            JST_SERDES_VAL("frequency", frequency);
            JST_SERDES_VAL("sampleRate", sampleRate);
            JST_SERDES_VAL("outputShape", outputShape);
            JST_SERDES_VAL("bufferMultiplier", bufferMultiplier);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Vector<D, T, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Vector<D, T, 2>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "soapy";
    }

    std::string_view prettyName() const {
        return "Soapy";
    }

    void summary() const final;

    // Constructor

    Result create();
    Result destroy();

    // Miscellaneous

    constexpr Memory::CircularBuffer<T>& getCircularBuffer() {
        return buffer;
    }

    constexpr const std::string& getDeviceName() const {
        return deviceName;
    }

    constexpr const std::string& getDeviceHardwareKey() const {
        return deviceHardwareKey;
    }

    F32 setTunerFrequency(const F32& frequency);

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;
    Result computeReady() final;

 private:
    std::thread producer;
    bool streaming = false;
    std::string deviceName;
    std::string deviceHardwareKey;
    Memory::CircularBuffer<T> buffer;

    SoapySDR::Device* soapyDevice;
    SoapySDR::Stream* soapyStream;

    void soapyThreadLoop();

    JST_DEFINE_MODULE_IO();
};

}  // namespace Jetstream

#endif
