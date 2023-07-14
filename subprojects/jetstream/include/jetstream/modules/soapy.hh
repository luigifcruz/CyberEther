#ifndef JETSTREAM_MODULES_SOAPY_HH
#define JETSTREAM_MODULES_SOAPY_HH

#include <thread>

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/graph/base.hh"

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Modules.hpp>

namespace Jetstream {

template<Device D, typename T = CF32>
class Soapy : public Module, public Compute {
 public:
    struct Config {
        std::string deviceString;
        F32 frequency;
        F32 sampleRate;
        U64 batchSize;
        U64 outputBufferSize;
        U64 bufferMultiplier = 4;
    };

    struct Input {
    };

    struct Output {
        Vector<D, T, 2> buffer;
    };

    explicit Soapy(const Config& config,
                   const Input& input);
    ~Soapy();

    constexpr Device device() const {
        return D;
    }

    const std::string name() const {
        return "Soapy";
    }

    void summary() const final;

    constexpr const Vector<D, T, 2>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr Config getConfig() const {
        return config;
    }

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

    static Result Factory(std::unordered_map<std::string, std::any>& config,
                          std::unordered_map<std::string, std::any>& input,
                          std::unordered_map<std::string, std::any>& output,
                          std::shared_ptr<Soapy<D, T>>& module);

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
    Config config;
    const Input input;
    Output output;

    std::thread producer;
    bool streaming = false;
    std::string deviceName;
    std::string deviceHardwareKey;
    Memory::CircularBuffer<T> buffer;

    SoapySDR::Device* soapyDevice;
    SoapySDR::Stream* soapyStream;

    void soapyThreadLoop();
};

}  // namespace Jetstream

#endif
