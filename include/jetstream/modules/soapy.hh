#ifndef JETSTREAM_MODULES_SOAPY_HH
#define JETSTREAM_MODULES_SOAPY_HH

#include <thread>

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory2/tensor.hh"
#include "jetstream/memory/utils/circular_buffer.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_SOAPY_CPU(MACRO) \
    MACRO(Soapy, CPU, CF32)

template<Device D, typename T = CF32>
class Soapy : public Module, public Compute {
 public:
    Soapy();
    ~Soapy();

    // Types

    struct DeviceEntry : public std::map<std::string, std::string> {
     public:
        DeviceEntry() = default;
        DeviceEntry(const std::map<std::string, std::string>& m) : std::map<std::string, std::string>(m) {}

        std::string toString() const;
    };

    typedef std::map<std::string, DeviceEntry> DeviceList;

    // Configuration

    struct Config {
        std::string deviceString = "";
        std::string streamString = "";
        F32 frequency = 96.9e6;
        F32 sampleRate = 2.0e6;
        bool automaticGain = true;
        U64 numberOfBatches = 8;
        U64 numberOfTimeSamples = 8192;
        U64 bufferMultiplier = 4;

        JST_SERDES(deviceString, streamString,
                   frequency, sampleRate, automaticGain,
                   numberOfBatches, numberOfTimeSamples,
                   bufferMultiplier);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES_INPUT();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();
    Result destroy();

    // Miscellaneous

    Memory::CircularBuffer<T>& getCircularBuffer();
    const std::string& getDeviceName() const;
    const std::string& getDeviceHardwareKey() const;
    const std::string& getDeviceLabel() const;

    Result setTunerFrequency(F32& frequency);
    Result setSampleRate(F32& sampleRate);
    Result setAutomaticGain(bool& automaticGain);

    static DeviceList ListAvailableDevices(const std::string& filter = "");

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;
    Result computeReady() final;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE
JST_SOAPY_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
