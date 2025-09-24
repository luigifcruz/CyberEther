#ifndef JETSTREAM_MODULES_AUDIO_HH
#define JETSTREAM_MODULES_AUDIO_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory2/tensor.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_AUDIO_CPU(MACRO) \
    MACRO(Audio, CPU, F32) \

template<Device D, typename T = F32>
class Audio : public Module, public Compute {
 public:
    Audio();
    ~Audio();

    // Types

    typedef std::vector<std::string> DeviceList;

    // Configuration

    struct Config {
        std::string deviceName = "Default";
        F32 inSampleRate = 48e3;
        F32 outSampleRate = 48e3;

        JST_SERDES(deviceName, inSampleRate, outSampleRate);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        mem2::Tensor buffer;

        JST_SERDES_INPUT(buffer);
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

    const std::string& getDeviceName() const;

    static DeviceList ListAvailableDevices();

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_AUDIO_CPU_AVAILABLE
JST_AUDIO_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
