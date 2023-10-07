#ifndef JETSTREAM_MODULES_AUDIO_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

#include "jetstream/tools/miniaudio.h"

namespace Jetstream {

template<Device D, typename T = F32>
class Audio : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        F32 inSampleRate = 48e3;
        F32 outSampleRate = 48e3;

        JST_SERDES(
            JST_SERDES_VAL("inSampleRate", inSampleRate);
            JST_SERDES_VAL("outSampleRate", outSampleRate);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Vector<D, T, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Vector<D, T, 1> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Vector<D, T, 1>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "audio";
    }

    std::string_view prettyName() const {
        return "Audio";
    }

    void summary() const final;

    // Constructor

    Result create();
    Result destroy();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
    ma_device_config deviceConfig;
    ma_device deviceCtx;
    ma_resampler_config resamplerConfig;
    ma_resampler resamplerCtx;

    Memory::CircularBuffer<F32> buffer;

    static void callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);

    JST_DEFINE_IO();
};

}  // namespace Jetstream

#endif
