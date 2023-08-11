#ifndef JETSTREAM_MODULES_AUDIO_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

#include "jetstream/tools/miniaudio.h"

namespace Jetstream {

template<Device D, typename T = CF32>
class Audio : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        F32 sampleRate = 48e3;

        JST_SERDES(
            JST_SERDES_VAL("sampleRate", sampleRate);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        const Vector<D, T, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        JST_SERDES();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    constexpr std::string name() const {
        return "audio";
    }

    constexpr std::string prettyName() const {
        return "Audio";
    }

    void summary() const final;

    // Constructor

    explicit Audio(const Config& config, const Input& input);
    ~Audio();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
    const Config config;
    const Input input;
    Output output;

    ma_device_config deviceConfig;
    ma_device deviceCtx;
    Memory::CircularBuffer<F32> buffer;

    static void callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);
};

}  // namespace Jetstream

#endif
