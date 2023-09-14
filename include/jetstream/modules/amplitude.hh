#ifndef JETSTREAM_MODULES_AMPLITUDE_HH
#define JETSTREAM_MODULES_AMPLITUDE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

template<Device D, typename IT = CF32, typename OT = F32>
class Amplitude : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        JST_SERDES();
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Vector<D, IT, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Vector<D, OT, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Vector<D, OT, 2>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    constexpr std::string name() const {
        return "amplitude";
    }

    constexpr std::string prettyName() const {
        return "Amplitude";
    }

    void summary() const final; 

    // Constructor

    Result create();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct MetalConstants {
        F32 scalingSize;
    };

    struct {
        MTL::ComputePipelineState* state;
        Vector<Device::Metal, U8> constants;
    } metal;
#endif

    JST_DEFINE_MODULE_IO();
};

}  // namespace Jetstream

#endif
