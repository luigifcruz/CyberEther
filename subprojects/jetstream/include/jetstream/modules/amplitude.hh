#ifndef JETSTREAM_MODULES_AMPLITUDE_HH
#define JETSTREAM_MODULES_AMPLITUDE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/graph/base.hh"

namespace Jetstream {

template<Device D, typename IT = CF32, typename OT = F32>
class Amplitude : public Module, public Compute {
 public:
    struct Config {
    };

    struct Input {
        const Vector<D, IT, 2>& buffer;
    };

    struct Output {
        Vector<D, OT, 2> buffer;
    };

    explicit Amplitude(const Config& config,
                       const Input& input);

    constexpr Device device() const {
        return D;
    }

    const std::string name() const {
        return "Amplitude";
    }

    void summary() const final;

    constexpr Vector<D, OT, 2>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr Config getConfig() const {
        return config;
    }

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
    const Config config;
    const Input input;
    Output output;

#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct MetalConstants {
        F32 scalingSize;
    };

    struct {
        MTL::ComputePipelineState* state;
        Vector<Device::Metal, U8> constants;
    } metal;
#endif
};

}  // namespace Jetstream

#endif
