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
        U64 size;
    };

    struct Input {
        const Vector<D, IT>& buffer;
    };

    struct Output {
        Vector<D, OT> buffer;
    };

    explicit Amplitude(const Config& config,
                       const Input& input);

    constexpr const Device device() const {
        return D;
    }

    constexpr const Taint taints() const {
        return Taint::None;
    }

    void summary() const final;

    constexpr const U64 getBufferSize() const {
        return this->config.size;
    }

    constexpr const Vector<D, OT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr const Config getConfig() const {
        return config;
    }

 protected:
    const Result createCompute(const RuntimeMetadata& meta) final;
    const Result compute(const RuntimeMetadata& meta) final;

 private:
    const Config config;
    const Input input;
    Output output;

#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct MetalConstants {
        float scalingSize;
    };

    struct {
        MTL::ComputePipelineState* state;
        Vector<Device::Metal, U8> constants;
    } metal;
#endif
};

}  // namespace Jetstream

#endif
