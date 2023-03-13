#ifndef JETSTREAM_MODULES_MULTIPLY_HH
#define JETSTREAM_MODULES_MULTIPLY_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

template<Device D, typename T = CF32>
class Multiply : public Module, public Compute {
 public:
    struct Config {
        U64 size;
    };

    struct Input {
        // TODO: Change this to Device::Metal. 
        const Vector<Device::CPU, T>& factorA;
        const Vector<Device::CPU, T>& factorB;
    };

    struct Output {
        // TODO: Change this to Device::Metal. 
        Vector<Device::CPU, T> product;
    };

    explicit Multiply(const Config& config,
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

    // TODO: Change this to Device::Metal. 
    constexpr const Vector<Device::CPU, T>& getProductBuffer() const {
        return this->output.product;
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
    struct {
        MTL::ComputePipelineState* state;
    } metal;
#endif
};

}  // namespace Jetstream

#endif
