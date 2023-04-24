#ifndef JETSTREAM_MODULES_MULTIPLY_HH
#define JETSTREAM_MODULES_MULTIPLY_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/graph/base.hh"

namespace Jetstream {

template<Device D, typename T = CF32>
class Multiply : public Module, public Compute {
 public:
    struct Config {
    };

    struct Input {
        const Vector<D, T, 2>& factorA;
        const Vector<D, T, 2>& factorB;
    };

    struct Output {
        Vector<D, T, 2> product;
    };

    explicit Multiply(const Config& config,
                      const Input& input);

    constexpr const Device device() const {
        return D;
    }

    const std::string name() const {
        return "Multiply";
    }

    void summary() const final;

    constexpr const Vector<D, T, 2>& getProductBuffer() const {
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
