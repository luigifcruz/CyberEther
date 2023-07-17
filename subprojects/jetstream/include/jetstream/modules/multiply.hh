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
        const Vector<D, T, 2> factorA;
        const Vector<D, T, 2> factorB;
    };

    struct Output {
        Vector<D, T, 2> product;
    };

    explicit Multiply(const Config& config,
                      const Input& input);

    constexpr Device device() const {
        return D;
    }

    const std::string name() const {
        return "Multiply";
    }

    void summary() const final;

    constexpr const Vector<D, T, 2>& getProductBuffer() const {
        return this->output.product;
    }

    constexpr Config getConfig() const {
        return config;
    }

    static Result Factory(std::unordered_map<std::string, std::any>& config,
                          std::unordered_map<std::string, std::any>& input,
                          std::unordered_map<std::string, std::any>& output,
                          std::shared_ptr<Multiply<D, T>>& module,
                          const bool& castFromString = false);

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

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
