#ifndef JETSTREAM_MODULES_MULTIPLY_HH
#define JETSTREAM_MODULES_MULTIPLY_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

template<Device D, typename T = CF32>
class Multiply : public Module, public Compute {
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
        Vector<D, T, 2> factorA;
        Vector<D, T, 2> factorB;

        JST_SERDES(
            JST_SERDES_VAL("factorA", factorA);
            JST_SERDES_VAL("factorB", factorB);
        );
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Vector<D, T, 2> product;

        JST_SERDES(
            JST_SERDES_VAL("product", product);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Vector<D, T, 2>& getOutputProduct() const {
        return this->output.product;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "multiply";
    }

    std::string_view prettyName() const {
        return "Multiply";
    }

    void summary() const final;

    // Constructor

    Result create();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct {
        MTL::ComputePipelineState* state;
    } metal;
#endif

    JST_DEFINE_MODULE_IO();
};

}  // namespace Jetstream

#endif
