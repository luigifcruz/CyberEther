#ifndef JETSTREAM_MODULES_TENSOR_MODIFIER_HH
#define JETSTREAM_MODULES_TENSOR_MODIFIER_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory2/tensor.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

template<Device D, typename T>
class TensorModifier : public Module, public Compute {
 public:
    // Configuration

    struct Config {
        std::function<Result(mem2::Tensor&)> callback;

        JST_SERDES();
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

    constexpr Taint taint() const {
        return Taint::DISCONTIGUOUS;
    }

    void info() const final {
        JST_DEBUG("  None");
    }

    // Constructor

    Result create() {
        JST_DEBUG("Initializing Tensor Modifier module.");
        JST_INIT_IO();

        auto outputCandidate = input.buffer;
        JST_CHECK(config.callback(outputCandidate));
        output.buffer = outputCandidate;

        return Result::SUCCESS;
    }

 protected:
    Result compute(const Context&) final {
        return Result::SUCCESS;
    }

 private:
    JST_DEFINE_IO()
};

}  // namespace Jetstream

#endif
