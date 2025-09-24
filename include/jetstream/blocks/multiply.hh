#ifndef JETSTREAM_BLOCK_MULTIPLY_BASE_HH
#define JETSTREAM_BLOCK_MULTIPLY_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/multiply.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Multiply : public Block {
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
        mem2::Tensor factorA;
        mem2::Tensor factorB;

        JST_SERDES(factorA, factorB);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor product;

        JST_SERDES(product);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputProduct() const {
        return this->output.product;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "multiply";
    }

    std::string name() const {
        return "Multiply";
    }

    std::string summary() const {
        return "Element-wise multiplication.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Takes 'factorA' and 'factorB' as inputs and outputs the result as 'product'.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            multiply, "multiply", {}, {
                .factorA = input.factorA,
                .factorB = input.factorB,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("product", output.product, multiply->getOutputProduct()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (multiply) {
            JST_CHECK(instance().eraseModule(multiply->locale()));
        }

        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<Jetstream::Multiply<D, IT>> multiply;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Multiply, is_specialized<Jetstream::Multiply<D, IT>>::value &&
                           std::is_same<OT, void>::value)

#endif
