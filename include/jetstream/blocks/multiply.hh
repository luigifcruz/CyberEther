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
        Tensor<D, IT> factorA;
        Tensor<D, IT> factorB;

        JST_SERDES(factorA, factorB);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> product;

        JST_SERDES(product);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputProduct() const {
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
        return "Performs element-wise multiplication of two input tensors, producing an output tensor with the product values.\n\n"
               "The Multiply block takes two input tensors and multiplies them together element by element, similar to "
               "the Hadamard product in linear algebra. This operation is fundamental for many signal processing tasks, "
               "including modulation, windowing, and custom gain adjustments.\n\n"
               "Inputs:\n"
               "- factorA: First input tensor to be multiplied.\n"
               "- factorB: Second input tensor to be multiplied.\n"
               "  - Both tensors must have compatible shapes for element-wise operations.\n"
               "  - Supported types include real (F32) and complex (CF32) tensors.\n"
               "  - Broadcasting is supported for tensors of different shapes.\n\n"
               "Outputs:\n"
               "- product: Output tensor containing the element-wise product of the inputs.\n"
               "  - The data type follows standard multiplication rules:\n"
               "    - F32 × F32 → F32\n"
               "    - F32 × CF32 → CF32\n"
               "    - CF32 × CF32 → CF32\n\n"
               "Mathematical Operation:\n"
               "- For real values: product[i] = factorA[i] × factorB[i]\n"
               "- For complex values: follows complex multiplication rules\n"
               "- When shapes differ, the smaller tensor is broadcast to match the larger one\n\n"
               "Key Applications:\n"
               "- Signal mixing and modulation\n"
               "- Applying window functions to signals\n"
               "- Gain control using variable factors\n"
               "- Point-wise signal masking\n"
               "- Implementing custom signal operations\n\n"
               "Performance Notes:\n"
               "- Hardware accelerated on supported platforms\n"
               "- More efficient when inputs have the same shape and stride\n"
               "- Broadcasting may introduce additional overhead";
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
        JST_CHECK(instance().eraseModule(multiply->locale()));

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
