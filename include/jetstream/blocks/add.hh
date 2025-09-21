#ifndef JETSTREAM_BLOCK_ADD_BASE_HH
#define JETSTREAM_BLOCK_ADD_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/add.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Add : public Block {
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
        Tensor<D, IT> addendA;
        Tensor<D, IT> addendB;

        JST_SERDES(addendA, addendB);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> sum;

        JST_SERDES(sum);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputSum() const {
        return this->output.sum;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "add";
    }

    std::string name() const {
        return "Add";
    }

    std::string summary() const {
        return "Element-wise addition.";
    }

    std::string description() const {
        return "The Add block performs element-wise addition of two input tensors with automatic broadcasting support. "
               "It takes two tensors of potentially different shapes and produces their sum by broadcasting the smaller "
               "tensor to match the larger one's dimensions, following standard NumPy-style broadcasting rules.\n\n"

               "## Arguments\n"
               "- **addendA**: The first input tensor to be added.\n"
               "- **addendB**: The second input tensor to be added.\n\n"

               "## Useful For:\n"
               "- Combining signal components in digital signal processing applications.\n"
               "- Implementing offset corrections in sensor data processing.\n"
               "- Merging multiple data streams with element-wise summation.\n\n"

               "## Examples:\n"
               "- Vector addition:\n"
               "  Input A: F32[1024] + Input B: F32[1024] → Output: F32[1024]\n"
               "- Matrix addition with broadcasting:\n"
               "  Input A: CF32[512, 256] + Input B: CF32[1, 256] → Output: CF32[512, 256]\n\n"

               "## Implementation:\n"
               "Input A → Broadcast → Add → Output\n"
               "Input B → Broadcast ↗\n"
               "1. Input tensors are validated for broadcasting compatibility.\n"
               "2. Output tensor shape is determined using broadcasting rules.\n"
               "3. Input tensors are broadcast to match the output shape.\n"
               "4. Element-wise addition is performed using optimized CPU iterators.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            add, "add", {}, {
                .addendA = input.addendA,
                .addendB = input.addendB,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("sum", output.sum, add->getOutputSum()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (add) {
            JST_CHECK(instance().eraseModule(add->locale()));
        }

        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<Jetstream::Add<D, IT>> add;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Add, is_specialized<Jetstream::Add<D, IT>>::value &&
                      std::is_same<OT, void>::value)

#endif
