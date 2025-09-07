#ifndef JETSTREAM_BLOCK_DECIMATOR_BASE_HH
#define JETSTREAM_BLOCK_DECIMATOR_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/arithmetic.hh"
#include "jetstream/modules/tensor_modifier.hh"
#include "jetstream/modules/duplicate.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Decimator : public Block {
 public:
    // Configuration

    struct Config {
        U64 axis = 1;

        JST_SERDES(axis);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "decimator";
    }

    std::string name() const {
        return "Decimator";
    }

    std::string summary() const {
        return "Decimates a signal by summing along an axis and slicing.";
    }

    std::string description() const {
        return "The Decimator block reduces tensor dimensionality by summing elements along a specified axis, "
               "then extracting the first element of the reduced dimension. This effectively collapses "
               "multi-dimensional data into lower-dimensional output while preserving accumulated values.\n\n"

               "## Parameters\n"
               "- **Axis**: The axis along which to sum and slice the input tensor.\n\n"

               "## Useful For:\n"
               "- Implementing simple decimation filters for signal processing.\n"
               "- Aggregating sensor data from multiple sources.\n\n"

               "## Examples:\n"
               "- Time-domain decimation:\n"
               "  Config: Axis=1\n"
               "  Input: CF32[8192, 128] → Output: CF32[8192]\n\n"

               "## Implementation:\n"
               "Input → Add Axis → Squeeze Axis → Duplicate → Output\n"
               "1. Arithmetic module sums all elements along the specified axis.\n"
               "2. Tensor modifier slices the result to extract index 0 from the reduced dimension.\n"
               "3. Duplicate module ensures proper output buffering and host accessibility.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            arithmetic, "arithmetic", {
                .operation = ArithmeticOp::Add,
                .axis = config.axis,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            slicer, "slicer", {
                .callback = [&](auto& mod) {
                    mod.squeeze_dims(config.axis);
                    return Result::SUCCESS;
                }
            }, {
                .buffer = arithmetic->getOutputBuffer(),
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            duplicate, "duplicate", {
                .hostAccessible = true,
            }, {
                .buffer = slicer->getOutputBuffer(),
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, duplicate->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        // Destroy modules in reverse order of creation with guards
        if (duplicate) {
            JST_CHECK(instance().eraseModule(duplicate->locale()));
        }

        if (slicer) {
            JST_CHECK(instance().eraseModule(slicer->locale()));
        }

        if (arithmetic) {
            JST_CHECK(instance().eraseModule(arithmetic->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Axis");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 axis = config.axis;
        if (ImGui::InputFloat("##axis", &axis, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (axis >= 0 && axis < input.buffer.rank()) {
                config.axis = static_cast<U64>(axis);

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Arithmetic<D, IT>> arithmetic;
    std::shared_ptr<Jetstream::TensorModifier<D, IT>> slicer;
    std::shared_ptr<Jetstream::Duplicate<D, IT>> duplicate;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Decimator, is_specialized<Jetstream::Arithmetic<D, IT>>::value &&
                            is_specialized<Jetstream::Duplicate<D, IT>>::value &&
                            std::is_same<OT, void>::value)

#endif
