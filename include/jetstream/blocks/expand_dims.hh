#ifndef JETSTREAM_BLOCK_EXPAND_DIMS_BASE_HH
#define JETSTREAM_BLOCK_EXPAND_DIMS_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/tensor_modifier.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class ExpandDims : public Block {
 public:
    // Configuration

    struct Config {
        U64 axis = 0;

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
        return "expand-dims";
    }

    std::string name() const {
        return "Expand Dims";
    }

    std::string summary() const {
        return "Expands the dimensions of a tensor.";
    }

    std::string description() const {
        return "Inserts new singleton dimensions (dimensions of size 1) into a tensor's shape, similar to NumPy's expand_dims() function.\n\n"
               "The Expand Dims block increases the dimensionality of a tensor by adding new axes with size 1 at specified positions. "
               "This operation is useful when working with blocks that expect inputs with specific numbers of dimensions, or when "
               "preparing data for operations like broadcasting or batch processing.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor to expand with new dimensions.\n"
               "  - Can be any supported data type and shape.\n\n"
               "Configuration Parameters:\n"
               "- axes: The positions at which to insert new singleton dimensions.\n"
               "  - Specified as a comma-separated list of integers.\n"
               "  - Positive values insert before the specified position (0 inserts at the beginning).\n"
               "  - Negative values count from the end (-1 inserts before the last dimension).\n\n"
               "Outputs:\n"
               "- buffer: Output tensor with added singleton dimensions.\n"
               "  - Same data type and total element count as the input.\n"
               "  - Shape has more dimensions than the input, with size 1 at the specified positions.\n\n"
               "Operation Behavior:\n"
               "- Inserts new dimensions with size 1 at the specified positions\n"
               "- Preserves all actual data values and their order\n"
               "- The rank of the output tensor equals (input rank + number of axes specified)\n"
               "- Does not create a copy of the data, only modifies the shape metadata\n\n"
               "Key Applications:\n"
               "- Preparing tensors for broadcasting operations\n"
               "- Adding batch dimensions for batch processing\n"
               "- Creating channel dimensions for multi-channel processing\n"
               "- Converting vectors to row or column matrices\n"
               "- Adapting tensor shapes for blocks with specific dimension requirements\n\n"
               "Complementary Operations:\n"
               "- Squeeze Dims: Removes singleton dimensions (inverse operation to Expand Dims)\n"
               "- Reshape: More general shape transformation that can also add dimensions\n\n"
               "Usage Notes:\n"
               "- This operation only modifies the shape metadata, not the underlying data\n"
               "- Multiple dimensions can be added in a single operation\n"
               "- The axes parameter can contain multiple values to add dimensions at multiple positions";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            modifier, "modifier", {
                .callback = [&](auto& mod) {
                    mod.expand_dims(config.axis);
                    return Result::SUCCESS;
                }
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, modifier->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(modifier->locale()));

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
        if (ImGui::InputFloat("##expand-dims-axis", &axis, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (axis >= 0 and axis <= input.buffer.rank()) {
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
    std::shared_ptr<Jetstream::TensorModifier<D, IT>> modifier;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(ExpandDims, !std::is_same<IT, void>::value &&
                              std::is_same<OT, void>::value)

#endif
