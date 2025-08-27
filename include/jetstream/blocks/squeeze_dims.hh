#ifndef JETSTREAM_BLOCK_SQUEEZE_DIMS_BASE_HH
#define JETSTREAM_BLOCK_SQUEEZE_DIMS_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/tensor_modifier.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class SqueezeDims : public Block {
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
        return "squeeze-dims";
    }

    std::string name() const {
        return "Squeeze Dims";
    }

    std::string summary() const {
        return "Squeezes the dimensions of a tensor.";
    }

    std::string description() const {
        return "Removes dimensions of size 1 from a tensor's shape, similar to NumPy's squeeze() function.\n\n"
               "The Squeeze Dims block simplifies tensor shapes by eliminating singleton dimensions (dimensions with size 1), "
               "which often result from operations that preserve dimensionality but reduce elements along certain axes. "
               "This operation is purely a shape transformation and doesn't change the actual data content or total element count.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor with potential singleton dimensions to remove.\n"
               "  - Can be any supported data type and shape.\n\n"
               "Configuration Parameters:\n"
               "- axes: Optional list of specific axes to squeeze (default: all axes with size 1).\n"
               "  - If specified, only the listed dimensions with size 1 will be removed.\n"
               "  - If omitted, all dimensions with size 1 will be removed.\n\n"
               "Outputs:\n"
               "- buffer: Output tensor with singleton dimensions removed.\n"
               "  - Same data type and total element count as the input.\n"
               "  - Shape has fewer dimensions if any singleton dimensions were removed.\n\n"
               "Operation Behavior:\n"
               "- Identifies all dimensions with size 1 (or only those specified in axes)\n"
               "- Removes these dimensions from the shape metadata\n"
               "- Preserves all actual data values and their order\n"
               "- If no dimensions can be squeezed, returns a tensor with the same shape\n\n"
               "Key Applications:\n"
               "- Simplifying tensor shapes after operations that create singleton dimensions\n"
               "- Preparing data for blocks that expect specific dimension counts\n"
               "- Cleaning up after expand_dims operations\n"
               "- Converting between column/row vectors and 1D arrays\n"
               "- Standardizing data structures in a processing pipeline\n\n"
               "Complementary Operations:\n"
               "- Expand Dims: Adds singleton dimensions (inverse operation to Squeeze Dims)\n"
               "- Reshape: More general shape transformation that can also remove singleton dimensions\n\n"
               "Usage Notes:\n"
               "- This operation modifies only the shape metadata, not the underlying data\n"
               "- The resulting tensor will be contiguous in memory\n"
               "- For tensors with no singleton dimensions, this block has no effect";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            modifier, "modifier", {
                .callback = [&](auto& mod) {
                    if (mod.shape()[config.axis] != 1) {
                        JST_ERROR("Cannot squeeze axis '{}' because it is not '1'.", config.axis);
                        return Result::ERROR;
                    }
                    mod.squeeze_dims(config.axis);
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
        if (ImGui::InputFloat("##squeeze-dims-axis", &axis, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
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

JST_BLOCK_ENABLE(SqueezeDims, !std::is_same<IT, void>::value &&
                               std::is_same<OT, void>::value)

#endif
