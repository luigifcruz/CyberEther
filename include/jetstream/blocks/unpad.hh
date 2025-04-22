#ifndef JETSTREAM_BLOCK_UNPAD_BASE_HH
#define JETSTREAM_BLOCK_UNPAD_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/unpad.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Unpad : public Block {
 public:
    // Configuration

    struct Config {
        U64 size = 0;
        U64 axis = 0;

        JST_SERDES(size, axis);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> padded;

        JST_SERDES(padded);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> unpadded;
        Tensor<D, IT> pad;

        JST_SERDES(unpadded, pad);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputUnpadded() const {
        return this->output.unpadded;
    }

    constexpr const Tensor<D, IT>& getOutputPad() const {
        return this->output.pad;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "unpad";
    }

    std::string name() const {
        return "Unpad";
    }

    std::string summary() const {
        return "Removes padding from a tensor.";
    }

    std::string description() const {
        return "Removes elements from the end of a tensor along a specified axis, reducing its size.\n\n"
               "The Unpad block truncates a tensor by removing a specified number of elements from the end of a "
               "designated dimension. This block is the complementary operation to the Pad block and is essential "
               "for restoring original data dimensions after processing operations that require padding.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor to be unpadded (truncated).\n"
               "  - Can be any supported data type and shape.\n\n"
               "Configuration Parameters:\n"
               "- axis: The dimension along which to remove elements (default is the last dimension).\n"
               "- padding: The number of elements to remove from the end of the specified axis.\n\n"
               "Outputs:\n"
               "- buffer: Unpadded (truncated) output tensor.\n"
               "  - Same data type as the input.\n"
               "  - All dimensions are the same as input except the unpadded dimension, which is reduced by the padding amount.\n\n"
               "Mathematical Operation:\n"
               "- Extracts a subset of the original tensor\n"
               "- Discards specified number of elements from the end of the chosen dimension\n"
               "- Result size along unpadded axis = original_size - padding\n\n"
               "Key Applications:\n"
               "- Extracting useful data after FFT-based filtering operations\n"
               "- Removing guard bands after overlap-add processing\n"
               "- Restoring original signal dimensions after padded operations\n"
               "- Extracting only the valid portion of convolution results\n"
               "- Discarding algorithmic artifacts from processing operations\n\n"
               "Usage Notes:\n"
               "- Typically used after a Pad block to restore the original tensor dimensions\n"
               "- Critical for maintaining consistent data sizes in multi-stage processing\n"
               "- Care must be taken to ensure the padding value doesn't exceed the dimension size\n"
               "- Often used in filter implementations based on the overlap-add method";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            unpad, "unpad", {
                .size = config.size,
                .axis = config.axis,
            }, {
                .padded = input.padded,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("unpadded", output.unpadded, unpad->getOutputUnpadded()));
        JST_CHECK(Block::LinkOutput("pad", output.pad, unpad->getOutputPad()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(unpad->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Pad Axis");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 axis = config.axis;
        if (ImGui::InputFloat("##pad-axis", &axis, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (axis >= 0 and axis < input.padded.rank()) {
                config.axis = static_cast<U64>(axis);

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Pad Size");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 size = config.size;
        if (ImGui::InputFloat("##pad-size", &size, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (size >= 0) {
                config.size = static_cast<U64>(size);

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
    std::shared_ptr<Jetstream::Unpad<D, IT>> unpad;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Unpad, is_specialized<Jetstream::Unpad<D, IT>>::value &&
                        std::is_same<OT, void>::value)

#endif
