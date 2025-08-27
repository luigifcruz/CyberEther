#ifndef JETSTREAM_BLOCK_PAD_BASE_HH
#define JETSTREAM_BLOCK_PAD_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/pad.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Pad : public Block {
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
        Tensor<D, IT> unpadded;

        JST_SERDES(unpadded);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> padded;

        JST_SERDES(padded);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputPadded() const {
        return this->output.padded;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "pad";
    }

    std::string name() const {
        return "Pad";
    }

    std::string summary() const {
        return "Adds zeros to the end of a tensor.";
    }

    std::string description() const {
        return "Adds zero-padding to the end of a tensor along a specified axis, increasing its size.\n\n"
               "The Pad block extends a tensor by adding zeros to the end of a specified dimension. This operation "
               "is essential for many signal processing algorithms, particularly those involving convolution, filtering, "
               "and frequency-domain processing, where specific tensor sizes are often required.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor to be padded.\n"
               "  - Can be any supported data type and shape.\n\n"
               "Configuration Parameters:\n"
               "- axis: The dimension along which to add padding (default is the last dimension).\n"
               "- padding: The number of zero elements to add along the specified axis.\n\n"
               "Outputs:\n"
               "- buffer: Padded output tensor.\n"
               "  - Same data type as the input.\n"
               "  - All dimensions are the same as input except the padded dimension, which is increased by the padding amount.\n\n"
               "Mathematical Operation:\n"
               "- Preserves all original values from the input tensor\n"
               "- Adds zeros to the end of the specified dimension\n"
               "- Result size along padded axis = original_size + padding\n\n"
               "Key Applications:\n"
               "- Preparing data for FFT operations (power-of-two sizing)\n"
               "- Signal processing filter operations\n"
               "- Zero-padding for convolution operations\n"
               "- Preparing data for batch processing\n"
               "- Creating guard bands for overlap-add method\n\n"
               "Usage Notes:\n"
               "- Often used in combination with the Unpad block to restore original dimensions\n"
               "- Commonly paired with the Overlap-Add block for efficient filtering operations\n"
               "- For multi-stage processing, padding should be carefully calculated to ensure correct results";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            pad, "pad", {
                .size = config.size,
                .axis = config.axis,
            }, {
                .unpadded = input.unpadded,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("padded", output.padded, pad->getOutputPadded()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(pad->locale()));

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
            if (axis >= 0 and axis < input.unpadded.rank()) {
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
    std::shared_ptr<Jetstream::Pad<D, IT>> pad;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Pad, is_specialized<Jetstream::Pad<D, IT>>::value &&
                      std::is_same<OT, void>::value)

#endif
