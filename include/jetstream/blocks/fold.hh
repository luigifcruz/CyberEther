#ifndef JETSTREAM_BLOCK_FOLD_BASE_HH
#define JETSTREAM_BLOCK_FOLD_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/fold.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Fold : public Block {
 public:
    // Configuration

    struct Config {
        U64 axis = 0;
        U64 offset = 0;
        U64 size = 0;

        JST_SERDES(axis, offset, size);
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
        return "fold";
    }

    std::string name() const {
        return "Fold";
    }

    std::string summary() const {
        return "Folds the input signal.";
    }

    std::string description() const {
        return "Reshapes a tensor by folding one dimension into multiple dimensions, transforming a flat array into a multi-dimensional structure.\n\n"
               "The Fold block performs a dimensional restructuring operation that transforms a single dimension of a tensor "
               "into multiple dimensions. This operation is particularly useful for converting one-dimensional signals into "
               "multi-dimensional forms for processing with algorithms that expect structured data, such as image or matrix operations.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor to be folded.\n"
               "  - Can be any supported data type and shape.\n\n"
               "Configuration Parameters:\n"
               "- axis: The dimension to fold (split into multiple dimensions).\n"
               "- shape: The shape to fold the selected dimension into (must multiply to the original dimension size).\n\n"
               "Outputs:\n"
               "- buffer: Folded output tensor with higher dimensionality.\n"
               "  - Same data type as the input but with transformed dimensions.\n"
               "  - Total number of elements remains the same, but organized differently.\n\n"
               "Mathematical Operation:\n"
               "- Takes a single dimension and restructures it into multiple dimensions\n"
               "- The product of the new dimensions must equal the original dimension size\n"
               "- Data ordering is preserved, just viewed with a different dimensional structure\n\n"
               "Key Applications:\n"
               "- Converting time-series data to 2D matrices for image-based processing\n"
               "- Restructuring data for efficient batch processing\n"
               "- Preparing data for multi-dimensional FFT operations\n"
               "- Signal segmentation for frame-based processing\n"
               "- Implementation of stride-based operations\n\n"
               "Technical Details:\n"
               "- Doesn't involve copying data, just changes the view of the data\n"
               "- The resulting tensor may have non-standard strides\n"
               "- The original dimension is replaced with the new dimensions in the same position\n\n"
               "Usage Notes:\n"
               "- Often paired with the Unfold block to restore the original structure\n"
               "- The specified shape must multiply to exactly match the original dimension size\n"
               "- Useful for implementing frame-based operations on streaming signals";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            fold, "fold", {
                .axis = config.axis,
                .offset = config.offset,
                .size = config.size,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, fold->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(fold->locale()));

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
            if (axis >= 0 and axis < input.buffer.rank()) {
                config.axis = static_cast<U64>(axis);

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Offset");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 offset = config.offset;
        if (ImGui::InputFloat("##offset", &offset, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (offset >= 0 and offset <= input.buffer.shape()[config.axis]) {
                config.offset = static_cast<U64>(offset);

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Size");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 size = config.size;
        if (ImGui::InputFloat("##size", &size, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (size >= 0 and size <= input.buffer.shape()[config.axis]) {
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
    std::shared_ptr<Jetstream::Fold<D, IT>> fold;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Fold, is_specialized<Jetstream::Fold<D, IT>>::value &&
                       std::is_same<OT, void>::value)

#endif
