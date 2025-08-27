#ifndef JETSTREAM_BLOCK_TAKE_BASE_HH
#define JETSTREAM_BLOCK_TAKE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/take.hh"
#include "jetstream/modules/tensor_modifier.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Take : public Block {
 public:
    // Configuration

    struct Config {
        U64 index = 0;
        U64 axis = 0;

        JST_SERDES(index, axis);
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
        return "take";
    }

    std::string name() const {
        return "Take";
    }

    std::string summary() const {
        return "Takes a slice of the input tensor.";
    }

    std::string description() const {
        return "Extracts elements from a tensor along a specified axis according to provided indices, similar to NumPy's take() function.\n\n"
               "The Take block selects specific elements from the input tensor based on an array of indices along a given axis. "
               "This allows for advanced indexing operations beyond simple slicing, enabling non-contiguous element selection, "
               "reordering, and custom extraction patterns for specialized processing needs.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor from which to extract elements.\n"
               "  - Can be any supported data type and shape.\n\n"
               "Configuration Parameters:\n"
               "- indices: The list of indices to extract along the specified axis.\n"
               "  - Specified as a comma-separated list of integers.\n"
               "  - Negative indices are supported and count from the end of the dimension.\n"
               "- axis: The dimension along which to take elements (default is 0).\n"
               "  - If negative, counts from the end of the dimensions list.\n\n"
               "Outputs:\n"
               "- buffer: Output tensor containing the selected elements.\n"
               "  - Same data type as the input.\n"
               "  - Shape is the same as input except along the specified axis, which becomes the length of the indices list.\n\n"
               "Operation Behavior:\n"
               "- For each index in the indices list, extracts the corresponding slice along the specified axis\n"
               "- Indices can be repeated, allowing duplication of elements\n"
               "- Indices can be in any order, enabling custom reordering operations\n"
               "- Bounds checking ensures all indices are valid\n\n"
               "Key Applications:\n"
               "- Custom decimation or downsampling patterns\n"
               "- Reordering data elements (e.g., bit-reversed ordering for FFT)\n"
               "- Extracting specific channels from multi-channel data\n"
               "- Implementing permutation operations\n"
               "- Creating custom lookup patterns\n\n"
               "Differences from Slice Block:\n"
               "- Take allows non-contiguous and repeated selection of elements\n"
               "- Take uses explicit indices rather than start/stop/step parameters\n"
               "- Take enables more flexible selection patterns\n\n"
               "Usage Example:\n"
               "- With indices=[0,2,4] and axis=0, takes the 1st, 3rd, and 5th elements along axis 0\n"
               "- For a 2D tensor, this would extract the 1st, 3rd, and 5th rows";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            take, "take", {
                .index = config.index,
                .axis = config.axis,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            squeezeDims, "squeeze-dims", {
                .callback = [&](auto& mod) {
                    mod.squeeze_dims(config.axis);
                    return Result::SUCCESS;
                }
            }, {
                .buffer = take->getOutputBuffer(),
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, squeezeDims->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(squeezeDims->locale()));
        JST_CHECK(instance().eraseModule(take->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Index");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 index = config.index;
        if (ImGui::InputFloat("##index", &index, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (index >= 0 and index < input.buffer.shape()[config.axis]) {
                config.index = static_cast<U64>(index);

                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }

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
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Take<D, IT>> take;
    std::shared_ptr<Jetstream::TensorModifier<D, IT>> squeezeDims;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Take, is_specialized<Jetstream::Take<D, IT>>::value &&
                       std::is_same<OT, void>::value)

#endif
