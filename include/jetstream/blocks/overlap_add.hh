#ifndef JETSTREAM_BLOCK_OVERLAP_ADD_BASE_HH
#define JETSTREAM_BLOCK_OVERLAP_ADD_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/overlap_add.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class OverlapAdd : public Block {
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
        Tensor<D, IT> overlap;

        JST_SERDES(buffer, overlap);
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
        return "overlap-add";
    }

    std::string name() const {
        return "Overlap Add";
    }

    std::string summary() const {
        return "Sums overlap with buffer.";
    }

    std::string description() const {
        return "Implements the overlap-add method, combining overlapping segments of a signal by adding their common parts.\n\n"
               "The Overlap-Add block performs a key operation in block-based signal processing, particularly for efficient "
               "FFT-based filtering. It combines consecutive data blocks by adding their overlapping regions, which is essential "
               "for reconstructing continuous output signals when processing is done in segments.\n\n"
               "Inputs:\n"
               "- buffer: Main input tensor containing the current data segment.\n"
               "- overlap: Tensor containing the overlap data from previous segments.\n"
               "  - Both tensors should have the same data type and compatible shapes.\n\n"
               "Configuration Parameters:\n"
               "- axis: The dimension along which to perform the overlap-add operation (default is the last dimension).\n\n"
               "Outputs:\n"
               "- buffer: Output tensor containing the combined data.\n"
               "- overlap: Output tensor containing the portion of the current input that will overlap with the next segment.\n\n"
               "Mathematical Operation:\n"
               "- Splits the input buffer into two parts: the part that overlaps with previous data and the part that will overlap with future data\n"
               "- Adds the overlapping part to the stored overlap from the previous iteration\n"
               "- Outputs the combined result and stores the future overlap portion\n\n"
               "Key Applications:\n"
               "- Efficient FFT-based filtering using the overlap-add method\n"
               "- Continuous processing of streaming data in block-based algorithms\n"
               "- Convolution operations with long impulse responses\n"
               "- Real-time audio processing\n"
               "- Efficient frequency-domain operations on large datasets\n\n"
               "Technical Details:\n"
               "- The overlap size is determined by the size of the overlap input tensor\n"
               "- State is maintained between processing iterations to ensure continuity\n"
               "- Critical for avoiding edge artifacts in block-wise signal processing\n\n"
               "Usage Notes:\n"
               "- Typically used in combination with Pad, FFT, and Unpad blocks for efficient filtering\n"
               "- The overlap size should match the filter impulse response length minus one\n"
               "- Proper synchronization of processing blocks is essential for correct operation";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            overlap_add, "overlap_add", {
                .axis = config.axis,
            }, {
                .buffer = input.buffer,
                .overlap = input.overlap,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, overlap_add->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(overlap_add->locale()));

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
        if (ImGui::InputFloat("##overlap-axis", &axis, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
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
    std::shared_ptr<Jetstream::OverlapAdd<D, IT>> overlap_add;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(OverlapAdd, is_specialized<Jetstream::OverlapAdd<D, IT>>::value &&
                             std::is_same<OT, void>::value)

#endif
