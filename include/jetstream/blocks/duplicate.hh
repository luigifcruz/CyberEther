#ifndef JETSTREAM_BLOCK_DUPLICATE_BASE_HH
#define JETSTREAM_BLOCK_DUPLICATE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/duplicate.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Duplicate : public Block {
 public:
    // Configuration

    struct Config {
        bool hostAccessible = true;

        JST_SERDES(hostAccessible);
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
        return "duplicate";
    }

    std::string name() const {
        return "Duplicate";
    }

    std::string summary() const {
        return "Copies the input signal.";
    }

    std::string description() const {
        return "Creates an exact copy of the input tensor, with optional memory management optimizations.\n\n"
               "The Duplicate block creates a complete copy of the input tensor in the output tensor. While seemingly "
               "simple, this block provides essential functionality for memory management, data transfer between devices, "
               "and ensuring contiguous memory layouts for optimal processing.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor to be duplicated.\n"
               "  - Can be any supported data type and shape.\n\n"
               "Outputs:\n"
               "- buffer: Output tensor containing an exact copy of the input.\n"
               "  - Same data type, shape, and values as the input tensor.\n"
               "  - Always in contiguous memory layout, even if the input was non-contiguous.\n\n"
               "Key Functions:\n"
               "- Basic duplication of data for parallel processing chains\n"
               "- Converting non-contiguous tensors to contiguous memory layout\n"
               "- Transferring data between different memory domains (CPU/GPU)\n"
               "- Creating independent copies that can be modified without affecting the original\n"
               "- Memory management optimization through the Host Accessible option\n\n"
               "Configuration Options:\n"
               "- Host Accessible: When enabled, ensures the output data is accessible from the host (CPU) memory.\n"
               "  - Useful for transferring data between host and device (GPU) memory.\n"
               "  - May involve additional memory transfers depending on the input location.\n\n"
               "Usage Scenarios:\n"
               "- Creating branches in a flowgraph where data needs to follow multiple paths\n"
               "- Ensuring data is in the optimal memory format before intensive processing\n"
               "- Moving data between different processing devices\n"
               "- Creating data snapshots that won't be affected by downstream modifications\n"
               "- Implementing efficient buffer management strategies\n\n"
               "Performance Considerations:\n"
               "- Involves a full memory copy, which can be expensive for large tensors\n"
               "- Host/device transfers may introduce latency\n"
               "- Creates additional memory usage proportional to the input size";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            duplicate, "duplicate", {
                .hostAccessible = config.hostAccessible,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, duplicate->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(duplicate->locale()));

        return Result::SUCCESS;
    }

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Host Accessible");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Checkbox("##HostAccessible", &config.hostAccessible)) {
            JST_DISPATCH_ASYNC([&](){ \
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." }); \
                JST_CHECK_NOTIFY(instance().reloadBlock(locale())); \
            });
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Duplicate<D, IT>> duplicate;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Duplicate, is_specialized<Jetstream::Duplicate<D, IT>>::value &&
                            std::is_same<OT, void>::value)

#endif
