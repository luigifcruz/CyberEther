#ifndef JETSTREAM_BLOCK_RESHAPE_BASE_HH
#define JETSTREAM_BLOCK_RESHAPE_BASE_HH

#include <fcntl.h>
#include <regex>

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/tensor_modifier.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Reshape : public Block {
 public:
    // Configuration

    struct Config {
        std::string shape = "";

        JST_SERDES(shape);
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
        return "reshape";
    }

    std::string name() const {
        return "Reshape";
    }

    std::string summary() const {
        return "Reshapes a tensor to a new shape.";
    }

    std::string description() const {
        return "Transforms a tensor to a new shape while preserving the total number of elements.\n\n"
               "The Reshape block changes the dimensional structure of a tensor without altering its data content. "
               "This operation is analogous to NumPy's reshape function, allowing flexible restructuring of data "
               "for various algorithmic needs while maintaining the same total element count.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor to be reshaped.\n"
               "  - Can be any supported data type and shape.\n\n"
               "Configuration Parameters:\n"
               "- shape: The target shape for the output tensor.\n"
               "  - Must specify the size for each dimension, with one optional -1 dimension that will be automatically calculated.\n"
               "  - Total number of elements must match the input tensor.\n\n"
               "Outputs:\n"
               "- buffer: Output tensor with the new shape.\n"
               "  - Same data type and total element count as the input.\n"
               "  - Data values are preserved but arranged according to the new dimensions.\n\n"
               "Reshape Behavior:\n"
               "- Total number of elements must remain unchanged\n"
               "- Data is traversed in row-major order (C-style) and filled into the new shape\n"
               "- A special value of -1 can be used for one dimension, which will be automatically calculated\n"
               "- Data order is preserved; only the dimensional view changes\n\n"
               "Key Differences from Fold Block:\n"
               "- Reshape can transform the entire tensor structure, not just one dimension\n"
               "- Reshape creates a contiguous tensor, potentially copying data if the input is non-contiguous\n"
               "- Reshape is more flexible for arbitrary dimension changes\n\n"
               "Common Applications:\n"
               "- Converting between 1D and 2D representations\n"
               "- Preparing data for blocks that expect specific dimensions\n"
               "- Batching or un-batching operations\n"
               "- Matrix transposition-like operations\n"
               "- Interleaving or de-interleaving channels\n\n"
               "Usage Notes:\n"
               "- Use -1 for one dimension to have it automatically calculated\n"
               "- Reshape does not change the underlying data, just how it's interpreted\n"
               "- For more specialized dimension manipulation, consider the Fold block";
    }

    // Constructor

    Result create() {
        const auto& parser = [&](const std::string& shapeStr, std::vector<U64>& shape) {
            // Validate the overall structure of the shape string.

            if (shapeStr.front() != '[' || shapeStr.back() != ']') {
                JST_ERROR("Invalid shape syntax: Missing brackets.");
                return Result::ERROR;
            }

            // Parse the shape string.

            std::regex re(R"(\d+)");
            std::sregex_iterator next(shapeStr.begin(), shapeStr.end(), re);
            std::sregex_iterator end;

            while (next != end) {
                std::smatch match = *next;
                shape.push_back(std::stoull(match.str()));
                next++;
            }

            if (shape.empty()) {
                JST_ERROR("Invalid shape syntax: No dimensions found.");
                return Result::ERROR;
            }

            JST_TRACE("[SLICE] Parsed shape string {} to shape {}.", shapeStr, shape);

            return Result::SUCCESS;
        };

        JST_CHECK(instance().addModule(
            modifier, "modifier", {
                .callback = [&](auto& mod) {
                    if (config.shape.empty()) {
                        return Result::SUCCESS;
                    }

                    std::vector<U64> shape;
                    JST_CHECK(parser(config.shape, shape));
                    JST_CHECK(mod.reshape(shape));
                
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
        ImGui::TextUnformatted("Shape");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##shape", &config.shape, ImGuiInputTextFlags_EnterReturnsTrue)) {
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
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

JST_BLOCK_ENABLE(Reshape, !std::is_same<IT, void>::value &&
                           std::is_same<OT, void>::value)

#endif
