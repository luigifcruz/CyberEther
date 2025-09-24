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
        U64 ratio = 4;

        JST_SERDES(axis, ratio);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        mem2::Tensor buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputBuffer() const {
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
        return "The Decimator block reduces tensor dimensionality by first reshaping the input to separate "
               "the specified axis into ratio chunks, then summing elements along the new axis, and finally "
               "extracting the first element of the reduced dimension. This effectively decimates the signal "
               "by the specified ratio while preserving accumulated values.\n\n"

               "## Parameters\n"
               "- **Axis**: The axis along which to reshape and decimate the input tensor.\n"
               "- **Ratio**: The decimation ratio that determines how many chunks to create from the axis.\n\n"

               "## Useful For:\n"
               "- Implementing decimation filters for signal processing.\n"
               "- Downsampling data by a fixed ratio.\n"
               "- Aggregating sensor data from multiple sources.\n\n"

               "## Examples:\n"
               "- Time-domain decimation:\n"
               "  Config: Axis=1, Ratio=4\n"
               "  Input: CF32[8192] → Output: CF32[2048]\n"

               "## Implementation:\n"
               "Input → Reshape → Add Axis → Squeeze Axis → Duplicate → Output\n"
               "1. Reshape module separates the specified axis into ratio chunks.\n"
               "2. Arithmetic module sums all elements along the new ratio axis.\n"
               "3. Tensor modifier slices the result to extract index 0 from the reduced dimension.\n"
               "4. Duplicate module ensures proper output buffering and host accessibility.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            reshape, "reshape", {
                .callback = [&](auto& mod) {
                    // Separate the axis into ratio chunks
                    // e.g. [8192] -> [2048, 4] where axis size / ratio = new size
                    const auto& shape = input.buffer.shape();
                    auto new_shape = shape;

                    if (config.axis < shape.size()) {
                        U64 axis_size = shape[config.axis];
                        U64 new_axis_size = axis_size / config.ratio;

                        new_shape[config.axis] = new_axis_size;
                        new_shape.insert(new_shape.begin() + config.axis + 1, config.ratio);

                        mod.reshape(new_shape);
                    }

                    return Result::SUCCESS;
                }
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            arithmetic, "arithmetic", {
                .operation = ArithmeticOp::Add,
                .axis = config.axis + 1,  // Adjust axis after reshape
            }, {
                .buffer = reshape->getOutputBuffer(),
            },
            locale()
        ));

        JST_CHECK(instance().addModule(
            slicer, "slicer", {
                .callback = [&](auto& mod) {
                    mod.squeeze_dims(config.axis + 1);
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
        if (duplicate) {
            JST_CHECK(instance().eraseModule(duplicate->locale()));
        }

        if (slicer) {
            JST_CHECK(instance().eraseModule(slicer->locale()));
        }

        if (arithmetic) {
            JST_CHECK(instance().eraseModule(arithmetic->locale()));
        }

        if (reshape) {
            JST_CHECK(instance().eraseModule(reshape->locale()));
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

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Ratio");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 ratio = config.ratio;
        if (ImGui::InputFloat("##ratio", &ratio, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (ratio >= 1) {
                config.ratio = static_cast<U64>(ratio);

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
    std::shared_ptr<Jetstream::TensorModifier<D, IT>> reshape;
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
