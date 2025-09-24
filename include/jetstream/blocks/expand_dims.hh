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
        return "expand-dims";
    }

    std::string name() const {
        return "Expand Dims";
    }

    std::string summary() const {
        return "Expands the dimensions of a tensor.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Expands the dimensions of a tensor. Similar to numpy.expand_dims().";
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
        if (modifier) {
            JST_CHECK(instance().eraseModule(modifier->locale()));
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
