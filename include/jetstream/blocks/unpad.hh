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
        // TODO: Add decent block description describing internals and I/O.
        return "Removes padding from the end of a tensor along a given axis.";
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
        if (ImGui::InputFloat("##pad-axis", &axis, 1.0f, 1.0f, "%.0f")) {
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
        if (ImGui::InputFloat("##pad-size", &size, 1.0f, 1.0f, "%.0f")) {
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
