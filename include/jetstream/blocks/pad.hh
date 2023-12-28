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
        // TODO: Add decent block description describing internals and I/O.
        return "Adds padding to the end of a tensor along a given axis.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().template addModule<Jetstream::Pad, D, IT>(
            pad, "pad", {
                .size = config.size,
                .axis = config.axis,
            }, {
                .unpadded = input.unpadded,
            },
            locale().blockId
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

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Pad, is_specialized<Jetstream::Pad<D, IT>>::value &&
                      std::is_same<OT, void>::value)

#endif
