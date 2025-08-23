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
        // TODO: Add decent block description describing internals and I/O.
        return "Takes a slice of the input tensor. Similar to numpy.take().";
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
        if (ImGui::InputFloat("##index", &index, 1.0f, 1.0f, "%.0f")) {
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
        if (ImGui::InputFloat("##axis", &axis, 1.0f, 1.0f, "%.0f")) {
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
