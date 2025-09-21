#ifndef JETSTREAM_BLOCK_ARITHMETIC_BASE_HH
#define JETSTREAM_BLOCK_ARITHMETIC_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/arithmetic.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Arithmetic : public Block {
 public:
    // Configuration

    struct Config {
        ArithmeticOp operation = ArithmeticOp::Add;
        U64 axis = 0;
        bool squeeze = false;

        JST_SERDES(operation, axis, squeeze);
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
        return "arithmetic";
    }

    std::string name() const {
        return "Arithmetic";
    }

    std::string summary() const {
        return "Performs arithmetic operations on a signal.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Performs arithmetic operations on a signal.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            arithmetic, "arithmetic", {
                .operation = config.operation,
                .axis = config.axis,
                .squeeze = config.squeeze,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, arithmetic->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (arithmetic) {
            JST_CHECK(instance().eraseModule(arithmetic->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Operation");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##DeviceList", config.operation.string().c_str())) {
            for (const auto& [key, value] : config.operation.rmap()) {
                bool isSelected = (config.operation == key);
                if (ImGui::Selectable(value.c_str(), isSelected)) {
                    config.operation = key;

                    JST_DISPATCH_ASYNC([&](){
                        ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                        JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                    });
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();

                }
            }
            ImGui::EndCombo();
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

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Squeeze");
        ImGui::TableSetColumnIndex(1);
        if (ImGui::Checkbox("##squeeze", &config.squeeze)) {
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
    std::shared_ptr<Jetstream::Arithmetic<D, IT>> arithmetic;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Arithmetic, is_specialized<Jetstream::Arithmetic<D, IT>>::value &&
                             std::is_same<OT, void>::value)

#endif
