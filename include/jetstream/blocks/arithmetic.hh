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

        JST_SERDES(operation, axis);
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
        return "Performs basic arithmetic operations (addition, subtraction, multiplication, division) on input tensors.\n\n"
               "The Arithmetic block provides a flexible way to perform element-wise arithmetic operations between "
               "two input tensors. It supports all four basic operations: addition, subtraction, multiplication, and "
               "division, making it a powerful tool for signal processing, data normalization, and custom algorithm implementation.\n\n"
               "Inputs:\n"
               "- a: First input tensor (operand A).\n"
               "- b: Second input tensor (operand B).\n"
               "  - Both tensors must have compatible shapes for element-wise operations.\n"
               "  - Broadcasting is supported for tensors of different shapes.\n"
               "  - Supported types include real (F32) and complex (CF32) tensors.\n\n"
               "Configuration Parameters:\n"
               "- operation: The arithmetic operation to perform:\n"
               "  - ADD: a + b (addition)\n"
               "  - SUB: a - b (subtraction)\n"
               "  - MUL: a * b (multiplication) - similar to the Multiply block but with more operations available\n"
               "  - DIV: a / b (division)\n\n"
               "Outputs:\n"
               "- result: Output tensor containing the result of the arithmetic operation.\n"
               "  - Data type follows standard arithmetic rules for the selected operation.\n"
               "  - Shape follows broadcasting rules when input shapes differ.\n\n"
               "Mathematical Operations:\n"
               "- Addition: result[i] = a[i] + b[i]\n"
               "- Subtraction: result[i] = a[i] - b[i]\n"
               "- Multiplication: result[i] = a[i] * b[i] (element-wise, not matrix multiplication)\n"
               "- Division: result[i] = a[i] / b[i]\n"
               "- Complex operations follow standard complex arithmetic rules\n\n"
               "Key Applications:\n"
               "- Signal combining and mixing\n"
               "- Background subtraction\n"
               "- Normalization by division\n"
               "- Custom algorithm implementation\n"
               "- Mathematical transformations\n\n"
               "Performance Notes:\n"
               "- All operations are optimized and hardware-accelerated where possible\n"
               "- Division is typically the most computationally expensive operation\n"
               "- Broadcasting may introduce additional overhead for tensors of different shapes";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            arithmetic, "arithmetic", {
                .operation = config.operation,
                .axis = config.axis,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, arithmetic->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(arithmetic->locale()));

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
