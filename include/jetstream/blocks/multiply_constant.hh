#ifndef JETSTREAM_BLOCK_MULTIPLY_CONSTANT_BASE_HH
#define JETSTREAM_BLOCK_MULTIPLY_CONSTANT_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/multiply_constant.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class MultiplyConstant : public Block {
 public:
    // Configuration

    struct Config {
        IT constant;

        JST_SERDES(constant);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> factor;

        JST_SERDES(factor);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> product;

        JST_SERDES(product);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputProduct() const {
        return this->output.product;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "multiply-const";
    }

    std::string name() const {
        return "Multiply Const";
    }

    std::string summary() const {
        return "Multiplies input by factor.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Takes 'factor' as input and multiplied it with the const value producing the result as 'product'.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            multiply, "multiply", {}, {
                .factor = input.factor,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("product", output.product, multiply->getOutputProduct()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(multiply->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Constant");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 constant = config.constant;
        if (ImGui::InputFloat("##constant-val", &constant, 1.0f, 2.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.constant = constant;

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
    std::shared_ptr<Jetstream::MultiplyConstant<D, IT>> multiply;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(MultiplyConstant, !IsComplex<IT>::value &&
                                   is_specialized<Jetstream::MultiplyConstant<D, IT>>::value &&
                                   std::is_same<OT, void>::value)

#endif
