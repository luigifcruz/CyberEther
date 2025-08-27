#ifndef JETSTREAM_BLOCK_CAST_BASE_HH
#define JETSTREAM_BLOCK_CAST_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/cast.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Cast : public Block {
 public:
    // Configuration

    struct Config {
        F32 scaler = 0.0f;

        JST_SERDES(scaler);
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
        Tensor<D, OT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, OT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "cast";
    }

    std::string name() const {
        return "Cast";
    }

    std::string summary() const {
        return "Casts the input to a type.";
    }

    std::string description() const {
        return "Converts data from one numeric type to another, enabling interoperability between different data formats.\n\n"
               "The Cast block performs type conversion of tensor data, allowing signals to be transformed between "
               "different numeric representations. This operation is essential for interfacing between blocks with different "
               "type requirements, handling data from external sources, or optimizing memory usage and computational efficiency.\n\n"
               "Inputs:\n"
               "- buffer: Input tensor to be converted to a new data type.\n"
               "  - Can be any supported data type (F32, CF32, I32, I16, etc.).\n\n"
               "Outputs:\n"
               "- buffer: Output tensor with the same data but converted to the target type.\n"
               "  - Data shape remains unchanged, only the representation changes.\n"
               "  - Value precision may change based on the conversion (e.g., float to int).\n\n"
               "Supported Conversions:\n"
               "- Real to Real: F32 → I16, I16 → F32, etc.\n"
               "- Complex to Real: CF32 → F32 (takes real part only)\n"
               "- Real to Complex: F32 → CF32 (imaginary part set to zero)\n"
               "- Various bit-depth conversions: I16 → I8, etc.\n\n"
               "Key Applications:\n"
               "- Interface with hardware that requires specific formats (e.g., audio devices, SDRs)\n"
               "- Reduce memory usage by converting to smaller data types\n"
               "- Prepare data for specialized processing blocks\n"
               "- Convert incoming data from external sources to internal formats\n"
               "- Extract real components from complex signals\n\n"
               "Technical Details:\n"
               "- Conversion is done element-wise across the entire tensor\n"
               "- Type conversions follow C++ casting rules with appropriate scaling\n"
               "- When converting to smaller bit-depth types, values outside the representable range are clamped\n"
               "- Complex to real conversion extracts only the real component\n\n"
               "Performance Considerations:\n"
               "- Type conversion adds some computational overhead\n"
               "- Converting to smaller types can improve performance in downstream blocks\n"
               "- Implemented with hardware acceleration where available";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            cast, "cast", {
                .scaler = config.scaler,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, cast->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(cast->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Scaler");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 scaler = config.scaler;
        if (ImGui::InputFloat("##scaler", &scaler, 1.0f, 1.0f, "%.0f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (scaler >= 0.0f) {
                config.scaler = scaler;

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
    std::shared_ptr<Jetstream::Cast<D, IT, OT>> cast;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Cast, is_specialized<Jetstream::Cast<D, IT, OT>>::value)

#endif
