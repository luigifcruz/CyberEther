#ifndef JETSTREAM_BLOCK_CAST_BASE_HH
#define JETSTREAM_BLOCK_CAST_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/cast.hh"
#include <type_traits>

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
        return "The Cast block converts input tensors from one numeric type to another, enabling "
               "seamless data type transformations within the processing pipeline. It supports "
               "various conversion operations including complex to real extraction and integer to "
               "floating-point transformations.\n\n"

               "## Arguments:\n"
               "- **Scaler**: Scaling factor applied during conversion. For CI8 to CF32, default "
               "is 128.0 to normalize integer values to floating-point range. For CF32 to F32, "
               "no scaling is applied as only the real part is extracted.\n\n"

               "## Useful For:\n"
               "- Converting complex signals to real-valued data by extracting the real component.\n"
               "- Normalizing integer-based complex data to floating-point for further processing.\n"
               "- Adapting data types between different processing stages in the flowgraph.\n"
               "- Preparing data for modules that require specific numeric formats.\n\n"

               "## Examples:\n"
               "- Complex to real conversion:\n"
               "  Config: Scaler=1.0\n"
               "  Input: CF32[8192] → Output: F32[8192]\n"
               "- Integer complex normalization:\n"
               "  Config: Scaler=128.0\n"
               "  Input: CI8[4096] → Output: CF32[4096]\n\n"

               "## Implementation:\n"
               "The module performs direct type conversion with optional scaling. For CF32 to F32, "
               "it extracts the real component and discards the imaginary part. For CI8 to CF32, "
               "it converts integer components to floating-point and applies the scaler for "
               "normalization. Other conversions apply scaling and clamping to prevent overflow.";
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
        if (cast) {
            JST_CHECK(instance().eraseModule(cast->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        if constexpr (std::is_same<IT, CF32>::value && std::is_same<OT, F32>::value) {
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
