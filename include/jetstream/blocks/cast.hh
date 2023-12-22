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
        return "Casts from one numeric type to another.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().template addModule<Jetstream::Cast, D, IT, OT>(
            cast, "cast", {
                .scaler = config.scaler,
            }, {
                .buffer = input.buffer,
            },
            locale().blockId
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

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Cast, is_specialized<Jetstream::Cast<D, IT, OT>>::value)

#endif
