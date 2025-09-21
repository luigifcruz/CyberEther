#ifndef JETSTREAM_BLOCK_FM_BASE_HH
#define JETSTREAM_BLOCK_FM_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/fm.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class FM : public Block {
 public:
    // Configuration

    struct Config {
        F32 sampleRate = 240e3f;

        JST_SERDES(sampleRate);
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
        return "fm";
    }

    std::string name() const {
        return "FM";
    }

    std::string summary() const {
        return "Demodulates a FM signal.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Demodulates a complex-valued frequency modulated signal.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            fm, "fm", {
                .sampleRate = config.sampleRate,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, fm->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (fm) {
            JST_CHECK(instance().eraseModule(fm->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 sampleRate = config.sampleRate / 1e6f;
        if (ImGui::InputFloat("##sample-rate", &sampleRate, 0.1f, 0.2f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.sampleRate = sampleRate * 1e6;

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
    std::shared_ptr<Jetstream::FM<D, IT, OT>> fm;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(FM, is_specialized<Jetstream::FM<D, IT, OT>>::value &&
                     !std::is_same<OT, void>::value)

#endif
