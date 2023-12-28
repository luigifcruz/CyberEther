#ifndef JETSTREAM_BLOCK_AUDIO_BASE_HH
#define JETSTREAM_BLOCK_AUDIO_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/audio.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Audio : public Block {
 public:
    // Configuration

    struct Config {
        F32 inSampleRate = 48e3;
        F32 outSampleRate = 48e3;

        JST_SERDES(inSampleRate, outSampleRate);
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
        return "audio";
    }

    std::string name() const {
        return "Audio";
    }

    std::string summary() const {
        return "Plays input on the speaker.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Downsamples the input to the output sample rate and plays it on the speaker.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().template addModule<Jetstream::Audio, D, IT>(
            audio, "audio", {
                .inSampleRate = config.inSampleRate,
                .outSampleRate = config.outSampleRate,
            }, {
                .buffer = input.buffer,
            },
            locale().blockId
        ));
        JST_CHECK(Block::LinkOutput("buffer", output.buffer, audio->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(audio->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 inSampleRate = config.inSampleRate / 1e6f;
        if (ImGui::InputFloat("##in-sample-rate", &inSampleRate, 0.1f, 0.2f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.inSampleRate = inSampleRate * 1e6;

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
    std::shared_ptr<Jetstream::Audio<D, IT>> audio;

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Audio, is_specialized<Jetstream::Audio<D, IT>>::value &&
                        std::is_same<OT, void>::value)

#endif
