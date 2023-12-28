#ifndef JETSTREAM_BLOCK_SPEECH_RECOGNITION_BASE_HH
#define JETSTREAM_BLOCK_SPEECH_RECOGNITION_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/speech_recognition.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class SpeechRecognition : public Block {
 public:
    // Configuration

    struct Config {
        JST_SERDES();
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
        JST_SERDES();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "speech-recognition";
    }

    std::string name() const {
        return "Speech Recognition";
    }

    std::string summary() const {
        return "Converts audio to text.";
    }

    std::string description() const {
        // TODO: Add decent block description describing internals and I/O.
        return "Converts audio to text using Whisper.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().template addModule<Jetstream::SpeechRecognition, D, IT>(
            tts, "tts", {}, {
                .buffer = input.buffer,
            },
            locale().blockId
        ));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(tts->locale()));

        return Result::SUCCESS;
    }

    // Interface

    void drawPreview(const F32& maxWidth) {
        ImVec2 textBoxSize(maxWidth, 200);
        if (ImGui::InputTextMultiline("##output", (char*)tts->getTextBuffer().c_str(), tts->getTextBuffer().size(),
                                      textBoxSize, ImGuiInputTextFlags_ReadOnly | 
                                                   ImGuiInputTextFlags_NoHorizontalScroll)) {
        }
        ImGuiContext& g = *GImGui;
        const char* child_window_name = NULL;
        ImFormatStringToTempBuffer(&child_window_name, NULL, "%s/%s_%08X", g.CurrentWindow->Name, "##output", ImGui::GetID("##output"));
        ImGuiWindow* child_window = ImGui::FindWindowByName(child_window_name);
        ImGui::SetScrollY(child_window, child_window->ScrollMax.y);
    }

    constexpr bool shouldDrawPreview() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::SpeechRecognition<D, IT>> tts;

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(SpeechRecognition, is_specialized<Jetstream::SpeechRecognition<D, IT>>::value &&
                                    std::is_same<OT, void>::value)

#endif
