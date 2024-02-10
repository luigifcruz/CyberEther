#ifndef JETSTREAM_BLOCK_NOTE_BASE_HH
#define JETSTREAM_BLOCK_NOTE_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Note : public Block {
 public:
    // Configuration

    struct Config {
        std::string note;

        JST_SERDES(note);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
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
        return "note";
    }

    std::string name() const {
        return "Note";
    }

    std::string summary() const {
        return "Displays a note.";
    }

    std::string description() const {
        // TODO: Add support for markdown with links and images.
        return "Just a simple flowgraph note.";
    }

    // Constructor

    Result create() {
        return Result::SUCCESS;
    }

    Result destroy() {
        return Result::SUCCESS;
    }

    // Interface

    void drawPreview(const F32& maxWidth) {
        if (editing) {
            const I32 numActualLines = std::count(config.note.begin(), config.note.end(), '\n');
            const I32 textHeight = std::min(std::max(5, numActualLines + 2) * ImGui::GetTextLineHeight(), 500.0f);
            // TODO: Implement automatic line wrapping.
            ImGui::InputTextMultiline("##note", 
                                      &config.note, 
                                      ImVec2(maxWidth, textHeight), 
                                      ImGuiInputTextFlags_NoHorizontalScroll);
        } else {
            ImGui::SetNextWindowSizeConstraints(ImVec2(maxWidth, 50.0f), ImVec2(maxWidth, 500.0f));
            // UPDATE-ME: Markdown rendering.
            ImGui::BeginChild("##note-markdown", ImVec2(maxWidth, 0.0f), ImGuiChildFlags_AutoResizeY | 
                                                                         ImGuiChildFrags_AlwaysUseParentDrawList, 
                                                                         ImGuiWindowFlags_NoBackground);
            ImGui::Markdown(config.note.c_str(), config.note.length(), instance().window().markdownConfig());
            ImGui::EndChild();
        }
        if (ImGui::Button((editing) ? "Done" : "Edit", ImVec2(maxWidth, 0))) {
            editing = !editing;
        }
    }

    constexpr bool shouldDrawPreview() const {
        return true;
    }

 private:
    bool editing = false;

    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Note, std::is_same<IT, void>::value &&
                       std::is_same<OT, void>::value && 
                       D == Device::CPU);

#endif
