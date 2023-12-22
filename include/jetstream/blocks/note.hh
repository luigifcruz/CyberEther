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
        const I32 numActualLines = std::count(config.note.begin(), config.note.end(), '\n');
        const I32 numLines = std::min(std::max(5, numActualLines + 2), 20);
        ImGui::InputTextMultiline("##note", &config.note, ImVec2(maxWidth, numLines * ImGui::GetTextLineHeight()), ImGuiInputTextFlags_NoHorizontalScroll);
    }

    constexpr bool shouldDrawPreview() const {
        return true;
    }

 private:
    JST_DEFINE_IO();
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Note, std::is_same<IT, void>::value &&
                       std::is_same<OT, void>::value && 
                       D == Device::CPU);

#endif
