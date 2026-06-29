#ifndef JETSTREAM_RENDER_SAKURA_NODE_CODE_EDITOR_HH
#define JETSTREAM_RENDER_SAKURA_NODE_CODE_EDITOR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/code_editor.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeCodeEditor {
    using StatusTone = Retained::CodeEditor::StatusTone;
    using Language = Retained::CodeEditor::Language;

    struct Config {
        std::string id;
        std::string value;
        std::vector<std::string> consoleOutput;
        std::string status;
        StatusTone statusTone = StatusTone::Info;
        bool consoleVisible = false;
        bool collapsible = false;
        bool autoHeight = false;
        F32 maxAutoHeightWindowRatio = 0.5f;
        Language language = Language::Python;
        bool lineNumbers = true;
        bool lineWrapping = false;
        F32 editorFontSize = 15.0f;
        std::function<void(std::string)> onChange;
        std::function<void(std::string)> onSubmit;
    };

    NodeCodeEditor();
    ~NodeCodeEditor();

    NodeCodeEditor(NodeCodeEditor&&) noexcept;
    NodeCodeEditor& operator=(NodeCodeEditor&&) noexcept;

    NodeCodeEditor(const NodeCodeEditor&) = delete;
    NodeCodeEditor& operator=(const NodeCodeEditor&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_CODE_EDITOR_HH
