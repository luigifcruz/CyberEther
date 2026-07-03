#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_CODE_EDITOR_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_CODE_EDITOR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct CodeEditor {
    enum class Language : U8 {
        Python,
        Markdown,
    };

    enum class StatusTone : U8 {
        Info,
        Success,
        Warning,
        Error,
    };

    struct Config {
        std::string id;
        std::string value;
        std::vector<std::string> consoleOutput;
        std::string status;
        Extent2D<F32> size = {0.0f, 200.0f};
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

    CodeEditor();
    ~CodeEditor();

    CodeEditor(CodeEditor&&) noexcept;
    CodeEditor& operator=(CodeEditor&&) noexcept;

    CodeEditor(const CodeEditor&) = delete;
    CodeEditor& operator=(const CodeEditor&) = delete;

    bool update(Config config);
    void render(const Sakura::Context& ctx);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_CODE_EDITOR_HH
