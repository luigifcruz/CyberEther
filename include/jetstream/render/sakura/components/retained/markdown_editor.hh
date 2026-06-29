#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_MARKDOWN_EDITOR_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_MARKDOWN_EDITOR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura::Retained {

struct MarkdownEditor {
    struct Config {
        std::string id;
        std::string value;
        bool editing = false;
        F32 fontSize = 15.0f;
        std::string backgroundColorKey = "editor_background";
        F32 maxAutoHeightWindowRatio = 0.5f;
        std::function<void(std::string)> onChange;
        std::function<void()> onEdit;
        std::function<void()> onDone;
    };

    MarkdownEditor();
    ~MarkdownEditor();

    MarkdownEditor(MarkdownEditor&&) noexcept;
    MarkdownEditor& operator=(MarkdownEditor&&) noexcept;

    MarkdownEditor(const MarkdownEditor&) = delete;
    MarkdownEditor& operator=(const MarkdownEditor&) = delete;

    bool update(Config config);
    void render(const Sakura::Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_MARKDOWN_EDITOR_HH
