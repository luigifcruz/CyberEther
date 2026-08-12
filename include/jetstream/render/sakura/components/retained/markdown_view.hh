#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_MARKDOWN_VIEW_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_MARKDOWN_VIEW_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura::Retained {

struct MarkdownView {
    struct Config {
        std::string id;
        std::string value;
        F32 fontSize = 15.0f;
        std::string backgroundColorKey = "background";
    };

    MarkdownView();
    ~MarkdownView();

    MarkdownView(MarkdownView&&) noexcept;
    MarkdownView& operator=(MarkdownView&&) noexcept;

    MarkdownView(const MarkdownView&) = delete;
    MarkdownView& operator=(const MarkdownView&) = delete;

    bool update(Config config);
    void render(const Sakura::Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_MARKDOWN_VIEW_HH
