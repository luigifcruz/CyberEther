#ifndef JETSTREAM_RENDER_SAKURA_MARKDOWN_EDITOR_HH
#define JETSTREAM_RENDER_SAKURA_MARKDOWN_EDITOR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct MarkdownEditor : public Component {
    struct Config {
        std::string id;
        std::string value;
        bool editing = false;
        F32 fontSize = Typography::FontSize;
        std::string backgroundColorKey = "editor_background";
        F32 maxAutoHeightWindowRatio = 0.5f;
        Callback<void(std::string)> onChange;
        Callback<void()> onEdit;
        Callback<void()> onDone;

        bool operator==(const Config&) const = default;
    };

    MarkdownEditor();
    ~MarkdownEditor();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MARKDOWN_EDITOR_HH
