#include <jetstream/render/sakura/components/retained/markdown_view.hh>

#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/canvas.hh>
#include <jetstream/render/sakura/components/retained/text_markdown.hh>

#include "../../helpers.hh"

#include <algorithm>
#include <string>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kReferenceFontSize = 15.0f;
constexpr F32 kPadding = 0.0f;

}  // namespace

struct MarkdownBody : public Component {
    Box background;
    TextMarkdown markdown;

    std::string value;
    F32 fontSizePixels = kReferenceFontSize;
    F32 pad = 0.0f;
    std::string backgroundColorKey = "background";

    MarkdownBody() {
        setClipsChildren(true);
        add(background);
        add(markdown);
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        const F32 innerWidth = std::max(0.0f, available.x - 2.0f * pad);
        const Extent2D<F32> inner = measureChild(markdown, ctx, {innerWidth, available.y});
        return {available.x, inner.y + 2.0f * pad};
    }

    void layout(const Context& ctx) override {
        const Rect bounds = frame();

        const ColorRGBA<F32> bg = ctx.color(backgroundColorKey);
        background.update({
            .id = "markdown:bg",
            .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = bg}},
        });

        const Rect content = bounds.inset(pad);
        markdown.update({
            .id = "markdown:md",
            .value = value,
            .fontSize = fontSizePixels,
            .textColorKey = "text_primary",
            .selectionColorKey = "editor_selection",
            .selectionMatchColorKey = "editor_selection_match",
            .scrollbarTrackColorKey = "editor_scrollbar_track",
            .scrollbarThumbColorKey = "editor_scrollbar_thumb",
        });

        layoutChild(ctx, background, bounds);
        layoutChild(ctx, markdown, content);

    }
};

struct MarkdownView::Impl {
    Config config;

    Canvas canvas;
    MarkdownBody body;

    F32 fontSizePixels = kReferenceFontSize;

    F32 pixelRatio() const { return config.fontSize > 0.0f ? fontSizePixels / config.fontSize : 1.0f; }

    Impl() {
        canvas.mount(body);
    }
};

MarkdownView::MarkdownView() {
    this->impl = std::make_unique<Impl>();
}

MarkdownView::~MarkdownView() = default;
MarkdownView::MarkdownView(MarkdownView&&) noexcept = default;
MarkdownView& MarkdownView::operator=(MarkdownView&&) noexcept = default;

bool MarkdownView::update(Config config) {
    impl->config = std::move(config);

    impl->canvas.update({
        .id = impl->config.id + ":canvas",
        .size = {0.0f, 200.0f},
        .autoHeight = true,
        .onLayout = [impl = this->impl.get()](const Canvas::Layout& layout) {
            impl->fontSizePixels = impl->config.fontSize * layout.pixelRatio;
        },
    });

    impl->body.value = impl->config.value;
    impl->body.fontSizePixels = impl->fontSizePixels;
    impl->body.pad = kPadding * impl->pixelRatio();
    impl->body.backgroundColorKey = impl->config.backgroundColorKey;
    impl->body.markdown.update({
        .id = impl->config.id + ":md",
        .value = impl->config.value,
        .fontSize = impl->fontSizePixels,
    });
    return true;
}

void MarkdownView::render(const Sakura::Context& ctx) const {
    impl->canvas.render(ctx);
}

}  // namespace Jetstream::Sakura::Retained
