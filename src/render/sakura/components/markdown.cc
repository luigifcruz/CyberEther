#include <jetstream/render/sakura/components/markdown.hh>

#include <jetstream/render/sakura/components/text_markdown.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <optional>
#include <utility>

namespace Jetstream::Sakura {

struct Markdown::Impl {
    Config config;
    TextMarkdown markdown;
    F32 fontSizePixels = 0.0f;
    bool stale = true;

    void sync(const Context& ctx) {
        const F32 pixels = config.fontSize * ctx.pixelRatio;
        if (!stale && pixels == fontSizePixels) {
            return;
        }
        stale = false;
        fontSizePixels = pixels;
        std::optional<Padding> padding;
        if (config.padding.has_value()) {
            const F32 inset = *config.padding * ctx.pixelRatio;
            padding = Padding{inset, inset, inset, inset};
        }
        markdown.update({
            .id = config.id.empty() ? jst::fmt::format("markdown:{}", static_cast<const void*>(this)) : config.id,
            .value = config.value,
            .fontSize = pixels,
            .scrollbar = config.scrollbar,
            .scrollbarInset = {config.scrollbarInset.x * ctx.pixelRatio, config.scrollbarInset.y * ctx.pixelRatio},
            .padding = padding,
        });
    }
};

Markdown::Markdown() {
    impl = std::make_unique<Impl>();
    addChild(impl->markdown);
}

Markdown::~Markdown() = default;

void Markdown::update(Config config) {
    if (config == impl->config) {
        impl->config = std::move(config);
        return;
    }
    impl->config = std::move(config);
    impl->stale = true;
    invalidate(Dirty::Layout);
}

Extent2D<F32> Markdown::measure(const Context& ctx, Extent2D<F32> available) {
    impl->sync(ctx);
    return {FillLength(available.x), FitLength(measureChild(impl->markdown, ctx, available).y, available.y)};
}

void Markdown::layout(const Context& ctx) {
    impl->sync(ctx);
    layoutChild(ctx, impl->markdown, frame());
}

}  // namespace Jetstream::Sakura
