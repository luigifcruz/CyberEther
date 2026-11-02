#include <jetstream/render/sakura/components/markdown.hh>

#include <jetstream/render/sakura/components/text_markdown.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <cmath>
#include <optional>
#include <utility>

namespace Jetstream::Sakura {

struct Markdown::Impl {
    Config config;
    TextMarkdown markdown;
    F32 fontSizePixels = 0.0f;
    F32 centering = 0.0f;
    bool stale = true;

    void sync(const Context& ctx, F32 width) {
        const F32 pixels = config.fontSize * ctx.pixelRatio;
        const F32 side = config.padding.has_value() ? config.padding->left + config.padding->right : 0.0f;
        const F32 limit = config.maxWidth > 0.0f ? (config.maxWidth + side) * ctx.pixelRatio : 0.0f;
        const F32 extra = width < 0.0f ? centering
                          : limit > 0.0f && width > limit ? std::floor((width - limit) * 0.5f) : 0.0f;
        if (!stale && pixels == fontSizePixels && extra == centering) {
            return;
        }
        stale = false;
        fontSizePixels = pixels;
        centering = extra;
        std::optional<Padding> padding;
        if (config.padding.has_value() || extra > 0.0f) {
            const auto inset = config.padding.value_or(Padding{});
            padding = Padding{inset.left * ctx.pixelRatio + extra, inset.top * ctx.pixelRatio,
                              inset.right * ctx.pixelRatio + extra, inset.bottom * ctx.pixelRatio};
        }
        Callback<bool(const std::string&)> onLink;
        if (config.onLink) {
            onLink = [this](const std::string& url) {
                return config.onLink && config.onLink(url);
            };
        }
        markdown.update({
            .id = config.id.empty() ? jst::fmt::format("markdown:{}", static_cast<const void*>(this)) : config.id,
            .value = config.value,
            .fontSize = pixels,
            .scrollbar = config.scrollbar,
            .scrollbarInset = {config.scrollbarInset.x * ctx.pixelRatio, config.scrollbarInset.y * ctx.pixelRatio},
            .padding = padding,
            .onLink = std::move(onLink),
            .hero = config.hero,
            .softBreaks = config.softBreaks,
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

const std::vector<TextMarkdown::Heading>& Markdown::headings() const {
    static const std::vector<TextMarkdown::Heading> pending;
    return impl->stale ? pending : impl->markdown.headings();
}

const TextGrid::Metrics& Markdown::metrics() const {
    return impl->markdown.metrics();
}

void Markdown::scrollToLine(U64 line) {
    impl->markdown.scrollToLine(line);
}

Extent2D<F32> Markdown::measure(const Context& ctx, Extent2D<F32> available) {
    impl->sync(ctx, std::isfinite(available.x) ? available.x : -1.0f);
    return {FillLength(available.x), FitLength(measureChild(impl->markdown, ctx, available).y, available.y)};
}

void Markdown::layout(const Context& ctx) {
    impl->sync(ctx, frame().width);
    layoutChild(ctx, impl->markdown, frame());
}

}  // namespace Jetstream::Sakura
