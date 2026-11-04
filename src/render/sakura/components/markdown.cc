#include <jetstream/render/sakura/components/markdown.hh>

#include <jetstream/render/sakura/components/text_markdown.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kHeadingActiveReach = 2.0f;
constexpr F32 kHeadingEndSlack = 1.0f;

}  // namespace

struct Markdown::Impl {
    Config config;
    TextMarkdown markdown;
    F32 fontSizePixels = 0.0f;
    F32 centering = 0.0f;
    bool stale = true;
    bool headingsStale = true;
    std::optional<U64> reflowLine;
    std::optional<U64> pinned;
    F32 pinnedScroll = 0.0f;
    F32 pinnedContentHeight = 0.0f;
    F32 pinnedViewportHeight = 0.0f;

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
        headingsStale = false;
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
            .textColorKey = config.textColorKey,
            .styleColorKeys = {config.boldColorKey, "", config.boldColorKey, "", "", ""},
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
    if (config.value != impl->config.value || config.softBreaks != impl->config.softBreaks ||
        config.hero != impl->config.hero) {
        impl->headingsStale = true;
        impl->pinned.reset();
    }
    impl->config = std::move(config);
    impl->stale = true;
    invalidate(Dirty::Layout);
}

const std::vector<TextMarkdown::Heading>& Markdown::headings() const {
    static const std::vector<TextMarkdown::Heading> pending;
    return impl->headingsStale ? pending : impl->markdown.headings();
}

const TextGrid::Metrics& Markdown::metrics() const {
    return impl->markdown.metrics();
}

std::optional<U64> Markdown::activeHeading(U64 minimumLevel, U64 maximumLevel) {
    auto& s = *impl;
    const auto& entries = headings();
    const auto included = [&](U64 index) {
        return index < entries.size() && entries[index].level >= minimumLevel && entries[index].level <= maximumLevel;
    };
    std::vector<U64> candidates;
    for (U64 index = 0; index < entries.size(); ++index) {
        if (included(index)) {
            candidates.push_back(index);
        }
    }
    if (candidates.empty()) {
        s.pinned.reset();
        return std::nullopt;
    }
    const auto& metrics = s.markdown.metrics();
    if (s.pinned.has_value()) {
        const bool sameLayout = metrics.contentHeight == s.pinnedContentHeight &&
                                metrics.viewportHeight == s.pinnedViewportHeight;
        if (included(*s.pinned) && sameLayout && (s.markdown.gliding() || metrics.scrollY == s.pinnedScroll)) {
            return s.pinned;
        }
        s.pinned.reset();
    }
    const F32 scrollable = metrics.contentHeight - metrics.viewportHeight;
    if (scrollable > 0.0f && metrics.scrollY >= scrollable - kHeadingEndSlack) {
        return candidates.back();
    }
    U64 active = candidates.front();
    for (const U64 index : candidates) {
        const U64 line = entries[index].line;
        if (line >= metrics.sourceLines.size()) {
            break;
        }
        const auto& row = metrics.sourceLines[line];
        if (row.top - metrics.viewportTop > row.height * kHeadingActiveReach) {
            break;
        }
        active = index;
    }
    return active;
}

void Markdown::revealHeading(U64 index) {
    const auto& entries = headings();
    if (index >= entries.size()) {
        return;
    }
    impl->pinned = index;
    impl->pinnedScroll = impl->markdown.glideToLine(entries[index].line);
    impl->pinnedContentHeight = impl->markdown.metrics().contentHeight;
    impl->pinnedViewportHeight = impl->markdown.metrics().viewportHeight;
}

void Markdown::scrollToLine(U64 line) {
    impl->pinned.reset();
    impl->markdown.scrollToLine(line);
    impl->reflowLine = impl->stale ? std::optional<U64>(line) : std::nullopt;
}

Extent2D<F32> Markdown::measure(const Context& ctx, Extent2D<F32> available) {
    impl->sync(ctx, std::isfinite(available.x) ? available.x : -1.0f);
    const auto content = measureChild(impl->markdown, ctx, available);
    return {FillLength(available.x, content.x), FitLength(content.y, available.y)};
}

void Markdown::layout(const Context& ctx) {
    impl->sync(ctx, frame().width);
    layoutChild(ctx, impl->markdown, frame());
    if (impl->reflowLine.has_value()) {
        impl->markdown.scrollToLine(*impl->reflowLine);
        impl->reflowLine.reset();
    }
}

}  // namespace Jetstream::Sakura
