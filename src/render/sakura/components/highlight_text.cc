#include <jetstream/render/sakura/components/highlight_text.hh>

#include <jetstream/render/sakura/components/text_grid.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kLineHeight = 1.6f;

}  // namespace

struct HighlightText::Impl {
    Config config;
    TextGrid grid;
    std::vector<std::vector<TextGrid::StyleId>> styles;
    U64 revision = 0;
    F32 fontSizePixels = 0.0f;
    bool stale = true;

    void sync(const Context& ctx) {
        const F32 pixels = Px(ctx, config.fontSize);
        if (!stale && pixels == fontSizePixels) {
            return;
        }
        stale = false;
        fontSizePixels = pixels;
        styles.clear();
        std::vector<U64> starts;
        U64 start = 0;
        while (true) {
            const U64 end = std::min<U64>(config.str.find('\n', start), config.str.size());
            starts.push_back(start);
            styles.emplace_back(end - start, 0);
            if (end == config.str.size()) {
                break;
            }
            start = end + 1;
        }
        for (const auto& mark : config.marks) {
            const U64 end = std::min<U64>(mark.end, config.str.size());
            for (U64 line = 0; line < styles.size(); ++line) {
                const U64 first = std::max(mark.begin, starts[line]);
                const U64 last = std::min(end, starts[line] + styles[line].size());
                if (first < last) {
                    std::fill(styles[line].begin() + (first - starts[line]), styles[line].begin() + (last - starts[line]), 1);
                }
            }
        }
        grid.update({
            .id = config.id + ":grid",
            .value = config.str,
            .fontSize = pixels,
            .lineHeight = kLineHeight,
            .fontName = config.fontName,
            .monospace = false,
            .showActiveLine = false,
            .scrollbar = false,
            .scrollPastEnd = false,
            .wrap = TextGrid::Wrap::Word,
            .padding = Padding{},
            .textColorKey = config.colorKey,
            .styleColorKeys = {config.markColorKey},
            .styleBackgroundColorKeys = {config.markBackgroundColorKey},
            .styleRevision = ++revision,
            .styler = [this](const std::vector<std::string>&, U64)
                          -> const std::vector<std::vector<TextGrid::StyleId>>& {
                return styles;
            },
            .onPositionClick = [this](TextGrid::Position) {
                if (!config.onClick) {
                    return false;
                }
                config.onClick();
                return true;
            },
            .isPositionInteractive = [this](TextGrid::Position) {
                return static_cast<bool>(config.onClick);
            },
        });
    }
};

HighlightText::HighlightText() {
    impl = std::make_unique<Impl>();
    addChild(impl->grid);
}

HighlightText::~HighlightText() = default;

void HighlightText::update(Config config) {
    if (config == impl->config) {
        impl->config = std::move(config);
        return;
    }
    impl->config = std::move(config);
    impl->stale = true;
    invalidate(Dirty::Layout);
}

Extent2D<F32> HighlightText::measure(const Context& ctx, Extent2D<F32> available) {
    impl->sync(ctx);
    return FitExtent(measureChild(impl->grid, ctx, available), available);
}

void HighlightText::layout(const Context& ctx) {
    impl->sync(ctx);
    layoutChild(ctx, impl->grid, frame());
}

}  // namespace Jetstream::Sakura
