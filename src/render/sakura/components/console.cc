#include <jetstream/render/sakura/components/console.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/status_bar.hh>
#include <jetstream/render/sakura/components/text_view.hh>

#include "../helpers.hh"
#include "../metrics.hh"
#include "../detail/corner_mask.hh"
#include "../detail/text_lines.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace Jetstream::Sakura {

struct Console::Impl {
    Config config;

    Box backgroundBox;
    TextView outputView;
    Label emptyLabel;
    StatusBar statusBar;
    Polyline cornerMask;

    Rect viewRect;
    F32 fontSizePixels = Typography::FontSize;
    F32 ratio = 1.0f;
    bool stale = true;
    F32 syncedRatio = 0.0f;
    F32 barHeight = 0.0f;

    F32 pixelRatio() const {
        return ratio;
    }
    bool statusVisible() const { return !config.status.empty(); }
    F32 statusBarHeightPixels() const {
        return statusVisible() ? barHeight : 0.0f;
    }
    Rect outputRect() const {
        return {viewRect.x, viewRect.y, viewRect.width,
                std::max(0.0f, viewRect.height - statusBarHeightPixels())};
    }
    Rect statusBarRect() const {
        const F32 height = statusBarHeightPixels();
        return {viewRect.x, std::max(viewRect.y, viewRect.bottom() - height), viewRect.width, height};
    }

    void sync(const Context& ctx) {
        ratio = ctx.pixelRatio;
        fontSizePixels = config.fontSize * ratio;
        if (!stale && syncedRatio == ratio) {
            return;
        }
        stale = false;
        syncedRatio = ratio;
        outputView.update({
            .id = config.id + ":output",
            .value = JoinLines(config.output),
            .fontSize = fontSizePixels,
            .fontScale = Typography::ConsoleScale,
            .stickToBottom = true,
            .wrap = TextGrid::Wrap::Character,
            .backgroundColorKey = config.backgroundColorKey,
            .textColorKey = "editor_text",
            .selectionColorKey = "editor_selection",
            .selectionMatchColorKey = "editor_selection_match",
            .scrollbarTrackColorKey = "editor_scrollbar_track",
            .scrollbarThumbColorKey = "editor_scrollbar_thumb",
        });
    }
};

Console::Console() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->backgroundBox);
    addChild(impl->outputView);
    addChild(impl->emptyLabel);
    addChild(impl->statusBar);
    addChild(impl->cornerMask);
}

Console::~Console() = default;

void Console::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const auto& current = impl->config;
    impl->stale = impl->stale || config.id != current.id || config.output != current.output ||
                  config.fontSize != current.fontSize || config.backgroundColorKey != current.backgroundColorKey;
    impl->config = std::move(config);
}

Extent2D<F32> Console::measure(const Context& ctx, Extent2D<F32> available) {
    impl->sync(ctx);
    impl->barHeight = measureChild(impl->statusBar, ctx, {std::numeric_limits<F32>::infinity(), std::numeric_limits<F32>::infinity()}).y;
    const F32 width = FillLength(available.x);
    F32 natural = 0.0f;
    if (impl->config.height == 0.0f) {
        natural = measureChild(impl->outputView, ctx, {width, std::numeric_limits<F32>::infinity()}).y +
                  impl->statusBarHeightPixels();
    }
    return {width, ResolveLength(ctx, impl->config.height, natural, available.y)};
}

void Console::layout(const Context& ctx) {
    const auto& config = impl->config;
    impl->sync(ctx);
    impl->barHeight = measureChild(impl->statusBar, ctx, {std::numeric_limits<F32>::infinity(), std::numeric_limits<F32>::infinity()}).y;
    impl->viewRect = frame();

    const Rect viewRect = impl->viewRect;
    const F32 fontSizePixels = impl->fontSizePixels;
    const bool visible = !viewRect.empty();

    impl->backgroundBox.update({
        .instances = {{.rect = viewRect, .visible = visible, .backgroundColor = ctx.color(config.backgroundColorKey)}},
    });

    const auto output = impl->outputRect();
    impl->emptyLabel.update({
        .instances = {{
            .rect = output,
            .str = config.emptyText,
            .visible = visible && config.output.empty() && !config.emptyText.empty(),
            .color = ctx.color("editor_line_number"),
            .fontSize = fontSizePixels * Typography::ConsoleScale,
            .alignment = {1, 1},
        }},
        .fontName = Typography::MonoFont,
    });

    impl->statusBar.update({
        .id = config.id + ":status",
        .status = config.status,
        .tone = config.statusTone,
        .fontSize = fontSizePixels,
    });

    layoutChild(ctx, impl->backgroundBox, viewRect);
    layoutChild(ctx, impl->outputView, output);
    layoutChild(ctx, impl->emptyLabel, output);
    layoutChild(ctx, impl->statusBar, impl->statusBarRect());
    UpdateCornerMask(ctx, impl->cornerMask, viewRect, clip());
}

}  // namespace Jetstream::Sakura
