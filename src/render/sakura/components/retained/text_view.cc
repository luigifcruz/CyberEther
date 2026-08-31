#include <jetstream/render/sakura/components/retained/text_view.hh>

#include <jetstream/render/sakura/components/retained/box.hh>

#include "../../context.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura::Retained {

struct TextView::Impl {
    Config config;
    Box background;
    TextGrid grid;
};

TextView::TextView() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    add(this->impl->background);
    add(this->impl->grid);
}

TextView::~TextView() = default;

bool TextView::update(Config config) {
    impl->config = std::move(config);

    impl->grid.update({
        .id = impl->config.id + ":grid",
        .value = impl->config.value,
        .editable = false,
        .fontSize = impl->config.fontSize,
        .fontScale = impl->config.fontScale,
        .fontName = impl->config.fontName,
        .monospace = impl->config.monospace,
        .lineNumbers = impl->config.lineNumbers,
        .showActiveLine = false,
        .stickToBottom = impl->config.stickToBottom,
        .scrollbar = impl->config.scrollbar,
        .wrap = impl->config.wrap,
        .backgroundColorKey = impl->config.contentPadding > 0.0f
                                  ? "transparent"
                                  : impl->config.backgroundColorKey,
        .textColorKey = impl->config.textColorKey,
        .lineNumberColorKey = impl->config.lineNumberColorKey,
        .gutterSeparatorColorKey = impl->config.gutterSeparatorColorKey,
        .selectionColorKey = impl->config.selectionColorKey,
        .selectionMatchColorKey = impl->config.selectionMatchColorKey,
        .activeLineColorKey = impl->config.activeLineColorKey,
        .cursorColorKey = impl->config.cursorColorKey,
        .scrollbarTrackColorKey = impl->config.scrollbarTrackColorKey,
        .scrollbarThumbColorKey = impl->config.scrollbarThumbColorKey,
        .styleColorKeys = impl->config.styleColorKeys,
        .styleFonts = impl->config.styleFonts,
        .styleBackgroundColorKeys = impl->config.styleBackgroundColorKeys,
        .maxLineSegments = impl->config.maxLineSegments,
        .styler = impl->config.styler,
        .onPositionClick = impl->config.onPositionClick,
        .isPositionInteractive = impl->config.isPositionInteractive,
    });
    return true;
}

Extent2D<F32> TextView::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 padding = impl->config.contentPadding * ctx.pixelRatio;
    const Extent2D<F32> innerAvailable = {
        available.x,
        std::max(0.0f, available.y - 2.0f * padding),
    };
    const auto inner = measureChild(this->impl->grid, ctx, innerAvailable);
    return {inner.x, inner.y + 2.0f * padding};
}

void TextView::layout(const Context& ctx) {
    const F32 padding = impl->config.contentPadding * ctx.pixelRatio;
    impl->background.update({
        .id = impl->config.id + ":background",
        .instances = {{
            .rect = frame(),
            .visible = impl->config.contentPadding > 0.0f && !frame().empty(),
            .backgroundColor = ctx.color(impl->config.backgroundColorKey),
        }},
    });
    layoutChild(ctx, this->impl->background, frame());
    const Rect content = {
        frame().x,
        frame().y + padding,
        frame().width,
        std::max(0.0f, frame().height - 2.0f * padding),
    };
    layoutChild(ctx, this->impl->grid, content);
}

}  // namespace Jetstream::Sakura::Retained
