#include <jetstream/render/sakura/components/retained/text_view.hh>

#include <utility>

namespace Jetstream::Sakura::Retained {

struct TextView::Impl {
    Config config;
    TextGrid grid;
};

TextView::TextView() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
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
        .backgroundColorKey = impl->config.backgroundColorKey,
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
    return measureChild(this->impl->grid, ctx, available);
}

void TextView::layout(const Context& ctx) {
    layoutChild(ctx, this->impl->grid, frame());
}

}  // namespace Jetstream::Sakura::Retained
