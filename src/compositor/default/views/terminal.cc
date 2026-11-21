#include "terminal.hh"

namespace Jetstream {

void TerminalView::update(Config config) {
    this->config = std::move(config);
    auto view = std::move(this->config.terminal);
    const auto* screen = view.screen;
    tint = screen && screen->fill.a > 0.0f && screen->fill != screen->base ? screen->fill : ColorRGBA<F32>{};
    view.padding = Sakura::Dock::ContentPadding;
    view.rounding = Sakura::Dock::TileRounding;
    view.corners = Sakura::Box::BottomLeft | Sakura::Box::BottomRight;
    terminal.update(std::move(view));
}

void TerminalView::retire() {
    config = {};
    terminal.update({});
}

const std::optional<std::string>& TerminalView::flowgraph() const {
    return config.flowgraph;
}

const std::optional<std::string>& TerminalView::tab() const {
    return config.tab;
}

Sakura::Dock::Pane TerminalView::pane() {
    const auto split = [onSplit = config.onSplit](bool stacked) {
        return [onSplit, stacked]() {
            if (onSplit) {
                onSplit(stacked);
            }
        };
    };
    return {
        .id = config.id,
        .title = config.title,
        .group = Group,
        .icon = ICON_LC_SQUARE_TERMINAL,
        .content = &terminal,
        .placement = config.placement,
        .fill = tint,
        .onClose = config.onClose,
        .actions = {
            {.icon = ICON_LC_COLUMNS_2, .onClick = split(false)},
            {.icon = ICON_LC_ROWS_2, .onClick = split(true)},
        },
    };
}

void TerminalTabView::update(Config config, std::vector<Sakura::Dock::Pane> panes) {
    this->config = std::move(config);
    frame.update({
        .accepts = TerminalView::Group,
        .panes = std::move(panes),
        .tiles = this->config.tiles,
        .onTiles = this->config.onTiles,
        .loneHeader = true,
    });
}

Sakura::Dock::Pane TerminalTabView::pane() {
    return {
        .id = config.id,
        .title = config.title,
        .group = TerminalView::Group,
        .icon = ICON_LC_SQUARE_TERMINAL,
        .content = &frame,
        .onClose = config.onClose,
        .onFocus = config.onFocus,
    };
}

}  // namespace Jetstream
