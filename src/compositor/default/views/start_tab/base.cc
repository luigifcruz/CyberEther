#include "base.hh"

namespace Jetstream {

void StartTabView::update(Config config) {
    closable = config.closable;
    onClose = std::move(config.onClose);
    onFocus = std::move(config.onFocus);
    welcome.update(std::move(config.welcome));
}

Sakura::Dock::Pane StartTabView::pane() {
    return {
        .id = PaneId,
        .title = "New Tab",
        .group = "main",
        .icon = ICON_LC_LAYOUT_GRID,
        .content = &welcome.component(),
        .closable = closable,
        .onClose = onClose,
        .onFocus = onFocus,
    };
}

}  // namespace Jetstream
