#include "sidebar_tiles.hh"

namespace Jetstream {

void SidebarTiles::update(Config config) {
    const bool interactive = static_cast<bool>(this->config.onSidebarOpen) &&
                             static_cast<bool>(config.onSidebarOpen);
    if (interactive) {
        if (config.sidebarOpen != this->config.sidebarOpen) {
            if (pendingOpen == config.sidebarOpen) {
                pendingOpen.reset();
            } else {
                autoCollapsed = false;
                pendingOpen.reset();
            }
        } else if (pendingOpen == config.sidebarOpen) {
            pendingOpen.reset();
        }
    }
    this->config = std::move(config);

    Dock::update({
        .panes = {
            {
                .id = SidebarTileId,
                .group = SidebarTileId,
                .content = this->config.sidebar,
                .closable = false,
                .maximizable = false,
                .hidden = !this->config.sidebarOpen,
            },
            {
                .id = PageTileId,
                .group = PageTileId,
                .content = this->config.page,
                .closable = false,
                .maximizable = false,
                .card = this->config.pageCard,
            },
        },
        .tiles = Sakura::DockTiles{
            .sizes = {SidebarWidth, 0.0f},
            .children = {{.pane = SidebarTileId}, {.pane = PageTileId}},
        },
        .onSize = this->config.autoCollapseWidth > 0.0f && this->config.onSidebarOpen
                      ? Sakura::Callback<void(Extent2D<F32>)>([this](Extent2D<F32> size) {
                            handleSize(size);
                        })
                      : Sakura::Callback<void(Extent2D<F32>)>(),
        .columns = true,
        .flexPane = PageTileId,
        .headers = false,
        .resizable = false,
    });
}

void SidebarTiles::handleSize(Extent2D<F32> logicalSize) {
    if (config.autoCollapseWidth <= 0.0f || !config.onSidebarOpen || logicalSize.x <= 0.0f) {
        return;
    }

    const bool nextCompact = logicalSize.x < config.autoCollapseWidth;
    if (compact == nextCompact) {
        return;
    }
    compact = nextCompact;

    const bool effectiveOpen = pendingOpen.value_or(config.sidebarOpen);
    if (nextCompact) {
        if (effectiveOpen) {
            autoCollapsed = true;
            pendingOpen = false;
            config.onSidebarOpen(false);
        }
    } else if (autoCollapsed && !effectiveOpen) {
        autoCollapsed = false;
        pendingOpen = true;
        config.onSidebarOpen(true);
    }
}

}  // namespace Jetstream
