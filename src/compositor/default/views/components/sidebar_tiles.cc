#include "sidebar_tiles.hh"

namespace Jetstream {

void SidebarTiles::update(Config config) {
    Dock::update({
        .panes = {
            {
                .id = SidebarTileId,
                .group = SidebarTileId,
                .content = config.sidebar,
                .closable = false,
                .maximizable = false,
                .hidden = !config.sidebarOpen,
            },
            {
                .id = PageTileId,
                .group = PageTileId,
                .content = config.page,
                .closable = false,
                .maximizable = false,
                .card = config.pageCard,
            },
        },
        .tiles = Sakura::DockTiles{
            .sizes = {SidebarWidth, 0.0f},
            .children = {{.pane = SidebarTileId}, {.pane = PageTileId}},
        },
        .columns = true,
        .flexPane = PageTileId,
        .headers = false,
        .resizable = false,
    });
}

}  // namespace Jetstream
