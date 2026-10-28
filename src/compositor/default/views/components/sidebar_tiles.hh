#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_TILES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_TILES_HH

#include "jetstream/render/sakura/base.hh"

namespace Jetstream {

struct SidebarTiles : public Sakura::Dock {
    static constexpr F32 SidebarWidth = 330.0f;
    static constexpr const char* SidebarTileId = "sidebar";
    static constexpr const char* PageTileId = "page";

    struct Config {
        Sakura::Component* sidebar = nullptr;
        Sakura::Component* page = nullptr;
        bool pageCard = true;
    };

    void update(Config config) {
        Dock::update({
            .panes = {
                {
                    .id = SidebarTileId,
                    .group = SidebarTileId,
                    .content = config.sidebar,
                    .closable = false,
                    .maximizable = false,
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
                .weights = {SidebarWidth, 0.0f},
                .children = {{.pane = SidebarTileId}, {.pane = PageTileId}},
            },
            .columns = true,
            .flexPane = PageTileId,
            .headers = false,
            .resizable = false,
        });
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_TILES_HH
