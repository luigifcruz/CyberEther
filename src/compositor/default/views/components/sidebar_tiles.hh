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
        bool sidebarOpen = true;
    };

    void update(Config config);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_SIDEBAR_TILES_HH
