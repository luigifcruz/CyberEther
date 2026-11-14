#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FLOWGRAPH_CELL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FLOWGRAPH_CELL_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphCell : public Sakura::Div {
    static constexpr F32 StarSize = 22.0f;

    struct Config {
        std::string title;
        bool favorite = false;
        std::string empty;
        Sakura::Callback<void()> onOpen;
        Sakura::Callback<void()> onToggleFavorite;
        Sakura::Callback<void()> onContextMenu;
    };

    void update(Config config);

 private:
    Config config;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Button star;
    Sakura::Stack row;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FLOWGRAPH_CELL_HH
