#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_HH

#include "hud/welcome.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct StartTabView {
    static constexpr const char* PaneId = "start-tab";
    static constexpr const char* TileId = "start-tab:welcome";
    static constexpr F32 ContentWidth = 560.0f;

    struct Config {
        WelcomeHudView::Config welcome;
        std::function<void()> onClose;
        std::function<void()> onFocus;
    };

    void update(Config config) {
        onClose = std::move(config.onClose);
        onFocus = std::move(config.onFocus);
        welcome.update(std::move(config.welcome));
        center.update({
            .items = {{
                .child = &welcome.component(),
                .anchor = Sakura::Anchor::Center,
                .size = {ContentWidth, 0.0f},
            }},
        });
        tile.update({
            .panes = {{
                .id = TileId,
                .title = "Welcome",
                .group = TileId,
                .content = &center,
                .closable = false,
                .maximizable = false,
            }},
            .columns = true,
            .flexPane = TileId,
        });
    }

    Sakura::Dock::Pane pane() {
        return {
            .id = PaneId,
            .title = "New Tab",
            .group = "main",
            .content = &tile,
            .onClose = onClose,
            .onFocus = onFocus,
        };
    }

 private:
    std::function<void()> onClose;
    std::function<void()> onFocus;
    WelcomeHudView welcome;
    Sakura::Overlay center;
    Sakura::Dock tile;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_HH
