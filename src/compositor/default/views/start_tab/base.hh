#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_BASE_HH

#include "welcome.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <utility>

namespace Jetstream {

struct StartTabView {
    static constexpr const char* PaneId = "start-tab";

    struct Config {
        WelcomeView::Config welcome;
        bool closable = true;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void()> onFocus;
    };

    void update(Config config) {
        closable = config.closable;
        onClose = std::move(config.onClose);
        onFocus = std::move(config.onFocus);
        welcome.update(std::move(config.welcome));
    }

    Sakura::Dock::Pane pane() {
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

 private:
    bool closable = true;
    Sakura::Callback<void()> onClose;
    Sakura::Callback<void()> onFocus;
    WelcomeView welcome;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_BASE_HH
