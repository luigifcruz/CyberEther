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

    void update(Config config);
    Sakura::Dock::Pane pane();

 private:
    bool closable = true;
    Sakura::Callback<void()> onClose;
    Sakura::Callback<void()> onFocus;
    WelcomeView welcome;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_BASE_HH
