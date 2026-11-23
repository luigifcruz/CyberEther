#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_MISSING_PLUGINS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_MISSING_PLUGINS_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/types.hh"
#include "tools/text.hh"

#include <string>
#include <vector>

namespace Jetstream {

struct MissingPluginsView {
    static constexpr F32 Width = 480.0f;
    static constexpr F32 ButtonHeight = 34.0f;

    struct Config {
        std::vector<std::string> plugins;
        bool installing = false;
        Sakura::Callback<void()> onInstall;
        Sakura::Callback<void()> onCancel;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Stack titleRow;
    Sakura::Text description;
    Sakura::Stack header;
    Sakura::Spacer footerSpacer;
    Sakura::PillButton cancel;
    Sakura::PillButton install;
    Sakura::Stack footer;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_MISSING_PLUGINS_HH
