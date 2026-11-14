#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_BREADCRUMB_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_BREADCRUMB_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct SettingsBreadcrumb : public Sakura::Stack {
    struct Config {
        std::string root;
        std::string current;
        Sakura::Callback<void()> onRoot;
    };

    void update(Config config);

 private:
    Sakura::Button rootButton;
    Sakura::Text chevron;
    Sakura::Text current;
    Sakura::Stack trail;
    Sakura::Spacer spacing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_BREADCRUMB_HH
