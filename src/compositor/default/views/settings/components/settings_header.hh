#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_HEADER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_HEADER_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct SettingsHeader : public Sakura::Stack {
    struct Config {
        std::string title;
        std::string description;
        Sakura::Component* action = nullptr;
    };

    void update(Config config);

 private:
    Sakura::Text title;
    Sakura::Stack textColumn;
    Sakura::Spacer titleSpacing;
    Sakura::Stack titleRow;
    Sakura::Text description;
    Sakura::Spacer trailing;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_SETTINGS_HEADER_HH
