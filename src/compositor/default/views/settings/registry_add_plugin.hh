#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_ADD_PLUGIN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_ADD_PLUGIN_HH

#include "../components/file_field.hh"
#include "components/settings_breadcrumb.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct RegistryAddPluginView {
    struct Config {
        Sakura::Callback<void(const std::string&, Sakura::Callback<void(std::string)>)> onBrowse;
        Sakura::Callback<void(const std::string&)> onRegister;
        Sakura::Callback<void()> onBack;
    };

    void reset();
    void update(Config config);
    Sakura::Component& content();

 private:
    void setPendingPath(std::string path);

    Config config;
    std::string pendingPath;
    bool trusted = false;
    SettingsBreadcrumb breadcrumb;
    SettingsHeader header;
    Sakura::Button registerButton;
    FileField pathField;
    Sakura::Toggle trustToggle;
    SectionGroup sourceGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_ADD_PLUGIN_HH
