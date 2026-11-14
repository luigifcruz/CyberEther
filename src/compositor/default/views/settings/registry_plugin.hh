#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_PLUGIN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_PLUGIN_HH

#include "../components/compact_button.hh"
#include "components/settings_breadcrumb.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RegistryPluginView {
    struct Item {
        std::string title;
        std::string summary;
    };

    struct Plugin {
        std::string name;
        std::string version;
        StatusBadge::Config status;
        std::string path;
        std::string minimumVersion;
        std::vector<Item> blocks;
        std::vector<Item> examples;
    };

    struct Config {
        Plugin plugin;
        Sakura::Callback<void(const std::string&)> onReload;
        Sakura::Callback<void(const std::string&)> onRemove;
        Sakura::Callback<void()> onBack;
    };

    static std::string VersionLabel(const Plugin& plugin);
    void update(Config config);
    Sakura::Component& content();

 private:
    static std::vector<Sakura::SettingCard::Row> Rows(const std::vector<Item>& items);

    Config config;
    SettingsBreadcrumb breadcrumb;
    SettingsHeader header;
    CompactButton reloadButton;
    CompactButton removeButton;
    Sakura::Stack actions;
    StatusBadge statusBadge;
    std::array<Sakura::Text, 2> aboutValues;
    SectionGroup aboutGroup;
    SectionGroup blocksGroup;
    SectionGroup examplesGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_PLUGIN_HH
