#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_HH

#include "registry_add_plugin.hh"
#include "registry_plugin.hh"
#include "../components/compact_button.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"

#include "../../model/ui.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RegistrySettingsView {
    using PluginItem = RegistryPluginView::Item;
    using PluginRow = RegistryPluginView::Plugin;

    struct DomainRow {
        std::string domain;
        std::vector<std::string> blocks;
    };

    struct Config {
        std::vector<DomainRow> domains;
        std::vector<PluginRow> plugins;
        RegistryPage page = RegistryPage::Overview;
        std::string plugin;
        std::string updatePolicy = "prompt";
        Sakura::Callback<void(RegistryPage, const std::string&)> onNavigate;
        Sakura::Callback<void(const std::string&, Sakura::Callback<void(std::string)>)> onBrowse;
        Sakura::Callback<void(const std::string&)> onRegister;
        Sakura::Callback<void(const std::string&)> onRemovePlugin;
        Sakura::Callback<void(const std::string&)> onReloadPlugin;
        Sakura::Callback<void(const std::string&)> onUpdatePlugin;
        Sakura::Callback<void(const std::string&)> onUpdatePolicyChange;
        Sakura::Callback<void()> onBrowsePlugins;
    };

    void update(Config config);
    Sakura::Component& content();

 private:
    struct PluginEntry {
        CompactButton updateButton;
        Sakura::Text chevron;
        Sakura::Stack control;
    };

    static std::string updatePolicyLabel(const std::string& value);
    static std::string updatePolicyValue(const std::string& label);
    void updateOverview();
    Sakura::Callback<void()> back();
    void navigate(RegistryPage page, const std::string& plugin) const;
    const PluginRow* selectedPlugin() const;

    Config config;
    RegistryPage lastPage = RegistryPage::Overview;

    SettingsHeader header;
    SectionGroup domainGroup;
    std::vector<std::unique_ptr<PluginEntry>> entries;
    CompactButton browseButton;
    CompactButton addButton;
    Sakura::Stack pluginActions;
    SectionGroup pluginGroup;
    Sakura::Combo updatePolicyCombo;
    SectionGroup updatesGroup;
    RegistryAddPluginView addPage;
    RegistryPluginView pluginPage;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_HH
