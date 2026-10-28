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

#include <functional>
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
        std::function<void(RegistryPage, const std::string&)> onNavigate;
        std::function<void(const std::string&, std::function<void(std::string)>)> onBrowse;
        std::function<void(const std::string&)> onRegister;
        std::function<void(const std::string&)> onRemovePlugin;
        std::function<void(const std::string&)> onReloadPlugin;
    };

    void update(Config config) {
        this->config = std::move(config);
        const PluginRow* plugin = selectedPlugin();
        RegistryPage page = this->config.page;
        if (page == RegistryPage::Plugin && !plugin) {
            page = RegistryPage::Overview;
        }
        if (page == RegistryPage::AddPlugin && lastPage != RegistryPage::AddPlugin) {
            addPage.reset();
        }
        lastPage = page;

        switch (page) {
            case RegistryPage::Overview:
                updateOverview();
                break;
            case RegistryPage::AddPlugin:
                addPage.update({
                    .onBrowse = this->config.onBrowse,
                    .onRegister = this->config.onRegister,
                    .onBack = back(),
                });
                stack.update({.items = {{.child = &addPage.content()}}});
                break;
            case RegistryPage::Plugin:
                pluginPage.update({
                    .plugin = *plugin,
                    .onReload = this->config.onReloadPlugin,
                    .onRemove = this->config.onRemovePlugin,
                    .onBack = back(),
                });
                stack.update({.items = {{.child = &pluginPage.content()}}});
                break;
        }
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    struct PluginEntry {
        Sakura::Text chevron;
    };

    void updateOverview() {
        header.update({
            .title = "Registry",
            .description = "Block domains available in this installation and the plugins that provide them.",
        });

        std::vector<Sakura::SettingCard::Row> domainRows;
        domainRows.reserve(config.domains.size());
        for (const auto& domain : config.domains) {
            domainRows.push_back({
                .label = domain.domain,
                .description = jst::fmt::format("{}", jst::fmt::join(domain.blocks, ", ")),
            });
        }
        domainGroup.update({
            .title = "Domains",
            .description = "Registered block domains and the blocks available in each domain.",
            .rows = std::move(domainRows),
            .empty = "No block domains are registered.",
        });

        while (entries.size() < config.plugins.size()) {
            entries.push_back(std::make_unique<PluginEntry>());
        }
        std::vector<Sakura::SettingCard::Row> pluginRows;
        pluginRows.reserve(config.plugins.size());
        for (U64 index = 0; index < config.plugins.size(); ++index) {
            const auto& plugin = config.plugins[index];
            auto& entry = *entries[index];
            entry.chevron.update({
                .str = ICON_LC_CHEVRON_RIGHT,
                .tone = Sakura::Text::Tone::Secondary,
                .scale = 0.8f,
            });
            pluginRows.push_back({
                .label = plugin.name,
                .description = RegistryPluginView::VersionLabel(plugin),
                .control = &entry.chevron,
                .onClick = [this, path = plugin.path]() {
                    navigate(RegistryPage::Plugin, path);
                },
            });
        }
        addButton.update({
            .str = "Add",
            .tone = CompactButton::Tone::Primary,
            .onClick = [this]() {
                navigate(RegistryPage::AddPlugin, "");
            },
        });
        pluginGroup.update({
            .title = "Plugins",
            .description = "Load additional block domains from plugins at startup.",
            .rows = std::move(pluginRows),
            .empty = "No plugins registered yet. Use Add to load one.",
            .action = &addButton,
        });

        stack.update({
            .spacing = 4.0f,
            .items = {
                {.child = &header},
                {.child = &domainGroup},
                {.child = &pluginGroup},
            },
        });
    }

    std::function<void()> back() {
        return [this]() {
            navigate(RegistryPage::Overview, "");
        };
    }

    void navigate(RegistryPage page, const std::string& path) const {
        if (config.onNavigate) {
            config.onNavigate(page, path);
        }
    }

    const PluginRow* selectedPlugin() const {
        for (const auto& plugin : config.plugins) {
            if (plugin.path == config.plugin) {
                return &plugin;
            }
        }
        return nullptr;
    }

    Config config;
    RegistryPage lastPage = RegistryPage::Overview;

    SettingsHeader header;
    SectionGroup domainGroup;
    std::vector<std::unique_ptr<PluginEntry>> entries;
    CompactButton addButton;
    SectionGroup pluginGroup;
    RegistryAddPluginView addPage;
    RegistryPluginView pluginPage;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_HH
