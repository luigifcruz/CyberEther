#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_PLUGIN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_PLUGIN_HH

#include "components/settings_breadcrumb.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"

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

    static std::string VersionLabel(const Plugin& plugin) {
        if (plugin.version.empty()) {
            return "Unknown version";
        }
        return jst::fmt::format("Version {}", plugin.version);
    }

    void update(Config config) {
        this->config = std::move(config);
        const auto& plugin = this->config.plugin;
        breadcrumb.update({
            .root = "Registry",
            .current = plugin.name,
            .onRoot = this->config.onBack,
        });
        reloadButton.update({
            .str = "Reload",
            .onClick = [this]() {
                if (this->config.onReload) {
                    this->config.onReload(this->config.plugin.path);
                }
            },
            .font = Sakura::Text::Font::Sans,
        });
        removeButton.update({
            .str = "Remove",
            .variant = Sakura::Button::Variant::Destructive,
            .onClick = [this]() {
                if (this->config.onRemove) {
                    this->config.onRemove(this->config.plugin.path);
                }
            },
            .font = Sakura::Text::Font::Sans,
        });
        actions.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &reloadButton},
                {.child = &removeButton},
            },
        });
        header.update({
            .title = plugin.name,
            .description = VersionLabel(plugin),
            .action = &actions,
        });
        statusBadge.update(plugin.status);
        const std::array<std::string, 2> values = {
            plugin.version.empty() ? "Unknown" : plugin.version,
            plugin.minimumVersion.empty() ? "Unknown" : plugin.minimumVersion,
        };
        for (U64 index = 0; index < values.size(); ++index) {
            aboutValues[index].update({
                .str = values[index],
                .font = Sakura::Text::Font::Sans,
            });
        }
        aboutGroup.update({
            .title = "About",
            .description = "Details reported by the plugin bundle.",
            .rows = {
                {.label = "Status", .control = &statusBadge},
                {.label = "Version", .control = &aboutValues[0]},
                {.label = "Minimum CyberEther", .control = &aboutValues[1]},
                {.label = "Location", .description = plugin.path},
            },
        });
        blocksGroup.update({
            .title = "Blocks",
            .description = "Blocks this plugin adds to the block catalog.",
            .rows = Rows(plugin.blocks),
            .empty = "This plugin does not register any blocks.",
        });
        examplesGroup.update({
            .title = "Examples",
            .description = "Example flowgraphs bundled with this plugin.",
            .rows = Rows(plugin.examples),
            .empty = "This plugin does not include any examples.",
        });
        stack.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = {
                {.child = &breadcrumb},
                {.child = &header},
                {.child = &aboutGroup},
                {.child = &blocksGroup},
                {.child = &examplesGroup},
            },
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    static std::vector<Sakura::SettingCard::Row> Rows(const std::vector<Item>& items) {
        std::vector<Sakura::SettingCard::Row> rows;
        rows.reserve(items.size());
        for (const auto& item : items) {
            rows.push_back({.label = item.title, .description = item.summary});
        }
        return rows;
    }

    Config config;
    SettingsBreadcrumb breadcrumb;
    SettingsHeader header;
    Sakura::Button reloadButton;
    Sakura::Button removeButton;
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
