#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_HH

#include "components/settings_breadcrumb.hh"
#include "components/settings_group.hh"
#include "components/settings_header.hh"

#include "../../model/ui.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RegistrySettingsPanel {
    struct DomainRow {
        std::string domain;
        std::vector<std::string> blocks;
    };

    struct PluginItem {
        std::string title;
        std::string summary;
    };

    struct PluginRow {
        std::string name;
        std::string version;
        std::string status;
        std::string path;
        std::string minimumVersion;
        std::vector<PluginItem> blocks;
        std::vector<PluginItem> examples;
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
            pendingPath.clear();
            trusted = false;
        }
        lastPage = page;

        switch (page) {
            case RegistryPage::Overview:
                updateOverview();
                break;
            case RegistryPage::AddPlugin:
                updateAddPlugin();
                break;
            case RegistryPage::Plugin:
                updatePlugin(*plugin);
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
                .description = joinBlocks(domain.blocks),
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
                .str = ICON_FA_CHEVRON_RIGHT,
                .tone = Sakura::Text::Tone::Secondary,
                .scale = 0.8f,
            });
            pluginRows.push_back({
                .label = plugin.name,
                .description = versionLabel(plugin),
                .control = &entry.chevron,
                .onClick = [this, path = plugin.path]() {
                    navigate(RegistryPage::Plugin, path);
                },
            });
        }
        addButton.update(primaryButton("Add", false, [this]() {
            navigate(RegistryPage::AddPlugin, "");
        }));
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

    void updateAddPlugin() {
        updateBreadcrumb("Add Plugin");
        const bool ready = trusted && !pendingPath.empty();
        registerButton.update({
            .str = "Register Plugin",
            .size = {Sakura::Fill, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .disabled = !ready,
            .rounding = 12.0f,
            .onClick = [this]() {
                if (trusted && !pendingPath.empty() && config.onRegister) {
                    config.onRegister(pendingPath);
                }
            },
            .font = Sakura::Text::Font::SansBold,
        });
        pageHeader.update({
            .title = "Add Plugin",
            .description = "Load a plugin bundle that registers additional blocks.",
        });
        pathInput.update({
            .value = pendingPath,
            .hint = "/path/to/plugin.cep",
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = [this](const std::string& value) {
                setPendingPath(value);
            },
            .borderColorKey = "border",
        });
        browseButton.update({
            .str = "Browse",
            .onClick = [this]() {
                if (config.onBrowse) {
                    config.onBrowse(pendingPath, [this](std::string path) {
                        setPendingPath(std::move(path));
                    });
                }
            },
            .font = Sakura::Text::Font::Sans,
        });
        pathControls.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .items = {
                {.child = &pathInput, .size = 1.0f, .grow = 1.0f},
                {.child = &browseButton},
            },
        });
        pathField.update({
            .content = &pathControls,
            .border = false,
            .colorKey = "transparent",
            .hoveredColorKey = "transparent",
        });
        trustToggle.update({
            .value = trusted,
            .onChange = [this](bool value) {
                trusted = value;
            },
        });
        sourceGroup.update({
            .warning = "Only register plugins you built yourself or received from a source you trust.",
            .rows = {
                {
                    .label = "Plugin Path",
                    .description = "Path to a .cep plugin bundle that registers additional blocks.",
                    .control = &pathField,
                },
                {
                    .label = "Trust Source",
                    .description = "Confirm that you trust the source of this plugin.",
                    .control = &trustToggle,
                },
            },
            .footer = &registerButton,
        });
        stack.update({
            .spacing = 4.0f,
            .items = {
                {.child = &breadcrumb},
                {.child = &pageHeader},
                {.child = &sourceGroup},
            },
        });
    }

    void updatePlugin(const PluginRow& plugin) {
        updateBreadcrumb(plugin.name);
        reloadButton.update({
            .str = "Reload",
            .onClick = [this, path = plugin.path]() {
                if (config.onReloadPlugin) {
                    config.onReloadPlugin(path);
                }
            },
            .font = Sakura::Text::Font::Sans,
        });
        removeButton.update({
            .str = "Remove",
            .variant = Sakura::Button::Variant::Destructive,
            .onClick = [this, path = plugin.path]() {
                if (config.onRemovePlugin) {
                    config.onRemovePlugin(path);
                }
            },
            .font = Sakura::Text::Font::Sans,
        });
        pluginActions.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 6.0f,
            .items = {
                {.child = &reloadButton},
                {.child = &removeButton},
            },
        });
        pageHeader.update({
            .title = plugin.name,
            .description = versionLabel(plugin),
            .action = &pluginActions,
        });
        statusBadge.update(statusBadgeConfig(plugin.status));
        const std::array<std::string, 2> values = {
            plugin.version.empty() || plugin.version == "-" ? "Unknown" : plugin.version,
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
            .rows = itemRows(plugin.blocks),
            .empty = "This plugin does not register any blocks.",
        });
        examplesGroup.update({
            .title = "Examples",
            .description = "Example flowgraphs bundled with this plugin.",
            .rows = itemRows(plugin.examples),
            .empty = "This plugin does not include any examples.",
        });
        stack.update({
            .spacing = 4.0f,
            .items = {
                {.child = &breadcrumb},
                {.child = &pageHeader},
                {.child = &aboutGroup},
                {.child = &blocksGroup},
                {.child = &examplesGroup},
            },
        });
    }

    void updateBreadcrumb(const std::string& current) {
        breadcrumb.update({
            .root = "Registry",
            .current = current,
            .onRoot = [this]() {
                navigate(RegistryPage::Overview, "");
            },
        });
    }

    void navigate(RegistryPage page, const std::string& path) const {
        if (config.onNavigate) {
            config.onNavigate(page, path);
        }
    }

    void setPendingPath(std::string path) {
        if (path != pendingPath) {
            trusted = false;
        }
        pendingPath = std::move(path);
    }

    const PluginRow* selectedPlugin() const {
        for (const auto& plugin : config.plugins) {
            if (plugin.path == config.plugin) {
                return &plugin;
            }
        }
        return nullptr;
    }

    static Sakura::Button::Config primaryButton(std::string label, bool disabled, std::function<void()> onClick) {
        return {
            .str = std::move(label),
            .disabled = disabled,
            .colorKey = "primary_btn",
            .hoveredColorKey = "primary_btn_hovered",
            .activeColorKey = "primary_btn_hovered",
            .borderColorKey = "primary_btn",
            .textColorKey = "primary_btn_text",
            .padding = {14.0f, 5.0f},
            .onClick = std::move(onClick),
            .font = Sakura::Text::Font::Sans,
        };
    }

    static std::vector<Sakura::SettingCard::Row> itemRows(const std::vector<PluginItem>& items) {
        std::vector<Sakura::SettingCard::Row> rows;
        rows.reserve(items.size());
        for (const auto& item : items) {
            rows.push_back({.label = item.title, .description = item.summary});
        }
        return rows;
    }

    static std::string versionLabel(const PluginRow& plugin) {
        if (plugin.version.empty() || plugin.version == "-") {
            return "Unknown version";
        }
        return jst::fmt::format("Version {}", plugin.version);
    }

    static std::string joinBlocks(const std::vector<std::string>& blocks) {
        std::string value;
        for (U64 i = 0; i < blocks.size(); ++i) {
            if (i > 0) {
                value += ", ";
            }
            value += blocks[i];
        }
        return value;
    }

    static Sakura::Badge::Config statusBadgeConfig(const std::string& status) {
        if (status == "Not loaded") {
            return {
                .label = status,
                .backgroundColorKey = "header_hovered",
                .borderColorKey = "border",
                .textColorKey = "text_secondary",
            };
        }
        const std::string tone = status == "Loaded" ? "banner_success" : "banner_warning";
        return {
            .label = status,
            .backgroundColorKey = tone + "_bg",
            .borderColorKey = tone + "_border",
            .textColorKey = tone + "_text",
        };
    }

    Config config;
    RegistryPage lastPage = RegistryPage::Overview;
    std::string pendingPath;
    bool trusted = false;

    SettingsHeader header;
    SettingsGroup domainGroup;
    std::vector<std::unique_ptr<PluginEntry>> entries;
    Sakura::Button addButton;
    SettingsGroup pluginGroup;

    SettingsBreadcrumb breadcrumb;
    SettingsHeader pageHeader;

    Sakura::Button registerButton;
    Sakura::TextInput pathInput;
    Sakura::Button browseButton;
    Sakura::Stack pathControls;
    Sakura::Div pathField;
    Sakura::Toggle trustToggle;
    SettingsGroup sourceGroup;

    Sakura::Button reloadButton;
    Sakura::Button removeButton;
    Sakura::Stack pluginActions;
    Sakura::Badge statusBadge;
    std::array<Sakura::Text, 2> aboutValues;
    SettingsGroup aboutGroup;
    SettingsGroup blocksGroup;
    SettingsGroup examplesGroup;

    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_REGISTRY_HH
