#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH

#include "jetstream/render/sakura/base.hh"

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

    struct PluginRow {
        std::string name;
        std::string version;
        std::string status;
        std::string path;
    };

    struct Config {
        std::vector<DomainRow> domains;
        std::vector<PluginRow> plugins;
        std::function<void()> onAddPlugin;
        std::function<void(const std::string&)> onRemovePlugin;
        std::function<void(const std::string&)> onReloadPlugin;
    };

    void update(Config config) {
        using Text = Sakura::Text;
        using Stack = Sakura::Stack;
        using Button = Sakura::Button;

        this->config = std::move(config);
        title.update({
            .str = "Registry",
            .font = Text::Font::Bold,
            .scale = 1.2f,
        });
        description.update({
            .str = "Registered block domains and the blocks available in each domain.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });
        divider.update({});
        pluginDivider.update({});
        table.update({
            .columns = {
                "Domain",
                "Blocks",
            },
            .rows = buildRows(),
            .fixedColumnWidths = {
                180.0f,
                0.0f,
            },
            .wrapped = true,
        });
        pluginTitle.update({.str = "Plugins"});
        pluginDescription.update({
            .str = "Load additional block domains from plugins at startup.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });
        pluginButton.update({
            .str = "Add Plugin",
            .size = {Sakura::Fill, 34.0f},
            .onClick = this->config.onAddPlugin,
        });
        emptyPluginText.update({
            .str = "No plugins registered.",
            .tone = Text::Tone::Disabled,
        });
        emptyText.update({
            .str = "No registered blocks.",
            .tone = Text::Tone::Disabled,
        });

        while (plugins.size() < this->config.plugins.size()) {
            plugins.push_back(std::make_unique<PluginCells>());
        }
        std::vector<std::vector<Sakura::Component*>> cellRows;
        if (this->config.plugins.empty()) {
            cellRows.push_back({&emptyPluginText});
        }
        for (U64 i = 0; i < this->config.plugins.size(); ++i) {
            const auto& plugin = this->config.plugins[i];
            auto& cells = *plugins[i];
            cells.name.update({
                .str = plugin.name,
                .wrapped = true,
            });
            cells.version.update({
                .str = plugin.version,
                .wrapped = true,
            });
            cells.status.update({
                .str = plugin.status,
                .tone = statusTone(plugin.status),
                .wrapped = true,
            });
            cells.reload.update({
                .str = "Reload",
                .variant = Button::Variant::Text,
                .onClick = [this, path = plugin.path]() {
                    if (this->config.onReloadPlugin) {
                        this->config.onReloadPlugin(path);
                    }
                },
            });
            cells.remove.update({
                .str = "Delete",
                .variant = Button::Variant::Text,
                .onClick = [this, path = plugin.path]() {
                    if (this->config.onRemovePlugin) {
                        this->config.onRemovePlugin(path);
                    }
                },
            });
            cells.actions.update({
                .axis = Stack::Axis::Horizontal,
                .spacing = 16.0f,
                .items = {
                    {.child = &cells.reload},
                    {.child = &cells.remove},
                },
            });
            cellRows.push_back({&cells.name, &cells.version, &cells.status, &cells.actions});
        }
        pluginTable.update({
            .columns = {
                "Plugin",
                "Version",
                "Status",
                "Action",
            },
            .cellRows = std::move(cellRows),
            .fixedColumnWidths = {
                0.0f,
                80.0f,
                110.0f,
                128.0f,
            },
            .wrapped = true,
        });

        stack.update({
            .items = {
                {.child = &title},
                {.child = &description},
                {.child = &divider},
                {.child = this->config.domains.empty() ? static_cast<Sakura::Component*>(&emptyText) : &table},
                {.child = &pluginDivider},
                {.child = &pluginTitle},
                {.child = &pluginTable},
                {.child = &pluginButton},
                {.child = &pluginDescription},
            },
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
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

    static Sakura::Text::Tone statusTone(const std::string& status) {
        if (status == "Loaded") {
            return Sakura::Text::Tone::Success;
        }

        if (status == "Not loaded") {
            return Sakura::Text::Tone::Disabled;
        }

        return Sakura::Text::Tone::Warning;
    }

    std::vector<std::vector<std::string>> buildRows() const {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(config.domains.size());
        for (const auto& domain : config.domains) {
            rows.push_back({
                domain.domain,
                joinBlocks(domain.blocks),
            });
        }
        return rows;
    }

    struct PluginCells {
        Sakura::Text name;
        Sakura::Text version;
        Sakura::Text status;
        Sakura::Button reload;
        Sakura::Button remove;
        Sakura::Stack actions;
    };

    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Table table;
    Sakura::Text emptyText;
    Sakura::Divider pluginDivider;
    Sakura::Text pluginTitle;
    Sakura::Text pluginDescription;
    Sakura::Button pluginButton;
    Sakura::Text emptyPluginText;
    std::vector<std::unique_ptr<PluginCells>> plugins;
    Sakura::Table pluginTable;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
