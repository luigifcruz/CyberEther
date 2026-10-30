#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH

#include "components/action_table.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
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
        emptyText.update({
            .str = "No registered blocks.",
            .tone = Text::Tone::Disabled,
        });

        std::vector<ActionTable::Row> pluginRows;
        pluginRows.reserve(this->config.plugins.size());
        for (const auto& plugin : this->config.plugins) {
            pluginRows.push_back({
                .cells = {
                    {.str = plugin.name},
                    {.str = plugin.version},
                    {.str = plugin.status, .tone = statusTone(plugin.status)},
                },
                .actions = {
                    {
                        .str = "Reload",
                        .onClick = [this, path = plugin.path]() {
                            if (this->config.onReloadPlugin) {
                                this->config.onReloadPlugin(path);
                            }
                        },
                    },
                    {
                        .str = "Delete",
                        .onClick = [this, path = plugin.path]() {
                            if (this->config.onRemovePlugin) {
                                this->config.onRemovePlugin(path);
                            }
                        },
                    },
                },
            });
        }
        pluginTable.update({
            .columns = {
                "Plugin",
                "Version",
                "Status",
                "Action",
            },
            .fixedColumnWidths = {
                0.0f,
                80.0f,
                110.0f,
                128.0f,
            },
            .empty = "No plugins registered.",
            .rows = std::move(pluginRows),
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
    ActionTable pluginTable;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
