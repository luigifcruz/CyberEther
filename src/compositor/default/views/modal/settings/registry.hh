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
        this->config = std::move(config);

        title.update({
            .id = "RegistryTitle",
            .str = "Registry",
            .font = Sakura::Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .id = "RegistryDescription",
            .str = "Registered block domains and the blocks available in each domain.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({
            .id = "RegistryHeaderDivider",
        });

        table.update({
            .id = "RegistryDomainTable",
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

        pluginTitle.update({
            .id = "RegistryPluginTitle",
            .str = "Plugins",
        });

        pluginDescription.update({
            .id = "RegistryPluginDescription",
            .str = "Load additional block domains from plugins at startup.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        pluginButton.update({
            .id = "RegistryPluginButton",
            .str = "Add Plugin",
            .size = {-1.0f, 34.0f},
            .onClick = this->config.onAddPlugin,
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
            .id = "RegistryPluginTable",
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

        emptyText.update({
            .id = "RegistryEmptyText",
            .str = "No registered blocks.",
            .tone = Sakura::Text::Tone::Disabled,
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);

        if (config.domains.empty()) {
            emptyText.render(ctx);
        } else {
            table.render(ctx);
        }

        divider.render(ctx);
        pluginTitle.render(ctx);
        pluginTable.render(ctx);
        pluginButton.render(ctx);
        pluginDescription.render(ctx);
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
    Sakura::Text pluginTitle;
    Sakura::Text pluginDescription;
    Sakura::Button pluginButton;
    ActionTable pluginTable;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
