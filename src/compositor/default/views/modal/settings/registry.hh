#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH

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

        pluginTable.update({
            .id = "RegistryPluginTable",
            .columns = {
                "Path",
                "Action",
            },
            .fixedColumnWidths = {
                0.0f,
                128.0f,
            },
            .wrapped = true,
        });

        pluginPathTexts.resize(this->config.plugins.size());
        pluginActionRows.resize(this->config.plugins.size());
        pluginReloadButtons.resize(this->config.plugins.size());
        pluginDeleteButtons.resize(this->config.plugins.size());
        for (U64 i = 0; i < this->config.plugins.size(); ++i) {
            const auto& plugin = this->config.plugins[i];
            pluginPathTexts[i].update({
                .id = "RegistryPluginPathText" + std::to_string(i),
                .str = plugin.path,
                .wrapped = true,
            });
            pluginActionRows[i].update({
                .id = "RegistryPluginActions" + std::to_string(i),
                .spacing = 8.0f,
            });
            pluginReloadButtons[i].update({
                .id = "RegistryPluginReload" + std::to_string(i),
                .str = "Reload",
                .variant = Sakura::Button::Variant::Text,
                .onClick = [this, path = plugin.path]() {
                    if (this->config.onReloadPlugin) {
                        this->config.onReloadPlugin(path);
                    }
                },
            });
            pluginDeleteButtons[i].update({
                .id = "RegistryPluginDelete" + std::to_string(i),
                .str = "Delete",
                .variant = Sakura::Button::Variant::Text,
                .onClick = [this, path = plugin.path]() {
                    if (this->config.onRemovePlugin) {
                        this->config.onRemovePlugin(path);
                    }
                },
            });
        }

        emptyPluginText.update({
            .id = "RegistryPluginEmptyText",
            .str = "No plugins registered.",
            .tone = Sakura::Text::Tone::Disabled,
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
        renderPluginTable(ctx);
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

    void renderPluginTable(const Sakura::Context& ctx) const {
        Sakura::Table::Rows rows;

        if (config.plugins.empty()) {
            Sakura::Table::Row row;
            row.push_back([this](const Sakura::Context& ctx) {
                emptyPluginText.render(ctx);
            });
            rows.push_back(std::move(row));
            pluginTable.render(ctx, std::move(rows));
            return;
        }

        rows.reserve(config.plugins.size());
        for (U64 i = 0; i < config.plugins.size(); ++i) {
            Sakura::Table::Row row;
            row.push_back([this, i](const Sakura::Context& ctx) {
                pluginPathTexts[i].render(ctx);
            });
            row.push_back([this, i](const Sakura::Context& ctx) {
                pluginActionRows[i].render(ctx, {
                    [this, i](const Sakura::Context& ctx) { pluginReloadButtons[i].render(ctx); },
                    [this, i](const Sakura::Context& ctx) { pluginDeleteButtons[i].render(ctx); },
                });
            });
            rows.push_back(std::move(row));
        }
        pluginTable.render(ctx, std::move(rows));
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
    Sakura::Table pluginTable;
    Sakura::Text emptyPluginText;
    std::vector<Sakura::Text> pluginPathTexts;
    std::vector<Sakura::HStack> pluginActionRows;
    std::vector<Sakura::Button> pluginReloadButtons;
    std::vector<Sakura::Button> pluginDeleteButtons;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
