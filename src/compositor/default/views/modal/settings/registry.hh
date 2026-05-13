#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RegistrySettingsPanel : public Sakura::Component {
    struct DomainRow {
        std::string domain;
        std::vector<std::string> blocks;
    };

    struct LibraryRow {
        std::string path;
    };

    struct Config {
        std::vector<DomainRow> domains;
        std::vector<LibraryRow> dynamicLibraries;
        std::function<void()> onAddLibrary;
        std::function<void(const std::string&)> onRemoveLibrary;
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = "RegistryTitle",
            .str = "Registry",
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

        libraryTitle.update({
            .id = "RegistryLibraryTitle",
            .str = "Dynamic Libraries",
        });

        libraryDescription.update({
            .id = "RegistryLibraryDescription",
            .str = "Load additional block domains from shared libraries at startup.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        libraryButton.update({
            .id = "RegistryLibraryButton",
            .str = ICON_FA_FOLDER_OPEN " Add Library",
            .size = {-1.0f, 34.0f},
            .onClick = this->config.onAddLibrary,
        });

        libraryTable.update({
            .id = "RegistryLibraryTable",
            .columns = {
                "Path",
                "Action",
            },
            .fixedColumnWidths = {
                0.0f,
                90.0f,
            },
            .wrapped = true,
        });

        libraryPathTexts.resize(this->config.dynamicLibraries.size());
        libraryDeleteButtons.resize(this->config.dynamicLibraries.size());
        for (U64 i = 0; i < this->config.dynamicLibraries.size(); ++i) {
            const auto& library = this->config.dynamicLibraries[i];
            libraryPathTexts[i].update({
                .id = "RegistryLibraryPathText" + std::to_string(i),
                .str = library.path,
                .wrapped = true,
            });
            libraryDeleteButtons[i].update({
                .id = "RegistryLibraryDelete" + std::to_string(i),
                .str = "Delete",
                .variant = Sakura::Button::Variant::Text,
                .onClick = [this, path = library.path]() {
                    if (this->config.onRemoveLibrary) {
                        this->config.onRemoveLibrary(path);
                    }
                },
            });
        }

        emptyLibraryText.update({
            .id = "RegistryLibraryEmptyText",
            .str = "No dynamic libraries registered.",
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
        libraryTitle.render(ctx);
        renderLibraryTable(ctx);
        libraryButton.render(ctx);
        libraryDescription.render(ctx);
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

    void renderLibraryTable(const Sakura::Context& ctx) const {
        Sakura::Table::Rows rows;

        if (config.dynamicLibraries.empty()) {
            Sakura::Table::Row row;
            row.push_back([this](const Sakura::Context& ctx) {
                emptyLibraryText.render(ctx);
            });
            rows.push_back(std::move(row));
            libraryTable.render(ctx, std::move(rows));
            return;
        }

        rows.reserve(config.dynamicLibraries.size());
        for (U64 i = 0; i < config.dynamicLibraries.size(); ++i) {
            Sakura::Table::Row row;
            row.push_back([this, i](const Sakura::Context& ctx) {
                libraryPathTexts[i].render(ctx);
            });
            row.push_back([this, i](const Sakura::Context& ctx) {
                libraryDeleteButtons[i].render(ctx);
            });
            rows.push_back(std::move(row));
        }
        libraryTable.render(ctx, std::move(rows));
    }

    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Table table;
    Sakura::Text emptyText;
    Sakura::Text libraryTitle;
    Sakura::Text libraryDescription;
    Sakura::Button libraryButton;
    Sakura::Table libraryTable;
    Sakura::Text emptyLibraryText;
    std::vector<Sakura::Text> libraryPathTexts;
    std::vector<Sakura::Button> libraryDeleteButtons;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
