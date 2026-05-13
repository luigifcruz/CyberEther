#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH

#include "jetstream/render/sakura/sakura.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RegistrySettingsPanel : public Sakura::Component {
    struct DomainRow {
        std::string domain;
        std::vector<std::string> blocks;
    };

    struct Config {
        std::vector<DomainRow> domains;
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
            return;
        }

        table.render(ctx);
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

    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Table table;
    Sakura::Text emptyText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_REGISTRY_HH
