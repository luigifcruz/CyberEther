#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_INTELLIGENCE_MODEL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_INTELLIGENCE_MODEL_HH

#include "../components/modal_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"
#include "jetstream/settings.hh"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceModelView {
    using Option = Settings::Intelligence::Model;

    struct Config {
        std::string search;
        std::vector<Option> models;
        std::string status;
        bool warning = false;
        std::function<void(const std::string&)> onSearch;
        std::function<void(const Option&)> onCommit;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        this->config = std::move(config);
        const auto& options = this->config.models;

        header.update({
            .title = ICON_FA_ROBOT " Add Model",
            .description = "Pick a model from the configured providers.",
        });

        filterInput.update({
            .value = this->config.search,
            .hint = "Filter models...",
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = [this](const std::string& value) {
                if (this->config.onSearch) {
                    this->config.onSearch(value);
                }
            },
        });

        while (optionCells.size() < options.size()) {
            optionCells.push_back(std::make_unique<OptionCells>());
        }
        std::vector<std::vector<Sakura::Component*>> cellRows;
        cellRows.reserve(options.size());
        for (U64 i = 0; i < options.size(); ++i) {
            auto& cells = *optionCells[i];
            cells.provider.update({
                .str = options[i].provider,
                .wrapped = true,
            });
            cells.id.update({
                .str = options[i].id,
                .wrapped = true,
            });
            cells.select.update({
                .str = "Add",
                .variant = Sakura::Button::Variant::Text,
                .onClick = [this, option = options[i]]() {
                    if (this->config.onCommit) {
                        this->config.onCommit(option);
                    }
                },
            });
            cellRows.push_back({&cells.provider, &cells.id, &cells.select});
        }
        table.update({
            .columns = {
                "Provider",
                "Model",
                "Action",
            },
            .cellRows = std::move(cellRows),
            .fixedColumnWidths = {
                150.0f,
                0.0f,
                80.0f,
            },
            .size = {Sakura::Fill, 320.0f},
            .wrapped = true,
        });

        statusText.update({
            .str = this->config.status,
            .tone = this->config.warning ? Text::Tone::Warning : Text::Tone::Disabled,
            .wrapped = true,
        });

        stack.update({
            .items = {
                {.child = &header},
                {.child = &filterInput},
                {.child = &table},
                {.child = &statusText},
            },
        });
        panel.update({
            .id = "IntelligenceModel",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

 private:
    struct OptionCells {
        Sakura::Text provider;
        Sakura::Text id;
        Sakura::Button select;
    };

    Config config;
    ModalHeader header;
    Sakura::TextInput filterInput;
    std::vector<std::unique_ptr<OptionCells>> optionCells;
    Sakura::Table table;
    Sakura::Text statusText;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_INTELLIGENCE_MODEL_HH
