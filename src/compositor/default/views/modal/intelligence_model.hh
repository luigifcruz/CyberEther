#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_INTELLIGENCE_MODEL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_INTELLIGENCE_MODEL_HH

#include "../components/modal_header.hh"

#include "jetstream/logger.hh"
#include "jetstream/settings.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
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
        this->config = std::move(config);
        const auto& options = this->config.models;

        header.update({
            .id = "IntelligenceModelHeader",
            .title = ICON_FA_ROBOT " Add Model",
            .description = "Pick a model from the configured providers.",
        });

        filterInput.update({
            .id = "##intelligence-model-filter",
            .value = this->config.search,
            .hint = "Filter models...",
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = this->config.onSearch,
        });

        table.update({
            .id = "IntelligenceModelTable",
            .columns = {
                "Provider",
                "Model",
                "Action",
            },
            .fixedColumnWidths = {
                150.0f,
                0.0f,
                80.0f,
            },
            .size = {0.0f, 320.0f},
            .wrapped = true,
        });

        optionProviderTexts.resize(options.size());
        optionIdTexts.resize(options.size());
        optionSelectButtons.resize(options.size());
        for (U64 i = 0; i < options.size(); ++i) {
            optionProviderTexts[i].update({
                .id = jst::fmt::format("IntelligenceModelOptionProvider{}", i),
                .str = options[i].provider,
                .wrapped = true,
            });
            optionIdTexts[i].update({
                .id = jst::fmt::format("IntelligenceModelOptionId{}", i),
                .str = options[i].id,
                .wrapped = true,
            });
            optionSelectButtons[i].update({
                .id = jst::fmt::format("IntelligenceModelOptionSelect{}", i),
                .str = "Add",
                .variant = Sakura::Button::Variant::Text,
                .onClick = [this, option = options[i]]() {
                    if (this->config.onCommit) {
                        this->config.onCommit(option);
                    }
                },
            });
        }

        statusText.update({
            .id = "IntelligenceModelStatusText",
            .str = this->config.status,
            .tone = this->config.warning ? Sakura::Text::Tone::Warning : Sakura::Text::Tone::Disabled,
            .wrapped = true,
        });
    }

    void render(const Sakura::Context& ctx) const {
        header.render(ctx);
        filterInput.render(ctx);
        Sakura::Table::Rows rows;
        rows.reserve(config.models.size());
        for (U64 i = 0; i < config.models.size(); ++i) {
            Sakura::Table::Row row;
            row.push_back([this, i](const Sakura::Context& ctx) {
                optionProviderTexts[i].render(ctx);
            });
            row.push_back([this, i](const Sakura::Context& ctx) {
                optionIdTexts[i].render(ctx);
            });
            row.push_back([this, i](const Sakura::Context& ctx) {
                optionSelectButtons[i].render(ctx);
            });
            rows.push_back(std::move(row));
        }
        table.render(ctx, std::move(rows));
        statusText.render(ctx);
    }

 private:
    Config config;
    ModalHeader header;
    Sakura::TextInput filterInput;
    Sakura::Table table;
    std::vector<Sakura::Text> optionProviderTexts;
    std::vector<Sakura::Text> optionIdTexts;
    std::vector<Sakura::Button> optionSelectButtons;
    Sakura::Text statusText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_INTELLIGENCE_MODEL_HH
