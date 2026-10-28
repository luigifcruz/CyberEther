#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_MODEL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_MODEL_HH

#include "../components/compact_button.hh"
#include "components/settings_breadcrumb.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/settings.hh"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceModelView {
    using Option = Settings::Intelligence::Model;

    struct Item {
        Option model;
        std::string label;
    };

    struct Config {
        std::string search;
        std::vector<Item> models;
        std::string status;
        bool warning = false;
        std::function<void(const std::string&)> onSearch;
        std::function<void(const Option&)> onCommit;
        std::function<void()> onBack;
    };

    void update(Config config) {
        this->config = std::move(config);
        const auto& models = this->config.models;

        breadcrumb.update({
            .root = "Intelligence",
            .current = "Add Model",
            .onRoot = this->config.onBack,
        });
        header.update({
            .title = "Add Model",
            .description = "Pick a model from the configured providers.",
        });
        filterInput.update({
            .value = this->config.search,
            .hint = "Filter models",
            .submit = Sakura::TextInput::Submit::OnEdit,
            .height = 34.0f,
            .onChange = [this](const std::string& value) {
                if (this->config.onSearch) {
                    this->config.onSearch(value);
                }
            },
            .borderColorKey = "border",
            .rounding = Sakura::Rounding::Card,
        });

        while (entries.size() < models.size()) {
            entries.push_back(std::make_unique<Entry>());
        }
        std::vector<Sakura::SettingCard::Row> rows;
        rows.reserve(models.size());
        for (U64 index = 0; index < models.size(); ++index) {
            const auto& item = models[index];
            auto& entry = *entries[index];
            entry.id.update({.label = item.model.id});
            entry.add.update({
                .str = "Add",
                .onClick = [this, model = item.model]() {
                    if (this->config.onCommit) {
                        this->config.onCommit(model);
                    }
                },
            });
            entry.row.update({
                .axis = Sakura::Stack::Axis::Horizontal,
                .spacing = Sakura::Spacing::Small,
                .items = {
                    {.child = &entry.id, .shrink = 1.0f},
                    {.child = &entry.add},
                },
            });
            rows.push_back({
                .label = item.label,
                .description = item.model.provider,
                .control = &entry.row,
                .wide = true,
            });
        }
        modelsGroup.update({
            .title = "Available Models",
            .description = this->config.warning ? "" : this->config.status,
            .warning = this->config.warning ? this->config.status : "",
            .rows = std::move(rows),
            .empty = "No models match the filter.",
            .toolbar = &filterInput,
        });

        stack.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = {
                {.child = &breadcrumb},
                {.child = &header},
                {.child = &modelsGroup},
            },
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    struct Entry {
        StatusBadge id;
        CompactButton add;
        Sakura::Stack row;
    };

    Config config;
    SettingsBreadcrumb breadcrumb;
    SettingsHeader header;
    Sakura::TextInput filterInput;
    std::vector<std::unique_ptr<Entry>> entries;
    SectionGroup modelsGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_MODEL_HH
