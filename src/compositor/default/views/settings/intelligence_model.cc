#include "intelligence_model.hh"

namespace Jetstream {

void IntelligenceModelView::update(Config config) {
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
        .icon = ICON_LC_SEARCH,
        .value = this->config.search,
        .hint = "Filter models",
        .submit = Sakura::TextInput::Submit::OnEdit,
        .onChange = [this](const std::string& value) {
            if (this->config.onSearch) {
                this->config.onSearch(value);
            }
        },
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
            .icon = ICON_LC_PLUS,
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

Sakura::Component& IntelligenceModelView::content() {
    return stack;
}

}  // namespace Jetstream
