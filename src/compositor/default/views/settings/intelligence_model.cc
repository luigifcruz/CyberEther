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
    std::vector<std::string> providers;
    std::vector<std::vector<Sakura::SettingCard::Row>> groups;
    for (U64 index = 0; index < models.size(); ++index) {
        const auto& item = models[index];
        auto& entry = *entries[index];
        entry.id.update({
            .label = item.model.id,
            .icon = ICON_LC_COPY,
            .onIcon = [this, id = item.model.id]() {
                if (this->config.onCopy) {
                    this->config.onCopy(id);
                }
            },
        });
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
        const auto found = std::find(providers.begin(), providers.end(), item.model.provider);
        const auto group = static_cast<U64>(found - providers.begin());
        if (found == providers.end()) {
            providers.push_back(item.model.provider);
            groups.emplace_back();
        }
        groups[group].push_back({
            .label = item.label,
            .control = &entry.row,
            .wide = true,
        });
    }
    modelsGroup.update({
        .title = "Available Models",
        .description = this->config.warning ? "" : this->config.status,
        .warning = this->config.warning ? this->config.status : "",
        .empty = models.empty() ? "No models match the filter." : "",
        .toolbar = &filterInput,
    });

    while (providerGroups.size() < groups.size()) {
        providerGroups.push_back(std::make_unique<SectionGroup>());
    }
    std::vector<Sakura::Stack::Item> items{
        {.child = &breadcrumb},
        {.child = &header},
        {.child = &modelsGroup},
    };
    for (U64 index = 0; index < groups.size(); ++index) {
        const U64 count = groups[index].size();
        providerGroups[index]->update({
            .title = providers[index],
            .description = jst::fmt::format("{} {}", count, count == 1 ? "model" : "models"),
            .rows = std::move(groups[index]),
        });
        items.push_back({.child = providerGroups[index].get()});
    }

    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = std::move(items),
    });
}

Sakura::Component& IntelligenceModelView::content() {
    return stack;
}

}  // namespace Jetstream
