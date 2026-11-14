#include "environment.hh"

namespace Jetstream {

void FlowgraphEnvironment::update(Config config) {
    this->config = std::move(config);

    searchInput.update({
        .value = this->config.search,
        .hint = this->config.searchHint,
        .submit = Sakura::TextInput::Submit::OnEdit,
        .onChange = this->config.onSearchChange,
    });
    countText.update({
        .str = this->config.entryCount,
        .font = Sakura::Text::Font::Sans,
        .tone = Sakura::Text::Tone::Secondary,
    });
    table.update({
        .columns = {"Key", "Value"},
        .nodes = this->config.nodes,
        .fixedColumnWidths = {180.0f},
        .size = {Sakura::Fill, Sakura::Fill},
        .wrapped = true,
        .resizable = true,
    });
    stack.update({
        .items = {
            {.child = &searchInput},
            {.child = &countText},
            {.child = &table, .grow = 1.0f},
        },
    });
    panel.update({
        .id = this->config.id + ":panel",
        .content = &stack,
        .size = {Sakura::Fill, Sakura::Fill},
        .transparent = true,
    });
}

Sakura::Component& FlowgraphEnvironment::component() {
    return panel;
}

const std::string& FlowgraphEnvironment::id() const {
    return config.id;
}

const std::string& FlowgraphEnvironment::title() const {
    return config.title;
}

Sakura::Callback<void()> FlowgraphEnvironment::closeCallback() const {
    return config.onClose;
}

}  // namespace Jetstream
