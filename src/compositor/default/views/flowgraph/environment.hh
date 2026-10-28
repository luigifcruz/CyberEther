#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_ENVIRONMENT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_ENVIRONMENT_HH

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphEnvironment {
    struct Config {
        std::string id;
        std::string title;
        std::string search;
        std::string searchHint;
        std::string entryCount;
        Sakura::Table::Nodes nodes;
        std::function<void(const std::string&)> onSearchChange;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);

        searchInput.update({
            .value = this->config.search,
            .hint = this->config.searchHint,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = this->config.onSearchChange,
        });
        countText.update({
            .str = this->config.entryCount,
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

    Sakura::Component& component() {
        return panel;
    }

    const std::string& id() const {
        return config.id;
    }

    const std::string& title() const {
        return config.title;
    }

    std::function<void()> closeCallback() const {
        return config.onClose;
    }

 private:
    Config config;
    Sakura::TextInput searchInput;
    Sakura::Text countText;
    Sakura::Table table;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_ENVIRONMENT_HH
