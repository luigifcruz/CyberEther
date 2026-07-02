#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_KEY_VALUE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_KEY_VALUE_HH

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphKeyValueWindow {
    struct Config {
        std::string id;
        std::string title;
        std::string search;
        std::string searchHint = "Search keys...";
        std::string entryCount;
        std::vector<std::vector<std::string>> rows;
        std::function<void(const std::string&)> onSearchChange;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);

        window.update({
            .id = this->config.id,
            .title = this->config.title,
            .size = {500.0f, 700.0f},
            .padding = Extent2D<F32>{8.0f, 8.0f},
            .onClose = this->config.onClose,
        });
        searchInput.update({
            .id = this->config.id + ":search",
            .value = this->config.search,
            .hint = this->config.searchHint,
            .submit = Sakura::TextInput::Submit::OnEdit,
            .onChange = this->config.onSearchChange,
        });
        countText.update({
            .id = this->config.id + ":count",
            .str = this->config.entryCount,
            .tone = Sakura::Text::Tone::Secondary,
        });
        table.update({
            .id = this->config.id + ":table",
            .columns = {"Key", "Value"},
            .rows = this->config.rows,
            .fixedColumnWidths = {180.0f},
            .size = {0.0f, -1.0f},
            .wrapped = true,
        });
    }

    void render(const Sakura::Context& ctx) {
        window.render(ctx, [this](const Sakura::Context& ctx) {
            searchInput.render(ctx);
            countText.render(ctx);
            table.render(ctx);
        });
    }

 private:
    Config config;
    Sakura::Window window;
    Sakura::TextInput searchInput;
    Sakura::Text countText;
    Sakura::Table table;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_KEY_VALUE_HH
