#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_EXAMPLES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_EXAMPLES_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <vector>

namespace Jetstream {

struct FlowgraphExampleCell : public Sakura::Component {
    struct Config {
        std::string id;
        std::string title;
        std::string summary;
        std::function<void()> onOpen;
    };

    void update(Config config) {
        this->config = std::move(config);

        div.update({
            .id = this->config.id + "Div",
            .padding = 12.0f,
            .rounding = 8.0f,
            .border = false,
            .scrollbar = false,
            .mouseScroll = false,
            .onClick = [this]() {
                if (this->config.onOpen) {
                    this->config.onOpen();
                }
            },
        });
        title.update({
            .id = this->config.id + "Title",
            .str = this->config.title,
            .font = Sakura::Text::Font::Bold,
        });
        subtitle.update({
            .id = this->config.id + "Subtitle",
            .str = this->config.summary,
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });
    }

    void render(const Sakura::Context& ctx) const {
        div.render(ctx, [this](const Sakura::Context& ctx) {
            title.render(ctx);
            subtitle.render(ctx);
        });
    }

 private:
    Config config;
    Sakura::Div div;
    Sakura::Text title;
    Sakura::Text subtitle;
};

struct FlowgraphExamplesView : public Sakura::Component {
    struct Example {
        std::string key;
        std::string title;
        std::string summary;
        std::string content;
    };

    struct Config {
        std::vector<Example> examples;
        std::function<void(const Example&)> onOpen;
    };

    void update(Config config) {
        this->config = std::move(config);
        header.update({
            .id = "FlowgraphExamplesHeader",
            .title = ICON_FA_STORE " Flowgraph Examples",
            .description = "Pick an example to bootstrap a new flowgraph. Examples open in a fresh tab so your work stays intact.",
        });
        emptyText.update({
            .id = "FlowgraphExamplesEmpty",
            .str = "No example flowgraphs registered.",
            .tone = Sakura::Text::Tone::Disabled,
            .align = Sakura::Text::Align::Center,
        });
        grid.update({
            .id = "FlowgraphExamplesGrid",
            .columns = 2,
            .size = {0.0f, 300.0f},
        });
        exampleCells.resize(this->config.examples.size());
        for (U64 i = 0; i < exampleCells.size(); ++i) {
            const auto& example = this->config.examples[i];
            exampleCells[i].update({
                .id = "FlowgraphExample" + example.key,
                .title = example.title,
                .summary = example.summary,
                .onOpen = [this, i]() {
                    if (i < this->config.examples.size() && this->config.onOpen) {
                        this->config.onOpen(this->config.examples[i]);
                    }
                },
            });
        }
    }

    void render(const Sakura::Context& ctx) const {
        header.render(ctx);
        if (config.examples.empty()) {
            emptyText.render(ctx);
        } else {
            Sakura::Grid::Children children;
            children.reserve(exampleCells.size());
            for (U64 i = 0; i < exampleCells.size(); ++i) {
                children.emplace_back([this, i](const Sakura::Context& ctx) {
                    exampleCells[i].render(ctx);
                });
            }
            grid.render(ctx, std::move(children));
        }
    }

 private:
    Config config;
    ModalHeader header;
    Sakura::Text emptyText;
    Sakura::Grid grid;
    std::vector<FlowgraphExampleCell> exampleCells;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_EXAMPLES_HH
