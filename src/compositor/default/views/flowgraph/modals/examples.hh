#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_EXAMPLES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_EXAMPLES_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream {

struct FlowgraphExampleCell : public Sakura::Div {
    struct Config {
        std::string title;
        std::string summary;
        std::function<void()> onOpen;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        this->config = std::move(config);
        title.update({
            .str = this->config.title,
            .font = Text::Font::Bold,
        });
        subtitle.update({
            .str = this->config.summary,
            .tone = Text::Tone::Secondary,
        });
        content.update({
            .items = {
                {.child = &title},
                {.child = &subtitle},
            },
        });
        Div::update({
            .content = &content,
            .padding = 11.0f,
            .rounding = 8.0f,
            .border = true,
            .onClick = [this]() {
                if (this->config.onOpen) {
                    this->config.onOpen();
                }
            },
        });
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Text subtitle;
    Sakura::Stack content;
};

struct FlowgraphExamplesView {
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
        using Text = Sakura::Text;

        this->config = std::move(config);
        header.update({
            .title = ICON_FA_STORE " Flowgraph Examples",
            .description = "Examples open in a fresh tab and start processing automatically once their inputs and devices are ready.",
        });
        emptyText.update({
            .str = "No example flowgraphs registered.",
            .tone = Text::Tone::Disabled,
            .align = Text::Align::Center,
        });
        while (exampleCells.size() < this->config.examples.size()) {
            exampleCells.push_back(std::make_unique<FlowgraphExampleCell>());
        }
        std::vector<Sakura::Component*> cells;
        for (U64 i = 0; i < this->config.examples.size(); ++i) {
            const auto& example = this->config.examples[i];
            exampleCells[i]->update({
                .title = example.title,
                .summary = example.summary,
                .onOpen = [this, i]() {
                    if (i < this->config.examples.size() && this->config.onOpen) {
                        this->config.onOpen(this->config.examples[i]);
                    }
                },
            });
            cells.push_back(exampleCells[i].get());
        }
        grid.update({
            .columns = 2,
            .size = {Sakura::Fill, 300.0f},
            .cellPadding = {4.0f, 4.0f},
            .items = std::move(cells),
        });
        stack.update({
            .items = {
                {.child = &header},
                {.child = this->config.examples.empty() ? static_cast<Sakura::Component*>(&emptyText) : &grid},
            },
        });
        panel.update({
            .id = "FlowgraphExamples",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

 private:
    Config config;
    ModalHeader header;
    Sakura::Text emptyText;
    std::vector<std::unique_ptr<FlowgraphExampleCell>> exampleCells;
    Sakura::Grid grid;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_EXAMPLES_HH
