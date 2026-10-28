#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_BENCHMARK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_BENCHMARK_HH

#include "../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include "jetstream/benchmark.hh"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct BenchmarkView {
    struct Config {
        bool running = false;
        F32 progress = 0.0f;
        std::vector<std::string> moduleOptions;
        std::string selectedModule;
        Benchmark::ResultMapType results;
        std::function<void()> onRun;
        std::function<void()> onReset;
        std::function<void(const std::string&)> onModuleChange;
    };

    void update(Config config) {
        using Text = Sakura::Text;
        using Button = Sakura::Button;
        using Stack = Sakura::Stack;

        this->config = std::move(config);
        header.update({
            .title = ICON_LC_GAUGE " Module Benchmarks",
            .description = "Run performance benchmarks for registered modules. Results show operations per second and timing information.",
        });
        moduleLabel.update({
            .str = "Module",
            .font = Text::Font::Bold,
        });
        moduleCombo.update({
            .id = "benchmark:module",
            .options = this->config.moduleOptions,
            .value = this->config.selectedModule,
            .disabled = this->config.running,
            .onChange = [this](const std::string& value) {
                if (this->config.onModuleChange) {
                    this->config.onModuleChange(value);
                }
            },
        });
        runButton.update({
            .str = ICON_LC_PLAY " Run Benchmarks",
            .size = {Sakura::Fill, 36.0f},
            .variant = Button::Variant::Action,
            .disabled = this->config.running,
            .onClick = [this]() {
                if (this->config.onRun) {
                    this->config.onRun();
                }
            },
        });
        resetButton.update({
            .str = ICON_LC_ROTATE_CCW " Reset Results",
            .size = {Sakura::Fill, 36.0f},
            .disabled = this->config.running || this->config.results.empty(),
            .onClick = [this]() {
                if (this->config.onReset) {
                    this->config.onReset();
                }
            },
        });
        progressBar.update({
            .value = this->config.progress,
            .overlay = jst::fmt::format("{:.0f}%", this->config.progress * 100.0f),
        });
        emptyText.update({
            .str = "No results yet. Click 'Run Benchmarks' to start.",
            .tone = Text::Tone::Disabled,
            .align = Text::Align::Center,
        });
        divider.update({});
        progressDivider.update({});

        while (results.size() < this->config.results.size()) {
            results.push_back(std::make_unique<ResultBlock>());
        }
        std::vector<Stack::Item> resultItems;
        U64 i = 0;
        for (const auto& [module, entries] : this->config.results) {
            auto& block = *results[i++];
            block.title.update({
                .str = module,
                .font = Text::Font::H2,
                .scale = Sakura::Typography::Scale::Lead,
            });
            std::vector<std::vector<std::string>> rows;
            rows.reserve(entries.size());
            for (const auto& entry : entries) {
                rows.push_back({
                    entry.name,
                    jst::fmt::format("{:.2f}", entry.opsPerSec),
                    jst::fmt::format("{:.4f}", entry.msPerOp),
                    jst::fmt::format("{:.2f}%", entry.error * 100.0),
                });
            }
            block.table.update({
                .columns = {"Variant", "Ops/sec", "ms/Op", "Error %"},
                .rows = std::move(rows),
            });
            block.divider.update({});
            resultItems.push_back({.child = &block.title});
            resultItems.push_back({.child = &block.table});
            resultItems.push_back({.child = &block.divider});
        }
        resultsContent.update({.items = std::move(resultItems)});
        resultsArea.update({
            .content = &resultsContent,
            .height = 300.0f,
            .alwaysScrollbar = false,
        });

        std::vector<Stack::Item> items = {
            {.child = &header},
            {.child = &moduleLabel},
            {.child = &moduleCombo},
            {.child = &runButton},
            {.child = &resetButton},
            {.child = &divider},
        };
        if (this->config.running) {
            items.push_back({.child = &progressBar});
            items.push_back({.child = &progressDivider});
        }
        if (this->config.results.empty() && !this->config.running) {
            items.push_back({.child = &emptyText});
        } else if (!this->config.results.empty()) {
            items.push_back({.child = &resultsArea});
        }
        stack.update({.items = std::move(items)});
        panel.update({
            .id = "Benchmark",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

 private:
    struct ResultBlock {
        Sakura::Text title;
        Sakura::Table table;
        Sakura::Divider divider;
    };

    Config config;
    ModalHeader header;
    Sakura::Text moduleLabel;
    Sakura::Combo moduleCombo;
    Sakura::Button runButton;
    Sakura::Button resetButton;
    Sakura::Divider divider;
    Sakura::ProgressBar progressBar;
    Sakura::Divider progressDivider;
    Sakura::Text emptyText;
    std::vector<std::unique_ptr<ResultBlock>> results;
    Sakura::Stack resultsContent;
    Sakura::ScrollArea resultsArea;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_BENCHMARK_HH
