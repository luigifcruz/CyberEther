#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_BENCHMARK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_BENCHMARK_HH

#include "../components/modal_header.hh"
#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include "jetstream/benchmark.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct BenchmarkView : public Sakura::Component {
    struct Config {
        bool running = false;
        F32 progress = 0.0f;
        Benchmark::ResultMapType results;
        std::function<void()> onRun;
        std::function<void()> onReset;
    };

    void update(Config config) {
        this->config = std::move(config);

        header.update({
            .id = "BenchmarkHeader",
            .title = ICON_FA_GAUGE_HIGH " Module Benchmarks",
            .description = "Run performance benchmarks for registered modules. Results show operations per second and timing information.",
        });
        runButton.update({
            .id = "BenchmarkRun",
            .str = ICON_FA_PLAY " Run Benchmarks",
            .size = {-1.0f, 36.0f},
            .variant = Sakura::Button::Variant::Action,
            .disabled = this->config.running,
            .onClick = [this]() {
                if (this->config.onRun) {
                    this->config.onRun();
                }
            },
        });
        resetButton.update({
            .id = "BenchmarkReset",
            .str = ICON_FA_ROTATE_LEFT " Reset Results",
            .size = {-1.0f, 36.0f},
            .disabled = this->config.running || this->config.results.empty(),
            .onClick = [this]() {
                if (this->config.onReset) {
                    this->config.onReset();
                }
            },
        });
        progressBar.update({
            .id = "BenchmarkProgress",
            .value = this->config.progress,
            .overlay = jst::fmt::format("{:.0f}%", this->config.progress * 100.0f),
        });
        emptyText.update({
            .id = "BenchmarkEmpty",
            .str = "No results yet. Click 'Run Benchmarks' to start.",
            .tone = Sakura::Text::Tone::Disabled,
            .align = Sakura::Text::Align::Center,
        });
        resultsDiv.update({
            .id = "BenchmarkResults",
            .size = {0.0f, 300.0f},
        });
        resultTitles.resize(this->config.results.size());
        resultTables.resize(this->config.results.size());
        U64 i = 0;
        for (const auto& [module, entries] : this->config.results) {
            resultTitles[i].update({
                .id = "BenchmarkTitle" + std::to_string(i),
                .str = module,
                .font = Sakura::Text::Font::H2,
                .scale = 1.05f,
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

            resultTables[i].update({
                .id = "BenchmarkResults" + std::to_string(i),
                .columns = {"Variant", "Ops/sec", "ms/Op", "Error %"},
                .rows = rows,
            });
            i += 1;
        }
    }

    void render(const Sakura::Context& ctx) const {
        header.render(ctx);
        runButton.render(ctx);
        resetButton.render(ctx);
        divider.render(ctx);
        if (config.running) {
            progressBar.render(ctx);
            divider.render(ctx);
        }
        if (config.results.empty() && !config.running) {
            emptyText.render(ctx);
        } else if (!config.results.empty()) {
            resultsDiv.render(ctx, [this](const Sakura::Context& ctx) {
                for (U64 i = 0; i < resultTables.size(); ++i) {
                    resultTitles[i].render(ctx);
                    resultTables[i].render(ctx);
                    divider.render(ctx);
                }
            });
        }
    }

 private:
    Config config;
    ModalHeader header;
    Sakura::Button runButton;
    Sakura::Button resetButton;
    Sakura::ProgressBar progressBar;
    Sakura::Text emptyText;
    Sakura::Div resultsDiv;
    Sakura::Divider divider;
    std::vector<Sakura::Text> resultTitles;
    std::vector<Sakura::Table> resultTables;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_BENCHMARK_HH
