#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_BENCHMARK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_BENCHMARK_HH

#include "../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include "jetstream/benchmark.hh"

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
        Sakura::Callback<void()> onRun;
        Sakura::Callback<void()> onReset;
        Sakura::Callback<void(const std::string&)> onModuleChange;
    };

    void update(Config config);
    Sakura::Component& component();

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
