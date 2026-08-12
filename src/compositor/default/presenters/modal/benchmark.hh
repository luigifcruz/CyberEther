#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_BENCHMARK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_BENCHMARK_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/benchmark.hh"

#include "jetstream/registry.hh"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Jetstream {

struct BenchmarkModalPresenter {
    const PresenterContext& context;

    explicit BenchmarkModalPresenter(const PresenterContext& context) : context(context) {}

    BenchmarkView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;

        static constexpr const char* kAll = "All";

        auto labelToModuleType = std::make_shared<std::unordered_map<std::string, std::string>>();
        labelToModuleType->emplace(kAll, "");

        std::vector<std::string> moduleOptions = {kAll};
        std::string selectedLabel = kAll;

        for (const auto& benchmark : Registry::ListAvailableBenchmarks()) {
            const auto blocks = Registry::ListAvailableBlocks(benchmark.moduleType);
            std::string label = blocks.empty() ? benchmark.moduleType : blocks.front().title;

            if (label == kAll || labelToModuleType->count(label) != 0) {
                continue;
            }
            labelToModuleType->emplace(label, benchmark.moduleType);
            moduleOptions.push_back(label);

            if (context.state.benchmark.selectedModule == benchmark.moduleType) {
                selectedLabel = label;
            }
        }

        return BenchmarkView::Config{
            .running = context.state.benchmark.running,
            .progress = context.state.benchmark.progress,
            .moduleOptions = std::move(moduleOptions),
            .selectedModule = std::move(selectedLabel),
            .results = context.state.benchmark.results,
            .onRun = [enqueue]() {
                enqueue(MailRunBenchmark{});
            },
            .onReset = [enqueue]() {
                enqueue(MailResetBenchmark{});
            },
            .onModuleChange = [enqueue, labelToModuleType](const std::string& label) {
                std::string moduleType;
                if (const auto it = labelToModuleType->find(label); it != labelToModuleType->end()) {
                    moduleType = it->second;
                }
                enqueue(MailSetBenchmarkModule{.moduleType = std::move(moduleType)});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_BENCHMARK_HH
