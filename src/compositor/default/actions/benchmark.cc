#include "benchmark.hh"

namespace Jetstream {

BenchmarkActions::BenchmarkActions(DefaultCompositorState& state,
                                   DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

Result BenchmarkActions::handle(const MailRunBenchmark&) {
    if (state.benchmark.running) {
        return Result::SUCCESS;
    }

    state.benchmark.running = true;
    state.benchmark.progress = 0.0f;
    state.benchmark.results.clear();
    state.benchmark.output.str("");
    state.benchmark.output.clear();
    Benchmark::ResetResults();
    auto* output = &state.benchmark.output;
    const auto moduleType = state.benchmark.selectedModule;
    state.benchmark.future = std::async(std::launch::async, [output, moduleType]() {
        Benchmark::Run("quiet", moduleType, *output);
    });

    return Result::SUCCESS;
}

Result BenchmarkActions::handle(const MailResetBenchmark&) {
    if (state.benchmark.running) {
        return Result::SUCCESS;
    }

    Benchmark::ResetResults();
    state.benchmark.progress = 0.0f;
    state.benchmark.results.clear();

    return Result::SUCCESS;
}

Result BenchmarkActions::handle(const MailSetBenchmarkModule& mail) {
    if (state.benchmark.running) {
        return Result::SUCCESS;
    }

    state.benchmark.selectedModule = mail.moduleType;

    return Result::SUCCESS;
}

}  // namespace Jetstream
