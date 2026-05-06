#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BENCHMARK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BENCHMARK_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/benchmark.hh"

#include <future>
#include <tuple>

namespace Jetstream {

struct BenchmarkActions {
    using Filter = std::tuple<MailRunBenchmark, MailResetBenchmark>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    BenchmarkActions(DefaultCompositorState& state,
                     DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    Result handle(const MailRunBenchmark&) {
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
        state.benchmark.future = std::async(std::launch::async, [output]() {
            Benchmark::Run("quiet", *output);
        });

        return Result::SUCCESS;
    }

    Result handle(const MailResetBenchmark&) {
        if (state.benchmark.running) {
            return Result::SUCCESS;
        }

        Benchmark::ResetResults();
        state.benchmark.progress = 0.0f;
        state.benchmark.results.clear();

        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BENCHMARK_HH
