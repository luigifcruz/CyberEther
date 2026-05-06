#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_BENCHMARK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_BENCHMARK_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../views/modal/benchmark.hh"

namespace Jetstream {

struct DefaultBenchmarkPresenter {
    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    DefaultBenchmarkPresenter(DefaultCompositorState& state,
                              DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    BenchmarkView::Config build() const {
        const auto enqueue = callbacks.enqueueMail;
        return BenchmarkView::Config{
            .running = state.benchmark.running,
            .progress = state.benchmark.progress,
            .results = state.benchmark.results,
            .onRun = [enqueue]() {
                enqueue(MailRunBenchmark{});
            },
            .onReset = [enqueue]() {
                enqueue(MailResetBenchmark{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_BENCHMARK_HH
