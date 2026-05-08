#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_BENCHMARK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_BENCHMARK_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/benchmark.hh"

namespace Jetstream {

struct BenchmarkModalPresenter {
    const PresenterContext& context;

    explicit BenchmarkModalPresenter(const PresenterContext& context) : context(context) {}

    BenchmarkView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return BenchmarkView::Config{
            .running = context.state.benchmark.running,
            .progress = context.state.benchmark.progress,
            .results = context.state.benchmark.results,
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

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_BENCHMARK_HH
