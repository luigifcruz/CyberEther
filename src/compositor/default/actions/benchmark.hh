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
    using Filter = std::tuple<MailRunBenchmark, MailResetBenchmark, MailSetBenchmarkModule>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    BenchmarkActions(DefaultCompositorState& state,
                     DefaultCompositorCallbacks& callbacks);
    Result handle(const MailRunBenchmark&);
    Result handle(const MailResetBenchmark&);
    Result handle(const MailSetBenchmarkModule& mail);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BENCHMARK_HH
