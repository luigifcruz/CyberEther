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

    explicit BenchmarkModalPresenter(const PresenterContext& context);
    BenchmarkView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_BENCHMARK_HH
