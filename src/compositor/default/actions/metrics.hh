#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_METRICS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_METRICS_HH

#include "../model/state.hh"

#include <any>
#include <chrono>
#include <map>
#include <ratio>
#include <set>
#include <string>
#include <tuple>

namespace Jetstream {

struct MetricActions {
    using FlowgraphState = DefaultCompositorState::FlowgraphState;
    using Clock = std::chrono::steady_clock;
    using Key = std::tuple<std::string, std::string, std::string>;

    static constexpr auto RefreshInterval =
        std::chrono::ceil<Clock::duration>(std::chrono::duration<I64, std::ratio<1, 60>>{1});

    struct Entry {
        Flowgraph::View::MetricEntry metric;
        Clock::time_point sampled;

        Entry(const Flowgraph::View::MetricEntry& metric, Clock::time_point sampled);
    };

    FlowgraphState& state;
    std::map<Key, Entry> entries;

    explicit MetricActions(FlowgraphState& state);
    void refresh(Clock::time_point now = Clock::now());
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_METRICS_HH
