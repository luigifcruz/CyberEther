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
    };

    FlowgraphState& state;
    std::map<Key, Entry> entries;

    explicit MetricActions(FlowgraphState& state) : state(state) {}

    void refresh(Clock::time_point now = Clock::now()) {
        std::set<Key> seen;
        for (auto& [flowgraphId, blocks] : state.blocks) {
            for (auto& block : blocks) {
                for (auto& metric : block.data.metrics) {
                    const Key key{flowgraphId, block.name, metric.name};
                    seen.insert(key);
                    const auto [it, inserted] = entries.try_emplace(key, metric, now);
                    if (inserted) {
                        continue;
                    }
                    auto& entry = it->second;
                    if (entry.metric.label != metric.label || entry.metric.help != metric.help ||
                        entry.metric.format != metric.format || entry.metric.value.type() != metric.value.type() ||
                        now - entry.sampled >= RefreshInterval) {
                        entry = {metric, now};
                    } else {
                        metric.value = entry.metric.value;
                    }
                }
            }
        }
        std::erase_if(entries, [&](const auto& item) { return !seen.contains(item.first); });
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_METRICS_HH
