#include "metrics.hh"

namespace Jetstream {

MetricActions::Entry::Entry(const Flowgraph::View::MetricEntry& metric, Clock::time_point sampled)
    : metric(metric), sampled(sampled) {}

MetricActions::MetricActions(FlowgraphState& state) : state(state) {}

void MetricActions::refresh(Clock::time_point now) {
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

}  // namespace Jetstream
