#include <mutex>
#include <shared_mutex>

#include <jetstream/detail/flowgraph_impl.hh>

namespace Jetstream {

Flowgraph::Environment::Environment(const std::shared_ptr<Flowgraph::Impl>& impl) : impl(impl) {}

bool Flowgraph::Environment::has(const std::string& key, U64 timestamp) const {
    const auto graph = impl.lock();
    if (!graph) {
        return false;
    }

    std::shared_lock lock(graph->environmentMutex);

    if (!graph->environmentValues.contains(key)) {
        return false;
    }

    for (const auto& entry : graph->environmentValues.at(key)) {
        if (timestamp >= entry.start && timestamp <= entry.end) {
            return true;
        }
    }

    return false;
}

Result Flowgraph::Environment::get(const std::string& key, Parser::Map& data, U64 timestamp) const {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] Environment is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::shared_lock lock(graph->environmentMutex);

    if (!graph->environmentValues.contains(key)) {
        return Result::SUCCESS;
    }

    const Impl::EnvironmentEntry* selected = nullptr;
    for (const auto& entry : graph->environmentValues.at(key)) {
        if (timestamp < entry.start || timestamp > entry.end) {
            continue;
        }

        if (selected == nullptr || entry.sequence > selected->sequence) {
            selected = &entry;
        }
    }

    if (selected != nullptr) {
        data = selected->data;
    }

    return Result::SUCCESS;
}

Result Flowgraph::Environment::set(const std::string& key, const Parser::Map& data, U64 start, U64 end) {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] Environment is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    if (end < start) {
        JST_ERROR("[FLOWGRAPH] Environment value '{}' has invalid timestamp range [{}, {}].", key, start, end);
        return Result::ERROR;
    }

    std::unique_lock lock(graph->environmentMutex);
    const U64 sequence = ++graph->environmentSequence;
    auto& entries = graph->environmentValues[key];

    for (auto& entry : entries) {
        if (entry.start == start && entry.end == end) {
            entry.sequence = sequence;
            entry.data = data;
            return Result::SUCCESS;
        }
    }

    entries.push_back({start, end, sequence, data});
    return Result::SUCCESS;
}

Result Flowgraph::Environment::clear(const std::string& key) {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] Environment is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::unique_lock lock(graph->environmentMutex);
    graph->environmentValues.erase(key);
    return Result::SUCCESS;
}

Result Flowgraph::Environment::clearAll() {
    const auto graph = impl.lock();
    if (!graph) {
        JST_ERROR("[FLOWGRAPH] Environment is no longer attached to a flowgraph.");
        return Result::ERROR;
    }

    std::unique_lock lock(graph->environmentMutex);
    graph->environmentValues.clear();
    graph->environmentSequence = 0;
    return Result::SUCCESS;
}

}  // namespace Jetstream
