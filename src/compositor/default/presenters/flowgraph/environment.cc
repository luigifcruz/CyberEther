#include "environment.hh"

namespace Jetstream {

FlowgraphEnvironmentPresenter::FlowgraphEnvironmentPresenter(const PresenterContext& context) : context(context) {}

std::optional<FlowgraphEnvironment::Config> FlowgraphEnvironmentPresenter::build(const std::string& flowgraphId,
                                                                                 const std::shared_ptr<Flowgraph>& flowgraph) const {
    const auto entry = context.state.flowgraph.environment.find(flowgraphId);
    if (entry == context.state.flowgraph.environment.end()) {
        return std::nullopt;
    }

    const std::string& search = entry->second;
    const std::string filter = Text::ToLower(search);

    std::vector<std::string> keys;
    if (flowgraph->environment().keys(keys) != Result::SUCCESS) {
        keys.clear();
    }

    Sakura::Table::Nodes nodes;
    for (const auto& key : keys) {
        if (!FlowgraphKeyValueDetail::KeyMatches(key, filter)) {
            continue;
        }

        Parser::Map value;
        if (flowgraph->environment().get(key, value) != Result::SUCCESS) {
            continue;
        }
        auto node = FlowgraphKeyValueDetail::AnyToNode(key, key, value);
        node.open = true;
        nodes.push_back(std::move(node));
    }
    FlowgraphKeyValueDetail::SortNodes(nodes);

    const auto enqueue = context.callbacks.enqueueMail;
    return FlowgraphEnvironment::Config{
        .id = kLayoutEnvironmentPane,
        .title = "Environment",
        .search = search,
        .searchHint = "Search environment keys...",
        .entryCount = FlowgraphKeyValueDetail::EntryCount(nodes.size(), keys.size()),
        .nodes = std::move(nodes),
        .onSearchChange = [enqueue, flowgraphId](const std::string& value) {
            enqueue(MailSetFlowgraphEnvironmentSearch{.flowgraph = flowgraphId, .value = value});
        },
        .onClose = [enqueue, flowgraphId]() {
            enqueue(MailSetFlowgraphEnvironmentVisible{.flowgraph = flowgraphId, .value = false});
        },
    };
}

}  // namespace Jetstream
