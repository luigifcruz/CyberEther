#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_ENVIRONMENT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_ENVIRONMENT_HH

#include "../context.hh"
#include "key_value.hh"
#include "labels.hh"

#include "../../model/messages.hh"
#include "../../views/flowgraph/key_value.hh"

#include "jetstream/flowgraph_environment.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

struct FlowgraphEnvironmentWindowPresenter {
    const PresenterContext& context;

    explicit FlowgraphEnvironmentWindowPresenter(const PresenterContext& context) : context(context) {}

    std::optional<FlowgraphKeyValueWindow::Config> build() const {
        if (!context.state.interface.flowgraphEnvironmentVisible ||
            !context.state.interface.focusedFlowgraph.has_value()) {
            return std::nullopt;
        }

        const std::string flowgraphId = context.state.interface.focusedFlowgraph.value();
        if (!context.state.flowgraph.items.contains(flowgraphId)) {
            return std::nullopt;
        }

        const auto& flowgraph = context.state.flowgraph.items.at(flowgraphId);
        const std::string filter = FlowgraphKeyValueDetail::NormalizeFilter(
            context.state.interface.flowgraphEnvironmentSearch);

        std::vector<std::string> keys;
        if (flowgraph->environment().keys(keys) != Result::SUCCESS) {
            keys.clear();
        }

        std::vector<std::vector<std::string>> rows;
        for (const auto& key : keys) {
            if (!FlowgraphKeyValueDetail::KeyMatches(key, filter)) {
                continue;
            }

            Parser::Map value;
            if (flowgraph->environment().get(key, value) != Result::SUCCESS) {
                continue;
            }
            rows.push_back({key, FlowgraphKeyValueDetail::AnyToString(value)});
        }
        std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
            return lhs[0] < rhs[0];
        });

        const auto enqueue = context.callbacks.enqueueMail;
        return FlowgraphKeyValueWindow::Config{
            .id = "flowgraph-environment-window",
            .title = "Flowgraph Environment (" + MakeFlowgraphWindowTitle(flowgraphId, flowgraph) + ")",
            .search = context.state.interface.flowgraphEnvironmentSearch,
            .searchHint = "Search environment keys...",
            .entryCount = FlowgraphKeyValueDetail::EntryCount(rows.size(), keys.size()),
            .rows = std::move(rows),
            .onSearchChange = [enqueue](const std::string& value) {
                enqueue(MailSetFlowgraphEnvironmentSearch{.value = value});
            },
            .onClose = [enqueue]() {
                enqueue(MailSetFlowgraphEnvironmentVisible{.value = false});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_ENVIRONMENT_HH
