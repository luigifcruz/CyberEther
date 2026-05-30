#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_METADATA_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_METADATA_HH

#include "../context.hh"
#include "key_value.hh"
#include "labels.hh"

#include "../../model/messages.hh"
#include "../../views/flowgraph/key_value.hh"

#include "jetstream/flowgraph_metadata.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

struct FlowgraphMetadataWindowPresenter {
    const PresenterContext& context;

    explicit FlowgraphMetadataWindowPresenter(const PresenterContext& context) : context(context) {}

    std::optional<FlowgraphKeyValueWindow::Config> build() const {
        if (!context.state.interface.flowgraphMetadataVisible ||
            !context.state.interface.focusedFlowgraph.has_value()) {
            return std::nullopt;
        }

        const std::string flowgraphId = context.state.interface.focusedFlowgraph.value();
        if (!context.state.flowgraph.items.contains(flowgraphId)) {
            return std::nullopt;
        }

        const auto& flowgraph = context.state.flowgraph.items.at(flowgraphId);
        const std::string filter = FlowgraphKeyValueDetail::NormalizeFilter(
            context.state.interface.flowgraphMetadataSearch);

        std::vector<std::vector<std::string>> rows;
        U64 totalEntries = 0;
        const auto appendRows = [&rows, &filter, &flowgraph](const std::vector<std::string>& keys,
                                                             const std::string& prefix,
                                                             const std::string& block) {
            for (const auto& key : keys) {
                const std::string displayKey = prefix.empty() ? key : prefix + "." + key;
                if (!FlowgraphKeyValueDetail::KeyMatches(displayKey, filter)) {
                    continue;
                }

                Parser::Map value;
                if (flowgraph->metadata().get(key, value, block) != Result::SUCCESS) {
                    continue;
                }
                rows.push_back({displayKey, FlowgraphKeyValueDetail::AnyToString(value)});
            }
        };

        std::vector<std::string> metadataKeys;
        if (flowgraph->metadata().keys(metadataKeys) == Result::SUCCESS) {
            totalEntries += metadataKeys.size();
            appendRows(metadataKeys, "", "");
        }
        for (const auto& [blockName, _] : flowgraph->blockList()) {
            std::vector<std::string> blockMetadataKeys;
            if (flowgraph->metadata().keys(blockMetadataKeys, blockName) != Result::SUCCESS) {
                continue;
            }
            totalEntries += blockMetadataKeys.size();
            appendRows(blockMetadataKeys, blockName, blockName);
        }

        std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
            return lhs[0] < rhs[0];
        });

        const auto enqueue = context.callbacks.enqueueMail;
        return FlowgraphKeyValueWindow::Config{
            .id = "flowgraph-metadata-window",
            .title = "Flowgraph Metadata (" + MakeFlowgraphWindowTitle(flowgraphId, flowgraph) + ")",
            .search = context.state.interface.flowgraphMetadataSearch,
            .searchHint = "Search metadata keys...",
            .entryCount = FlowgraphKeyValueDetail::EntryCount(rows.size(), totalEntries),
            .rows = std::move(rows),
            .onSearchChange = [enqueue](const std::string& value) {
                enqueue(MailSetFlowgraphMetadataSearch{.value = value});
            },
            .onClose = [enqueue]() {
                enqueue(MailSetFlowgraphMetadataVisible{.value = false});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_METADATA_HH
