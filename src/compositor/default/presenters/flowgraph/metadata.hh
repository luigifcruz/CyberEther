#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_METADATA_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_METADATA_HH

#include "../context.hh"
#include "key_value.hh"
#include "labels.hh"

#include "../../model/messages.hh"
#include "../../views/flowgraph/key_value.hh"

#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"

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

        Sakura::Table::Nodes nodes;
        U64 totalEntries = 0;
        const auto appendNodes = [&nodes, &filter, &flowgraph](const std::vector<std::string>& keys,
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
                const std::string id = block.empty() ? "flowgraph/" + key :
                    jst::fmt::format("block/{}:{}/{}", block.size(), block, key);
                auto node = FlowgraphKeyValueDetail::AnyToNode(id, displayKey, value);
                node.open = true;
                nodes.push_back(std::move(node));
            }
        };

        std::vector<std::string> metadataKeys;
        if (flowgraph->metadata().keys(metadataKeys) == Result::SUCCESS) {
            totalEntries += metadataKeys.size();
            appendNodes(metadataKeys, "", "");
        }
        std::vector<std::string> blocks;
        if (flowgraph->view().keys(blocks) != Result::SUCCESS) {
            return std::nullopt;
        }

        for (const auto& blockName : blocks) {
            std::vector<std::string> blockMetadataKeys;
            if (flowgraph->metadata().keys(blockMetadataKeys, blockName) != Result::SUCCESS) {
                continue;
            }
            totalEntries += blockMetadataKeys.size();
            appendNodes(blockMetadataKeys, blockName, blockName);
        }

        FlowgraphKeyValueDetail::SortNodes(nodes);

        const auto enqueue = context.callbacks.enqueueMail;
        return FlowgraphKeyValueWindow::Config{
            .id = "flowgraph-metadata-window",
            .title = "Flowgraph Metadata (" + MakeFlowgraphWindowTitle(flowgraphId, flowgraph) + ")",
            .search = context.state.interface.flowgraphMetadataSearch,
            .searchHint = "Search metadata keys...",
            .entryCount = FlowgraphKeyValueDetail::EntryCount(nodes.size(), totalEntries),
            .nodes = std::move(nodes),
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
