#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_DETACHED_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_DETACHED_CONFIG_HH

#include "config.hh"
#include "labels.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../model/meta.hh"
#include "../../views/flowgraph/detached_config.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedConfigPresenter {
    const PresenterContext& context;

    explicit FlowgraphDetachedConfigPresenter(const PresenterContext& context) : context(context) {}

    std::vector<FlowgraphDetachedConfig::Config> build(const std::string& flowgraphId,
                                                       const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
        std::vector<FlowgraphDetachedConfig::Config> configs;
        if (!flowgraph) {
            return configs;
        }

        const auto referencedBlocks = buildReferencedBlocks(flowgraphId);

        std::vector<std::string> blocks;
        if (flowgraph->view().keys(blocks) != Result::SUCCESS) {
            return configs;
        }

        for (const auto& blockName : blocks) {
            Flowgraph::View::BlockData blockData;
            if (flowgraph->view().block(blockName, blockData) != Result::SUCCESS) {
                continue;
            }
            const bool pending = blockData.state == Block::State::Creating ||
                                 blockData.state == Block::State::Destroying;
            if (!pending && blockData.interfaceConfigs.empty()) {
                continue;
            }

            ConfigMeta configMeta;
            flowgraph->metadata().get("config", configMeta, blockName);
            if (!configMeta.detached && !referencedBlocks.contains(blockName)) {
                continue;
            }

            const std::string windowId = MakeDetachedConfigWindowId(flowgraphId, blockName, configMeta);
            configs.push_back({
                .id = windowId,
                .title = MakeDetachedConfigWindowTitle(blockName, blockData.title),
                .configFields = pending
                    ? std::vector<FlowgraphConfigFieldConfig>{}
                    : BuildFlowgraphDetachedConfigFields(enqueue,
                                                         flowgraphId,
                                                         blockName,
                                                         windowId,
                                                         blockData),
                .onClose = [enqueue, flowgraphId, blockName]() {
                    enqueue(MailSetConfigDetached{
                        .flowgraph = flowgraphId,
                        .block = blockName,
                        .detached = false,
                    });
                },
            });
        }

        return configs;
    }

 private:
    std::unordered_set<std::string> buildReferencedBlocks(const std::string& flowgraphId) const {
        std::unordered_set<std::string> referenced;
        const auto stacksIt = context.state.flowgraph.stacks.find(flowgraphId);
        if (stacksIt == context.state.flowgraph.stacks.end()) {
            return referenced;
        }

        for (const auto& [_, stack] : stacksIt->second) {
            if (stack.meta.layout.has_value()) {
                collectReferencedBlocks(*stack.meta.layout, referenced);
            }
        }
        return referenced;
    }

    static void collectReferencedBlocks(const StackDockLayoutMeta& layout,
                                        std::unordered_set<std::string>& referenced) {
        if (layout.configs.has_value()) {
            for (const auto& config : *layout.configs) {
                if (!config.block.empty()) {
                    referenced.insert(config.block);
                }
            }
        }
        if (!layout.children.has_value()) {
            return;
        }
        for (const auto& child : *layout.children) {
            collectReferencedBlocks(child, referenced);
        }
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_DETACHED_CONFIG_HH
