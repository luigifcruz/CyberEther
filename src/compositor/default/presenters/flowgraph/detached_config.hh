#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_DETACHED_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_DETACHED_CONFIG_HH

#include "config.hh"
#include "labels.hh"
#include "layout.hh"

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

        const auto blocks = context.state.flowgraph.blocks.find(flowgraphId);
        if (blocks == context.state.flowgraph.blocks.end()) {
            return configs;
        }

        for (const auto& [blockName, blockData] : blocks->second) {
            const bool pending = blockData.state == Block::State::Creating ||
                                 blockData.state == Block::State::Destroying;
            if (!pending && blockData.interfaceConfigs.empty()) {
                continue;
            }

            ConfigMeta configMeta;
            flowgraph->metadata().get("config", configMeta, blockName);
            if (!configMeta.detached) {
                continue;
            }

            const std::string key = MakeConfigTileKey(blockName);
            configs.push_back({
                .id = key,
                .title = MakeConfigTileTitle(blockName, blockData.title),
                .metrics = pending
                    ? std::vector<FlowgraphMetricConfig>{}
                    : BuildFlowgraphMetrics(key, blockData),
                .configFields = pending
                    ? std::vector<FlowgraphConfigFieldConfig>{}
                    : BuildFlowgraphDetachedConfigFields(enqueue,
                                                         context.state.flowgraph,
                                                         flowgraphId,
                                                         blockName,
                                                         key,
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

};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_DETACHED_CONFIG_HH
