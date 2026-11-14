#include "detached_config.hh"

namespace Jetstream {

FlowgraphDetachedConfigPresenter::FlowgraphDetachedConfigPresenter(const PresenterContext& context) : context(context) {}

std::vector<FlowgraphDetachedConfig::Config> FlowgraphDetachedConfigPresenter::build(const std::string& flowgraphId,
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
            .metricsOpen = !configMeta.tileMetricsCollapsed,
            .onClose = [enqueue, flowgraphId, blockName]() {
                enqueue(MailSetConfigDetached{
                    .flowgraph = flowgraphId,
                    .block = blockName,
                    .detached = false,
                });
            },
            .onToggleMetricsOpen = [enqueue, flowgraphId, blockName](const bool open) {
                enqueue(MailSetMetricsCollapsed{
                    .flowgraph = flowgraphId,
                    .block = blockName,
                    .collapsed = !open,
                });
            },
        });
    }

    return configs;
}

}  // namespace Jetstream
