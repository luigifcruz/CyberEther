#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH

#include "picker.hh"
#include "surface.hh"
#include "../config.hh"

#include "../../context.hh"

#include "../../../model/meta.hh"
#include "../../../views/flowgraph/editor/base.hh"

#include "jetstream/block.hh"
#include "jetstream/block_interface.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/runtime_context.hh"

#include <algorithm>
#include <any>
#include <memory>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphNodePresenter {
    const PresenterContext& context;
    FlowgraphCatalogPresenter catalog;
    FlowgraphSurfacePresenter surfaces;

    explicit FlowgraphNodePresenter(const PresenterContext& context) : context(context),
                                                                       surfaces(context) {}

    void buildInputs(FlowgraphNode::BlockData& block, const Flowgraph::View::BlockData& blockData) const {
        for (const auto& interfaceInfo : blockData.interfaceInputs) {
            FlowgraphNode::Input input{
                .port = {
                    .id = interfaceInfo.name,
                    .label = interfaceInfo.label,
                    .help = interfaceInfo.help,
                },
            };

            const auto inputIt = blockData.inputs.find(interfaceInfo.name);
            if (inputIt != blockData.inputs.end() && inputIt->second.external.has_value()) {
                input.source = FlowgraphNode::Link{
                    .block = inputIt->second.external->block,
                    .port = inputIt->second.external->port,
                    .resolved = inputIt->second.resolved(),
                    .tensor = inputIt->second.tensor,
                };
            }

            block.inputs.push_back(std::move(input));
        }
    }

    void buildOutputs(FlowgraphNode::BlockData& block, const Flowgraph::View::BlockData& blockData) const {
        for (const auto& interfaceInfo : blockData.interfaceOutputs) {
            FlowgraphNode::Output output{
                .port = {
                    .id = interfaceInfo.name,
                    .label = interfaceInfo.label,
                    .help = interfaceInfo.help,
                },
            };

            const auto outputIt = blockData.outputs.find(interfaceInfo.name);
            if (outputIt != blockData.outputs.end()) {
                output.tensor = outputIt->second.tensor;
            }

            block.outputs.push_back(std::move(output));
        }
    }

    void buildConfigFields(FlowgraphNode::BlockData& block,
                           const std::string& flowgraphId,
                           const std::string& blockName,
                           const std::string& nodeViewId,
                           const Flowgraph::View::BlockData& blockData) const {
        block.config = blockData.config;
        block.configFields = BuildFlowgraphConfigFields(nodeViewId, blockData);
        ApplyLiveMarkdown(block.configFields, context.state.flowgraph, flowgraphId, blockName);
        ApplyConfigDrafts(block.configFields, context.callbacks.enqueueMail, context.state.flowgraph, flowgraphId, blockName);
    }

    void buildTiming(FlowgraphNode::BlockData& block,
                     const Flowgraph::View::BlockData& blockData) const {
        F32 totalTime = 0.0f;
        U64 maxCycles = 0;

        for (const auto& metric : blockData.metrics) {
            if (Parser::Get<std::string>(metric.format, "type") != "timing" ||
                Parser::Get<std::string>(metric.format, "visibility") != "internal") {
                continue;
            }

            Module::Timing timing;
            try {
                timing = std::any_cast<Module::Timing>(metric.value);
            } catch (const std::bad_any_cast&) {
                continue;
            }

            const F32 averageComputeTime = timing.cycles == 0
                ? 0.0f
                : timing.computeTime / static_cast<F32>(timing.cycles);
            const auto& label = metric.label.empty() ? metric.name : metric.label;

            block.timing.push_back(jst::fmt::format("{} ({}/{}/{}): {:.2f} ms",
                                                    label,
                                                    timing.backend,
                                                    timing.device,
                                                    timing.runtime,
                                                    averageComputeTime));
            totalTime += averageComputeTime;
            maxCycles = std::max(maxCycles, timing.cycles);
        }

        if (block.timing.empty()) {
            return;
        }

        const std::string cyclesStr = (maxCycles > 1000)
            ? jst::fmt::format("{:.1f}k", maxCycles / 1000.0f)
            : jst::fmt::format("{}", maxCycles);
        block.timing.push_back(jst::fmt::format("= {:.2f} ms ({})", totalTime, cyclesStr));
    }

    FlowgraphNode::BlockData build(const std::string& flowgraphId,
                                   const std::shared_ptr<Flowgraph>& flowgraph,
                                   const std::string& blockName,
                                   const Flowgraph::View::BlockData& blockData) const {
        const std::string nodeViewId = jst::fmt::format("{}:{}", flowgraphId, blockName);
        const Block::State blockState = blockData.state;
        const bool hasDiagnostic = (blockState == Block::State::Errored ||
                                     blockState == Block::State::Incomplete) &&
                                    !blockData.diagnostic.empty();
        FlowgraphNode::BlockData block{
            .name = blockName,
            .module = blockData.type,
            .title = blockData.title,
            .state = blockState,
            .nodeSize = blockData.nodeSize,
            .diagnostic = hasDiagnostic ? Sakura::CleanUserMessage(blockData.diagnostic) : "",
        };
        block.deviceOptions = catalog.buildDeviceOptions(block.module,
                                                         blockData.device,
                                                         blockData.runtime,
                                                         blockData.provider);

        NodeMeta nodeMeta;
        flowgraph->metadata().get("node", nodeMeta, blockName);
        block.layout = FlowgraphNode::Layout{
            .x = nodeMeta.x,
            .y = nodeMeta.y,
            .width = nodeMeta.width,
            .height = nodeMeta.height,
        };

        ConfigMeta configMeta;
        flowgraph->metadata().get("config", configMeta, blockName);
        block.configCollapsed = configMeta.collapsed;
        block.configDetached = configMeta.detached;

        buildInputs(block, blockData);
        buildOutputs(block, blockData);

        if (blockState != Block::State::Creating) {
            block.metrics = BuildFlowgraphMetrics(nodeViewId, blockData);
            buildConfigFields(block, flowgraphId, blockName, nodeViewId, blockData);
            if (context.state.debug.timingEnabled) {
                buildTiming(block, blockData);
            }
            surfaces.buildSurfaces(block, flowgraph, flowgraphId, blockName, nodeViewId, blockData);
        }

        return block;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH
