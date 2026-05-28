#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH

#include "picker.hh"
#include "surface.hh"

#include "../../context.hh"

#include "../../../model/meta.hh"
#include "../../../views/flowgraph/editor/base.hh"

#include "jetstream/block.hh"
#include "jetstream/block_interface.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/sakura/toast.hh"

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

    void buildInputs(FlowgraphNode::BlockData& block, const std::shared_ptr<Block>& blockPtr) const {
        for (const auto& [slot, interfaceInfo] : blockPtr->interface()->inputs()) {
            FlowgraphNode::Input input{
                .port = {
                    .id = slot,
                    .label = interfaceInfo.label,
                    .help = interfaceInfo.help,
                },
            };

            const auto inputIt = blockPtr->inputs().find(slot);
            if (inputIt != blockPtr->inputs().end() && inputIt->second.external.has_value()) {
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

    void buildOutputs(FlowgraphNode::BlockData& block, const std::shared_ptr<Block>& blockPtr) const {
        for (const auto& [slot, interfaceInfo] : blockPtr->interface()->outputs()) {
            FlowgraphNode::Output output{
                .port = {
                    .id = slot,
                    .label = interfaceInfo.label,
                    .help = interfaceInfo.help,
                },
            };

            const auto outputIt = blockPtr->outputs().find(slot);
            if (outputIt != blockPtr->outputs().end()) {
                output.tensor = outputIt->second.tensor;
            }

            block.outputs.push_back(std::move(output));
        }
    }

    void buildBlockMetrics(FlowgraphNode::BlockData& block,
                           const std::string& nodeViewId,
                           const std::shared_ptr<Block>& blockPtr) const {
        for (const auto& [name, entry] : blockPtr->interface()->metrics()) {
            if (entry.format.starts_with("private-")) {
                continue;
            }

            std::any value;
            if (entry.metric) {
                value = entry.metric();
            }
            block.metrics.push_back({
                .id = nodeViewId + ":metric:" + name,
                .label = entry.label,
                .help = entry.help,
                .format = entry.format,
                .value = std::move(value),
            });
        }
    }

    void buildConfigFields(FlowgraphNode::BlockData& block,
                           const std::string& nodeViewId,
                           const std::shared_ptr<Block>& blockPtr) const {
        Parser::Map cfg;
        if (blockPtr->config(cfg) != Result::SUCCESS) {
            return;
        }

        block.config = cfg;
        for (const auto& [name, entry] : blockPtr->interface()->configs()) {
            std::string encoded;
            if (cfg.contains(name)) {
                Parser::TypedToString(cfg[name], encoded);
            }

            block.configFields.push_back({
                .id = nodeViewId + ":config:" + name,
                .name = name,
                .label = entry.label.empty() ? name : entry.label,
                .help = entry.help,
                .format = entry.format,
                .encoded = encoded,
                .values = cfg,
            });
        }
    }

    void buildRuntimeMetrics(FlowgraphNode::BlockData& block,
                             const std::shared_ptr<Flowgraph>& flowgraph,
                             const std::string& blockName,
                             const std::shared_ptr<Block>& blockPtr) const {
        if (!flowgraph->metrics().contains(blockName)) {
            return;
        }

        const auto& blockMetrics = flowgraph->metrics().at(blockName);
        F32 totalTime = 0.0f;
        U64 maxCycles = 0;

        block.runtimeMetrics.push_back(jst::fmt::format("Runtime #{} ({}/{})",
                                                        blockMetrics->runtime,
                                                        blockMetrics->device,
                                                        blockMetrics->backend));

        for (const auto& moduleName : blockPtr->modules()) {
            const auto& fullModuleName = jst::fmt::format("{}-{}", blockName, moduleName);
            if (!blockMetrics->averageComputeTime.contains(fullModuleName) ||
                !blockMetrics->cycles.contains(fullModuleName)) {
                continue;
            }

            const auto& averageComputeTime = blockMetrics->averageComputeTime.at(fullModuleName);
            const auto& cycles = blockMetrics->cycles.at(fullModuleName);
            block.runtimeMetrics.push_back(jst::fmt::format("+ {}: {:.3f} ms", moduleName, averageComputeTime));
            totalTime += averageComputeTime;
            maxCycles = std::max(maxCycles, cycles);
        }

        const std::string cyclesStr = (maxCycles > 1000)
            ? jst::fmt::format("{:.1f}k", maxCycles / 1000.0f)
            : jst::fmt::format("{}", maxCycles);
        block.runtimeMetrics.push_back(jst::fmt::format("= {:.3f} ms ({})", totalTime, cyclesStr));
    }

    FlowgraphNode::BlockData build(const std::string& flowgraphId,
                                   const std::shared_ptr<Flowgraph>& flowgraph,
                                   const std::string& blockName,
                                   const std::shared_ptr<Block>& blockPtr) const {
        const std::string nodeViewId = flowgraphId + ":" + blockName;
        const Block::State blockState = blockPtr->state();
        const bool hasDiagnostic = (blockState == Block::State::Errored ||
                                    blockState == Block::State::Incomplete) &&
                                   !blockPtr->diagnostic().empty();
        FlowgraphNode::BlockData block{
            .name = blockName,
            .module = blockPtr->config().type(),
            .title = blockPtr->config().title(),
            .documentation = blockPtr->config().description(),
            .device = blockPtr->device(),
            .runtime = blockPtr->runtime(),
            .provider = blockPtr->provider(),
            .state = blockState,
            .diagnostic = hasDiagnostic ? Sakura::CleanUserMessage(blockPtr->diagnostic()) : "",
        };
        block.deviceOptions = catalog.buildDeviceOptions(block.module,
                                                         block.device,
                                                         block.runtime,
                                                         block.provider);

        NodeMeta nodeMeta;
        flowgraph->getPersistentMeta("node", nodeMeta, blockName);
        block.layout = FlowgraphNode::Layout{
            .x = nodeMeta.x,
            .y = nodeMeta.y,
            .width = nodeMeta.width,
            .height = nodeMeta.height,
        };

        buildInputs(block, blockPtr);
        buildOutputs(block, blockPtr);

        if (blockState != Block::State::Creating) {
            buildBlockMetrics(block, nodeViewId, blockPtr);
            buildConfigFields(block, nodeViewId, blockPtr);
            buildRuntimeMetrics(block, flowgraph, blockName, blockPtr);
            surfaces.buildSurfaces(block, flowgraph, flowgraphId, blockName, nodeViewId, blockPtr);
        }

        return block;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH
