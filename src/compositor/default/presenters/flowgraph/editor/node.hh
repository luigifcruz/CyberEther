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

    void buildBlockMetrics(FlowgraphNode::BlockData& block,
                           const std::string& nodeViewId,
                           const Flowgraph::View::BlockData& blockData) const {
        for (const auto& metric : blockData.metrics) {
            if (metric.format.starts_with("private-")) {
                continue;
            }

            block.metrics.push_back({
                .id = nodeViewId + ":metric:" + metric.name,
                .label = metric.label,
                .help = metric.help,
                .format = metric.format,
                .value = metric.value,
            });
        }
    }

    void buildConfigFields(FlowgraphNode::BlockData& block,
                           const std::string& nodeViewId,
                           const Flowgraph::View::BlockData& blockData) const {
        block.config = blockData.config;
        for (const auto& entry : blockData.interfaceConfigs) {
            std::string encoded;
            if (blockData.config.contains(entry.name)) {
                Parser::TypedToString(blockData.config.at(entry.name), encoded);
            }

            FlowgraphConfigFieldConfig field{
                .id = nodeViewId + ":config:" + entry.name,
                .name = entry.name,
                .label = entry.label.empty() ? entry.name : entry.label,
                .help = entry.help,
                .format = entry.format,
                .encoded = encoded,
                .values = blockData.config,
            };

            const auto formatParts = Parser::SplitString(entry.format, ":");
            if (!formatParts.empty() && formatParts[0] == "python") {
                for (const auto& metric : blockData.metrics) {
                    if (metric.format != "private-python-diagnostic" || !metric.value.has_value()) {
                        continue;
                    }

                    try {
                        const auto diagnostic = std::any_cast<Runtime::Context::Diagnostic>(metric.value);
                        field.status = diagnostic.status;
                        field.statusTone = diagnostic.healthy
                            ? Sakura::NodeCodeEditor::StatusTone::Success
                            : Sakura::NodeCodeEditor::StatusTone::Error;
                        field.consoleOutput = diagnostic.console;
                        field.consoleVisible = !field.consoleOutput.empty();
                    } catch (const std::bad_any_cast&) {
                    }
                    break;
                }
            }

            block.configFields.push_back(std::move(field));
        }
    }

    void buildTiming(FlowgraphNode::BlockData& block,
                     const Flowgraph::View::BlockData& blockData) const {
        F32 totalTime = 0.0f;
        U64 maxCycles = 0;

        for (const auto& metric : blockData.metrics) {
            if (metric.format != "private-timing") {
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

            block.timing.push_back(jst::fmt::format("{} ({}/{}/{}): {:.3f} ms",
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
        block.timing.push_back(jst::fmt::format("= {:.3f} ms ({})", totalTime, cyclesStr));
    }

    FlowgraphNode::BlockData build(const std::string& flowgraphId,
                                   const std::shared_ptr<Flowgraph>& flowgraph,
                                   const std::string& blockName,
                                   const Flowgraph::View::BlockData& blockData) const {
        const std::string nodeViewId = flowgraphId + ":" + blockName;
        const Block::State blockState = blockData.state;
        const bool hasDiagnostic = (blockState == Block::State::Errored ||
                                     blockState == Block::State::Incomplete) &&
                                    !blockData.diagnostic.empty();
        FlowgraphNode::BlockData block{
            .name = blockName,
            .module = blockData.type,
            .title = blockData.title,
            .documentation = blockData.description,
            .device = blockData.device,
            .runtime = blockData.runtime,
            .provider = blockData.provider,
            .state = blockState,
            .diagnostic = hasDiagnostic ? Sakura::CleanUserMessage(blockData.diagnostic) : "",
        };
        block.deviceOptions = catalog.buildDeviceOptions(block.module,
                                                         block.device,
                                                         block.runtime,
                                                         block.provider);

        NodeMeta nodeMeta;
        flowgraph->metadata().get("node", nodeMeta, blockName);
        block.layout = FlowgraphNode::Layout{
            .x = nodeMeta.x,
            .y = nodeMeta.y,
            .width = nodeMeta.width,
            .height = nodeMeta.height,
        };

        buildInputs(block, blockData);
        buildOutputs(block, blockData);

        if (blockState != Block::State::Creating) {
            buildBlockMetrics(block, nodeViewId, blockData);
            buildConfigFields(block, nodeViewId, blockData);
            buildTiming(block, blockData);
            surfaces.buildSurfaces(block, flowgraph, flowgraphId, blockName, nodeViewId, blockData);
        }

        return block;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH
