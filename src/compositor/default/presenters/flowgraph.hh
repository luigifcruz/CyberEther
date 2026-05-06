#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/meta.hh"
#include "../model/state.hh"
#include "../views/flowgraph/editor/node.hh"
#include "../views/flowgraph/editor/picker.hh"
#include "../views/flowgraph/modals/close.hh"
#include "../views/flowgraph/modals/examples.hh"
#include "../views/flowgraph/modals/info.hh"
#include "../views/flowgraph/modals/rename.hh"
#include "../views/flowgraph/window.hh"

#include "jetstream/block.hh"
#include "jetstream/block_interface.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/registry.hh"

#include <algorithm>
#include <any>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Jetstream {

struct DefaultFlowgraphPresenter {
    using ModalContent = DefaultCompositorState::ModalState::Content;

    struct ModalConfigs {
        FlowgraphExamplesView::Config examples;
        std::optional<FlowgraphInfoView::Config> info;
        std::optional<FlowgraphCloseView::Config> close;
        std::optional<RenameBlockView::Config> renameBlock;
    };

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    DefaultFlowgraphPresenter(DefaultCompositorState& state,
                              DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    FlowgraphExamplesView::Config buildFlowgraphExamples() const {
        const auto enqueue = callbacks.enqueueMail;
        std::vector<FlowgraphExamplesView::Example> examples;
        for (const auto& entry : Registry::ListAvailableFlowgraphs()) {
            examples.push_back({
                .key = entry.key,
                .title = entry.title,
                .summary = entry.summary,
                .content = entry.content,
            });
        }

        return FlowgraphExamplesView::Config{
            .examples = std::move(examples),
            .onOpen = [enqueue](const FlowgraphExamplesView::Example& flowgraph) {
                std::vector<char> blob(flowgraph.content.begin(), flowgraph.content.end());
                enqueue(MailOpenFlowgraphBlob{std::move(blob)});
                enqueue(MailCloseModal{});
            },
        };
    }

    std::optional<std::string> buildTargetFlowgraph() const {
        return state.modal.flowgraph.has_value()
            ? state.modal.flowgraph
            : state.interface.focusedFlowgraph;
    }

    std::optional<FlowgraphInfoView::Config> buildFlowgraphInfo() const {
        const auto targetFlowgraph = buildTargetFlowgraph();
        if (!targetFlowgraph.has_value() || !state.flowgraph.items.contains(targetFlowgraph.value())) {
            return std::nullopt;
        }

        const auto enqueue = callbacks.enqueueMail;
        const std::string flowgraphId = targetFlowgraph.value();
        const auto& flowgraph = state.flowgraph.items.at(flowgraphId);
        return FlowgraphInfoView::Config{
            .flowgraphId = flowgraphId,
            .title = flowgraph->title(),
            .summary = flowgraph->summary(),
            .author = flowgraph->author(),
            .license = flowgraph->license(),
            .description = flowgraph->description(),
            .path = flowgraph->path(),
            .onTitleChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .title = value});
            },
            .onSummaryChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .summary = value});
            },
            .onAuthorChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .author = value});
            },
            .onLicenseChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .license = value});
            },
            .onDescriptionChange = [enqueue, flowgraphId](const std::string& value) {
                enqueue(MailSetFlowgraphInfo{.flowgraph = flowgraphId, .description = value});
            },
            .onBrowse = [enqueue](const std::string& currentPath, std::function<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = currentPath,
                    .save = true,
                    .extensions = {"yaml", "yml"},
                    .onSelect = std::move(onSelect),
                });
            },
            .onSave = [enqueue, flowgraphId](const std::string& filename) {
                enqueue(MailSaveFlowgraphPath{.flowgraph = flowgraphId, .path = filename});
            },
        };
    }

    std::optional<FlowgraphCloseView::Config> buildFlowgraphClose() const {
        const auto targetFlowgraph = buildTargetFlowgraph();
        if (!targetFlowgraph.has_value()) {
            return std::nullopt;
        }

        const auto enqueue = callbacks.enqueueMail;
        const std::string flowgraphId = targetFlowgraph.value();
        return FlowgraphCloseView::Config{
            .onSave = [enqueue, flowgraphId]() {
                enqueue(MailOpenModal{
                    .content = ModalContent::FlowgraphInfo,
                    .flowgraph = flowgraphId,
                });
            },
            .onDontSave = [enqueue, flowgraphId]() {
                enqueue(MailCloseFlowgraph{.flowgraph = flowgraphId, .force = true});
                enqueue(MailCloseModal{});
            },
        };
    }

    std::optional<RenameBlockView::Config> buildRenameBlock() const {
        if (!state.interface.focusedFlowgraph.has_value() || !state.modal.renameBlockOldName.has_value()) {
            return std::nullopt;
        }

        const auto enqueue = callbacks.enqueueMail;
        const std::string focusedFlowgraph = state.interface.focusedFlowgraph.value();
        const std::string oldName = state.modal.renameBlockOldName.value();
        return RenameBlockView::Config{
            .oldName = oldName,
            .onRename = [enqueue, focusedFlowgraph, oldName](const std::string& newName) {
                enqueue(MailRenameBlock{focusedFlowgraph, oldName, newName});
                enqueue(MailCloseModal{});
            },
        };
    }

    ModalConfigs buildModalConfigs() const {
        ModalConfigs configs;
        if (state.modal.content == ModalContent::FlowgraphExamples) {
            configs.examples = buildFlowgraphExamples();
        } else if (state.modal.content == ModalContent::FlowgraphInfo) {
            configs.info = buildFlowgraphInfo();
        } else if (state.modal.content == ModalContent::FlowgraphClose) {
            configs.close = buildFlowgraphClose();
        } else if (state.modal.content == ModalContent::RenameBlock) {
            configs.renameBlock = buildRenameBlock();
        }
        return configs;
    }

    std::vector<DeviceType> buildDeviceList(const std::string& blockType, const DeviceType& currentDevice) const {
        std::vector<DeviceType> devices;
        for (const auto& module : Registry::ListAvailableModules(blockType)) {
            if (std::find(devices.begin(), devices.end(), module.device) == devices.end()) {
                devices.push_back(module.device);
            }
        }

        if (currentDevice != DeviceType::None &&
            std::find(devices.begin(), devices.end(), currentDevice) == devices.end()) {
            devices.push_back(currentDevice);
        }

        return devices;
    }

    std::optional<Registry::ModuleRegistration> buildDeviceImplementation(const std::string& blockType,
                                                                          const DeviceType& device,
                                                                          const RuntimeType& preferredRuntime,
                                                                          const ProviderType& preferredProvider) const {
        const auto implementations = Registry::ListAvailableModules(blockType, std::optional<DeviceType>{device});
        if (implementations.empty()) {
            return std::nullopt;
        }

        const auto exactMatch = std::find_if(implementations.begin(), implementations.end(), [&](const auto& entry) {
            return entry.runtime == preferredRuntime && entry.provider == preferredProvider;
        });
        if (exactMatch != implementations.end()) {
            return *exactMatch;
        }

        const auto runtimeMatch = std::find_if(implementations.begin(), implementations.end(), [&](const auto& entry) {
            return entry.runtime == preferredRuntime;
        });
        if (runtimeMatch != implementations.end()) {
            return *runtimeMatch;
        }

        const auto providerMatch = std::find_if(implementations.begin(), implementations.end(), [&](const auto& entry) {
            return entry.provider == preferredProvider;
        });
        if (providerMatch != implementations.end()) {
            return *providerMatch;
        }

        return implementations.front();
    }

    std::vector<FlowgraphNode::DeviceOption> buildDeviceOptions(const std::string& blockType,
                                                               const DeviceType& currentDevice,
                                                               const RuntimeType& currentRuntime,
                                                               const ProviderType& currentProvider) const {
        std::vector<FlowgraphNode::DeviceOption> options;
        for (const auto& device : buildDeviceList(blockType, currentDevice)) {
            if (device == currentDevice) {
                options.push_back({
                    .label = GetDevicePrettyName(device),
                    .selected = true,
                    .device = currentDevice,
                    .runtime = currentRuntime,
                    .provider = currentProvider,
                });
                continue;
            }

            const auto implementation = buildDeviceImplementation(blockType,
                                                                  device,
                                                                  currentRuntime,
                                                                  currentProvider);
            if (!implementation.has_value()) {
                continue;
            }

            options.push_back({
                .label = GetDevicePrettyName(device),
                .selected = false,
                .device = implementation->device,
                .runtime = implementation->runtime,
                .provider = implementation->provider,
            });
        }

        return options;
    }

    std::vector<FlowgraphBlockPicker::BlockOption> buildBlockCatalog() const {
        std::vector<FlowgraphBlockPicker::BlockOption> options;
        for (const auto& entry : Registry::ListAvailableBlocks("")) {
            DeviceType device = DeviceType::CPU;
            RuntimeType runtime = RuntimeType::NATIVE;
            ProviderType provider = "generic";
            std::vector<FlowgraphBlockPicker::DeviceOption> devices;

            const auto modules = Registry::ListAvailableModules(entry.type);
            if (!modules.empty()) {
                device = modules.front().device;
                runtime = modules.front().runtime;
                provider = modules.front().provider;
                for (const auto& module : modules) {
                    const auto duplicate = std::find_if(devices.begin(), devices.end(), [&](const auto& option) {
                        return option.device == module.device;
                    });
                    if (duplicate == devices.end()) {
                        devices.push_back({
                            .device = module.device,
                            .runtime = module.runtime,
                            .provider = module.provider,
                        });
                    }
                }
            } else {
                devices.push_back({
                    .device = device,
                    .runtime = runtime,
                    .provider = provider,
                });
            }

            options.push_back({
                .type = entry.type,
                .title = entry.title,
                .summary = entry.summary,
                .description = entry.description,
                .category = entry.domain,
                .devices = std::move(devices),
                .device = device,
                .runtime = runtime,
                .provider = provider,
            });
        }

        return options;
    }

    void buildInputs(FlowgraphEditor::Block& block, const std::shared_ptr<Block>& blockPtr) const {
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

    void buildOutputs(FlowgraphEditor::Block& block, const std::shared_ptr<Block>& blockPtr) const {
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

    void buildBlockMetrics(FlowgraphEditor::Block& block,
                           const std::string& nodeViewId,
                           const std::shared_ptr<Block>& blockPtr) const {
        for (const auto& [name, entry] : blockPtr->interface()->metrics()) {
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

    void buildConfigFields(FlowgraphEditor::Block& block,
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

    void buildRuntimeMetrics(FlowgraphEditor::Block& block,
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

    void buildSurfaces(FlowgraphEditor::Block& block,
                       const std::shared_ptr<Flowgraph>& flowgraph,
                       const std::string& flowgraphId,
                       const std::string& blockName,
                       const std::string& nodeViewId,
                       const std::shared_ptr<Block>& blockPtr) const {
        const auto enqueue = callbacks.enqueueMail;
        for (const auto& surface : blockPtr->surfaces()) {
            for (const auto& manifest : surface->manifests()) {
                if (!manifest.surface || manifest.surface->raw() == 0) {
                    continue;
                }

                const std::string surfaceMetaKey = "surface_" + manifest.id;
                SurfaceMeta surfaceMeta;
                flowgraph->getMeta(surfaceMetaKey, surfaceMeta, blockName);
                std::optional<Extent2D<F32>> aspectRatioSize;
                const SurfaceMeta defaultSurfaceMeta;
                if (surfaceMeta.attachedWidth == defaultSurfaceMeta.attachedWidth &&
                    surfaceMeta.attachedHeight == defaultSurfaceMeta.attachedHeight &&
                    surfaceMeta.attachedWidth > 0 && surfaceMeta.attachedHeight > 0) {
                    aspectRatioSize = Extent2D<F32>{
                        static_cast<F32>(surfaceMeta.attachedWidth),
                        static_cast<F32>(surfaceMeta.attachedHeight),
                    };
                }

                block.surfaces.push_back({
                    .id = nodeViewId + ":surface:" + manifest.id,
                    .texture = manifest.surface,
                    .logicalSize = {
                        static_cast<F32>(surfaceMeta.detachedWidth),
                        static_cast<F32>(surfaceMeta.detachedHeight),
                    },
                    .aspectRatioSize = aspectRatioSize,
                    .onAttachedSize = [enqueue,
                                       surface,
                                       flowgraphId,
                                       surfaceMetaKey,
                                       blockName](const Sakura::SurfaceSize& size) {
                        enqueue(MailResizeSurface{
                            .surface = surface,
                            .flowgraph = flowgraphId,
                            .block = blockName,
                            .metaKey = surfaceMetaKey,
                            .width = size.framebufferSize.x,
                            .height = size.framebufferSize.y,
                            .scale = size.scale,
                            .detachedSurface = false,
                            .attached = size.logicalSize,
                        });
                    },
                    .onDetachedSize = [enqueue,
                                       surface,
                                       flowgraphId,
                                       surfaceMetaKey,
                                       blockName](const Sakura::SurfaceSize& size) {
                        enqueue(MailResizeSurface{
                            .surface = surface,
                            .flowgraph = flowgraphId,
                            .block = blockName,
                            .metaKey = surfaceMetaKey,
                            .width = size.framebufferSize.x,
                            .height = size.framebufferSize.y,
                            .scale = size.scale,
                            .detachedSurface = true,
                            .detached = size.logicalSize,
                        });
                    },
                    .onMouse = [enqueue, surface](MouseEvent event) {
                        enqueue(MailSurfaceMouse{
                            .surface = surface,
                            .event = event,
                        });
                    },
                });
            }
        }
    }

    FlowgraphEditor::Block buildBlock(const std::string& flowgraphId,
                                      const std::shared_ptr<Flowgraph>& flowgraph,
                                      const std::string& blockName,
                                      const std::shared_ptr<Block>& blockPtr) const {
        const std::string nodeViewId = flowgraphId + ":" + blockName;
        const bool hasDiagnostic = (blockPtr->state() == Block::State::Errored ||
                                    blockPtr->state() == Block::State::Incomplete) &&
                                   !blockPtr->diagnostic().empty();
        FlowgraphEditor::Block block{
            .name = blockName,
            .module = blockPtr->config().type(),
            .title = blockPtr->config().title(),
            .documentation = blockPtr->config().description(),
            .device = blockPtr->device(),
            .runtime = blockPtr->runtime(),
            .provider = blockPtr->provider(),
            .state = blockPtr->state(),
            .diagnostic = hasDiagnostic ? Sakura::CleanUserMessage(blockPtr->diagnostic()) : "",
        };
        block.deviceOptions = buildDeviceOptions(block.module,
                                                 block.device,
                                                 block.runtime,
                                                 block.provider);

        NodeMeta nodeMeta;
        flowgraph->getMeta("node", nodeMeta, blockName);
        block.layout = FlowgraphNode::Layout{
            .x = nodeMeta.x,
            .y = nodeMeta.y,
            .width = nodeMeta.width,
            .height = nodeMeta.height,
        };

        buildInputs(block, blockPtr);
        buildOutputs(block, blockPtr);

        if (blockPtr->state() != Block::State::Creating) {
            buildBlockMetrics(block, nodeViewId, blockPtr);
            buildConfigFields(block, nodeViewId, blockPtr);
            buildRuntimeMetrics(block, flowgraph, blockName, blockPtr);
            buildSurfaces(block, flowgraph, flowgraphId, blockName, nodeViewId, blockPtr);
        }

        return block;
    }

    FlowgraphEditor::Config buildEditor(const std::string& flowgraphId,
                                        const std::shared_ptr<Flowgraph>& flowgraph,
                                        const std::unordered_map<std::string, std::shared_ptr<Block>>& blocks) const {
        const auto enqueue = callbacks.enqueueMail;
        FlowgraphEditor::Config editorConfig{
            .id = flowgraphId,
            .contextId = flowgraphId,
            .clipboardHasData = state.clipboard.hasData,
            .debugRuntimeMetricsEnabled = state.debug.runtimeMetricsEnabled,
            .blockOptions = buildBlockCatalog(),
            .title = flowgraph->title(),
            .summary = flowgraph->summary(),
            .author = flowgraph->author(),
            .license = flowgraph->license(),
            .description = flowgraph->description(),
            .path = flowgraph->path(),
            .onCreateBlock = [enqueue, flowgraphId](const std::string& moduleId,
                                                    std::optional<Extent2D<F32>> gridPosition,
                                                    DeviceType device,
                                                    RuntimeType runtime,
                                                    ProviderType provider) {
                enqueue(MailCreateBlock{flowgraphId, moduleId, gridPosition, device, runtime, provider});
            },
            .onCopyBlock = [enqueue, flowgraphId](const std::string& blockName) {
                enqueue(MailCopyBlock{flowgraphId, blockName});
            },
            .onPasteBlock = [enqueue, flowgraphId](std::optional<Extent2D<F32>> gridPosition) {
                enqueue(MailPasteBlock{flowgraphId, gridPosition});
            },
            .onReloadBlock = [enqueue, flowgraphId](const std::string& blockName) {
                enqueue(MailReloadBlock{flowgraphId, blockName});
            },
            .onDeleteBlock = [enqueue, flowgraphId](const std::string& blockName) {
                enqueue(MailDeleteBlock{flowgraphId, blockName});
            },
            .onChangeBlockDevice = [enqueue, flowgraphId](const std::string& blockName,
                                                          DeviceType device,
                                                          RuntimeType runtime,
                                                          ProviderType provider) {
                enqueue(MailChangeBlockDevice{flowgraphId, blockName, device, runtime, provider});
            },
            .onConnectBlock = [enqueue, flowgraphId](const std::string& blockName,
                                                     const std::string& inputPort,
                                                     const std::string& sourceBlock,
                                                     const std::string& sourcePort) {
                enqueue(MailConnectBlock{flowgraphId, blockName, inputPort, sourceBlock, sourcePort});
            },
            .onDisconnectBlock = [enqueue, flowgraphId](const std::string& blockName,
                                                        const std::string& inputPort) {
                enqueue(MailDisconnectBlock{flowgraphId, blockName, inputPort});
            },
            .onReconfigureBlock = [enqueue, flowgraphId](const std::string& blockName,
                                                         Parser::Map config,
                                                         const bool silent) {
                enqueue(MailReconfigureBlock{flowgraphId, blockName, std::move(config), silent});
            },
            .onConfigError = [enqueue](const Result result, const std::string& message) {
                enqueue(MailNotifyResult{.result = result, .message = message});
            },
            .onBrowseConfigPath = [enqueue](bool save,
                                            std::vector<std::string> extensions,
                                            std::function<void(std::string)> onSelect) {
                enqueue(MailBrowseConfigPath{
                    .path = "",
                    .save = save,
                    .extensions = std::move(extensions),
                    .onSelect = std::move(onSelect),
                });
            },
            .onNodeLayout = [enqueue, flowgraphId](const std::string& blockName,
                                                   F32 x,
                                                   F32 y,
                                                   F32 width,
                                                   F32 height) {
                enqueue(MailSetNodeMeta{flowgraphId, blockName, NodeMeta{x, y, width, height}});
            },
        };

        for (const auto& [blockName, blockPtr] : blocks) {
            if (!blockPtr || !blockPtr->interface()) {
                continue;
            }

            editorConfig.graph.push_back(buildBlock(flowgraphId, flowgraph, blockName, blockPtr));
        }

        return editorConfig;
    }

    std::vector<FlowgraphWindow::Config> build() const {
        std::vector<FlowgraphWindow::Config> flowgraphWindowConfigs;
        flowgraphWindowConfigs.reserve(state.flowgraph.items.size());

        for (const auto& [flowgraphId, flowgraph] : state.flowgraph.items) {
            const auto blocks = flowgraph->blockList();
            const auto enqueue = callbacks.enqueueMail;
            flowgraphWindowConfigs.push_back({
                .id = flowgraphId,
                .title = flowgraph->title(),
                .editor = buildEditor(flowgraphId, flowgraph, blocks),
                .empty = blocks.empty(),
                .onSave = [enqueue, flowgraphId]() {
                    enqueue(MailSaveFlowgraph{.flowgraph = flowgraphId});
                },
                .onClose = [enqueue, flowgraphId]() {
                    enqueue(MailCloseFlowgraph{flowgraphId});
                },
            });
        }

        return flowgraphWindowConfigs;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_HH
