#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_EDITOR_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_EDITOR_BASE_HH

#include "link.hh"
#include "node.hh"
#include "picker.hh"

#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphEditor : public Sakura::Component {
    using Block = FlowgraphNode::BlockData;

    struct Config {
        std::string id;
        std::string contextId;
        bool clipboardHasData = false;
        bool debugRuntimeMetricsEnabled = false;
        bool openBlockPicker = false;
        std::vector<FlowgraphBlockPicker::BlockOption> blockOptions;

        std::string title;
        std::string summary;
        std::string author;
        std::string license;
        std::string description;
        std::string path;
        std::vector<Block> graph;

        std::function<void(const std::string&, std::optional<Extent2D<F32>>, DeviceType, RuntimeType, ProviderType)> onCreateBlock;
        std::function<void(const std::string&)> onCopyBlock;
        std::function<void(std::optional<Extent2D<F32>>)> onPasteBlock;
        std::function<void(const std::string&)> onReloadBlock;
        std::function<void(const std::string&)> onDeleteBlock;
        std::function<void(const std::string&, DeviceType, RuntimeType, ProviderType)> onChangeBlockDevice;
        std::function<void(const std::string&, const std::string&, const std::string&, const std::string&)> onConnectBlock;
        std::function<void(const std::string&, const std::string&)> onDisconnectBlock;
        std::function<void(const std::string&, Parser::Map, bool)> onReconfigureBlock;
        std::function<void(Result, std::string)> onConfigError;
        std::function<void(bool, std::vector<std::string>, std::function<void(std::string)>)> onBrowseConfigPath;
        std::function<void(const std::string&, F32, F32, F32, F32)> onNodeLayout;
    };

    void update(Config config) {
        this->config = std::move(config);

        pinIdToInfo.clear();
        nodeIdToBlockName.clear();
        linkIdToConnection.clear();
        nodeViewIds.clear();
        linkViewIds.clear();

        nodeEditor.update({
            .id = this->config.id + "editor",
            .contextId = this->config.contextId,
            .pasteEnabled = this->config.clipboardHasData,
            .onEditorContextMenu = [this](Extent2D<F32> gridPosition, Extent2D<F32>) {
                flowgraphContextMenuGridPos = gridPosition;
                flowgraphContextMenuOpen = true;
            },
            .onEditorDoubleClick = [this](Extent2D<F32> gridPosition) {
                if (blockPickerOpen || suppressEditorDoubleClick) {
                    suppressEditorDoubleClick = false;
                    return;
                }
                openBlockPickerAtGrid(gridPosition);
            },
            .onLinkCreated = [this](Sakura::NodeEditor::PinRef startPin,
                                    Sakura::NodeEditor::PinRef endPin) {
                handleLinkCreation(startPin, endPin);
            },
            .onLinkDestroyed = [this](const std::string& linkId) {
                handleLinkDestruction(linkId);
            },
            .onCopyShortcut = [this](const std::vector<std::string>& selectedNodes) {
                if (!selectedNodes.empty() && nodeIdToBlockName.contains(selectedNodes[0]) && this->config.onCopyBlock) {
                    this->config.onCopyBlock(nodeIdToBlockName.at(selectedNodes[0]));
                }
            },
            .onPasteShortcut = [this](Extent2D<F32> gridPosition) {
                if (this->config.onPasteBlock) {
                    this->config.onPasteBlock(gridPosition);
                }
            },
            .onMouseGridPositionChange = [this](Extent2D<F32> gridPosition) {
                mouseGridPosition = gridPosition;
            },
            .onViewportGridCenterChange = [this](Extent2D<F32> gridPosition) {
                viewportGridCenter = gridPosition;
            },
        });

        if (this->config.openBlockPicker) {
            openBlockPickerAtGrid(viewportGridCenter);
        }
        if (blockPickerWasOpen && !blockPickerOpen) {
            blockPicker = FlowgraphBlockPicker();
        }

        blockPicker.update({
            .id = this->config.id + ":block-picker",
            .search = blockPickerSearch,
            .selectedIndex = blockPickerSelectedIndex,
            .blocks = this->config.blockOptions,
            .onResolveGridPosition = [this]() {
                return blockPickerGridPosition;
            },
            .onSearchChange = [this](const std::string& value) {
                blockPickerSearch = value;
            },
            .onSelectIndex = [this](const int index) {
                blockPickerSelectedIndex = index;
            },
            .onCreateBlock = [this](const std::string& moduleId,
                                    Extent2D<F32> gridPosition,
                                    DeviceType device,
                                    RuntimeType runtime,
                                    ProviderType provider) {
                suppressEditorDoubleClick = true;
                if (this->config.onCreateBlock) {
                    this->config.onCreateBlock(moduleId, gridPosition, device, runtime, provider);
                }
            },
            .onClose = [this]() {
                suppressEditorDoubleClick = true;
                blockPickerOpen = false;
            },
        });
        blockPickerWasOpen = blockPickerOpen;

        for (const auto& block : this->config.graph) {
            if (block.state == Jetstream::Block::State::Destroying ||
                block.state == Jetstream::Block::State::Destroyed) {
                continue;
            }

            const std::string nodeId = FlowgraphNodeId(block.name);
            nodeIdToBlockName[nodeId] = block.name;

            Block nodeBlock = block;
            auto onReconfigureBlock = this->config.onReconfigureBlock;
            for (auto& input : nodeBlock.inputs) {
                pinIdToInfo[nodeId][pinKey(FlowgraphPinId(input.port.id), true)] = {block.name, input.port.id, true};
            }
            for (auto& output : nodeBlock.outputs) {
                pinIdToInfo[nodeId][pinKey(FlowgraphPinId(output.port.id), false)] = {block.name, output.port.id, false};
            }
            for (auto& field : nodeBlock.configFields) {
                field.onApply = [onReconfigureBlock, blockName = block.name](Parser::Map values, const bool silent) {
                    if (onReconfigureBlock) {
                        onReconfigureBlock(blockName, std::move(values), silent);
                    }
                };
                field.onError = this->config.onConfigError;
                field.onBrowsePath = this->config.onBrowseConfigPath;
            }
            FlowgraphNode::Config nodeConfig{
                .id = block.name,
                .block = std::move(nodeBlock),
                .pasteEnabled = this->config.clipboardHasData,
                .runtimeMetricsEnabled = this->config.debugRuntimeMetricsEnabled,
                .onCopy = [this, blockName = block.name]() {
                    if (this->config.onCopyBlock) {
                        this->config.onCopyBlock(blockName);
                    }
                },
                .onPaste = [this](Extent2D<F32> pastePos) {
                    if (this->config.onPasteBlock) {
                        this->config.onPasteBlock(pastePos);
                    }
                },
                .onReload = [this, blockName = block.name]() {
                    if (this->config.onReloadBlock) {
                        this->config.onReloadBlock(blockName);
                    }
                },
                .onDelete = [this, blockName = block.name]() {
                    if (this->config.onDeleteBlock) {
                        this->config.onDeleteBlock(blockName);
                    }
                },
                .onDeviceSelect = [this, blockName = block.name](DeviceType device,
                                                                 RuntimeType runtime,
                                                                 ProviderType provider) {
                    if (!this->config.onChangeBlockDevice) {
                        return;
                    }

                    this->config.onChangeBlockDevice(blockName, device, runtime, provider);
                },
                .onLayout = [this, blockName = block.name](F32 x, F32 y, F32 width, F32 height) {
                    if (this->config.onNodeLayout) {
                        this->config.onNodeLayout(blockName, x, y, width, height);
                    }
                },
            };

            auto& nodeView = nodeViews[nodeConfig.id];
            nodeView.update(std::move(nodeConfig));
            nodeViewIds.push_back(block.name);
        }

        for (const auto& block : this->config.graph) {
            for (const auto& input : block.inputs) {
                if (!input.source.has_value()) {
                    continue;
                }

                const auto& source = input.source.value();
                const std::string inNodeId = FlowgraphNodeId(block.name);
                const std::string outNodeId = FlowgraphNodeId(source.block);
                const std::string inPinId = FlowgraphPinId(input.port.id);
                const std::string outPinId = FlowgraphPinId(source.port);
                const std::string linkId = FlowgraphLinkId(block.name, input.port.id);

                if (!hasPinInfo(inNodeId, inPinId, true) || !hasPinInfo(outNodeId, outPinId, false)) {
                    continue;
                }

                FlowgraphLink::Connection connection{
                    .consumerName = block.name,
                    .inputSlot = input.port.id,
                    .producerName = source.block,
                    .producerPort = source.port,
                    .resolved = source.resolved,
                    .tensor = source.tensor,
                };

                linkIdToConnection[linkId] = connection;
                auto& linkView = linkViews[linkId];
                linkView.update({
                    .id = linkId,
                    .start = {
                        .nodeId = outNodeId,
                        .pinId = outPinId,
                        .isInput = false,
                    },
                    .end = {
                        .nodeId = inNodeId,
                        .pinId = inPinId,
                        .isInput = true,
                    },
                    .connection = std::move(connection),
                });
                linkViewIds.push_back(linkId);
            }
        }

        std::vector<std::string> staleLinkIds;
        for (const auto& [linkId, _] : linkViews) {
            if (!linkIdToConnection.contains(linkId)) {
                staleLinkIds.push_back(linkId);
            }
        }
        for (const auto& linkId : staleLinkIds) {
            linkViews.erase(linkId);
        }

        flowgraphPasteMenuItem.update({
            .id = this->config.id + ":context-menu:paste",
            .label = ICON_FA_PASTE " Paste Block",
            .shortcut = "CTRL+V",
            .enabled = this->config.clipboardHasData,
            .onClick = [this]() {
                if (this->config.onPasteBlock) {
                    this->config.onPasteBlock(flowgraphContextMenuGridPos);
                }
            },
        });

        flowgraphContextMenu.update({
            .id = this->config.id + ":context-menu",
            .onClose = [this]() {
                flowgraphContextMenuOpen = false;
            },
        });
    }

    void render(const Sakura::Context& ctx) {
        if (!ctx.nodeContext(config.contextId).native) {
            return;
        }

        nodeEditor.render(ctx, [this](const Sakura::Context& ctx) {
            for (const auto& nodeViewId : nodeViewIds) {
                nodeViews.at(nodeViewId).render(ctx);
            }
            for (const auto& linkViewId : linkViewIds) {
                linkViews.at(linkViewId).render(ctx);
            }
            if (blockPickerOpen) {
                blockPicker.render(ctx);
            }
        });
        suppressEditorDoubleClick = false;

        if (flowgraphContextMenuOpen) {
            flowgraphContextMenu.render(ctx, [&](const Sakura::Context& ctx) {
                flowgraphPasteMenuItem.render(ctx);
            });
        }
    }

 private:
    void openBlockPickerAtGrid(const Extent2D<F32>& blockGridPosition) {
        blockPickerGridPosition = blockGridPosition;
        blockPickerSearch.clear();
        blockPickerSelectedIndex = 0;
        blockPickerOpen = true;
    }

    struct PinInfo {
        std::string block;
        std::string port;
        bool isInput = false;
    };

    static std::string pinKey(const std::string& pinId, const bool isInput) {
        return (isInput ? "in:" : "out:") + pinId;
    }

    bool hasPinInfo(const std::string& nodeId, const std::string& pinId, const bool isInput) const {
        const auto nodeIt = pinIdToInfo.find(nodeId);
        return nodeIt != pinIdToInfo.end() && nodeIt->second.contains(pinKey(pinId, isInput));
    }

    const PinInfo* resolvePinInfo(const Sakura::NodeEditor::PinRef& pin) const {
        const auto nodeIt = pinIdToInfo.find(pin.nodeId);
        if (nodeIt == pinIdToInfo.end()) {
            return nullptr;
        }

        const auto pinIt = nodeIt->second.find(pinKey(pin.pinId, pin.isInput));
        if (pinIt == nodeIt->second.end()) {
            return nullptr;
        }

        return &pinIt->second;
    }

    void handleLinkCreation(const Sakura::NodeEditor::PinRef& startPin,
                            const Sakura::NodeEditor::PinRef& endPin) const {
        const auto* startInfo = resolvePinInfo(startPin);
        const auto* endInfo = resolvePinInfo(endPin);
        if (!startInfo || !endInfo || !config.onConnectBlock) {
            return;
        }
        if (startInfo->isInput == endInfo->isInput) {
            return;
        }

        const auto& input = startInfo->isInput ? *startInfo : *endInfo;
        const auto& output = startInfo->isInput ? *endInfo : *startInfo;
        config.onConnectBlock(input.block, input.port, output.block, output.port);
    }

    void handleLinkDestruction(const std::string& linkId) const {
        if (!linkIdToConnection.contains(linkId) || !config.onDisconnectBlock) {
            return;
        }

        const auto& info = linkIdToConnection.at(linkId);
        config.onDisconnectBlock(info.consumerName, info.inputSlot);
    }

    Config config;
    Sakura::NodeEditor nodeEditor;
    FlowgraphBlockPicker blockPicker;
    bool blockPickerOpen = false;
    bool blockPickerWasOpen = false;
    bool suppressEditorDoubleClick = false;
    std::string blockPickerSearch;
    int blockPickerSelectedIndex = 0;
    Extent2D<F32> blockPickerGridPosition;
    Sakura::ContextMenu flowgraphContextMenu;
    Sakura::MenuItem flowgraphPasteMenuItem;
    bool flowgraphContextMenuOpen = false;
    Extent2D<F32> flowgraphContextMenuGridPos;
    Extent2D<F32> mouseGridPosition;
    Extent2D<F32> viewportGridCenter;
    std::unordered_map<std::string, std::unordered_map<std::string, PinInfo>> pinIdToInfo;
    std::unordered_map<std::string, std::string> nodeIdToBlockName;
    std::unordered_map<std::string, FlowgraphLink::Connection> linkIdToConnection;
    std::vector<std::string> nodeViewIds;
    std::vector<std::string> linkViewIds;
    std::unordered_map<std::string, FlowgraphLink> linkViews;
    std::unordered_map<std::string, FlowgraphNode> nodeViews;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_EDITOR_BASE_HH
