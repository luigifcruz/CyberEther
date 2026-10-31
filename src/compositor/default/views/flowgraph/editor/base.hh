#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_EDITOR_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_EDITOR_BASE_HH

#include "link.hh"
#include "node.hh"
#include "picker.hh"

#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphEditor {
    static constexpr F32 PillHeight = 40.0f;
    static constexpr F32 PillMargin = Sakura::Spacing::Medium;
    static constexpr F32 PillGap = Sakura::Spacing::Small;
    static constexpr F32 ZoomPillWidth = 116.0f;
    static constexpr F32 ZoomScale = Sakura::Typography::Scale::Caption;

    struct Measured {
        struct Node {
            Rect bounds;
            Extent2D<F32> size;
            std::optional<Extent2D<F32>> content;
        };

        std::unordered_map<std::string, Node> nodes;
        Rect viewport;
    };

    struct Config {
        std::string id;
        bool clipboardHasData = false;
        bool debugTimingEnabled = false;
        bool measure = false;
        bool paused = false;
        Sakura::Callback<std::vector<FlowgraphBlockPicker::BlockOption>()> onBuildBlockOptions;

        std::vector<FlowgraphNode::BlockData> graph;

        Sakura::Callback<void(const std::string&, std::optional<Extent2D<F32>>, DeviceType, RuntimeType, ProviderType)> onCreateBlock;
        Sakura::Callback<void(const std::string&)> onCopyBlock;
        Sakura::Callback<void(std::optional<Extent2D<F32>>)> onPasteBlock;
        Sakura::Callback<void(const std::string&)> onRenameBlock;
        Sakura::Callback<void(const std::string&)> onReloadBlock;
        Sakura::Callback<void(const std::string&)> onDeleteBlock;
        Sakura::Callback<void(const std::string&)> onOpenDocumentation;
        Sakura::Callback<void(const std::string&, DeviceType, RuntimeType, ProviderType)> onChangeBlockDevice;
        Sakura::Callback<void(const std::string&, const std::string&, const std::string&, const std::string&)> onConnectBlock;
        Sakura::Callback<void(const std::string&, const std::string&)> onDisconnectBlock;
        Sakura::Callback<void(const std::string&, Parser::Map, bool, Sakura::Callback<void(Result)>)> onReconfigureBlock;
        Sakura::Callback<void(Result, std::string)> onConfigError;
        Sakura::Callback<void(bool, std::vector<std::string>, Sakura::Callback<void(std::string)>)> onBrowseConfigPath;
        Sakura::Callback<void(const std::string&)> onOpenUrl;
        Sakura::Callback<void(const std::string&, F32, F32, F32, F32, std::optional<FlowgraphNode::Layout>)> onNodeLayout;
        Sakura::Callback<void(Measured)> onLayout;
        Sakura::Callback<void(const std::string&, bool)> onNodeConfigCollapse;
        Sakura::Callback<void(const std::string&, bool)> onNodeConfigDetach;
        Sakura::Callback<void()> onTogglePause;
        Sakura::Callback<void()> onSave;
        Sakura::Callback<void()> onToggleEnvironment;
        Sakura::Callback<void()> onOpenTerminal;
    };

    void update(Config config) {
        this->config = std::move(config);

        nodeViewIds.clear();
        linkViewIds.clear();
        liveBlocks.clear();

        blockPicker.update({
            .onCreateBlock = [this](const std::string& moduleId,
                                    Extent2D<F32> gridPosition,
                                    DeviceType device,
                                    RuntimeType runtime,
                                    ProviderType provider) {
                nodeEditor.stopFraming();
                if (this->config.onCreateBlock) {
                    this->config.onCreateBlock(moduleId, gridPosition, device, runtime, provider);
                }
            },
        });

        for (const auto& block : this->config.graph) {
            if (block.state != Block::State::Destroyed) {
                liveBlocks.insert(block.name);
            }
        }

        for (const auto& block : this->config.graph) {
            if (!liveBlocks.contains(block.name)) {
                continue;
            }
            for (const auto& input : block.inputs) {
                if (!input.source.has_value() || !liveBlocks.contains(input.source->block)) {
                    continue;
                }

                const auto& source = input.source.value();
                const std::string linkId = FlowgraphLinkId(block.name, input.port.id);
                auto& linkView = linkViews[linkId];
                linkView.update({
                    .id = linkId,
                    .start = {
                        .nodeId = source.block,
                        .pinId = source.port,
                        .isInput = false,
                    },
                    .end = {
                        .nodeId = block.name,
                        .pinId = input.port.id,
                        .isInput = true,
                    },
                    .connection = {
                        .consumerName = block.name,
                        .inputSlot = input.port.id,
                        .producerName = source.block,
                        .producerPort = source.port,
                        .resolved = source.resolved,
                        .tensor = source.tensor,
                    },
                });
                linkViewIds.push_back(linkId);
            }
        }

        std::erase_if(linkViews, [this](const auto& entry) {
            return std::find(linkViewIds.begin(), linkViewIds.end(), entry.first) == linkViewIds.end();
        });

        for (auto& block : this->config.graph) {
            if (!liveBlocks.contains(block.name)) {
                continue;
            }

            const std::string blockName = block.name;
            const auto base = block.layout;
            for (auto& field : block.configFields) {
                field.onApply = [this, blockName](Parser::Map values,
                                                  const bool silent,
                                                  Sakura::Callback<void(Result)> onComplete) {
                    if (this->config.onReconfigureBlock) {
                        this->config.onReconfigureBlock(blockName, std::move(values), silent, std::move(onComplete));
                    }
                };
                field.onError = [this](Result result, std::string message) {
                    if (this->config.onConfigError) {
                        this->config.onConfigError(result, std::move(message));
                    }
                };
                field.onBrowsePath = [this](bool save,
                                            std::vector<std::string> extensions,
                                            Sakura::Callback<void(std::string)> onPicked) {
                    if (this->config.onBrowseConfigPath) {
                        this->config.onBrowseConfigPath(save, std::move(extensions), std::move(onPicked));
                    }
                };
                field.onOpenUrl = [this](const std::string& url) {
                    if (this->config.onOpenUrl) {
                        this->config.onOpenUrl(url);
                    }
                };
            }
            FlowgraphNode::Config nodeConfig{
                .id = blockName,
                .inspectorId = jst::fmt::format("{}:{}:inspector", this->config.id, blockName),
                .block = std::move(block),
                .pasteEnabled = this->config.clipboardHasData,
                .timingEnabled = this->config.debugTimingEnabled,
                .onCopy = [this, blockName]() {
                    if (this->config.onCopyBlock) {
                        this->config.onCopyBlock(blockName);
                    }
                },
                .onPaste = [this](Extent2D<F32> pastePos) {
                    if (this->config.onPasteBlock) {
                        this->config.onPasteBlock(pastePos);
                    }
                },
                .onRename = [this, blockName]() {
                    if (this->config.onRenameBlock) {
                        this->config.onRenameBlock(blockName);
                    }
                },
                .onReload = [this, blockName]() {
                    if (this->config.onReloadBlock) {
                        this->config.onReloadBlock(blockName);
                    }
                },
                .onInspectApply = [this, blockName](Parser::Map values) {
                    if (this->config.onReconfigureBlock) {
                        this->config.onReconfigureBlock(blockName, std::move(values), false, {});
                    }
                },
                .onDelete = [this, blockName]() {
                    if (this->config.onDeleteBlock) {
                        this->config.onDeleteBlock(blockName);
                    }
                },
                .onDocumentation = [this, blockName]() {
                    if (this->config.onOpenDocumentation) {
                        this->config.onOpenDocumentation(blockName);
                    }
                },
                .onDeviceSelect = [this, blockName](DeviceType device,
                                                    RuntimeType runtime,
                                                    ProviderType provider) {
                    if (!this->config.onChangeBlockDevice) {
                        return;
                    }

                    this->config.onChangeBlockDevice(blockName, device, runtime, provider);
                },
                .onLayout = [this, blockName, base](F32 x, F32 y, F32 width, F32 height) {
                    if (this->config.onNodeLayout) {
                        this->config.onNodeLayout(blockName, x, y, width, height, base);
                    }
                },
                .onBounds = [this, blockName](Rect bounds, Extent2D<F32> size, std::optional<Extent2D<F32>> content) {
                    measured.nodes[blockName] = {bounds, size, content};
                },
                .onConfigCollapse = [this, blockName](bool collapsed) {
                    if (this->config.onNodeConfigCollapse) {
                        this->config.onNodeConfigCollapse(blockName, collapsed);
                    }
                },
                .onConfigDetach = [this, blockName](bool detached) {
                    if (this->config.onNodeConfigDetach) {
                        this->config.onNodeConfigDetach(blockName, detached);
                    }
                },
            };

            nodeViews[blockName].update(std::move(nodeConfig));
            nodeViewIds.push_back(blockName);
        }

        std::erase_if(nodeViews, [this](const auto& entry) {
            return !liveBlocks.contains(entry.first);
        });
        std::erase_if(measured.nodes, [this](const auto& entry) {
            return !liveBlocks.contains(entry.first);
        });

        std::vector<Sakura::NodeEditor::Node> nodes;
        nodes.reserve(nodeViewIds.size());
        for (const auto& nodeViewId : nodeViewIds) {
            nodes.push_back(nodeViews.at(nodeViewId).takeNode());
        }
        std::vector<Sakura::NodeEditor::Link> links;
        links.reserve(linkViewIds.size());
        for (const auto& linkViewId : linkViewIds) {
            links.push_back(linkViews.at(linkViewId).takeLink());
        }

        nodeEditor.update({
            .id = this->config.id + "editor",
            .pasteEnabled = this->config.clipboardHasData,
            .frameOnOpen = true,
            .scale = ZoomScale,
            .nodes = std::move(nodes),
            .links = std::move(links),
            .onEditorContextMenu = [this](Extent2D<F32> gridPosition) {
                flowgraphContextMenuGridPos = gridPosition;
                flowgraphContextMenuOpen = true;
            },
            .onEditorDoubleClick = [this](Extent2D<F32> gridPosition) {
                if (!blockPicker.isOpen()) {
                    openBlockPicker(gridPosition);
                }
            },
            .onLinkCreated = [this](Sakura::NodeEditor::PinRef startPin,
                                    Sakura::NodeEditor::PinRef endPin) {
                handleLinkCreation(startPin, endPin);
            },
            .onLinkDestroyed = [this](const std::string& linkId) {
                handleLinkDestruction(linkId);
            },
            .onCopyShortcut = [this](const std::vector<std::string>& selectedNodes) {
                if (!selectedNodes.empty() && liveBlocks.contains(selectedNodes[0]) && this->config.onCopyBlock) {
                    this->config.onCopyBlock(selectedNodes[0]);
                }
            },
            .onPasteShortcut = [this](Extent2D<F32> gridPosition) {
                if (this->config.onPasteBlock) {
                    this->config.onPasteBlock(gridPosition);
                }
            },
            .onViewportGridChange = [this](Rect viewport) {
                viewportGridCenter = {viewport.x + viewport.width * 0.5f, viewport.y + viewport.height * 0.5f};
                measured.viewport = viewport;
            },
        });

        flowgraphContextMenu.update({
            .entries = {
                {
                    .label = ICON_LC_CLIPBOARD_PASTE " Paste Block",
                    .shortcut = "CTRL+V",
                    .enabled = this->config.clipboardHasData,
                    .onClick = [this]() {
                        if (this->config.onPasteBlock) {
                            this->config.onPasteBlock(flowgraphContextMenuGridPos);
                        }
                    },
                },
            },
            .open = flowgraphContextMenuOpen,
            .onClose = [this]() {
                flowgraphContextMenuOpen = false;
            },
        });

        using Item = Sakura::Overlay::Item;
        std::vector<Item> items = {{.child = &nodeEditor}};
        for (const auto& nodeViewId : nodeViewIds) {
            items.push_back({.child = &nodeViews.at(nodeViewId).menuComponent()});
        }
        items.push_back({.child = &flowgraphContextMenu});
        items.push_back({.child = &blockPicker.component()});
        const F32 zoom = nodeEditor.zoom();
        runButton.update({
            .icon = this->config.paused ? ICON_LC_PLAY : ICON_LC_PAUSE,
            .iconColorKey = this->config.paused ? "run_state_paused" : "run_state_running",
            .tinted = true,
            .tooltip = this->config.paused ? "Resume flowgraph" : "Pause flowgraph",
            .size = {PillHeight, PillHeight},
            .onClick = this->config.onTogglePause,
        });
        saveButton.update({
            .icon = ICON_LC_SAVE,
            .tooltip = "Save flowgraph",
            .size = {PillHeight, PillHeight},
            .onClick = this->config.onSave,
        });
        addButton.update({
            .icon = ICON_LC_PACKAGE_PLUS,
            .tooltip = "Add block",
            .size = {PillHeight, PillHeight},
            .onClick = [this]() {
                nodeEditor.stopFraming();
                openBlockPicker(viewportGridCenter);
            },
        });
        environmentButton.update({
            .icon = ICON_LC_TABLE_PROPERTIES,
            .tooltip = "Environment variables",
            .size = {PillHeight, PillHeight},
            .onClick = this->config.onToggleEnvironment,
        });
        terminalButton.update({
            .icon = ICON_LC_SQUARE_TERMINAL,
            .tooltip = "Terminal",
            .size = {PillHeight, PillHeight},
            .onClick = this->config.onOpenTerminal,
        });
        zoomPill.update({
            .label = jst::fmt::format("{}%", std::lround(zoom * 100.0f)),
            .canDecrement = nodeEditor.canZoom(-1),
            .canIncrement = nodeEditor.canZoom(1),
            .size = {ZoomPillWidth, PillHeight},
            .onDecrement = [this]() {
                nodeEditor.zoomStep(-1);
            },
            .onIncrement = [this]() {
                nodeEditor.zoomStep(1);
            },
            .onReset = [this]() {
                nodeEditor.zoomReset();
            },
        });
        F32 edge = PillMargin;
        const auto place = [&](Sakura::Component& child, F32 width) {
            items.push_back({
                .child = &child,
                .anchor = Sakura::Anchor::BottomRight,
                .offset = {-edge, -PillMargin},
                .size = {width, PillHeight},
            });
            edge += width + PillGap;
        };
        place(zoomPill, ZoomPillWidth);
        if (this->config.onOpenTerminal) {
            place(terminalButton, PillHeight);
        }
        place(environmentButton, PillHeight);
        place(addButton, PillHeight);
        place(saveButton, PillHeight);
        place(runButton, PillHeight);
        overlay.update({.items = std::move(items)});

        if (this->config.measure && this->config.onLayout && nodeEditor.shown()) {
            this->config.onLayout(measured);
        }
    }

    Sakura::Component& component() {
        return overlay;
    }

    bool pickerOpen() const {
        return blockPicker.isOpen();
    }

    void appendWindows(std::vector<Sakura::FloatingLayer::Window>& windows) {
        for (const auto& nodeViewId : nodeViewIds) {
            nodeViews.at(nodeViewId).appendWindows(windows);
        }
    }

 private:
    void openBlockPicker(const Extent2D<F32>& gridPosition) {
        blockPicker.open(config.onBuildBlockOptions ? config.onBuildBlockOptions()
                                                    : std::vector<FlowgraphBlockPicker::BlockOption>{},
                         gridPosition);
    }

    void handleLinkCreation(const Sakura::NodeEditor::PinRef& startPin,
                            const Sakura::NodeEditor::PinRef& endPin) const {
        if (startPin.isInput == endPin.isInput || !config.onConnectBlock) {
            return;
        }
        if (!liveBlocks.contains(startPin.nodeId) || !liveBlocks.contains(endPin.nodeId)) {
            return;
        }

        const auto& input = startPin.isInput ? startPin : endPin;
        const auto& output = startPin.isInput ? endPin : startPin;
        config.onConnectBlock(input.nodeId, input.pinId, output.nodeId, output.pinId);
    }

    void handleLinkDestruction(const std::string& linkId) const {
        const auto it = linkViews.find(linkId);
        if (it == linkViews.end() || !config.onDisconnectBlock) {
            return;
        }

        const auto& info = it->second.connection();
        config.onDisconnectBlock(info.consumerName, info.inputSlot);
    }

    Config config;
    Sakura::Overlay overlay;
    Sakura::NodeEditor nodeEditor;
    FlowgraphBlockPicker blockPicker;
    Sakura::PillStepper zoomPill;
    Sakura::PillIconButton runButton;
    Sakura::PillIconButton saveButton;
    Sakura::PillIconButton addButton;
    Sakura::PillIconButton environmentButton;
    Sakura::PillIconButton terminalButton;
    Sakura::Menu flowgraphContextMenu;
    bool flowgraphContextMenuOpen = false;
    Extent2D<F32> flowgraphContextMenuGridPos;
    Extent2D<F32> viewportGridCenter;
    Measured measured;
    std::unordered_set<std::string> liveBlocks;
    std::vector<std::string> nodeViewIds;
    std::vector<std::string> linkViewIds;
    std::unordered_map<std::string, FlowgraphLink> linkViews;
    std::unordered_map<std::string, FlowgraphNode> nodeViews;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_EDITOR_BASE_HH
