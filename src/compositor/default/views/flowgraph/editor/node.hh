#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH

#include "config/base.hh"
#include "config/stack.hh"
#include "inspect.hh"
#include "menu.hh"
#include "metrics/base.hh"

#include "../../../model/meta.hh"
#include "jetstream/block.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

using FlowgraphTooltipRows = std::vector<std::vector<std::string>>;

inline FlowgraphTooltipRows FlowgraphTensorLayoutRows(const Tensor& tensor) {
    return {
        {"Device", jst::fmt::format("{}", tensor.device())},
        {"Type", jst::fmt::format("{}", tensor.dtype())},
        {"Shape", jst::fmt::format("{}", tensor.shape())},
        {"Strides", jst::fmt::format("{} ({})", tensor.stride(), tensor.contiguous() ? "Contiguous" : "Non-contiguous")},
        {"Offset", jst::fmt::format("{} bytes", tensor.offsetBytes())},
    };
}

inline FlowgraphTooltipRows FlowgraphTensorAttributeRows(const Tensor& tensor) {
    FlowgraphTooltipRows rows;
    for (const auto& key : tensor.attributeKeys()) {
        std::string encoded;
        const std::any value = tensor.attribute(key);
        if (!value.has_value() || Parser::TypedToString(value, encoded) != Result::SUCCESS) {
            encoded = "?";
        }
        rows.push_back({key, encoded});
    }
    return rows;
}

struct FlowgraphPinTooltip {
    void update(std::string id, std::string help, const std::optional<Tensor>& tensor) {
        this->id = std::move(id);
        const bool hasTensor = tensor.has_value() && tensor->shape().size() > 0;
        const bool asked = tooltip.requested();
        if (help != this->help || hasTensor != this->hasTensor) {
            ++revision;
        }
        this->help = std::move(help);
        this->hasTensor = hasTensor;
        if (!tooltip.prepared() || (this->help.empty() && !hasTensor)) {
            return;
        }
        auto layout = hasTensor ? FlowgraphTensorLayoutRows(*tensor) : FlowgraphTooltipRows{};
        auto attributes = hasTensor ? FlowgraphTensorAttributeRows(*tensor) : FlowgraphTooltipRows{};
        const bool changed = layout != layoutRows || attributes != attributeRows;
        if (changed) {
            ++revision;
            layoutRows = std::move(layout);
            attributeRows = std::move(attributes);
        }
        if (asked || changed) {
            prepare();
        }
    }

    Sakura::NodeOverlay overlay() {
        if (help.empty() && !hasTensor) {
            return {};
        }
        return {
            .content = [this]() {
                return tooltip.show();
            },
            .revision = revision,
        };
    }

 private:
    void prepare() {
        using Text = Sakura::Text;

        helpText.update({
            .str = help,
            .wrapped = true,
        });
        std::vector<Sakura::Stack::Item> items;
        if (!help.empty()) {
            items.push_back({.child = &helpText});
        }
        if (hasTensor) {
            tensorSection.update({
                .str = "Layout",
                .font = Text::Font::Bold,
            });
            tensorMetadataTable.update({
                .columns = {"", ""},
                .rows = layoutRows,
                .fixedColumnWidths = {72.0f},
                .showHeaders = false,
                .wrapped = true,
            });
            items.push_back({.child = &tensorSection});
            items.push_back({.child = &tensorMetadataTable});
            if (!attributeRows.empty()) {
                tensorAttributesSection.update({
                    .str = "Attributes",
                    .font = Text::Font::Bold,
                });
                tensorAttributeMetadataTable.update({
                    .columns = {"", ""},
                    .rows = attributeRows,
                    .fixedColumnWidths = {120.0f},
                    .showHeaders = false,
                    .wrapped = true,
                });
                items.push_back({.child = &tensorAttributesSection});
                items.push_back({.child = &tensorAttributeMetadataTable});
            }
        }
        content.update({.items = std::move(items)});
        tooltip.update({
            .id = "NodePin" + id,
            .content = &content,
            .wrapWidth = 560.0f,
        });
    }

    std::string id;
    std::string help;
    bool hasTensor = false;
    FlowgraphTooltipRows layoutRows;
    FlowgraphTooltipRows attributeRows;
    U64 revision = 0;
    Tooltip tooltip;
    Sakura::Text helpText;
    Sakura::Text tensorSection;
    Sakura::Table tensorMetadataTable;
    Sakura::Text tensorAttributesSection;
    Sakura::Table tensorAttributeMetadataTable;
    Sakura::Stack content;
};

struct FlowgraphDiagnosticTooltip {
    void update(const std::string& id, const std::string& message, const std::string& colorKey) {
        if (!tooltip.requested()) {
            return;
        }
        headerIcon.update({
            .str = ICON_LC_TRIANGLE_ALERT,
            .colorKey = colorKey,
        });
        headerLabel.update({.str = "Diagnostic"});
        header.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 12.0f,
            .items = {
                {.child = &headerIcon},
                {.child = &headerLabel},
            },
        });
        divider.update({.spacing = 0.0f});
        this->message.update({
            .str = message,
            .wrapped = true,
        });
        content.update({
            .items = {
                {.child = &header},
                {.child = &divider},
                {.child = &this->message},
            },
        });
        tooltip.update({
            .id = id + "DiagnosticTooltip",
            .content = &content,
            .wrapWidth = 420.0f,
        });
    }

    Sakura::NodeOverlay overlay() {
        return {
            .content = [this]() {
                return tooltip.show();
            },
        };
    }

 private:
    Tooltip tooltip;
    Sakura::Text headerIcon;
    Sakura::Text headerLabel;
    Sakura::Stack header;
    Sakura::Divider divider;
    Sakura::Text message;
    Sakura::Stack content;
};

struct FlowgraphNode {
    using NodeEditor = Sakura::NodeEditor;

    struct Port {
        std::string id;
        std::string label;
        std::string help;
    };

    struct Link {
        std::string block;
        std::string port;
        bool resolved = false;
        Tensor tensor;
    };

    struct Input {
        Port port;
        std::optional<Link> source;
    };

    struct Output {
        Port port;
        Tensor tensor;
    };

    struct Surface {
        std::string id;
        std::shared_ptr<const Render::Texture> texture;
        F32 height = 512.0f;
        bool detached = false;
        std::function<void(bool)> onDetach;
        std::function<void(const Sakura::SurfaceResize&)> onAttachedSize;
        std::function<void(InputEvent)> onInput;
        std::function<SurfaceCursor()> resolveCursor;
    };

    struct DeviceOption {
        std::string label;
        bool selected = false;
        DeviceType device = DeviceType::None;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider = "generic";
    };

    struct Layout {
        F32 x = 0.0f;
        F32 y = 0.0f;
        F32 width = 0.0f;
        F32 height = 0.0f;

        bool operator==(const Layout&) const = default;
    };

    struct BlockData {
        std::string name;
        std::string module;
        std::string title;
        DeviceType device = DeviceType::CPU;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider = "generic";
        Block::State state = Block::State::None;
        Block::NodeSize nodeSize = Block::NodeSize::S;
        std::string diagnostic;
        Parser::Map config;
        std::optional<Layout> layout;
        std::vector<Input> inputs;
        std::vector<Output> outputs;
        std::vector<FlowgraphMetricConfig> metrics;
        std::vector<std::string> timing;
        std::vector<FlowgraphConfigFieldConfig> configFields;
        std::vector<Surface> surfaces;
        std::vector<DeviceOption> deviceOptions;
        bool configCollapsed = false;
        bool configDetached = false;
    };

    struct Config {
        std::string id;
        std::string inspectorId;
        BlockData block;
        bool pasteEnabled = false;
        bool timingEnabled = false;
        std::function<void()> onCopy;
        std::function<void(Extent2D<F32>)> onPaste;
        std::function<void()> onRename;
        std::function<void()> onReload;
        std::function<void(Parser::Map)> onInspectApply;
        std::function<void()> onDelete;
        std::function<void()> onDocumentation;
        std::function<void(DeviceType, RuntimeType, ProviderType)> onDeviceSelect;
        std::function<void(F32, F32, F32, F32)> onLayout;
        std::function<void(Rect, Extent2D<F32>, std::optional<Extent2D<F32>>)> onBounds;
        std::function<void(bool)> onConfigCollapse;
        std::function<void(bool)> onConfigDetach;
    };

    static constexpr F32 MinimumNodeWidth = 120.0f;

    static constexpr FlowgraphNodeHeightSpec SurfaceHeightSpec() {
        return {
            .policy = FlowgraphNodeHeightPolicy::FillRemaining,
            .minimum = 150.0f,
            .grow = 1.0f,
        };
    }

    void update(Config config) {
        std::optional<Layout> previousLayout;
        std::function<void(F32, F32, F32, F32)> previousOnLayout;
        if (layoutChanged) {
            previousLayout = this->config.block.layout;
            previousOnLayout = this->config.onLayout;
        }
        this->config = std::move(config);
        if (layoutChanged && geometry.has_value() && this->config.onLayout) {
            layoutChanged = false;
            const auto& current = this->config.block.layout;
            const bool external = current != previousLayout && current != emittedLayout;
            const auto& emit = external && previousOnLayout ? previousOnLayout : this->config.onLayout;
            emittedLayout = Layout{geometry->gridPosition.x, geometry->gridPosition.y, dimensions.x, dimensions.y};
            emit(emittedLayout->x, emittedLayout->y, emittedLayout->width, emittedLayout->height);
        }
        auto& block = this->config.block;
        const U64 surfaceCount = block.surfaces.size();
        const bool hasSurfaces = surfaceCount > 0;
        const bool hasConfigFields = !block.configFields.empty();
        const bool isBusy = Busy(block.state);
        const bool isPending = isBusy ||
                               block.state == Block::State::Incomplete;
        const bool configHidden = block.configCollapsed || block.configDetached;
        const auto nodeState = block.state == Block::State::Errored
            ? NodeEditor::NodeState::Error
            : isBusy ? NodeEditor::NodeState::Loading
                         : isPending ? NodeEditor::NodeState::Pending : NodeEditor::NodeState::Normal;

        if (!isBusy) {
            fieldStack.update(std::move(block.configFields), [this]() {
                reportBounds();
            });
        }
        const bool hasFlexibleFields = fieldStack.hasFlexible();

        bool allSurfacesDetached = hasSurfaces;
        U64 attachedSurfaceCount = 0;
        for (const auto& surface : block.surfaces) {
            if (!surface.detached) {
                allSurfacesDetached = false;
                ++attachedSurfaceCount;
            }
        }

        const bool verticalResize = (!configHidden && hasFlexibleFields) ||
                                    attachedSurfaceCount > 0;
        const auto resizeAxes = verticalResize
            ? NodeEditor::ResizeAxes::XY
            : NodeEditor::ResizeAxes::X;

        if (dimensions.x <= 0.0f) {
            dimensions.x = DefaultNodeWidth(block.nodeSize);
        }

        if (block.layout.has_value()) {
            const auto& layout = block.layout.value();
            gridPosition = Extent2D<F32>{layout.x, layout.y};
            if (layout.width > 0.0f) {
                dimensions.x = layout.width;
            }
            if (layout.height > 0.0f) {
                dimensions.y = layout.height;
            }
        } else {
            gridPosition.reset();
        }
        dimensions.x = std::max(MinimumNodeWidth, dimensions.x);

        Extent2D<F32> nodeDimensions = dimensions;
        if ((!verticalResize && (!isBusy || configHidden)) || (allSurfacesDetached && !hasFlexibleFields)) {
            nodeDimensions.y = 0.0f;
            // Hidden flexible fields still own the expanded height. Render
            // at auto height without overwriting it in the saved layout.
            if (!hasFlexibleFields && !isBusy) {
                layoutChanged = layoutChanged || dimensions.y > 0.0f;
                dimensions.y = 0.0f;
            }
        }

        updateContent(block, isBusy, configHidden);

        for (U64 i = 0; i < surfaceCount; ++i) {
            const auto& surface = block.surfaces[i];
            surfaces[i]->update({
                .id = surface.id,
                .texture = surface.texture,
                .minimumHeight = SurfaceHeightSpec().minimum,
                .onSize = surface.onAttachedSize,
                .onInput = surface.detached ? nullptr : surface.onInput,
                .resolveCursor = surface.detached ? nullptr : surface.resolveCursor,
            });
        }

        const U64 pinCount = block.inputs.size() + block.outputs.size();
        if (!isBusy || pinTooltips.size() < pinCount) {
            pinTooltips.resize(pinCount);
        }
        for (auto& tooltip : pinTooltips) {
            if (!tooltip) {
                tooltip = std::make_unique<FlowgraphPinTooltip>();
            }
        }
        std::vector<NodeEditor::Pin> pins;
        pins.reserve(pinTooltips.size());
        U64 pinIndex = 0;
        for (const auto& input : block.inputs) {
            auto& tooltip = *pinTooltips[pinIndex++];
            tooltip.update(input.port.id, input.port.help, std::nullopt);
            pins.push_back({
                .id = input.port.id,
                .isInput = true,
                .label = input.port.label,
                .enableDetach = true,
                .tooltip = tooltip.overlay(),
            });
        }
        for (const auto& output : block.outputs) {
            auto& tooltip = *pinTooltips[pinIndex++];
            tooltip.update(output.port.id, output.port.help, output.tensor);
            pins.push_back({
                .id = output.port.id,
                .isInput = false,
                .label = output.port.label,
                .tooltip = tooltip.overlay(),
            });
        }

        const bool hasDiagnostic = !block.diagnostic.empty() && nodeState != NodeEditor::NodeState::Normal;
        diagnosticTooltip.update(this->config.id,
                                 block.diagnostic,
                                 nodeState == NodeEditor::NodeState::Error ? "node_outline_error"
                                                                           : "node_outline_pending");

        nodeSpec = {
            .id = this->config.id,
            .header = block.module != "note",
            .title = block.title,
            .subtitle = block.name,
            .state = nodeState,
            .diagnostic = hasDiagnostic ? block.diagnostic : std::string(),
            .diagnosticTooltip = hasDiagnostic ? diagnosticTooltip.overlay() : Sakura::NodeOverlay{},
            .collapsible = hasConfigFields && !block.configDetached,
            .collapsed = block.configCollapsed,
            .onToggleCollapse = [this]() {
                if (this->config.onConfigCollapse) {
                    this->config.onConfigCollapse(!this->config.block.configCollapsed);
                }
            },
            .onDetach = attachedSurfaceCount > 0 ? Sakura::Callback<void()>([this]() {
                setSurfacesDetached(true);
            }) : Sakura::Callback<void()>(),
            .pins = std::move(pins),
            .content = &content,
            .resize = resizeAxes,
            .dimensions = nodeDimensions,
            .gridPosition = gridPosition,
            .timing = this->config.timingEnabled ? block.timing : std::vector<std::string>{},
            .onContextMenu = [this]() {
                menuOpen = true;
            },
            .onGeometryChange = [this, verticalResize, isBusy](const NodeEditor::Geometry& next) {
                geometry = next;
                this->dimensions.x = next.contentDimensions.x;
                if (verticalResize && !isBusy) {
                    this->dimensions.y = next.contentDimensions.y;
                }
                layoutChanged = true;
                reportBounds();
            },
        };

        deviceOptions.clear();
        deviceOptions.reserve(block.deviceOptions.size());
        for (const auto& device : block.deviceOptions) {
            deviceOptions.push_back({
                .label = device.label,
                .selected = device.selected,
            });
        }

        menu.update({
            .id = this->config.id + ":context-menu",
            .open = menuOpen,
            .pasteEnabled = this->config.pasteEnabled,
            .configDetachEnabled = hasConfigFields,
            .configDetached = block.configDetached,
            .plotDetachEnabled = hasSurfaces,
            .plotDetached = allSurfacesDetached,
            .devices = deviceOptions,
            .onCopy = this->config.onCopy,
            .onPaste = [this]() {
                if (this->config.onPaste) {
                    Extent2D<F32> pastePosition = {50.0f, 50.0f};
                    if (geometry.has_value()) {
                        pastePosition = {
                            geometry->gridPosition.x + pastePosition.x,
                            geometry->gridPosition.y + pastePosition.y,
                        };
                    }
                    this->config.onPaste(pastePosition);
                }
            },
            .onRename = this->config.onRename,
            .onInspect = [this]() {
                inspector.open();
                inspectorOpen = true;
            },
            .onToggleConfigDetach = [this]() {
                if (this->config.onConfigDetach) {
                    this->config.onConfigDetach(!this->config.block.configDetached);
                }
            },
            .onTogglePlotDetach = [this, allSurfacesDetached]() {
                setSurfacesDetached(!allSurfacesDetached);
            },
            .onReload = this->config.onReload,
            .onDelete = this->config.onDelete,
            .onDocumentation = this->config.onDocumentation,
            .onDeviceSelect = [this](const U64 index) {
                if (index >= this->config.block.deviceOptions.size() || !this->config.onDeviceSelect) {
                    return;
                }

                const auto& option = this->config.block.deviceOptions.at(index);
                if (option.selected) {
                    return;
                }

                this->config.onDeviceSelect(option.device, option.runtime, option.provider);
            },
            .onClose = [this]() {
                menuOpen = false;
            },
        });
        inspector.update({
            .id = this->config.inspectorId,
            .name = block.name,
            .value = block.config,
            .onApply = this->config.onInspectApply,
            .onClose = [this]() {
                inspectorOpen = false;
            },
        });
    }

    void setSurfacesDetached(bool detached) {
        for (const auto& surface : config.block.surfaces) {
            if (surface.detached != detached && surface.onDetach) {
                surface.onDetach(detached);
            }
        }
    }

    NodeEditor::Node takeNode() {
        return std::move(nodeSpec);
    }

    Sakura::Component& menuComponent() {
        return menu.component();
    }

    void appendWindows(std::vector<Sakura::FloatingLayer::Window>& windows) {
        if (inspectorOpen) {
            windows.push_back(inspector.window());
        }
    }

 private:
    static bool Busy(const Block::State state) {
        return state == Block::State::Creating || state == Block::State::Destroying;
    }

    void reportBounds() const {
        const auto& block = config.block;
        if (!geometry.has_value() || Busy(block.state) || !config.onBounds) {
            return;
        }
        const auto& next = *geometry;
        const Extent2D<F32> inner = next.contentDimensions;
        std::optional<Extent2D<F32>> content;
        if (next.onscreen && !block.configCollapsed && !block.configDetached) {
            content = fieldStack.content(inner, next.minimumHeight);
        }
        config.onBounds({next.gridPosition.x, next.gridPosition.y, next.dimensions.x, next.dimensions.y},
                        inner,
                        content);
    }

    void updateContent(const BlockData& block, bool isBusy, bool configHidden) {
        using Stack = Sakura::NodeContentStack;

        std::vector<Stack::Item> items;

        if (!isBusy) {
            while (metrics.size() > block.metrics.size()) {
                metrics.pop_back();
            }
            while (metrics.size() < block.metrics.size()) {
                metrics.push_back(std::make_unique<FlowgraphMetricInstance>());
            }
            for (U64 i = 0; i < metrics.size(); ++i) {
                metrics[i]->update(block.metrics[i]);
            }
        }

        while (surfaces.size() > block.surfaces.size()) {
            surfaces.pop_back();
        }
        while (surfaces.size() < block.surfaces.size()) {
            surfaces.push_back(std::make_unique<Sakura::NodeSurface>());
        }

        if (!isBusy) {
            for (const auto& metric : metrics) {
                items.push_back({.child = &metric->component()});
            }
            if (!metrics.empty()) {
                items.back().trailingGap = Stack::Spacing;
            }

            if (!configHidden) {
                const U64 firstField = items.size();
                fieldStack.appendItems(items);
                const bool surfaceAttached = std::any_of(block.surfaces.begin(), block.surfaces.end(), [](const auto& surface) {
                    return !surface.detached;
                });
                for (U64 index = firstField; surfaceAttached && index < items.size(); ++index) {
                    if (items[index].flex.has_value()) {
                        items[index].flex->grow = 0.0f;
                    }
                }
            }

            const auto surfaceSpec = SurfaceHeightSpec();
            for (U64 i = 0; i < block.surfaces.size(); ++i) {
                if (block.surfaces[i].detached) {
                    continue;
                }
                items.push_back({
                    .child = surfaces[i].get(),
                    .flex = Stack::Flex{
                        .minimum = surfaceSpec.minimum,
                        .grow = surfaceSpec.grow,
                        .basis = std::max(surfaceSpec.minimum, block.surfaces[i].height),
                    },
                });
            }
        }

        content.update({
            .id = config.id + ":content",
            .items = std::move(items),
        });
    }

    Config config;
    Extent2D<F32> dimensions = {0.0f, 0.0f};
    std::optional<Extent2D<F32>> gridPosition;
    std::optional<NodeEditor::Geometry> geometry;
    NodeEditor::Node nodeSpec;
    Sakura::NodeContentStack content;
    FlowgraphConfigFieldStack fieldStack;
    std::vector<std::unique_ptr<FlowgraphMetricInstance>> metrics;
    std::vector<std::unique_ptr<Sakura::NodeSurface>> surfaces;
    std::vector<std::unique_ptr<FlowgraphPinTooltip>> pinTooltips;
    FlowgraphDiagnosticTooltip diagnosticTooltip;
    std::vector<FlowgraphNodeMenu::DeviceOption> deviceOptions;
    FlowgraphNodeMenu menu;
    FlowgraphNodeInspector inspector;
    bool menuOpen = false;
    bool inspectorOpen = false;
    bool layoutChanged = false;
    std::optional<Layout> emittedLayout;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
