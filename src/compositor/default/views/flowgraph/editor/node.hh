#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH

#include "config/base.hh"
#include "documentation.hh"
#include "inspect.hh"
#include "menu.hh"
#include "metrics/base.hh"

#include "jetstream/block.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/base/texture.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

inline std::string FlowgraphNodeId(const std::string& nodeName) {
    return nodeName + "node";
}

inline std::string FlowgraphPinId(const std::string& pinName) {
    return pinName + "pin";
}

struct FlowgraphNode {
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
        F32 rounding = 0.0f;
        F32 height = 512.0f;
        bool detached = false;
        std::function<void()> onDetach;
        std::function<void(const Sakura::SurfaceResize&)> onAttachedSize;
        std::function<void(MouseEvent)> onMouse;
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
    };

    struct BlockData {
        std::string name;
        std::string module;
        std::string title;
        std::string documentation;
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
        std::function<void(DeviceType, RuntimeType, ProviderType)> onDeviceSelect;
        std::function<void(F32, F32, F32, F32)> onLayout;
    };

    struct Geometry {
        Extent2D<F32> gridPosition;
        Extent2D<F32> screenPosition;
        Extent2D<F32> dimensions;
    };

    static constexpr F32 MinimumNodeWidth = 120.0f;
    static constexpr F32 MinimumNodeHeight = 100.0f;

    static F32 DefaultNodeWidth(const Block::NodeSize& size) {
        switch (size) {
            case Block::NodeSize::XS:
                return 120.0f;
            case Block::NodeSize::M:
                return 220.0f;
            case Block::NodeSize::L:
                return 320.0f;
            case Block::NodeSize::XL:
                return 460.0f;
            default:
                return 140.0f;
        }
    }

    static constexpr FlowgraphNodeHeightSpec SurfaceHeightSpec() {
        return {
            .policy = FlowgraphNodeHeightPolicy::FillRemaining,
            .minimum = 150.0f,
            .grow = 1.0f,
        };
    }

    void update(Config config) {
        this->config = std::move(config);
        const auto& block = this->config.block;
        const U64 surfaceCount = block.surfaces.size();
        const bool hasSurfaces = surfaceCount > 0;
        const bool isCreating = block.state == Block::State::Creating;
        const bool isPending = isCreating ||
                               block.state == Block::State::Incomplete;
        const auto nodeState = block.state == Block::State::Errored
            ? Sakura::Node::State::Error
            : isCreating ? Sakura::Node::State::Loading
                         : isPending ? Sakura::Node::State::Pending : Sakura::Node::State::Normal;

        fields.resize(block.configFields.size());
        for (U64 i = 0; i < fields.size(); ++i) {
            fields[i].update(block.configFields[i]);
        }

        U64 flexibleFieldCount = 0;
        F32 flexibleMinimumSum = 0.0f;
        for (const auto& field : fields) {
            const auto spec = field.heightSpec();
            if (spec.policy == FlowgraphNodeHeightPolicy::FillRemaining) {
                ++flexibleFieldCount;
                flexibleMinimumSum += std::max(0.0f, spec.minimum);
            }
        }
        const bool hasFlexibleFields = flexibleFieldCount > 0;

        bool allSurfacesDetached = hasSurfaces;
        U64 attachedSurfaceCount = 0;
        F32 restoredFlexibleHeight = flexibleMinimumSum;
        for (const auto& surface : block.surfaces) {
            if (!surface.detached) {
                allSurfacesDetached = false;
                ++attachedSurfaceCount;
                restoredFlexibleHeight += std::max(SurfaceHeightSpec().minimum,
                                                   surface.height);
            }
        }

        const bool verticalResize = hasFlexibleFields || attachedSurfaceCount > 0;
        const auto resizeAxes = verticalResize
            ? Sakura::Node::ResizeAxes::XY
            : Sakura::Node::ResizeAxes::X;

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
        if (!isCreating && verticalResize && dimensions.y <= 0.0f) {
            const auto* previousLayout = contentLayout.layout();
            const bool chromeMeasured = previousLayout && previousLayout->measured;
            const F32 fixedHeight = chromeMeasured ? previousLayout->fixedHeight : 0.0f;
            dimensions.y = std::max(MinimumNodeHeight, restoredFlexibleHeight) +
                           fixedHeight;
            if (attachedSurfaceCount > 0 && !chromeMeasured) {
                pendingFlexibleRestoreHeight = restoredFlexibleHeight;
            }
        } else if (!isCreating && !verticalResize) {
            dimensions.y = 0.0f;
        }
        dimensions.x = std::max(MinimumNodeWidth, dimensions.x);

        Extent2D<F32> nodeDimensions = dimensions;
        if (allSurfacesDetached && !hasFlexibleFields) {
            nodeDimensions.y = 0.0f;
            dimensions.y = 0.0f;
        }

        contentChildren.clear();
        std::vector<std::optional<U64>> fieldLayoutItems(fields.size());
        std::vector<std::optional<U64>> surfaceLayoutItems(surfaceCount);
        Sakura::VStack::Config contentLayoutConfig{
            .id = this->config.id + ":content",
            .height = nodeDimensions.y > 0.0f
                ? std::optional<F32>{nodeDimensions.y}
                : std::nullopt,
        };
        auto addContent = [this, &contentLayoutConfig](std::string id,
                                                       std::optional<Sakura::VStack::Flex> flex,
                                                       Sakura::VStack::Child child) {
            const U64 index = contentLayoutConfig.items.size();
            contentLayoutConfig.items.push_back({
                .id = std::move(id),
                .flex = flex,
            });
            contentChildren.push_back(std::move(child));
            return index;
        };

        if (block.module != "note") {
            addContent("title", std::nullopt, [this](const Sakura::Context& ctx) {
                title.render(ctx);
            });
            addContent("subtitle", std::nullopt, [this](const Sakura::Context& ctx) {
                subtitle.render(ctx);
            });
        }
        U64 layoutPinIndex = 0;
        for (const auto& input : block.inputs) {
            const U64 index = layoutPinIndex++;
            addContent("input:" + input.port.id,
                       std::nullopt,
                       [this, index](const Sakura::Context& ctx) {
                           pins[index].render(ctx);
                       });
        }
        for (const auto& output : block.outputs) {
            const U64 index = layoutPinIndex++;
            addContent("output:" + output.port.id,
                       std::nullopt,
                       [this, index](const Sakura::Context& ctx) {
                           pins[index].render(ctx);
                       });
        }

        if (!isCreating) {
            for (U64 i = 0; i < block.metrics.size(); ++i) {
                addContent("metric:" + block.metrics[i].id,
                           std::nullopt,
                           [this, i](const Sakura::Context& ctx) {
                               metrics[i].render(ctx);
                           });
            }
            if (!block.metrics.empty()) {
                addContent("metrics-spacing", std::nullopt, [this](const Sakura::Context& ctx) {
                    metricsSpacing.render(ctx);
                });
            }

            for (U64 i = 0; i < fields.size(); ++i) {
                const auto spec = fields[i].heightSpec();
                std::optional<Sakura::VStack::Flex> flex;
                if (spec.policy == FlowgraphNodeHeightPolicy::FillRemaining) {
                    flex = Sakura::VStack::Flex{
                        .minimum = spec.minimum,
                        .grow = spec.grow,
                    };
                }
                fieldLayoutItems[i] = addContent(
                    "field:" + block.configFields[i].id + ":" + block.configFields[i].format,
                    flex,
                    [this, i](const Sakura::Context& ctx) {
                        fields[i].render(ctx);
                    });
            }

            const auto surfaceSpec = SurfaceHeightSpec();
            for (U64 i = 0; i < surfaceCount; ++i) {
                if (block.surfaces[i].detached) {
                    continue;
                }
                surfaceLayoutItems[i] = addContent(
                    "surface:" + block.surfaces[i].id,
                    Sakura::VStack::Flex{
                        .minimum = surfaceSpec.minimum,
                        .grow = surfaceSpec.grow,
                        .basis = std::max(surfaceSpec.minimum,
                                          block.surfaces[i].height),
                    },
                    [this, i](const Sakura::Context& ctx) {
                        attachedSurfaces[i].render(ctx);
                    });
            }
        }

        contentLayout.update(std::move(contentLayoutConfig));
        const auto* contentLayoutState = contentLayout.layout();
        const F32 minimumNodeHeight = contentLayoutState &&
                                      contentLayoutState->minimumHeight.has_value()
            ? std::max(MinimumNodeHeight, *contentLayoutState->minimumHeight)
            : MinimumNodeHeight;
        if (verticalResize && !isCreating && dimensions.y < minimumNodeHeight) {
            dimensions.y = minimumNodeHeight;
            nodeDimensions.y = dimensions.y;
        }

        if (attachedSurfaceCount == 0) {
            pendingFlexibleRestoreHeight.reset();
        }
        if (pendingFlexibleRestoreHeight.has_value()) {
            if (contentLayoutState && contentLayoutState->measured) {
                const F32 target =
                    std::max(MinimumNodeHeight, *pendingFlexibleRestoreHeight) +
                    contentLayoutState->fixedHeight;
                if (verticalResize && !isCreating && dimensions.y < target - 0.5f) {
                    dimensions.y = target;
                    nodeDimensions.y = dimensions.y;
                } else {
                    pendingFlexibleRestoreHeight.reset();
                }
            }
        }

        std::vector<std::optional<F32>> fieldAllocatedHeights(fields.size());
        std::vector<std::optional<F32>> surfaceAllocatedHeights(surfaceCount);
        if (contentLayoutState) {
            for (U64 i = 0; i < fields.size(); ++i) {
                if (fieldLayoutItems[i].has_value()) {
                    const auto allocated = contentLayoutState->itemHeight(*fieldLayoutItems[i]);
                    if (!allocated.has_value()) {
                        fieldAllocatedHeights[i] = allocated;
                        continue;
                    }
                    const auto spec = fields[i].heightSpec();
                    fieldAllocatedHeights[i] =
                        spec.policy == FlowgraphNodeHeightPolicy::FillRemaining
                            ? std::max(*allocated, spec.minimum)
                            : *allocated;
                }
            }
            for (U64 i = 0; i < surfaceCount; ++i) {
                if (surfaceLayoutItems[i].has_value()) {
                    const auto allocated = contentLayoutState->itemHeight(*surfaceLayoutItems[i]);
                    if (allocated.has_value()) {
                        surfaceAllocatedHeights[i] =
                            std::max(*allocated, SurfaceHeightSpec().minimum);
                    } else {
                        surfaceAllocatedHeights[i] = allocated;
                    }
                }
            }
        }
        for (U64 i = 0; i < fields.size(); ++i) {
            fields[i].setAllocatedHeight(fieldAllocatedHeights[i]);
        }

        node.update({
            .id = FlowgraphNodeId(this->config.id),
            .state = nodeState,
            .resize = resizeAxes,
            .dimensions = nodeDimensions,
            .minimumDimensions = {MinimumNodeWidth, minimumNodeHeight},
            .gridPosition = gridPosition,
            .onContextMenu = [this]() {
                menuOpen = true;
            },
            .onGeometryChange = [this, verticalResize, isCreating](Extent2D<F32> gridPosition,
                                                                   Extent2D<F32> screenPosition,
                                                                   Extent2D<F32> dimensions,
                                                                   Extent2D<F32> contentDimensions) {
                geometry = {
                    .gridPosition = gridPosition,
                    .screenPosition = screenPosition,
                    .dimensions = dimensions,
                };

                this->dimensions.x = contentDimensions.x;
                if (verticalResize && !isCreating) {
                    this->dimensions.y = contentDimensions.y;
                }
                if (this->config.onLayout) {
                    this->config.onLayout(gridPosition.x,
                                          gridPosition.y,
                                          this->dimensions.x,
                                          this->dimensions.y);
                }
            },
        });
        title.update({
            .title = block.title,
            .diagnostic = {
                .state = block.diagnostic.empty() ? Sakura::Node::State::Normal : nodeState,
                .message = block.diagnostic,
            },
        });
        subtitle.update({.text = block.name});
        metricsSpacing.update({.id = this->config.id + "MetricsSpacing"});
        runtimeOverlay.update({
            .lines = block.timing,
            .onResolveGeometry = [this]() -> std::optional<Sakura::NodeRuntimeOverlay::Geometry> {
                if (!geometry.has_value()) {
                    return std::nullopt;
                }

                return Sakura::NodeRuntimeOverlay::Geometry{
                    .nodePos = geometry->screenPosition,
                    .nodeSize = geometry->dimensions,
                };
            },
        });

        pins.resize(block.inputs.size() + block.outputs.size());
        U64 pinIndex = 0;
        for (const auto& input : block.inputs) {
            pins[pinIndex++].update({
                .id = FlowgraphPinId(input.port.id),
                .direction = Sakura::NodePin::Direction::Input,
                .label = input.port.label,
                .help = input.port.help,
                .enableDetach = true,
                .dataShape = Shape(),
                .dataStride = Shape(),
                .dataType = DataType::None,
                .dataDevice = block.device,
                .dataOffsetBytes = 0,
                .dataContiguous = false,
            });
        }
        for (const auto& output : block.outputs) {
            const auto& tensor = output.tensor;
            std::vector<std::vector<std::string>> attributeRows;
            for (const auto& key : tensor.attributeKeys()) {
                std::string encoded;
                const std::any value = tensor.attribute(key);
                if (!value.has_value() ||
                    Parser::TypedToString(value, encoded) != Result::SUCCESS) {
                    encoded = "?";
                }
                attributeRows.push_back({key, encoded});
            }
            pins[pinIndex++].update({
                .id = FlowgraphPinId(output.port.id),
                .direction = Sakura::NodePin::Direction::Output,
                .label = output.port.label,
                .help = output.port.help,
                .dataShape = tensor.shape(),
                .dataStride = tensor.stride(),
                .dataType = tensor.dtype(),
                .dataDevice = tensor.device(),
                .dataOffsetBytes = tensor.offsetBytes(),
                .dataContiguous = tensor.contiguous(),
                .dataAttributes = std::move(attributeRows),
            });
        }

        metrics.resize(block.metrics.size());
        for (U64 i = 0; i < metrics.size(); ++i) {
            metrics[i].update(block.metrics[i]);
        }

        attachedSurfaces.resize(surfaceCount);
        for (U64 i = 0; i < surfaceCount; ++i) {
            const auto& surface = block.surfaces[i];
            const auto texture = surface.texture;
            attachedSurfaces[i].update({
                .id = surface.id,
                .size = {0.0f, 0.0f},
                .height = surfaceAllocatedHeights[i],
                .rounding = surface.rounding,
                .detachOverlay = true,
                .onResolveTexture = [texture]() {
                    return texture ? texture->raw() : 0;
                },
                .onSize = contentLayoutState && contentLayoutState->measured &&
                                  !pendingFlexibleRestoreHeight.has_value()
                    ? surface.onAttachedSize
                    : std::function<void(const Sakura::SurfaceResize&)>{},
                .onDetach = surface.onDetach,
            });
        }

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
            .pasteEnabled = this->config.pasteEnabled,
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
            .onReload = this->config.onReload,
            .onDelete = this->config.onDelete,
            .onDocumentation = [this]() {
                documentationOpen = true;
            },
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
        documentation.update({
            .id = this->config.id + ":documentation",
            .title = block.title,
            .name = block.name,
            .value = block.documentation,
            .onClose = [this]() {
                documentationOpen = false;
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

    void render(const Sakura::Context& ctx) {
        node.render(ctx, [this](const Sakura::Context& ctx) {
            contentLayout.render(ctx, contentChildren);
        });

        if (menuOpen) {
            menu.render(ctx);
        }
        if (documentationOpen) {
            documentation.render(ctx);
        }
        if (inspectorOpen) {
            inspector.render(ctx);
        }
        if (config.timingEnabled) {
            runtimeOverlay.render(ctx);
        }
    }

 private:
    Config config;
    Extent2D<F32> dimensions = {0.0f, 0.0f};
    std::optional<Extent2D<F32>> gridPosition;
    std::optional<Geometry> geometry;
    Sakura::Node node;
    Sakura::NodeTitle title;
    Sakura::NodeSubtitle subtitle;
    Sakura::NodeRuntimeOverlay runtimeOverlay;
    Sakura::Spacing metricsSpacing;
    Sakura::VStack contentLayout;
    Sakura::VStack::Children contentChildren;
    std::vector<Sakura::NodePin> pins;
    std::vector<FlowgraphMetricInstance> metrics;
    std::vector<FlowgraphConfigFieldInstance> fields;
    std::vector<Sakura::SurfaceView> attachedSurfaces;
    std::vector<FlowgraphNodeMenu::DeviceOption> deviceOptions;
    FlowgraphNodeMenu menu;
    FlowgraphNodeDocumentation documentation;
    FlowgraphNodeInspector inspector;
    bool menuOpen = false;
    bool documentationOpen = false;
    bool inspectorOpen = false;
    std::optional<F32> pendingFlexibleRestoreHeight;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
