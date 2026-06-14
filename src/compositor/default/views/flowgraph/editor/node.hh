#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH

#include "config/base.hh"
#include "documentation.hh"
#include "menu.hh"
#include "metrics/base.hh"

#include "jetstream/block.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/base/texture.hh"

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

struct FlowgraphNode : public Sakura::Component {
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
        Extent2D<F32> logicalSize = {512.0f, 512.0f};
        std::optional<Extent2D<F32>> aspectRatioSize;
        bool detached = false;
        std::function<void()> onDetach;
        std::function<void(const Sakura::SurfaceResize&)> onAttachedSize;
        std::function<void(const Sakura::SurfaceResize&)> onDetachedSize;
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
        BlockData block;
        bool pasteEnabled = false;
        bool timingEnabled = false;
        std::function<void()> onCopy;
        std::function<void(Extent2D<F32>)> onPaste;
        std::function<void()> onReload;
        std::function<void()> onDelete;
        std::function<void(DeviceType, RuntimeType, ProviderType)> onDeviceSelect;
        std::function<void(F32, F32, F32, F32)> onLayout;
    };

    struct Geometry {
        Extent2D<F32> gridPosition;
        Extent2D<F32> screenPosition;
        Extent2D<F32> dimensions;
    };

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

        bool allSurfacesDetached = hasSurfaces;
        for (const auto& surface : block.surfaces) {
            if (!surface.detached) {
                allSurfacesDetached = false;
                break;
            }
        }

        if (dimensions.x <= 0.0f) {
            dimensions.x = 140.0f;
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

        if (hasSurfaces) {
            if (dimensions.y <= 0.0f) {
                const Surface* firstAttachedSurface = nullptr;
                for (const auto& surface : block.surfaces) {
                    if (!surface.detached) {
                        firstAttachedSurface = &surface;
                        break;
                    }
                }
                if (firstAttachedSurface) {
                    if (firstAttachedSurface->aspectRatioSize.has_value()) {
                        dimensions.x = firstAttachedSurface->aspectRatioSize->x;
                        dimensions.y = firstAttachedSurface->aspectRatioSize->y;
                    } else {
                        dimensions.y = firstAttachedSurface->logicalSize.y;
                    }
                }
            }
        } else if (!isPending) {
            dimensions.y = 0.0f;
        }

        Extent2D<F32> nodeDimensions = dimensions;
        if (allSurfacesDetached) {
            nodeDimensions.y = 0.0f;
        }

        node.update({
            .id = FlowgraphNodeId(this->config.id),
            .state = nodeState,
            .verticalResize = hasSurfaces && !allSurfacesDetached,
            .dimensions = nodeDimensions,
            .gridPosition = gridPosition,
            .onContextMenu = [this]() {
                menuOpen = true;
            },
            .onGeometryChange = [this, hasSurfaces, allSurfacesDetached](Extent2D<F32> gridPosition,
                                                                         Extent2D<F32> screenPosition,
                                                                         Extent2D<F32> dimensions,
                                                                         Extent2D<F32> contentDimensions) {
                geometry = {
                    .gridPosition = gridPosition,
                    .screenPosition = screenPosition,
                    .dimensions = dimensions,
                };

                this->dimensions.x = contentDimensions.x;
                if (hasSurfaces && !allSurfacesDetached) {
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
                if (Parser::TypedToString(tensor.attribute(key), encoded) != Result::SUCCESS) {
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

        fields.resize(block.configFields.size());
        for (U64 i = 0; i < fields.size(); ++i) {
            fields[i].update(block.configFields[i]);
        }

        attachedSurfaces.resize(surfaceCount);
        for (U64 i = 0; i < surfaceCount; ++i) {
            const auto& surface = block.surfaces[i];
            const auto texture = surface.texture;
            attachedSurfaces[i].update({
                .id = surface.id,
                .size = {0.0f, 0.0f},
                .rounding = surface.rounding,
                .detachOverlay = true,
                .aspectRatioSize = surface.aspectRatioSize,
                .aspectLock = Sakura::SurfaceView::AspectLock::X,
                .onResolveTexture = [texture]() {
                    return texture ? texture->raw() : 0;
                },
                .onSize = surface.onAttachedSize,
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
    }

    void render(const Sakura::Context& ctx) {
        node.render(ctx, [this](const Sakura::Context& ctx) {
            title.render(ctx);
            subtitle.render(ctx);

            for (const auto& pin : pins) {
                pin.render(ctx);
            }

            if (config.block.state == Block::State::Creating) {
                return;
            }

            for (const auto& metric : metrics) {
                metric.render(ctx);
            }
            if (!metrics.empty()) {
                metricsSpacing.render(ctx);
            }

            for (const auto& field : fields) {
                field.render(ctx);
            }

            for (U64 i = 0; i < attachedSurfaces.size(); ++i) {
                if (i < config.block.surfaces.size() && config.block.surfaces[i].detached) {
                    continue;
                }
                attachedSurfaces[i].render(ctx);
            }
        });

        if (menuOpen) {
            menu.render(ctx);
        }
        if (documentationOpen) {
            documentation.render(ctx);
        }
        if (config.timingEnabled) {
            runtimeOverlay.render(ctx);
        }
    }

 private:
    Config config;
    Extent2D<F32> dimensions = {140.0f, 0.0f};
    std::optional<Extent2D<F32>> gridPosition;
    std::optional<Geometry> geometry;
    Sakura::Node node;
    Sakura::NodeTitle title;
    Sakura::NodeSubtitle subtitle;
    Sakura::NodeRuntimeOverlay runtimeOverlay;
    Sakura::Spacing metricsSpacing;
    std::vector<Sakura::NodePin> pins;
    std::vector<FlowgraphMetricInstance> metrics;
    std::vector<FlowgraphConfigFieldInstance> fields;
    std::vector<Sakura::SurfaceView> attachedSurfaces;
    std::vector<FlowgraphNodeMenu::DeviceOption> deviceOptions;
    FlowgraphNodeMenu menu;
    FlowgraphNodeDocumentation documentation;
    bool menuOpen = false;
    bool documentationOpen = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
