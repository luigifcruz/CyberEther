#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH

#include "config/base.hh"
#include "documentation.hh"
#include "menu.hh"
#include "metrics/base.hh"
#include "surface.hh"

#include "jetstream/block.hh"
#include "jetstream/render/base/texture.hh"

#include <algorithm>
#include <cmath>
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
        std::function<void(const Sakura::SurfaceSize&)> onAttachedSize;
        std::function<void(const Sakura::SurfaceSize&)> onDetachedSize;
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
        Jetstream::Block::State state = Jetstream::Block::State::None;
        std::string diagnostic;
        Parser::Map config;
        std::optional<Layout> layout;
        std::vector<Input> inputs;
        std::vector<Output> outputs;
        std::vector<FlowgraphMetricConfig> metrics;
        std::vector<std::string> runtimeMetrics;
        std::vector<FlowgraphConfigFieldConfig> configFields;
        std::vector<Surface> surfaces;
        std::vector<DeviceOption> deviceOptions;
    };

    struct Config {
        std::string id;
        BlockData block;
        bool pasteEnabled = false;
        bool runtimeMetricsEnabled = false;
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
        const auto nodeState = this->config.block.state == Jetstream::Block::State::Errored
            ? Sakura::Node::State::Error
            : (this->config.block.state == Jetstream::Block::State::Creating ||
               this->config.block.state == Jetstream::Block::State::Incomplete)
                ? Sakura::Node::State::Pending
                : Sakura::Node::State::Normal;
        if (dimensions.x <= 0.0f) {
            dimensions.x = 140.0f;
        }

        if (this->config.block.layout.has_value()) {
            const auto& layout = this->config.block.layout.value();
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

        const bool hasSurfaces = !this->config.block.surfaces.empty();
        const bool isReloading = nodeState == Sakura::Node::State::Pending;
        if (!hasSurfaces && !isReloading) {
            dimensions.y = 0.0f;
        } else if (hasSurfaces && dimensions.y <= 0.0f) {
            dimensions.y = this->config.block.surfaces.front().logicalSize.y;
        }

        const Extent2D<F32> requestedDimensions = dimensions;
        node.update({
            .id = FlowgraphNodeId(this->config.id),
            .state = nodeState,
            .verticalResize = !this->config.block.surfaces.empty(),
            .dimensions = dimensions,
            .gridPosition = gridPosition,
            .onContextMenu = [this]() {
                menuOpen = true;
            },
            .onGeometryChange = [this, requestedDimensions](Extent2D<F32> gridPosition,
                                                            Extent2D<F32> screenPosition,
                                                            Extent2D<F32> dimensions,
                                                            Extent2D<F32> contentDimensions) {
                geometry = {
                    .gridPosition = gridPosition,
                    .screenPosition = screenPosition,
                    .dimensions = dimensions,
                };

                if (std::abs(contentDimensions.x - requestedDimensions.x) > 0.5f ||
                    std::abs(contentDimensions.y - requestedDimensions.y) > 0.5f) {
                    this->dimensions = contentDimensions;
                }

                emitLayout({gridPosition.x,
                            gridPosition.y,
                            this->dimensions.x,
                            this->dimensions.y});
            },
        });
        title.update({
            .title = this->config.block.title,
            .diagnostic = {
                .state = this->config.block.diagnostic.empty() ? Sakura::Node::State::Normal : nodeState,
                .message = this->config.block.diagnostic,
            },
        });
        subtitle.update({.text = this->config.block.name});
        loadingBar.update({.id = this->config.id + "Loading"});
        metricsSpacing.update({.id = this->config.id + "MetricsSpacing"});
        runtimeOverlay.update({
            .lines = this->config.block.runtimeMetrics,
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

        pins.resize(this->config.block.inputs.size() + this->config.block.outputs.size());
        U64 pinIndex = 0;
        for (const auto& input : this->config.block.inputs) {
            pins[pinIndex++].update({
                .id = FlowgraphPinId(input.port.id),
                .direction = Sakura::NodePin::Direction::Input,
                .label = input.port.label,
                .help = input.port.help,
                .enableDetach = true,
            });
        }
        for (const auto& output : this->config.block.outputs) {
            pins[pinIndex++].update({
                .id = FlowgraphPinId(output.port.id),
                .direction = Sakura::NodePin::Direction::Output,
                .label = output.port.label,
                .help = output.port.help,
            });
        }

        metrics.resize(this->config.block.metrics.size());
        for (U64 i = 0; i < metrics.size(); ++i) {
            metrics[i].update(this->config.block.metrics[i]);
        }

        fields.resize(this->config.block.configFields.size());
        for (U64 i = 0; i < fields.size(); ++i) {
            fields[i].update(this->config.block.configFields[i]);
        }

        attachedSurfaces.resize(this->config.block.surfaces.size());
        detachedSurfaces.resize(this->config.block.surfaces.size());
        detachedSurfaceStates.resize(this->config.block.surfaces.size(), false);
        lastSurfaceSizes.resize(this->config.block.surfaces.size());
        for (U64 i = 0; i < attachedSurfaces.size(); ++i) {
            const auto texture = this->config.block.surfaces[i].texture;
            const void* surfaceIdentity = texture.get();
            auto onAttachedSize = this->config.block.surfaces[i].onAttachedSize;
            auto onDetachedSize = this->config.block.surfaces[i].onDetachedSize;
            attachedSurfaces[i].update({
                .id = this->config.block.surfaces[i].id,
                .onResolveTexture = [texture]() {
                    return texture ? texture->raw() : 0;
                },
                .size = {0.0f, 0.0f},
                .rounding = this->config.block.surfaces[i].rounding,
                .detachOverlay = true,
                .aspectRatioSize = this->config.block.surfaces[i].aspectRatioSize,
                .onSize = [this, i, onAttachedSize, surfaceIdentity](const Sakura::SurfaceSize& size) mutable {
                    if (size.resolvedLogicalSize.y > 0.0f && size.resolvedLogicalSize.y < size.availableLogicalSize.y) {
                        dimensions.y = std::max(0.0f,
                                                dimensions.y - size.availableLogicalSize.y + size.resolvedLogicalSize.y);
                    }

                    if (onAttachedSize && consumeSurfaceSize(i, size, false, surfaceIdentity)) {
                        onAttachedSize(size);
                    }
                },
                .onDetach = [this, i]() {
                    if (i < detachedSurfaceStates.size()) {
                        detachedSurfaceStates[i] = true;
                    }
                },
            });
            detachedSurfaces[i].update({
                .id = this->config.block.surfaces[i].id + ":detached",
                .title = this->config.block.title,
                .name = this->config.block.name,
                .onResolveTexture = [texture]() {
                    return texture ? texture->raw() : 0;
                },
                .logicalSize = this->config.block.surfaces[i].logicalSize,
                .onSize = [this, i, onDetachedSize, surfaceIdentity](const Sakura::SurfaceSize& size) mutable {
                    if (onDetachedSize && consumeSurfaceSize(i, size, true, surfaceIdentity)) {
                        onDetachedSize(size);
                    }
                },
                .onMouse = this->config.block.surfaces[i].onMouse,
                .onClose = [this, i]() {
                    if (i < detachedSurfaceStates.size()) {
                        detachedSurfaceStates[i] = false;
                    }
                },
            });
        }

        deviceOptions.clear();
        deviceOptions.reserve(this->config.block.deviceOptions.size());
        for (const auto& device : this->config.block.deviceOptions) {
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
                    this->config.onPaste(pasteGridPosition({50.0f, 50.0f}));
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
            .title = this->config.block.title,
            .name = this->config.block.name,
            .value = this->config.block.documentation,
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

            if (config.block.state == Jetstream::Block::State::Creating) {
                loadingBar.render(ctx);
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
                if (i < detachedSurfaceStates.size() && detachedSurfaceStates[i]) {
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
        for (U64 i = 0; i < detachedSurfaces.size(); ++i) {
            if (i < detachedSurfaceStates.size() && detachedSurfaceStates[i]) {
                detachedSurfaces[i].render(ctx);
            }
        }
        if (config.runtimeMetricsEnabled) {
            runtimeOverlay.render(ctx);
        }
    }

 private:
    struct SurfaceSizeRecord {
        Sakura::SurfaceResize resize;
        bool detached = false;
        const void* source = nullptr;
    };

    struct LayoutRecord {
        F32 x = 0.0f;
        F32 y = 0.0f;
        F32 width = 0.0f;
        F32 height = 0.0f;
    };

    void emitLayout(LayoutRecord layout) {
        if (lastLayout.has_value()) {
            const auto& last = *lastLayout;
            if (std::abs(last.x - layout.x) <= 0.5f &&
                std::abs(last.y - layout.y) <= 0.5f &&
                std::abs(last.width - layout.width) <= 0.5f &&
                std::abs(last.height - layout.height) <= 0.5f) {
                return;
            }
        }

        lastLayout = layout;
        if (config.onLayout) {
            config.onLayout(layout.x, layout.y, layout.width, layout.height);
        }
    }

    bool consumeSurfaceSize(const U64 index,
                            const Sakura::SurfaceSize& size,
                            const bool detached,
                            const void* source) {
        if (index >= lastSurfaceSizes.size()) {
            return false;
        }

        const Sakura::SurfaceResize resize{
            .logicalSize = size.logicalSize,
            .framebufferSize = size.framebufferSize,
            .scale = size.scale,
        };

        if (lastSurfaceSizes[index].has_value()) {
            const auto& last = *lastSurfaceSizes[index];
            if (last.source == source && last.detached == detached &&
                last.resize.logicalSize.x == resize.logicalSize.x &&
                last.resize.logicalSize.y == resize.logicalSize.y &&
                last.resize.framebufferSize.x == resize.framebufferSize.x &&
                last.resize.framebufferSize.y == resize.framebufferSize.y &&
                std::abs(last.resize.scale - resize.scale) <= 1e-6f) {
                return false;
            }
        }

        lastSurfaceSizes[index] = SurfaceSizeRecord{
            .resize = resize,
            .detached = detached,
            .source = source,
        };
        return true;
    }

    Extent2D<F32> pasteGridPosition(const Extent2D<F32>& offset) const {
        if (!geometry.has_value()) {
            return offset;
        }

        return {
            geometry->gridPosition.x + offset.x,
            geometry->gridPosition.y + offset.y,
        };
    }

    Config config;
    Extent2D<F32> dimensions = {140.0f, 0.0f};
    std::optional<Extent2D<F32>> gridPosition;
    std::optional<Geometry> geometry;
    Sakura::Node node;
    Sakura::NodeTitle title;
    Sakura::NodeSubtitle subtitle;
    Sakura::NodeLoadingBar loadingBar;
    Sakura::NodeRuntimeOverlay runtimeOverlay;
    Sakura::Spacing metricsSpacing;
    std::vector<Sakura::NodePin> pins;
    std::vector<FlowgraphMetricInstance> metrics;
    std::vector<FlowgraphConfigFieldInstance> fields;
    std::vector<Sakura::SurfaceView> attachedSurfaces;
    std::vector<FlowgraphDetachedSurface> detachedSurfaces;
    std::vector<bool> detachedSurfaceStates;
    std::vector<std::optional<SurfaceSizeRecord>> lastSurfaceSizes;
    std::optional<LayoutRecord> lastLayout;
    std::vector<FlowgraphNodeMenu::DeviceOption> deviceOptions;
    FlowgraphNodeMenu menu;
    FlowgraphNodeDocumentation documentation;
    bool menuOpen = false;
    bool documentationOpen = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_NODE_HH
