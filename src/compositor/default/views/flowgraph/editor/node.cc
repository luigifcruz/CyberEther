#include "node.hh"

namespace Jetstream {

FlowgraphTooltipRows FlowgraphTensorLayoutRows(const Tensor& tensor) {
    return {
        {"Device", jst::fmt::format("{}", tensor.device())},
        {"Type", jst::fmt::format("{}", tensor.dtype())},
        {"Shape", jst::fmt::format("{}", tensor.shape())},
        {"Strides", jst::fmt::format("{} ({})", tensor.stride(), tensor.contiguous() ? "Contiguous" : "Non-contiguous")},
        {"Offset", jst::fmt::format("{} bytes", tensor.offsetBytes())},
    };
}

FlowgraphTooltipRows FlowgraphTensorAttributeRows(const Tensor& tensor) {
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

void FlowgraphPinTooltip::update(std::string id, std::string help, const std::optional<Tensor>& tensor) {
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

Sakura::NodeOverlay FlowgraphPinTooltip::overlay() {
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

void FlowgraphPinTooltip::prepare() {
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

void FlowgraphDiagnosticTooltip::update(const std::string& id, const std::string& message, const std::string& colorKey) {
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
        .spacing = Sakura::Spacing::Medium,
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

Sakura::NodeOverlay FlowgraphDiagnosticTooltip::overlay() {
    return {
        .content = [this]() {
            return tooltip.show();
        },
    };
}

void FlowgraphNode::update(Config config) {
    std::optional<Layout> previousLayout;
    Sakura::Callback<void(F32, F32, F32, F32)> previousOnLayout;
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
    const bool configHidden = block.configCollapsed;
    const bool isErrored = block.state == Block::State::Errored;
    const auto nodeState = isErrored    ? NodeEditor::NodeState::Error
                         : isBusy       ? NodeEditor::NodeState::Loading
                         : isPending    ? NodeEditor::NodeState::Pending
                         : block.frozen ? NodeEditor::NodeState::Frozen
                                        : NodeEditor::NodeState::Normal;

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
        .collapsible = hasConfigFields,
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
        .open = menuOpen,
        .targets = this->config.targets,
        .configDetachEnabled = hasConfigFields,
        .configDetached = block.configDetached,
        .plotDetachEnabled = hasSurfaces,
        .plotDetached = allSurfacesDetached,
        .frozen = block.frozen,
        .devices = deviceOptions,
        .onCopy = this->config.onCopy,
        .onCut = this->config.onCut,
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
        .onDuplicate = this->config.onDuplicate,
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
        .onToggleFrozen = this->config.onToggleFrozen,
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

void FlowgraphNode::setSurfacesDetached(bool detached) {
    for (const auto& surface : config.block.surfaces) {
        if (surface.detached != detached && surface.onDetach) {
            surface.onDetach(detached);
        }
    }
}

FlowgraphNode::NodeEditor::Node FlowgraphNode::takeNode() {
    return std::move(nodeSpec);
}

Sakura::Component& FlowgraphNode::menuComponent() {
    return menu.component();
}

void FlowgraphNode::appendWindows(std::vector<Sakura::FloatingLayer::Window>& windows) {
    if (inspectorOpen) {
        windows.push_back(inspector.window());
    }
}

bool FlowgraphNode::Busy(const Block::State state) {
    return state == Block::State::Creating || state == Block::State::Destroying;
}

void FlowgraphNode::reportBounds() const {
    const auto& block = config.block;
    if (!geometry.has_value() || Busy(block.state) || !config.onBounds) {
        return;
    }
    const auto& next = *geometry;
    const Extent2D<F32> inner = next.contentDimensions;
    std::optional<Extent2D<F32>> content;
    if (next.onscreen && !block.configCollapsed) {
        content = fieldStack.content(inner, next.minimumHeight);
    }
    config.onBounds({next.gridPosition.x, next.gridPosition.y, next.dimensions.x, next.dimensions.y},
                    inner,
                    content);
}

void FlowgraphNode::updateContent(const BlockData& block, bool isBusy, bool configHidden) {
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

}  // namespace Jetstream
