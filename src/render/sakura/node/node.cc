#include <jetstream/render/sakura/node/node.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Node::Impl {
    Config config;
    std::optional<Extent2D<F32>> appliedGridPosition;
    F32 appliedGridPositionScale = 0.0f;
    int lastRenderedFrame = -1;

    bool shouldApplyGridPosition(const Context& ctx, const Extent2D<F32>& position) const {
        if (lastRenderedFrame >= 0 && ImGui::GetFrameCount() > lastRenderedFrame + 1) {
            return true;
        }
        if (!appliedGridPosition.has_value()) {
            return true;
        }
        if (appliedGridPositionScale != ScalingFactor(ctx)) {
            return true;
        }
        return appliedGridPosition->x != position.x || appliedGridPosition->y != position.y;
    }
};

Node::Node() {
    this->impl = std::make_unique<Impl>();
}

Node::~Node() = default;
Node::Node(Node&&) noexcept = default;
Node& Node::operator=(Node&&) noexcept = default;

bool Node::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Node::render(const Context& ctx, Child child) const {
    auto& impl = *this->impl;
    const auto& config = impl.config;

    Private::RegisterNodeEditorNode(config.id);
    const int imNodesId = Private::NodeEditorObjectId(config.id);

    if (config.gridPosition.has_value() && impl.shouldApplyGridPosition(ctx, config.gridPosition.value())) {
        ImNodes::SetNodeGridSpacePos(imNodesId, Private::ToImVec2(Scale(ctx, *config.gridPosition)));
        impl.appliedGridPosition = config.gridPosition;
        impl.appliedGridPositionScale = ScalingFactor(ctx);
    } else if (!config.gridPosition.has_value()) {
        impl.appliedGridPosition.reset();
        impl.appliedGridPositionScale = 0.0f;
    }

    ImNodes::SetNodeVerticalResizeEnabled(imNodesId, config.verticalResize);
    ImVec2 contentSize = Private::ToImVec2(Scale(ctx, config.dimensions));
    ImNodes::SetNodeDimensions(imNodesId, contentSize);

    if (config.state != State::Normal) {
        const ImU32 stateColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, config.state == State::Error ? "node_outline_error"
                                                                                                                   : "node_outline_pending"));
        ImNodes::PushColorStyle(ImNodesCol_NodeOutline, stateColor);
        ImNodes::PushColorStyle(ImNodesCol_Pin, stateColor);
        ImNodes::PushColorStyle(ImNodesCol_PinHovered, stateColor);
    }

    ImNodes::BeginNode(imNodesId);
    Private::NodeEditorNodeStack().push_back(config.id);
    if (child) {
        child(ctx);
    }
    Private::NodeEditorNodeStack().pop_back();
    ImNodes::EndNode();

    if (config.onGeometryChange) {
        config.onGeometryChange(Unscale(ctx, Private::ToExtent2D(ImNodes::GetNodeGridSpacePos(imNodesId))),
                                Private::ToExtent2D(ImNodes::GetNodeScreenSpacePos(imNodesId)),
                                Private::ToExtent2D(ImNodes::GetNodeDimensions(imNodesId)),
                                Unscale(ctx, Private::ToExtent2D(contentSize)));
    }

    if (config.onContextMenu && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        const ImVec2 pos = ImNodes::GetNodeScreenSpacePos(imNodesId);
        const ImVec2 size = ImNodes::GetNodeDimensions(imNodesId);
        const ImVec2 mousePos = ImGui::GetMousePos();
        if (mousePos.x >= pos.x && mousePos.x <= pos.x + size.x &&
            mousePos.y >= pos.y && mousePos.y <= pos.y + size.y) {
            config.onContextMenu();
        }
    }

    if (config.state != State::Normal) {
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }

    impl.lastRenderedFrame = ImGui::GetFrameCount();
}

}  // namespace Jetstream::Sakura
