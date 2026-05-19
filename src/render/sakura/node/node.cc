#include <jetstream/render/sakura/node/node.hh>

#include "base.hh"

#include <cmath>

namespace Jetstream::Sakura {

struct Node::Impl {
    struct GeometryRecord {
        Extent2D<F32> gridPosition;
        Extent2D<F32> screenPosition;
        Extent2D<F32> outerDimensions;
        Extent2D<F32> contentDimensions;
    };

    Config config;
    std::optional<Extent2D<F32>> appliedGridPosition;
    std::optional<GeometryRecord> lastEmittedGeometry;
    F32 appliedGridPositionScale = 0.0f;
    ImVec2 contentSize = ImVec2(0.0f, 0.0f);
    ImVec2 requestedContentSize = ImVec2(0.0f, 0.0f);
    F32 contentSizeScale = 0.0f;
    bool hasContentSize = false;
    bool hasRequestedContentSize = false;
    int lastRenderedFrame = -1;
};

Node::Node() {
    this->impl = std::make_unique<Impl>();
}

Node::~Node() = default;
Node::Node(Node&&) noexcept = default;
Node& Node::operator=(Node&&) noexcept = default;

bool Node::update(Config config) {
    if (this->impl->config.id != config.id) {
        this->impl->contentSize = ImVec2(0.0f, 0.0f);
        this->impl->requestedContentSize = ImVec2(0.0f, 0.0f);
        this->impl->contentSizeScale = 0.0f;
        this->impl->hasContentSize = false;
        this->impl->hasRequestedContentSize = false;
        this->impl->lastEmittedGeometry.reset();
    }
    this->impl->config = std::move(config);
    return true;
}

void Node::render(const Context& ctx, Child child) const {
    const auto& config = impl->config;

    Private::RegisterNodeEditorNode(config.id);
    const int imNodesId = Private::NodeEditorObjectId(config.id);

    if (config.gridPosition.has_value()) {
        const bool shouldApplyGridPosition =
            (impl->lastRenderedFrame >= 0 && ImGui::GetFrameCount() > impl->lastRenderedFrame + 1) ||
            !impl->appliedGridPosition.has_value() ||
            impl->appliedGridPositionScale != ScalingFactor(ctx) ||
            impl->appliedGridPosition->x != config.gridPosition->x ||
            impl->appliedGridPosition->y != config.gridPosition->y;
        if (shouldApplyGridPosition) {
            ImNodes::SetNodeGridSpacePos(imNodesId, Private::ToImVec2(Scale(ctx, *config.gridPosition)));
            impl->appliedGridPosition = config.gridPosition;
            impl->appliedGridPositionScale = ScalingFactor(ctx);
        }
    } else if (!config.gridPosition.has_value()) {
        impl->appliedGridPosition.reset();
        impl->appliedGridPositionScale = 0.0f;
    }

    ImNodes::SetNodeVerticalResizeEnabled(imNodesId, config.verticalResize);
    const ImVec2 requestedContentSize = Private::ToImVec2(Scale(ctx, config.dimensions));
    const bool requestedContentSizeChanged = !impl->hasContentSize ||
                                             !impl->hasRequestedContentSize ||
                                              impl->contentSizeScale != ScalingFactor(ctx) ||
                                              impl->requestedContentSize.x != requestedContentSize.x ||
                                              impl->requestedContentSize.y != requestedContentSize.y;
    if (requestedContentSizeChanged) {
        impl->contentSize = requestedContentSize;
    }
    if (requestedContentSize.x <= 0.0f) {
        impl->contentSize.x = 0.0f;
    }
    if (requestedContentSize.y <= 0.0f) {
        impl->contentSize.y = 0.0f;
    }
    impl->requestedContentSize = requestedContentSize;
    impl->contentSizeScale = ScalingFactor(ctx);
    impl->hasContentSize = true;
    impl->hasRequestedContentSize = true;
    // Keep this storage alive through EndNodeEditor(); ImNodes writes manual resize results and
    // measured auto-axis sizes after the node body has rendered.
    ImNodes::SetNodeDimensions(imNodesId, impl->contentSize);

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

    const Impl::GeometryRecord geometryRecord{
        .gridPosition = Unscale(ctx, Private::ToExtent2D(ImNodes::GetNodeGridSpacePos(imNodesId))),
        .screenPosition = Private::ToExtent2D(ImNodes::GetNodeScreenSpacePos(imNodesId)),
        .outerDimensions = Private::ToExtent2D(ImNodes::GetNodeDimensions(imNodesId)),
        .contentDimensions = Unscale(ctx, Private::ToExtent2D(impl->contentSize)),
    };
    if (config.onGeometryChange) {
        bool geometryChanged = !impl->lastEmittedGeometry.has_value();
        if (!geometryChanged) {
            const auto& last = *impl->lastEmittedGeometry;
            geometryChanged = !(std::abs(last.gridPosition.x - geometryRecord.gridPosition.x) <= 0.5f &&
                                std::abs(last.gridPosition.y - geometryRecord.gridPosition.y) <= 0.5f &&
                                std::abs(last.screenPosition.x - geometryRecord.screenPosition.x) <= 0.5f &&
                                std::abs(last.screenPosition.y - geometryRecord.screenPosition.y) <= 0.5f &&
                                std::abs(last.outerDimensions.x - geometryRecord.outerDimensions.x) <= 0.5f &&
                                std::abs(last.outerDimensions.y - geometryRecord.outerDimensions.y) <= 0.5f &&
                                std::abs(last.contentDimensions.x - geometryRecord.contentDimensions.x) <= 0.5f &&
                                std::abs(last.contentDimensions.y - geometryRecord.contentDimensions.y) <= 0.5f);
        }
        if (geometryChanged) {
            impl->lastEmittedGeometry = geometryRecord;
            config.onGeometryChange(geometryRecord.gridPosition,
                                    geometryRecord.screenPosition,
                                    geometryRecord.outerDimensions,
                                    geometryRecord.contentDimensions);
        }
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

    impl->lastRenderedFrame = ImGui::GetFrameCount();
}

}  // namespace Jetstream::Sakura
