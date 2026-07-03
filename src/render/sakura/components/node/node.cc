#include <jetstream/render/sakura/components/node/node.hh>

#include "base.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Sakura {

namespace {

constexpr F32 Pi = 3.14159265358979323846f;

ImVec2 PointOnRectPerimeter(const ImVec2& min, const ImVec2& max, F32 distance) {
    const F32 width = max.x - min.x;
    const F32 height = max.y - min.y;
    const F32 perimeter = 2.0f * (width + height);
    if (perimeter <= 0.0f) {
        return min;
    }

    F32 d = std::fmod(distance, perimeter);
    if (d < 0.0f) {
        d += perimeter;
    }

    if (d <= width) {
        return ImVec2(min.x + d, min.y);
    }
    d -= width;

    if (d <= height) {
        return ImVec2(max.x, min.y + d);
    }
    d -= height;

    if (d <= width) {
        return ImVec2(max.x - d, max.y);
    }
    d -= width;

    return ImVec2(min.x, max.y - d);
}

F32 RoundedRectPerimeter(const ImVec2& min, const ImVec2& max, F32 radius) {
    const F32 width = max.x - min.x;
    const F32 height = max.y - min.y;
    const F32 r = std::clamp(radius, 0.0f, std::min(width, height) * 0.5f);
    return 2.0f * (std::max(0.0f, width - 2.0f * r) + std::max(0.0f, height - 2.0f * r)) +
           2.0f * Pi * r;
}

ImVec2 PointOnRoundedRectPerimeter(const ImVec2& min, const ImVec2& max, F32 radius, F32 distance) {
    const F32 width = max.x - min.x;
    const F32 height = max.y - min.y;
    const F32 r = std::clamp(radius, 0.0f, std::min(width, height) * 0.5f);
    if (r <= 0.0f) {
        return PointOnRectPerimeter(min, max, distance);
    }

    const F32 straightWidth = std::max(0.0f, width - 2.0f * r);
    const F32 straightHeight = std::max(0.0f, height - 2.0f * r);
    const F32 arcLength = Pi * r * 0.5f;
    const F32 perimeter = RoundedRectPerimeter(min, max, r);
    if (perimeter <= 0.0f) {
        return min;
    }

    F32 d = std::fmod(distance, perimeter);
    if (d < 0.0f) {
        d += perimeter;
    }

    const auto arcPoint = [r](const ImVec2& center, F32 startAngle, F32 arcDistance) {
        const F32 angle = startAngle + arcDistance / r;
        return ImVec2(center.x + std::cos(angle) * r, center.y + std::sin(angle) * r);
    };

    if (d <= straightWidth) {
        return ImVec2(min.x + r + d, min.y);
    }
    d -= straightWidth;

    if (d <= arcLength) {
        return arcPoint(ImVec2(max.x - r, min.y + r), -Pi * 0.5f, d);
    }
    d -= arcLength;

    if (d <= straightHeight) {
        return ImVec2(max.x, min.y + r + d);
    }
    d -= straightHeight;

    if (d <= arcLength) {
        return arcPoint(ImVec2(max.x - r, max.y - r), 0.0f, d);
    }
    d -= arcLength;

    if (d <= straightWidth) {
        return ImVec2(max.x - r - d, max.y);
    }
    d -= straightWidth;

    if (d <= arcLength) {
        return arcPoint(ImVec2(min.x + r, max.y - r), Pi * 0.5f, d);
    }
    d -= arcLength;

    if (d <= straightHeight) {
        return ImVec2(min.x, max.y - r - d);
    }
    d -= straightHeight;

    return arcPoint(ImVec2(min.x + r, min.y + r), Pi, d);
}

ImU32 ColorWithAlpha(const ImVec4& color, F32 alpha) {
    return ImGui::ColorConvertFloat4ToU32(ImVec4(color.x,
                                                 color.y,
                                                 color.z,
                                                 color.w * std::clamp(alpha, 0.0f, 1.0f)));
}

void DrawLoadingBadge(const Context& ctx, const ImVec2& pos, const ImVec2& size) {
    const char* label = "LOADING";
    ImFont* font = Private::NativeFont(ctx.fonts.bold);
    if (!font) {
        font = ImGui::GetFont();
    }

    const F32 fontSize = ImGui::GetStyle().FontSizeBase;
    const ImVec2 padding(Scale(ctx, 10.0f), Scale(ctx, 5.0f));
    const F32 rounding = Scale(ctx, 8.0f);
    const ImVec4 color = Private::ImColor(ctx, "node_outline_pending");
    const ImU32 fillColor = ColorWithAlpha(color, 0.16f);
    const ImU32 textColor = ColorWithAlpha(color, 1.0f);
    const ImVec2 textSize = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, label);
    const ImVec2 badgeSize(textSize.x + padding.x * 2.0f, textSize.y + padding.y * 2.0f);
    const F32 titleClearance = Scale(ctx, 48.0f);
    const F32 bottomMargin = Scale(ctx, 8.0f);
    const F32 centeredY = pos.y + (size.y - badgeSize.y) * 0.5f + Scale(ctx, 12.0f);
    const F32 maxY = pos.y + size.y - badgeSize.y - bottomMargin;
    const F32 badgeY = maxY >= pos.y + titleClearance
        ? std::clamp(centeredY, pos.y + titleClearance, maxY)
        : centeredY;
    const ImVec2 badgePosition(pos.x + (size.x - badgeSize.x) * 0.5f, badgeY);
    const ImVec2 badgeMax(badgePosition.x + badgeSize.x, badgePosition.y + badgeSize.y);
    const ImVec2 textPosition(badgePosition.x + padding.x, badgePosition.y + padding.y);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(badgePosition, badgeMax, fillColor, rounding);
    drawList->AddText(font, fontSize, textPosition, textColor, label);
}

void DrawBorderChase(const Context& ctx, const ImVec2& pos, const ImVec2& size) {
    const F32 thickness = ImNodes::GetStyle().NodeBorderThickness;
    const F32 inset = std::max(thickness * 0.5f, Scale(ctx, 1.0f));
    if (size.x <= inset * 2.0f || size.y <= inset * 2.0f) {
        return;
    }

    const ImVec2 min(pos.x + inset, pos.y + inset);
    const ImVec2 max(pos.x + size.x - inset, pos.y + size.y - inset);
    const F32 radius = std::max(0.0f, ImNodes::GetStyle().NodeCornerRounding - inset);
    const F32 perimeter = RoundedRectPerimeter(min, max, radius);
    if (perimeter <= 0.0f) {
        return;
    }

    const F32 time = static_cast<F32>(ImGui::GetTime());
    const F32 visibleThickness = thickness + Scale(ctx, 0.75f);
    const F32 glowThickness = visibleThickness + Scale(ctx, 4.0f);
    const F32 chaseLength = std::min(perimeter * 0.42f, Scale(ctx, 360.0f));
    if (chaseLength <= 0.0f) {
        return;
    }

    const F32 step = std::max(Scale(ctx, 2.0f), chaseLength / 88.0f);
    const F32 speed = Scale(ctx, 320.0f);
    const F32 head = std::fmod(time * speed, perimeter);
    const F32 pulse = 0.55f + std::sin(time * 7.0f) * 0.25f;
    const ImVec4 pendingColor = Private::ImColor(ctx, "node_outline_pending");
    const ImVec4 highlightColor(std::min(pendingColor.x + 0.22f, 0.92f),
                                std::min(pendingColor.y + 0.22f, 0.92f),
                                std::min(pendingColor.z + 0.22f, 0.92f),
                                pendingColor.w);
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    drawList->AddRect(min,
                      max,
                      ColorWithAlpha(highlightColor, pulse * 0.45f),
                      radius,
                      ImDrawFlags_RoundCornersAll,
                      thickness);

    for (F32 offset = 0.0f; offset < chaseLength; offset += step) {
        const F32 nextOffset = std::min(offset + step, chaseLength);
        const ImVec2 p0 = PointOnRoundedRectPerimeter(min, max, radius, head - offset);
        const ImVec2 p1 = PointOnRoundedRectPerimeter(min, max, radius, head - nextOffset);
        const F32 dx = p1.x - p0.x;
        const F32 dy = p1.y - p0.y;
        if ((dx * dx + dy * dy) > (step * step * 4.0f)) {
            continue;
        }

        const F32 t = 1.0f - (offset / chaseLength);
        const F32 alpha = t * t;
        drawList->AddLine(p0, p1, ColorWithAlpha(pendingColor, alpha * 0.45f), glowThickness);
        drawList->AddLine(p0, p1, ColorWithAlpha(pendingColor, alpha * 0.95f), visibleThickness);
        drawList->AddLine(p0, p1, ColorWithAlpha(highlightColor, alpha * 0.85f), thickness);
    }
}

}  // namespace

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
    ImVec2 requestedContentSize = Private::ToImVec2(Scale(ctx, config.dimensions));
    if (config.state == State::Loading) {
        requestedContentSize.y = std::max(requestedContentSize.y, Scale(ctx, 96.0f));
    }
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

    const ImVec2 nodeGridPosition = ImNodes::GetNodeGridSpacePos(imNodesId);
    const ImVec2 nodeScreenPosition = ImNodes::GetNodeScreenSpacePos(imNodesId);
    const ImVec2 nodeDimensions = ImNodes::GetNodeDimensions(imNodesId);
    const F32 measuredContentWidth = std::max(0.0f, nodeDimensions.x - ImNodes::GetStyle().NodePadding.x * 2.0f);
    if (measuredContentWidth > impl->contentSize.x + Scale(ctx, 0.5f)) {
        impl->contentSize.x = measuredContentWidth;
    }
    if (config.state == State::Loading) {
        DrawBorderChase(ctx, nodeScreenPosition, nodeDimensions);
        DrawLoadingBadge(ctx, nodeScreenPosition, nodeDimensions);
    }

    const Impl::GeometryRecord geometryRecord{
        .gridPosition = Unscale(ctx, Private::ToExtent2D(nodeGridPosition)),
        .screenPosition = Private::ToExtent2D(nodeScreenPosition),
        .outerDimensions = Private::ToExtent2D(nodeDimensions),
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
