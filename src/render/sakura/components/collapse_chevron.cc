#include <jetstream/render/sakura/components/collapse_chevron.hh>

#include "../helpers.hh"

namespace Jetstream::Sakura {

struct CollapseChevron::Impl {
    Config config;
};

CollapseChevron::CollapseChevron() {
    this->impl = std::make_unique<Impl>();
}

CollapseChevron::~CollapseChevron() = default;
CollapseChevron::CollapseChevron(CollapseChevron&&) noexcept = default;
CollapseChevron& CollapseChevron::operator=(CollapseChevron&&) noexcept = default;

bool CollapseChevron::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

bool CollapseChevron::render(const Context& ctx, const bool expanded) const {
    const auto& config = impl->config;

    ImGui::PushID(config.id.c_str());

    const F32 stripHeight = Scale(ctx, ImMin(config.stripHeight, 10.0f));
    const ImVec2 rowStart = ImGui::GetCursorScreenPos();
    const F32 availWidth = ImGui::GetContentRegionAvail().x;

    const F32 halfWidth = Scale(ctx, 4.0f);
    const F32 halfHeight = Scale(ctx, 2.0f);
    const F32 thickness = ImMax(1.0f, Scale(ctx, 1.5f));
    const ImVec2 center(rowStart.x + availWidth * 0.5f,
                        rowStart.y + stripHeight * 0.5f);

    ImVec2 points[3];
    if (expanded) {
        points[0] = ImVec2(center.x - halfWidth, center.y + halfHeight);
        points[1] = ImVec2(center.x, center.y - halfHeight);
        points[2] = ImVec2(center.x + halfWidth, center.y + halfHeight);
    } else {
        points[0] = ImVec2(center.x - halfWidth, center.y - halfHeight);
        points[1] = ImVec2(center.x, center.y + halfHeight);
        points[2] = ImVec2(center.x + halfWidth, center.y - halfHeight);
    }

    const ImVec2 hitSize(availWidth, stripHeight);
    ImGui::SetCursorScreenPos(ImVec2(rowStart.x, rowStart.y));
    ImGui::InvisibleButton("##toggle", hitSize);
    const bool hovered = ImGui::IsItemHovered();
    bool toggled = false;
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        toggled = true;
    }

    ImGui::SetCursorScreenPos(ImVec2(rowStart.x, rowStart.y + stripHeight));
    ImGui::Dummy(ImVec2(availWidth, 0.0f));

    const char* colorKey = hovered ? "text_primary" : "text_secondary";
    const ImU32 color =
        ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, colorKey));
    ImGui::GetWindowDrawList()->AddPolyline(points, 3, color, ImDrawFlags_None, thickness);

    if (hovered) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }

    ImGui::PopID();
    return toggled;
}

}  // namespace Jetstream::Sakura
