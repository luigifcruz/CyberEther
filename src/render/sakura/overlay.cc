#include <jetstream/render/sakura/overlay.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Overlay::Impl {
    Config config;

    Extent2D<F32> anchorPosition(const Extent2D<F32>& origin,
                                 const Extent2D<F32>& available,
                                 const Extent2D<F32>& size) const {
        switch (config.anchor) {
            case Anchor::TopLeft:
                return origin;
            case Anchor::TopRight:
                return {origin.x + available.x - size.x, origin.y};
            case Anchor::BottomLeft:
                return {origin.x, origin.y + available.y - size.y};
            case Anchor::BottomRight:
                return {origin.x + available.x - size.x, origin.y + available.y - size.y};
            case Anchor::BottomCenter:
                return {origin.x + (available.x - size.x) * 0.5f, origin.y + available.y - size.y};
            case Anchor::Center:
                return {origin.x + (available.x - size.x) * 0.5f,
                        origin.y + (available.y - size.y) * 0.5f};
        }
        return origin;
    }
};

Overlay::Overlay() {
    this->impl = std::make_unique<Impl>();
}

Overlay::~Overlay() = default;
Overlay::Overlay(Overlay&&) noexcept = default;
Overlay& Overlay::operator=(Overlay&&) noexcept = default;

bool Overlay::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Overlay::render(const Context& ctx, Child child) const {
    const auto& config = this->impl->config;

    if (!child) {
        return;
    }

    const ImRect region = ImGui::GetCurrentWindow()->WorkRect;
    const ImVec2 regionMin = region.Min;
    const ImVec2 regionMax = region.Max;
    const Extent2D<F32> available = {
        std::max(0.0f, regionMax.x - regionMin.x),
        std::max(0.0f, regionMax.y - regionMin.y),
    };
    Extent2D<F32> size = Scale(ctx, config.size);
    if (size.x <= 0.0f) {
        size.x = available.x;
    }
    if (size.y <= 0.0f) {
        size.y = available.y;
    }

    Extent2D<F32> position = this->impl->anchorPosition(Private::ToExtent2D(regionMin), available, size);
    const Extent2D<F32> offset = Scale(ctx, config.offset);
    position.x += offset.x;
    position.y += offset.y;

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoBackground |
                                   ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoScrollbar |
                                   ImGuiWindowFlags_NoScrollWithMouse |
                                   ImGuiWindowFlags_NoNav;
    if (!config.inputs) {
        windowFlags |= ImGuiWindowFlags_NoInputs;
    }

    ImGuiWindow* parentWindow = ImGui::GetCurrentWindow();
    const ImVec2 cursorPos = parentWindow->DC.CursorPos;
    const ImVec2 cursorPosPrevLine = parentWindow->DC.CursorPosPrevLine;
    const ImVec2 cursorMaxPos = parentWindow->DC.CursorMaxPos;
    const ImVec2 idealMaxPos = parentWindow->DC.IdealMaxPos;
    const ImVec2 currLineSize = parentWindow->DC.CurrLineSize;
    const ImVec2 prevLineSize = parentWindow->DC.PrevLineSize;
    const F32 currLineTextBaseOffset = parentWindow->DC.CurrLineTextBaseOffset;
    const F32 prevLineTextBaseOffset = parentWindow->DC.PrevLineTextBaseOffset;
    const bool isSameLine = parentWindow->DC.IsSameLine;
    const bool isSetPos = parentWindow->DC.IsSetPos;

    ImGui::PushID(config.id.c_str());
    ImGui::SetCursorScreenPos(Private::ToImVec2(position));
    const F32 clipSlack = std::ceil(Scale(ctx, 2.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    if (ImGui::BeginChild("content", ImVec2(size.x + clipSlack, size.y + clipSlack), ImGuiChildFlags_None, windowFlags)) {
        child(ctx);
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopID();

    parentWindow = ImGui::GetCurrentWindow();
    parentWindow->DC.CursorPos = cursorPos;
    parentWindow->DC.CursorPosPrevLine = cursorPosPrevLine;
    parentWindow->DC.CursorMaxPos = cursorMaxPos;
    parentWindow->DC.IdealMaxPos = idealMaxPos;
    parentWindow->DC.CurrLineSize = currLineSize;
    parentWindow->DC.PrevLineSize = prevLineSize;
    parentWindow->DC.CurrLineTextBaseOffset = currLineTextBaseOffset;
    parentWindow->DC.PrevLineTextBaseOffset = prevLineTextBaseOffset;
    parentWindow->DC.IsSameLine = isSameLine;
    parentWindow->DC.IsSetPos = isSetPos;
}

}  // namespace Jetstream::Sakura
