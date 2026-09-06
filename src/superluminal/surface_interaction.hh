#ifndef JETSTREAM_SUPERLUMINAL_SURFACE_INTERACTION_HH
#define JETSTREAM_SUPERLUMINAL_SURFACE_INTERACTION_HH

#include <cmath>

#include <imgui.h>

#include <jetstream/surface.hh>

namespace Jetstream::detail {

// Called immediately after the surface's InvisibleButton so item state refers
// to the plot whose normalized coordinates are being forwarded.
template<typename Emit>
void ForwardSuperluminalSurfaceMouseEvents(const ImVec2& origin,
                                           const ImVec2& size,
                                           Emit&& emit) {
    if (!std::isfinite(size.x) || !std::isfinite(size.y) ||
        size.x <= 0.0f || size.y <= 0.0f) {
        return;
    }
    const bool hovered = ImGui::IsItemHovered();
    // The item loses its active state on release, so include that final frame
    // to complete a captured drag even when the pointer is outside the plot.
    if (!hovered && !ImGui::IsItemActive() && !ImGui::IsItemDeactivated()) {
        return;
    }

    const auto mouse = ImGui::GetMousePos();
    MouseEvent event{};
    event.position = {(mouse.x - origin.x) / size.x,
                      (mouse.y - origin.y) / size.y};

    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        event.type = MouseEventType::Click;
        event.button = MouseButton::Left;
        emit(event);
    } else if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        event.type = MouseEventType::Click;
        event.button = MouseButton::Right;
        emit(event);
    } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        event.type = MouseEventType::Release;
        event.button = MouseButton::Left;
        emit(event);
    } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        event.type = MouseEventType::Release;
        event.button = MouseButton::Right;
        emit(event);
    }

    const auto& io = ImGui::GetIO();
    if (hovered && (io.MouseWheel != 0.0f || io.MouseWheelH != 0.0f)) {
        event.type = MouseEventType::Scroll;
        event.scroll = {io.MouseWheelH, io.MouseWheel};
        emit(event);
    }

    event.type = MouseEventType::Move;
    event.scroll = {0.0f, 0.0f};
    emit(event);
}

}  // namespace Jetstream::detail

#endif  // JETSTREAM_SUPERLUMINAL_SURFACE_INTERACTION_HH
