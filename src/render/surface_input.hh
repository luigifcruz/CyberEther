#ifndef JETSTREAM_RENDER_SURFACE_INPUT_HH
#define JETSTREAM_RENDER_SURFACE_INPUT_HH

#include <array>
#include <functional>

#include <imgui.h>

#include <jetstream/surface.hh>

namespace Jetstream::detail {

struct JETSTREAM_API SurfaceInputState {
    SurfaceInputState() = default;
    SurfaceInputState(const SurfaceInputState&) = delete;
    SurfaceInputState& operator=(const SurfaceInputState&) = delete;
    SurfaceInputState(SurfaceInputState&&) = delete;
    SurfaceInputState& operator=(SurfaceInputState&&) = delete;
    ~SurfaceInputState();

    bool focused = false;
    bool macOS = false;
    int lastFrame = -1;
    std::array<bool, ImGuiKey_NamedKey_COUNT> heldKeys{};

    void observe(std::function<void(const InputEvent&)> emit);
    void reset(const std::function<void(const InputEvent&)>& emit);

 private:
    ImGuiContext* context = nullptr;
    ImGuiID endFrameHook = 0;
    ImGuiID shutdownHook = 0;
    std::function<void(const InputEvent&)> callback;

    void removeHooks();
};

inline ImGuiMouseCursor ToImGuiMouseCursor(const SurfaceCursor cursor) {
    switch (cursor) {
        case SurfaceCursor::Hand:
            return ImGuiMouseCursor_Hand;
        case SurfaceCursor::ResizeEW:
            return ImGuiMouseCursor_ResizeEW;
        case SurfaceCursor::ResizeNS:
            return ImGuiMouseCursor_ResizeNS;
        case SurfaceCursor::ResizeAll:
            return ImGuiMouseCursor_ResizeAll;
        case SurfaceCursor::ResizeNESW:
            return ImGuiMouseCursor_ResizeNESW;
        case SurfaceCursor::ResizeNWSE:
            return ImGuiMouseCursor_ResizeNWSE;
        case SurfaceCursor::TextInput:
            return ImGuiMouseCursor_TextInput;
        case SurfaceCursor::NotAllowed:
            return ImGuiMouseCursor_NotAllowed;
        default:
            return ImGuiMouseCursor_Arrow;
    }
}

JETSTREAM_API void ForwardSurfaceInputEvents(const ImVec2& origin,
                                             const ImVec2& size,
                                             SurfaceInputState& state,
                                             const std::function<void(const InputEvent&)>& emit);

}  // namespace Jetstream::detail

#endif  // JETSTREAM_RENDER_SURFACE_INPUT_HH
