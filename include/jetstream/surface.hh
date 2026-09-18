#ifndef JETSTREAM_SURFACE_HH
#define JETSTREAM_SURFACE_HH

#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <variant>

#include "jetstream/types.hh"
#include "jetstream/render/base/texture.hh"

namespace Jetstream {

//
// SurfaceManifest
//

struct JETSTREAM_API SurfaceManifest {
    std::string id;
    Extent2D<U64> size;
    std::shared_ptr<const Render::Texture> surface;
};

//
// InputEvent
//

struct JETSTREAM_API KeyModifiers {
    bool control = false;
    bool shift = false;
    bool alt = false;
    bool super = false;
};

enum class JETSTREAM_API KeyCode : U16 {
    Unknown = 0,
    Tab, LeftArrow, RightArrow, UpArrow, DownArrow,
    PageUp, PageDown, Home, End, Insert, Delete, Backspace, Space, Enter, Escape,
    LeftCtrl, LeftShift, LeftAlt, LeftSuper, RightCtrl, RightShift, RightAlt, RightSuper,
    Menu,
    Digit0, Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9,
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24,
    Apostrophe, Comma, Minus, Period, Slash, Semicolon, Equal,
    LeftBracket, Backslash, RightBracket, GraveAccent,
    CapsLock, ScrollLock, NumLock, PrintScreen, Pause,
    Keypad0, Keypad1, Keypad2, Keypad3, Keypad4, Keypad5, Keypad6, Keypad7, Keypad8, Keypad9,
    KeypadDecimal, KeypadDivide, KeypadMultiply, KeypadSubtract, KeypadAdd, KeypadEnter, KeypadEqual,
    AppBack, AppForward, Oem102,
};

enum class JETSTREAM_API KeyEventType : U8 {
    Press = 0,
    Release = 1,
};

struct JETSTREAM_API KeyEvent {
    KeyEventType type;
    KeyCode key;
    KeyModifiers modifiers;
    bool repeat = false;
};

enum class JETSTREAM_API MouseButton : U8 {
    Left = 0,
    Right = 1,
};

enum class JETSTREAM_API MouseEventType : U8 {
    Click = 0,
    Release = 1,
    Move = 2,
    Scroll = 3,
    Enter = 4,
    Leave = 5,
};

struct JETSTREAM_API MouseEvent {
    MouseEventType type;
    MouseButton button;
    Extent2D<F32> position;
    Extent2D<F32> scroll;
    KeyModifiers modifiers;
};

struct JETSTREAM_API FocusEvent {
    bool focused = false;
};

using InputEvent = std::variant<MouseEvent, KeyEvent, FocusEvent>;

inline std::optional<MouseEvent> SurfaceMouseEvent(const InputEvent& event) {
    if (const auto* mouse = std::get_if<MouseEvent>(&event)) {
        return *mouse;
    }
    if (const auto* focus = std::get_if<FocusEvent>(&event); focus && !focus->focused) {
        return MouseEvent{.type = MouseEventType::Leave};
    }
    return std::nullopt;
}

//
// SurfaceEvent
//

enum class JETSTREAM_API SurfaceEventType : U8 {
    Resize = 0,
};

enum class JETSTREAM_API SurfacePlacementType : U8 {
    Attached = 0,
    Detached = 1,
};

struct JETSTREAM_API SurfaceEvent {
    SurfaceEventType type;
    Extent2D<U64> size;
    F32 scale = 1.0f;
    ColorRGBA<F32> backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};
    SurfacePlacementType placement = SurfacePlacementType::Detached;
};

//
// EventBuffer
//

struct JETSTREAM_API EventBuffer {
    std::vector<InputEvent> inputEvents;
    std::vector<SurfaceEvent> surfaceEvents;

    void pushInput(const InputEvent& event) {
        inputEvents.push_back(event);
    }

    void pushSurface(const SurfaceEvent& event) {
        surfaceEvents.push_back(event);
    }

    std::vector<InputEvent> consumeInputEvents() {
        return std::exchange(inputEvents, {});
    }

    std::vector<SurfaceEvent> consumeSurfaceEvents() {
        return std::move(surfaceEvents);
    }
};

//
// SurfaceInteractionState
//

struct JETSTREAM_API SurfaceInteractionState {
    F32 zoom = 1.0f;
    F32 offset = 0.0f;
    F32 scale = 1.0f;
    Extent2D<U64> viewSize = {512, 512};
    ColorRGBA<F32> backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};

    Extent2D<F32> cursorNormalized = {0.0f, 0.0f};

    bool dragging = false;
    F32 dragAnchor = 0.0f;

    bool viewChanged = false;
    bool cursorMoved = false;
    SurfacePlacementType placement = SurfacePlacementType::Detached;
};

struct JETSTREAM_API SurfaceInteractionConfig {
    F32 zoomSpeed = 0.1f;
    F32 minZoom = 1.0f;
    F32 maxZoom = 10.0f;
    bool enableZoom = true;
    bool enablePan = true;
    bool enableCursor = true;
};

JETSTREAM_API SurfaceInteractionState ProcessSurfaceInteraction(SurfaceInteractionState state,
                                                                std::vector<SurfaceEvent>&& surfaceEvents,
                                                                std::vector<InputEvent>&& inputEvents,
                                                                const SurfaceInteractionConfig& config = {});

}  // namespace Jetstream

#endif  // JETSTREAM_SURFACE_HH
