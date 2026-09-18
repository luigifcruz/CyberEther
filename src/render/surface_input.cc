#include "surface_input.hh"

#include <cmath>

#include <imgui_internal.h>

namespace Jetstream::detail {

namespace {

ImGuiKey SurfaceNativeKey(ImGuiKey key, bool macOS) {
    if (macOS) {
        switch (key) {
            case ImGuiMod_Ctrl: return ImGuiMod_Super;
            case ImGuiMod_Super: return ImGuiMod_Ctrl;
            case ImGuiKey_LeftCtrl: return ImGuiKey_LeftSuper;
            case ImGuiKey_RightCtrl: return ImGuiKey_RightSuper;
            case ImGuiKey_LeftSuper: return ImGuiKey_LeftCtrl;
            case ImGuiKey_RightSuper: return ImGuiKey_RightCtrl;
            default: break;
        }
    }
    return key;
}

KeyCode SurfaceKeyCode(ImGuiKey key, bool macOS) {
    key = SurfaceNativeKey(key, macOS);
    if (key >= ImGuiKey_0 && key <= ImGuiKey_9) {
        return static_cast<KeyCode>(static_cast<U16>(KeyCode::Digit0) + key - ImGuiKey_0);
    }
    if (key >= ImGuiKey_A && key <= ImGuiKey_Z) {
        return static_cast<KeyCode>(static_cast<U16>(KeyCode::A) + key - ImGuiKey_A);
    }
    if (key >= ImGuiKey_F1 && key <= ImGuiKey_F24) {
        return static_cast<KeyCode>(static_cast<U16>(KeyCode::F1) + key - ImGuiKey_F1);
    }
    if (key >= ImGuiKey_Keypad0 && key <= ImGuiKey_Keypad9) {
        return static_cast<KeyCode>(static_cast<U16>(KeyCode::Keypad0) + key - ImGuiKey_Keypad0);
    }
    switch (key) {
#define JST_SURFACE_KEY(name) case ImGuiKey_##name: return KeyCode::name;
        JST_SURFACE_KEY(Tab)
        JST_SURFACE_KEY(LeftArrow)
        JST_SURFACE_KEY(RightArrow)
        JST_SURFACE_KEY(UpArrow)
        JST_SURFACE_KEY(DownArrow)
        JST_SURFACE_KEY(PageUp)
        JST_SURFACE_KEY(PageDown)
        JST_SURFACE_KEY(Home)
        JST_SURFACE_KEY(End)
        JST_SURFACE_KEY(Insert)
        JST_SURFACE_KEY(Delete)
        JST_SURFACE_KEY(Backspace)
        JST_SURFACE_KEY(Space)
        JST_SURFACE_KEY(Enter)
        JST_SURFACE_KEY(Escape)
        JST_SURFACE_KEY(LeftCtrl)
        JST_SURFACE_KEY(LeftShift)
        JST_SURFACE_KEY(LeftAlt)
        JST_SURFACE_KEY(LeftSuper)
        JST_SURFACE_KEY(RightCtrl)
        JST_SURFACE_KEY(RightShift)
        JST_SURFACE_KEY(RightAlt)
        JST_SURFACE_KEY(RightSuper)
        JST_SURFACE_KEY(Menu)
        JST_SURFACE_KEY(Apostrophe)
        JST_SURFACE_KEY(Comma)
        JST_SURFACE_KEY(Minus)
        JST_SURFACE_KEY(Period)
        JST_SURFACE_KEY(Slash)
        JST_SURFACE_KEY(Semicolon)
        JST_SURFACE_KEY(Equal)
        JST_SURFACE_KEY(LeftBracket)
        JST_SURFACE_KEY(Backslash)
        JST_SURFACE_KEY(RightBracket)
        JST_SURFACE_KEY(GraveAccent)
        JST_SURFACE_KEY(CapsLock)
        JST_SURFACE_KEY(ScrollLock)
        JST_SURFACE_KEY(NumLock)
        JST_SURFACE_KEY(PrintScreen)
        JST_SURFACE_KEY(Pause)
        JST_SURFACE_KEY(KeypadDecimal)
        JST_SURFACE_KEY(KeypadDivide)
        JST_SURFACE_KEY(KeypadMultiply)
        JST_SURFACE_KEY(KeypadSubtract)
        JST_SURFACE_KEY(KeypadAdd)
        JST_SURFACE_KEY(KeypadEnter)
        JST_SURFACE_KEY(KeypadEqual)
        JST_SURFACE_KEY(AppBack)
        JST_SURFACE_KEY(AppForward)
        JST_SURFACE_KEY(Oem102)
#undef JST_SURFACE_KEY
        default: return KeyCode::Unknown;
    }
}

KeyModifiers SurfaceKeyModifiers() {
    const auto& io = ImGui::GetIO();
    return {io.ConfigMacOSXBehaviors ? io.KeySuper : io.KeyCtrl,
            io.KeyShift, io.KeyAlt,
            io.ConfigMacOSXBehaviors ? io.KeyCtrl : io.KeySuper};
}

void UpdateSurfaceModifiers(KeyModifiers& modifiers, ImGuiKey key, bool down, bool macOS) {
    switch (SurfaceNativeKey(key, macOS)) {
        case ImGuiMod_Ctrl: modifiers.control = down; break;
        case ImGuiMod_Shift: modifiers.shift = down; break;
        case ImGuiMod_Alt: modifiers.alt = down; break;
        case ImGuiMod_Super: modifiers.super = down; break;
        default: break;
    }
}

}  // namespace

SurfaceInputState::~SurfaceInputState() {
    removeHooks();
    if (callback) reset(callback);
}

void SurfaceInputState::observe(std::function<void(const InputEvent&)> emit) {
    auto* current = ImGui::GetCurrentContext();
    if (context && context != current) {
        reset(callback);
        removeHooks();
    }
    callback = std::move(emit);
    lastFrame = current->FrameCount;
    if (context) return;

    context = current;
    ImGuiContextHook hook;
    hook.UserData = this;
    hook.Type = ImGuiContextHookType_EndFramePost;
    hook.Callback = [](ImGuiContext* ctx, ImGuiContextHook* hook) {
        auto& state = *static_cast<SurfaceInputState*>(hook->UserData);
        if (state.lastFrame != ctx->FrameCount) {
            state.reset(state.callback);
        }
    };
    endFrameHook = ImGui::AddContextHook(context, &hook);
    hook.Type = ImGuiContextHookType_Shutdown;
    hook.Callback = [](ImGuiContext*, ImGuiContextHook* hook) {
        auto& state = *static_cast<SurfaceInputState*>(hook->UserData);
        state.context = nullptr;
        state.reset(state.callback);
    };
    shutdownHook = ImGui::AddContextHook(context, &hook);
}

void SurfaceInputState::reset(const std::function<void(const InputEvent&)>& emit) {
    for (int key = ImGuiKey_Tab; key <= ImGuiKey_Oem102; ++key) {
        if (heldKeys[key - ImGuiKey_NamedKey_BEGIN]) {
            emit(InputEvent{KeyEvent{KeyEventType::Release,
                                    SurfaceKeyCode(static_cast<ImGuiKey>(key), macOS), {}}});
        }
    }
    heldKeys.fill(false);
    if (focused) {
        emit(InputEvent{FocusEvent{false}});
    }
    focused = false;
}

void SurfaceInputState::removeHooks() {
    if (!context) return;
    ImGui::RemoveContextHook(context, endFrameHook);
    ImGui::RemoveContextHook(context, shutdownHook);
    context = nullptr;
}

void ForwardSurfaceInputEvents(const ImVec2& origin, const ImVec2& size,
                               SurfaceInputState& state,
                               const std::function<void(const InputEvent&)>& emit) {
    state.observe(emit);
    if (!std::isfinite(size.x) || !std::isfinite(size.y) || size.x <= 0.0f || size.y <= 0.0f) {
        state.reset(emit);
        return;
    }

    auto& g = *ImGui::GetCurrentContext();
    const auto& io = ImGui::GetIO();
    state.macOS = io.ConfigMacOSXBehaviors;
    const auto id = ImGui::GetItemID();
    const bool hovered = ImGui::IsItemHovered();
    const bool mouseEligible = hovered || ImGui::IsItemActive() || ImGui::IsItemDeactivated();
    const bool clicked = hovered && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                                     ImGui::IsMouseClicked(ImGuiMouseButton_Right));
    bool outsideClick = false;
    for (int button = 0; button < ImGuiMouseButton_COUNT; ++button) {
        outsideClick |= !hovered && ImGui::IsMouseClicked(button);
    }
    const bool windowFocused = ImGui::IsWindowFocused() && !io.AppFocusLost &&
                               (g.ActiveId == 0 || g.ActiveId == id);
    const bool focused = (state.focused || clicked) && ImGui::IsItemFocused() &&
                         windowFocused && !outsideClick;
    const bool spaceFocused = windowFocused && !io.WantTextInput && g.OpenPopupStack.empty() &&
                              ImGui::SetShortcutRouting(ImGuiKey_Space, ImGuiInputFlags_RouteFocused, id);
    if (!focused && state.focused) {
        state.reset(emit);
    }
    auto& spaceHeld = state.heldKeys[ImGuiKey_Space - ImGuiKey_NamedKey_BEGIN];
    if (!spaceFocused && spaceHeld) {
        emit(KeyEvent{KeyEventType::Release, KeyCode::Space, {}});
        spaceHeld = false;
    }

    const auto position = ImGui::GetMousePos();
    MouseEvent mouse{};
    mouse.position = {(position.x - origin.x) / size.x, (position.y - origin.y) / size.y};
    bool buttonPending = mouseEligible;
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        mouse.type = MouseEventType::Click;
        mouse.button = MouseButton::Left;
    } else if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        mouse.type = MouseEventType::Click;
        mouse.button = MouseButton::Right;
    } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        mouse.type = MouseEventType::Release;
        mouse.button = MouseButton::Left;
    } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        mouse.type = MouseEventType::Release;
        mouse.button = MouseButton::Right;
    } else {
        buttonPending = false;
    }
    bool wheelPending = hovered && (io.MouseWheel != 0.0f || io.MouseWheelH != 0.0f);

    auto modifiers = SurfaceKeyModifiers();
    for (int i = g.InputEventsTrail.Size - 1; i >= 0; --i) {
        const auto& event = g.InputEventsTrail[i];
        if (event.Type == ImGuiInputEventType_Key) {
            UpdateSurfaceModifiers(modifiers, event.Key.Key, !event.Key.Down, state.macOS);
        }
    }
    const auto emitButton = [&] {
        if (focused && !state.focused) {
            state.focused = true;
            emit(InputEvent{FocusEvent{true}});
        }
        mouse.modifiers = modifiers;
        emit(InputEvent{mouse});
        buttonPending = false;
    };
    const auto emitWheel = [&](const Extent2D<F32>& scroll) {
        auto wheel = mouse;
        wheel.type = MouseEventType::Scroll;
        wheel.scroll = scroll;
        wheel.modifiers = modifiers;
        emit(InputEvent{wheel});
        wheelPending = false;
    };
    for (const auto& event : g.InputEventsTrail) {
        if (event.Type == ImGuiInputEventType_Key) {
            UpdateSurfaceModifiers(modifiers, event.Key.Key, event.Key.Down, state.macOS);
            const auto key = SurfaceKeyCode(event.Key.Key, state.macOS);
            if (key == KeyCode::Unknown || (key == KeyCode::Space ? !spaceFocused : !state.focused)) continue;
            auto& held = state.heldKeys[event.Key.Key - ImGuiKey_NamedKey_BEGIN];
            if (held == event.Key.Down) continue;
            held = event.Key.Down;
            emit(InputEvent{KeyEvent{held ? KeyEventType::Press : KeyEventType::Release, key, modifiers}});
        } else if (event.Type == ImGuiInputEventType_MouseButton && buttonPending &&
                   event.MouseButton.Button == static_cast<int>(mouse.button) &&
                   event.MouseButton.Down == (mouse.type == MouseEventType::Click)) {
            emitButton();
        } else if (event.Type == ImGuiInputEventType_MouseWheel && hovered) {
            emitWheel({event.MouseWheel.WheelX, event.MouseWheel.WheelY});
        }
    }

    modifiers = SurfaceKeyModifiers();
    if (buttonPending) emitButton();
    if (wheelPending) emitWheel({io.MouseWheelH, io.MouseWheel});
    if (state.focused || spaceFocused) {
        for (int value = ImGuiKey_Tab; value <= ImGuiKey_Oem102; ++value) {
            const auto key = static_cast<ImGuiKey>(value);
            if (key == ImGuiKey_Space ? !spaceFocused : !state.focused) continue;
            ImGui::SetKeyOwner(key, id);
            if (state.heldKeys[value - ImGuiKey_NamedKey_BEGIN] &&
                ImGui::GetKeyData(key)->DownDuration > 0.0f &&
                ImGui::IsKeyPressed(key, ImGuiInputFlags_Repeat, id)) {
                emit(InputEvent{KeyEvent{KeyEventType::Press, SurfaceKeyCode(key, state.macOS), modifiers, true}});
            }
        }
    }
    if (mouseEligible) {
        mouse.type = MouseEventType::Move;
        mouse.scroll = {0.0f, 0.0f};
        mouse.modifiers = modifiers;
        emit(InputEvent{mouse});
    }
}

}  // namespace Jetstream::detail
