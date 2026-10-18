#include <jetstream/render/sakura/clipboard.hh>

#include "input.hh"

#include "jetstream/render/tools/imgui_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

namespace Jetstream::Sakura {

namespace {

constexpr std::array<MouseButton, 3> kButtons = {MouseButton::Left, MouseButton::Right, MouseButton::Middle};

struct InputSource {
    InputFrame frame;
    struct Click {
        F64 time = -std::numeric_limits<F64>::infinity();
        Extent2D<F32> position{};
        U8 count = 0;
    };
    std::array<Click, 3> clicks;
};

void DestroyInputSource(ImGuiContext*, ImGuiContextHook* hook) {
    auto* source = static_cast<InputSource*>(hook->UserData);
    hook->UserData = nullptr;
    delete source;
}

InputSource* GetInputSource(ImGuiContext* context) {
    if (!context || !context->Initialized || context->FontAtlases.empty()) return nullptr;
    for (const auto& hook : context->Hooks) {
        if (hook.Callback == DestroyInputSource) return static_cast<InputSource*>(hook.UserData);
    }
    static std::atomic<U64> generation{0};
    auto* source = new InputSource;
    source->frame.source = context;
    source->frame.sourceGeneration = ++generation;
    ImGuiContextHook hook;
    hook.Type = ImGuiContextHookType_Shutdown;
    hook.Callback = DestroyInputSource;
    hook.UserData = source;
    ImGui::AddContextHook(context, &hook);
    return source;
}

ImGuiKey NativeKey(ImGuiKey key, bool macOS) {
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

void UpdateModifiers(KeyModifiers& modifiers, ImGuiKey key, bool down, bool macOS) {
    switch (NativeKey(key, macOS)) {
        case ImGuiMod_Ctrl: modifiers.control = down; break;
        case ImGuiMod_Shift: modifiers.shift = down; break;
        case ImGuiMod_Alt: modifiers.alt = down; break;
        case ImGuiMod_Super: modifiers.super = down; break;
        default: break;
    }
}

KeyCode ToKeyCode(ImGuiKey key, bool macOS) {
    key = NativeKey(key, macOS);
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
#define JST_BRIDGE_KEY(name) case ImGuiKey_##name: return KeyCode::name;
        JST_BRIDGE_KEY(Tab)
        JST_BRIDGE_KEY(LeftArrow)
        JST_BRIDGE_KEY(RightArrow)
        JST_BRIDGE_KEY(UpArrow)
        JST_BRIDGE_KEY(DownArrow)
        JST_BRIDGE_KEY(PageUp)
        JST_BRIDGE_KEY(PageDown)
        JST_BRIDGE_KEY(Home)
        JST_BRIDGE_KEY(End)
        JST_BRIDGE_KEY(Insert)
        JST_BRIDGE_KEY(Delete)
        JST_BRIDGE_KEY(Backspace)
        JST_BRIDGE_KEY(Space)
        JST_BRIDGE_KEY(Enter)
        JST_BRIDGE_KEY(Escape)
        JST_BRIDGE_KEY(LeftCtrl)
        JST_BRIDGE_KEY(LeftShift)
        JST_BRIDGE_KEY(LeftAlt)
        JST_BRIDGE_KEY(LeftSuper)
        JST_BRIDGE_KEY(RightCtrl)
        JST_BRIDGE_KEY(RightShift)
        JST_BRIDGE_KEY(RightAlt)
        JST_BRIDGE_KEY(RightSuper)
        JST_BRIDGE_KEY(Menu)
        JST_BRIDGE_KEY(Apostrophe)
        JST_BRIDGE_KEY(Comma)
        JST_BRIDGE_KEY(Minus)
        JST_BRIDGE_KEY(Period)
        JST_BRIDGE_KEY(Slash)
        JST_BRIDGE_KEY(Semicolon)
        JST_BRIDGE_KEY(Equal)
        JST_BRIDGE_KEY(LeftBracket)
        JST_BRIDGE_KEY(Backslash)
        JST_BRIDGE_KEY(RightBracket)
        JST_BRIDGE_KEY(GraveAccent)
        JST_BRIDGE_KEY(CapsLock)
        JST_BRIDGE_KEY(ScrollLock)
        JST_BRIDGE_KEY(NumLock)
        JST_BRIDGE_KEY(PrintScreen)
        JST_BRIDGE_KEY(Pause)
        JST_BRIDGE_KEY(KeypadDecimal)
        JST_BRIDGE_KEY(KeypadDivide)
        JST_BRIDGE_KEY(KeypadMultiply)
        JST_BRIDGE_KEY(KeypadSubtract)
        JST_BRIDGE_KEY(KeypadAdd)
        JST_BRIDGE_KEY(KeypadEnter)
        JST_BRIDGE_KEY(KeypadEqual)
        JST_BRIDGE_KEY(AppBack)
        JST_BRIDGE_KEY(AppForward)
        JST_BRIDGE_KEY(Oem102)
#undef JST_BRIDGE_KEY
        default: return KeyCode::Unknown;
    }
}

KeyModifiers CurrentModifiers() {
    const auto& io = ImGui::GetIO();
    return {io.ConfigMacOSXBehaviors ? io.KeySuper : io.KeyCtrl,
            io.KeyShift, io.KeyAlt,
            io.ConfigMacOSXBehaviors ? io.KeyCtrl : io.KeySuper};
}

ImGuiMouseCursor ToImGuiMouseCursor(const SurfaceCursor cursor) {
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

}  // namespace

void SetClipboardText(const std::string& value) {
    ImGui::SetClipboardText(value.c_str());
}

std::string GetClipboardText() {
    if (!ImGui::GetCurrentContext()) return {};
    const char* text = ImGui::GetClipboardText();
    return text ? std::string(text) : std::string();
}

namespace Private {

void SampleInput(InputFrame& destination) {
    ImGuiContext* context = ImGui::GetCurrentContext();
    auto* cached = GetInputSource(context);
    if (!cached) {
        destination.advance(0.0f);
        destination.source = nullptr;
        destination.sourceGeneration = 0;
        destination.sourceFrame = -1;
        destination.windowFocused = false;
        destination.pointer.reset();
        destination.buttons.fill(false);
        destination.modifiers = {};
        destination.events.push_back(MouseEvent{.type = MouseEventType::Cancel});
        return;
    }
    auto& source = *cached;
    auto& frame = source.frame;
    if (frame.sourceFrame == context->FrameCount) {
        destination = frame;
        return;
    }
    frame.sourceFrame = context->FrameCount;

    const ImGuiIO& io = ImGui::GetIO();
    const bool macOS = io.ConfigMacOSXBehaviors;

    bool focused = frame.windowFocused;
    bool pointerCancelled = !frame.pointer.has_value();
    frame.advance(io.DeltaTime);
    frame.time = ImGui::GetTime();
    frame.windowFocused = !io.AppFocusLost;
    frame.modifiers = CurrentModifiers();

    const ImVec2 position = io.MousePos;
    frame.pointer.reset();
    const auto valid = [](const ImVec2& point) {
        return ImGui::IsMousePosValid(&point) && std::isfinite(point.x) && std::isfinite(point.y);
    };
    if (valid(position) && frame.windowFocused) {
        frame.pointer = Extent2D<F32>{position.x, position.y};
    }
    ImVec2 point = position;
    for (U64 index = 0; index < kButtons.size(); ++index) {
        frame.buttons[index] = frame.windowFocused && io.MouseDown[index];
    }

    const auto& trail = context->InputEventsTrail;
    for (const auto& event : trail) {
        if (event.Type == ImGuiInputEventType_MousePos) {
            point = io.MousePosPrev;
            break;
        }
    }
    KeyModifiers modifiers = frame.modifiers;
    for (int index = trail.Size - 1; index >= 0; --index) {
        if (trail[index].Type == ImGuiInputEventType_Key) {
            UpdateModifiers(modifiers, trail[index].Key.Key, !trail[index].Key.Down, macOS);
        }
    }

    const auto button = [&](U64 index, bool down) {
        if (!focused || !valid(point)) return;
        MouseEvent event{};
        event.type = down ? MouseEventType::Click : MouseEventType::Release;
        event.button = kButtons[index];
        event.position = {point.x, point.y};
        event.modifiers = modifiers;
        if (down) {
            auto& click = source.clicks[index];
            const F32 dx = point.x - click.position.x;
            const F32 dy = point.y - click.position.y;
            const bool repeated = frame.time - click.time < io.MouseDoubleClickTime &&
                                  dx * dx + dy * dy < io.MouseDoubleClickMaxDist * io.MouseDoubleClickMaxDist;
            click.count = repeated ? static_cast<U8>(std::min<int>(click.count + 1,
                                                     std::numeric_limits<U8>::max())) : 1;
            click.time = frame.time;
            click.position = event.position;
            event.clicks = click.count;
        }
        frame.events.push_back(event);
    };
    const auto wheel = [&](F32 x, F32 y) {
        if (!focused || !valid(point)) return;
        MouseEvent event{};
        event.type = MouseEventType::Scroll;
        event.position = {point.x, point.y};
        event.scroll = {x, y};
        event.modifiers = modifiers;
        frame.events.push_back(event);
    };

    std::array<bool, 3> clicked{};
    std::array<bool, 3> released{};
    bool scrolled = false;
    for (const auto& event : trail) {
        switch (event.Type) {
            case ImGuiInputEventType_MousePos:
                point = {event.MousePos.PosX, event.MousePos.PosY};
                pointerCancelled = !valid(point);
                if (focused) {
                    frame.events.push_back(MouseEvent{
                        .type = valid(point) ? MouseEventType::Move : MouseEventType::Cancel,
                        .position = valid(point) ? Extent2D<F32>{point.x, point.y} : Extent2D<F32>{},
                        .modifiers = modifiers,
                    });
                }
                break;
            case ImGuiInputEventType_Focus:
                focused = event.AppFocused.Focused;
                frame.events.push_back(FocusEvent{focused});
                if (!focused) source.clicks = {};
                break;
            case ImGuiInputEventType_Key: {
                UpdateModifiers(modifiers, event.Key.Key, event.Key.Down, macOS);
                const KeyCode code = ToKeyCode(event.Key.Key, macOS);
                if (focused && code != KeyCode::Unknown) {
                    frame.events.push_back(KeyEvent{event.Key.Down ? KeyEventType::Press : KeyEventType::Release,
                                                    code, modifiers});
                }
                break;
            }
            case ImGuiInputEventType_MouseButton: {
                const int index = event.MouseButton.Button;
                if (index < 0 || index >= static_cast<int>(kButtons.size())) {
                    break;
                }
                button(static_cast<U64>(index), event.MouseButton.Down);
                (event.MouseButton.Down ? clicked : released)[static_cast<U64>(index)] = true;
                break;
            }
            case ImGuiInputEventType_MouseWheel:
                wheel(event.MouseWheel.WheelX, event.MouseWheel.WheelY);
                scrolled = true;
                break;
            case ImGuiInputEventType_Text:
                if (focused) frame.events.push_back(TextEvent{static_cast<char32_t>(event.Text.Char), modifiers});
                break;
            default:
                break;
        }
    }

    modifiers = frame.modifiers;
    point = position;
    focused = frame.windowFocused;
    for (U64 index = 0; index < kButtons.size(); ++index) {
        if (!clicked[index] && ImGui::IsMouseClicked(static_cast<int>(index))) {
            button(index, true);
        }
        if (!released[index] && ImGui::IsMouseReleased(static_cast<int>(index))) {
            button(index, false);
        }
    }
    if (!scrolled && (io.MouseWheel != 0.0f || io.MouseWheelH != 0.0f)) {
        wheel(io.MouseWheelH, io.MouseWheel);
    }

    for (int value = ImGuiKey_NamedKey_BEGIN; value < ImGuiKey_NamedKey_END; ++value) {
        const auto key = static_cast<ImGuiKey>(value);
        const KeyCode code = ToKeyCode(key, macOS);
        if (focused && code != KeyCode::Unknown && ImGui::IsKeyDown(key) && ImGui::GetKeyData(key)->DownDuration > 0.0f &&
            ImGui::IsKeyPressed(key, true)) {
            frame.events.push_back(KeyEvent{KeyEventType::Press, code, frame.modifiers, true});
        }
    }
    if (frame.windowFocused && !frame.pointer && !pointerCancelled) {
        frame.events.push_back(MouseEvent{.type = MouseEventType::Cancel});
    }
    destination = frame;
}

void ApplyCursor(SurfaceCursor cursor) {
    if (cursor != SurfaceCursor::Default) {
        ImGui::SetMouseCursor(ToImGuiMouseCursor(cursor));
    }
}

}  // namespace Private

}  // namespace Jetstream::Sakura
