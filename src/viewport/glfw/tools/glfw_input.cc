#include "glfw_input.hh"

#ifdef JST_OS_MAC
#include "cocoa_pointer.hh"
#endif

#ifdef JST_OS_BROWSER
#include "browser_wheel.hh"
#endif

#include <GLFW/glfw3.h>

#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Viewport {

namespace {

constexpr U64 kCursorCount = static_cast<U64>(SurfaceCursor::NotAllowed) + 1;

KeyModifiers CurrentModifiers(GLFWwindow* window) {
    const auto down = [&](int left, int right) {
        return glfwGetKey(window, left) == GLFW_PRESS || glfwGetKey(window, right) == GLFW_PRESS;
    };
    return {
        .control = down(GLFW_KEY_LEFT_CONTROL, GLFW_KEY_RIGHT_CONTROL),
        .shift = down(GLFW_KEY_LEFT_SHIFT, GLFW_KEY_RIGHT_SHIFT),
        .alt = down(GLFW_KEY_LEFT_ALT, GLFW_KEY_RIGHT_ALT),
        .super = down(GLFW_KEY_LEFT_SUPER, GLFW_KEY_RIGHT_SUPER),
    };
}

int UntranslateKey(int key, int scancode) {
    if (key >= GLFW_KEY_KP_0 && key <= GLFW_KEY_KP_EQUAL) {
        return key;
    }
    GLFWerrorfun previous = glfwSetErrorCallback(nullptr);
    const char* name = glfwGetKeyName(key, scancode);
    glfwSetErrorCallback(previous);
#ifndef __EMSCRIPTEN__
    (void)glfwGetError(nullptr);
#endif
    if (name && name[0] != 0 && name[1] == 0) {
        static constexpr char kNames[] = "`-=[]\\,;\'./";
        static constexpr std::array<int, 11> kKeys = {
            GLFW_KEY_GRAVE_ACCENT, GLFW_KEY_MINUS, GLFW_KEY_EQUAL, GLFW_KEY_LEFT_BRACKET,
            GLFW_KEY_RIGHT_BRACKET, GLFW_KEY_BACKSLASH, GLFW_KEY_COMMA, GLFW_KEY_SEMICOLON,
            GLFW_KEY_APOSTROPHE, GLFW_KEY_PERIOD, GLFW_KEY_SLASH,
        };
        if (name[0] >= '0' && name[0] <= '9') {
            key = GLFW_KEY_0 + (name[0] - '0');
        } else if (name[0] >= 'A' && name[0] <= 'Z') {
            key = GLFW_KEY_A + (name[0] - 'A');
        } else if (name[0] >= 'a' && name[0] <= 'z') {
            key = GLFW_KEY_A + (name[0] - 'a');
        } else if (const char* match = std::strchr(kNames, name[0])) {
            key = kKeys[static_cast<U64>(match - kNames)];
        }
    }
    return key;
}

KeyCode ToKeyCode(int key) {
    switch (key) {
        case GLFW_KEY_TAB: return KeyCode::Tab;
        case GLFW_KEY_LEFT: return KeyCode::LeftArrow;
        case GLFW_KEY_RIGHT: return KeyCode::RightArrow;
        case GLFW_KEY_UP: return KeyCode::UpArrow;
        case GLFW_KEY_DOWN: return KeyCode::DownArrow;
        case GLFW_KEY_PAGE_UP: return KeyCode::PageUp;
        case GLFW_KEY_PAGE_DOWN: return KeyCode::PageDown;
        case GLFW_KEY_HOME: return KeyCode::Home;
        case GLFW_KEY_END: return KeyCode::End;
        case GLFW_KEY_INSERT: return KeyCode::Insert;
        case GLFW_KEY_DELETE: return KeyCode::Delete;
        case GLFW_KEY_BACKSPACE: return KeyCode::Backspace;
        case GLFW_KEY_SPACE: return KeyCode::Space;
        case GLFW_KEY_ENTER: return KeyCode::Enter;
        case GLFW_KEY_ESCAPE: return KeyCode::Escape;
        case GLFW_KEY_APOSTROPHE: return KeyCode::Apostrophe;
        case GLFW_KEY_COMMA: return KeyCode::Comma;
        case GLFW_KEY_MINUS: return KeyCode::Minus;
        case GLFW_KEY_PERIOD: return KeyCode::Period;
        case GLFW_KEY_SLASH: return KeyCode::Slash;
        case GLFW_KEY_SEMICOLON: return KeyCode::Semicolon;
        case GLFW_KEY_EQUAL: return KeyCode::Equal;
        case GLFW_KEY_LEFT_BRACKET: return KeyCode::LeftBracket;
        case GLFW_KEY_BACKSLASH: return KeyCode::Backslash;
        case GLFW_KEY_WORLD_1: return KeyCode::Oem102;
        case GLFW_KEY_WORLD_2: return KeyCode::Oem102;
        case GLFW_KEY_RIGHT_BRACKET: return KeyCode::RightBracket;
        case GLFW_KEY_GRAVE_ACCENT: return KeyCode::GraveAccent;
        case GLFW_KEY_CAPS_LOCK: return KeyCode::CapsLock;
        case GLFW_KEY_SCROLL_LOCK: return KeyCode::ScrollLock;
        case GLFW_KEY_NUM_LOCK: return KeyCode::NumLock;
        case GLFW_KEY_PRINT_SCREEN: return KeyCode::PrintScreen;
        case GLFW_KEY_PAUSE: return KeyCode::Pause;
        case GLFW_KEY_KP_0: return KeyCode::Keypad0;
        case GLFW_KEY_KP_1: return KeyCode::Keypad1;
        case GLFW_KEY_KP_2: return KeyCode::Keypad2;
        case GLFW_KEY_KP_3: return KeyCode::Keypad3;
        case GLFW_KEY_KP_4: return KeyCode::Keypad4;
        case GLFW_KEY_KP_5: return KeyCode::Keypad5;
        case GLFW_KEY_KP_6: return KeyCode::Keypad6;
        case GLFW_KEY_KP_7: return KeyCode::Keypad7;
        case GLFW_KEY_KP_8: return KeyCode::Keypad8;
        case GLFW_KEY_KP_9: return KeyCode::Keypad9;
        case GLFW_KEY_KP_DECIMAL: return KeyCode::KeypadDecimal;
        case GLFW_KEY_KP_DIVIDE: return KeyCode::KeypadDivide;
        case GLFW_KEY_KP_MULTIPLY: return KeyCode::KeypadMultiply;
        case GLFW_KEY_KP_SUBTRACT: return KeyCode::KeypadSubtract;
        case GLFW_KEY_KP_ADD: return KeyCode::KeypadAdd;
        case GLFW_KEY_KP_ENTER: return KeyCode::KeypadEnter;
        case GLFW_KEY_KP_EQUAL: return KeyCode::KeypadEqual;
        case GLFW_KEY_LEFT_SHIFT: return KeyCode::LeftShift;
        case GLFW_KEY_LEFT_CONTROL: return KeyCode::LeftCtrl;
        case GLFW_KEY_LEFT_ALT: return KeyCode::LeftAlt;
        case GLFW_KEY_LEFT_SUPER: return KeyCode::LeftSuper;
        case GLFW_KEY_RIGHT_SHIFT: return KeyCode::RightShift;
        case GLFW_KEY_RIGHT_CONTROL: return KeyCode::RightCtrl;
        case GLFW_KEY_RIGHT_ALT: return KeyCode::RightAlt;
        case GLFW_KEY_RIGHT_SUPER: return KeyCode::RightSuper;
        case GLFW_KEY_MENU: return KeyCode::Menu;
        case GLFW_KEY_0: return KeyCode::Digit0;
        case GLFW_KEY_1: return KeyCode::Digit1;
        case GLFW_KEY_2: return KeyCode::Digit2;
        case GLFW_KEY_3: return KeyCode::Digit3;
        case GLFW_KEY_4: return KeyCode::Digit4;
        case GLFW_KEY_5: return KeyCode::Digit5;
        case GLFW_KEY_6: return KeyCode::Digit6;
        case GLFW_KEY_7: return KeyCode::Digit7;
        case GLFW_KEY_8: return KeyCode::Digit8;
        case GLFW_KEY_9: return KeyCode::Digit9;
        case GLFW_KEY_A: return KeyCode::A;
        case GLFW_KEY_B: return KeyCode::B;
        case GLFW_KEY_C: return KeyCode::C;
        case GLFW_KEY_D: return KeyCode::D;
        case GLFW_KEY_E: return KeyCode::E;
        case GLFW_KEY_F: return KeyCode::F;
        case GLFW_KEY_G: return KeyCode::G;
        case GLFW_KEY_H: return KeyCode::H;
        case GLFW_KEY_I: return KeyCode::I;
        case GLFW_KEY_J: return KeyCode::J;
        case GLFW_KEY_K: return KeyCode::K;
        case GLFW_KEY_L: return KeyCode::L;
        case GLFW_KEY_M: return KeyCode::M;
        case GLFW_KEY_N: return KeyCode::N;
        case GLFW_KEY_O: return KeyCode::O;
        case GLFW_KEY_P: return KeyCode::P;
        case GLFW_KEY_Q: return KeyCode::Q;
        case GLFW_KEY_R: return KeyCode::R;
        case GLFW_KEY_S: return KeyCode::S;
        case GLFW_KEY_T: return KeyCode::T;
        case GLFW_KEY_U: return KeyCode::U;
        case GLFW_KEY_V: return KeyCode::V;
        case GLFW_KEY_W: return KeyCode::W;
        case GLFW_KEY_X: return KeyCode::X;
        case GLFW_KEY_Y: return KeyCode::Y;
        case GLFW_KEY_Z: return KeyCode::Z;
        case GLFW_KEY_F1: return KeyCode::F1;
        case GLFW_KEY_F2: return KeyCode::F2;
        case GLFW_KEY_F3: return KeyCode::F3;
        case GLFW_KEY_F4: return KeyCode::F4;
        case GLFW_KEY_F5: return KeyCode::F5;
        case GLFW_KEY_F6: return KeyCode::F6;
        case GLFW_KEY_F7: return KeyCode::F7;
        case GLFW_KEY_F8: return KeyCode::F8;
        case GLFW_KEY_F9: return KeyCode::F9;
        case GLFW_KEY_F10: return KeyCode::F10;
        case GLFW_KEY_F11: return KeyCode::F11;
        case GLFW_KEY_F12: return KeyCode::F12;
        case GLFW_KEY_F13: return KeyCode::F13;
        case GLFW_KEY_F14: return KeyCode::F14;
        case GLFW_KEY_F15: return KeyCode::F15;
        case GLFW_KEY_F16: return KeyCode::F16;
        case GLFW_KEY_F17: return KeyCode::F17;
        case GLFW_KEY_F18: return KeyCode::F18;
        case GLFW_KEY_F19: return KeyCode::F19;
        case GLFW_KEY_F20: return KeyCode::F20;
        case GLFW_KEY_F21: return KeyCode::F21;
        case GLFW_KEY_F22: return KeyCode::F22;
        case GLFW_KEY_F23: return KeyCode::F23;
        case GLFW_KEY_F24: return KeyCode::F24;
        default: return KeyCode::Unknown;
    }
}

void WakeEvents() {
#ifndef __EMSCRIPTEN__
    glfwPostEmptyEvent();
#endif
}

KeyCode TranslateKey(int key, int scancode) {
    return ToKeyCode(UntranslateKey(key, scancode));
}

int ToStandardCursor(SurfaceCursor cursor) {
    switch (cursor) {
        case SurfaceCursor::Hand: return GLFW_HAND_CURSOR;
        case SurfaceCursor::ResizeEW: return GLFW_HRESIZE_CURSOR;
        case SurfaceCursor::ResizeNS: return GLFW_VRESIZE_CURSOR;
        case SurfaceCursor::ResizeAll: return GLFW_RESIZE_ALL_CURSOR;
        case SurfaceCursor::ResizeNESW: return GLFW_RESIZE_NESW_CURSOR;
        case SurfaceCursor::ResizeNWSE: return GLFW_RESIZE_NWSE_CURSOR;
        case SurfaceCursor::TextInput: return GLFW_IBEAM_CURSOR;
        case SurfaceCursor::NotAllowed: return GLFW_NOT_ALLOWED_CURSOR;
        default: return GLFW_ARROW_CURSOR;
    }
}

}  // namespace

struct GlfwInput::Impl {
    GLFWwindow* window = nullptr;
    Generic& viewport;
    std::atomic<bool> pointerInside = false;
    std::array<GLFWcursor*, kCursorCount> cursors{};
    std::optional<SurfaceCursor> cursor;
    std::atomic<SurfaceCursor> requestedCursor = SurfaceCursor::Default;
    mutable std::mutex clipboardMutex;
    std::string clipboard;
    std::optional<std::string> pendingClipboard;
#ifdef JST_OS_MAC
    std::unique_ptr<CocoaPointer> pointer;
#endif
#ifdef JST_OS_BROWSER
    std::unique_ptr<BrowserWheel> wheel;
#endif

    Impl(GLFWwindow* window, Generic& viewport) : window(window), viewport(viewport) {}

    Render::InputQueue::Wheel takeWheel(const Extent2D<F32>& delta) {
        Render::InputQueue::Wheel event{.delta = delta};
#if defined(JST_OS_MAC)
        if (pointer) {
            if (const auto scroll = pointer->take()) {
                viewport.input().push(Render::InputQueue::Modifiers{scroll->modifiers});
                event.phase = scroll->phase;
            }
        }
#elif defined(JST_OS_BROWSER)
        event.precise = wheel && wheel->take(event.delta);
#endif
        return event;
    }

    void readClipboard() {
        GLFWerrorfun previous = glfwSetErrorCallback(nullptr);
        const char* text = glfwGetClipboardString(nullptr);
        glfwSetErrorCallback(previous);
#ifndef __EMSCRIPTEN__
        (void)glfwGetError(nullptr);
#endif
        std::lock_guard lock(clipboardMutex);
        clipboard = text ? std::string(text) : std::string();
    }

    void applyCursor() {
        const SurfaceCursor next = requestedCursor.load();
        if (cursor == next) {
            return;
        }
        cursor = next;
        GLFWcursor* handle = cursors[static_cast<U64>(next)];
        if (!handle) {
            handle = cursors[static_cast<U64>(SurfaceCursor::Default)];
        }
        glfwSetCursor(window, handle);
    }

    void applyClipboard() {
        std::optional<std::string> text;
        {
            std::lock_guard lock(clipboardMutex);
            text = std::exchange(pendingClipboard, std::nullopt);
        }
        if (text.has_value()) {
            glfwSetClipboardString(nullptr, text->c_str());
        }
    }

    static Impl& From(GLFWwindow* window) {
        return *static_cast<Impl*>(glfwGetWindowUserPointer(window));
    }

    void pushModifiers() {
        viewport.input().push(Render::InputQueue::Modifiers{CurrentModifiers(window)});
    }

    static void OnFocus(GLFWwindow* window, int focused) {
        auto& impl = From(window);
        if (focused != 0) {
            impl.readClipboard();
        }
        impl.viewport.input().push(Render::InputQueue::Focus{focused != 0});
    }

    static void OnEnter(GLFWwindow* window, int entered) {
        auto& impl = From(window);
        impl.pointerInside = entered != 0;
        impl.viewport.input().push(Render::InputQueue::Hover{entered != 0});
    }

    static void OnCursor(GLFWwindow* window, double x, double y) {
        auto& impl = From(window);
        impl.viewport.input().push(Render::InputQueue::Pointer{
            Extent2D<F32>{static_cast<F32>(x), static_cast<F32>(y)}});
    }

    static void OnButton(GLFWwindow* window, int button, int action, int) {
        auto& impl = From(window);
        if (action == GLFW_PRESS) {
            impl.readClipboard();
        }
        impl.pushModifiers();
        if (const auto pressed = Render::InputQueue::ButtonFromIndex(button)) {
            impl.viewport.input().push(Render::InputQueue::Button{*pressed, action == GLFW_PRESS});
        }
    }

    static void OnScroll(GLFWwindow* window, double x, double y) {
        auto& impl = From(window);
        impl.viewport.input().push(impl.takeWheel({static_cast<F32>(x), static_cast<F32>(y)}));
    }

    static void OnKey(GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto& impl = From(window);
        if (action != GLFW_PRESS && action != GLFW_RELEASE) {
            return;
        }
        if (action == GLFW_PRESS && (mods & (GLFW_MOD_CONTROL | GLFW_MOD_SUPER)) != 0) {
            impl.readClipboard();
        }
        impl.pushModifiers();
        const KeyCode code = TranslateKey(key, scancode);
        if (code != KeyCode::Unknown) {
            impl.viewport.input().push(Render::InputQueue::Key{code, action == GLFW_PRESS});
        }
    }

    static void OnChar(GLFWwindow* window, unsigned int codepoint) {
        auto& impl = From(window);
        impl.viewport.input().push(Render::InputQueue::Text{static_cast<char32_t>(codepoint)});
    }

    static void OnRefresh(GLFWwindow* window) {
        auto& viewport = From(window).viewport;
        viewport.refreshSnapshot();
        viewport.requestRedraw();
    }

    static void OnResize(GLFWwindow* window, int, int) {
        OnRefresh(window);
    }

    static void OnScale(GLFWwindow* window, float, float) {
        OnRefresh(window);
    }

    static void OnIconify(GLFWwindow* window, int) {
        From(window).viewport.requestRedraw();
    }

    static void OnDrop(GLFWwindow* window, int count, const char** paths) {
        auto& impl = From(window);
        if (count <= 0 || !paths) {
            return;
        }
        std::vector<std::string> files;
        files.reserve(static_cast<U64>(count));
        for (int i = 0; i < count; ++i) {
            if (paths[i]) {
                files.emplace_back(paths[i]);
            }
        }
        (void)impl.viewport.addFileDropEvent(std::move(files));
    }
};

GlfwInput::GlfwInput(GLFWwindow* window, Generic& viewport) {
    impl = std::make_unique<Impl>(window, viewport);

    glfwSetWindowUserPointer(window, impl.get());
    glfwSetWindowFocusCallback(window, &Impl::OnFocus);
    glfwSetCursorEnterCallback(window, &Impl::OnEnter);
    glfwSetCursorPosCallback(window, &Impl::OnCursor);
    glfwSetMouseButtonCallback(window, &Impl::OnButton);
    glfwSetScrollCallback(window, &Impl::OnScroll);
    glfwSetKeyCallback(window, &Impl::OnKey);
    glfwSetCharCallback(window, &Impl::OnChar);
    glfwSetDropCallback(window, &Impl::OnDrop);
    glfwSetWindowRefreshCallback(window, &Impl::OnRefresh);
    glfwSetWindowSizeCallback(window, &Impl::OnResize);
    glfwSetFramebufferSizeCallback(window, &Impl::OnResize);
    glfwSetWindowContentScaleCallback(window, &Impl::OnScale);
    glfwSetWindowIconifyCallback(window, &Impl::OnIconify);

    GLFWerrorfun previous = glfwSetErrorCallback(nullptr);
    for (U64 index = 0; index < kCursorCount; ++index) {
        impl->cursors[index] = glfwCreateStandardCursor(ToStandardCursor(static_cast<SurfaceCursor>(index)));
    }
    glfwSetErrorCallback(previous);
#ifndef __EMSCRIPTEN__
    (void)glfwGetError(nullptr);
#endif
    impl->readClipboard();
#ifdef JST_OS_MAC
    impl->pointer = std::make_unique<CocoaPointer>(window, [input = impl.get()]() {
        Impl::OnScroll(input->window, 0.0, 0.0);
    });
#endif
#ifdef JST_OS_BROWSER
    impl->wheel = std::make_unique<BrowserWheel>();
#endif
}

GlfwInput::~GlfwInput() {
#ifdef JST_OS_MAC
    impl->pointer.reset();
#endif
#ifdef JST_OS_BROWSER
    impl->wheel.reset();
#endif
    glfwSetWindowFocusCallback(impl->window, nullptr);
    glfwSetCursorEnterCallback(impl->window, nullptr);
    glfwSetCursorPosCallback(impl->window, nullptr);
    glfwSetMouseButtonCallback(impl->window, nullptr);
    glfwSetScrollCallback(impl->window, nullptr);
    glfwSetKeyCallback(impl->window, nullptr);
    glfwSetCharCallback(impl->window, nullptr);
    glfwSetDropCallback(impl->window, nullptr);
    glfwSetWindowRefreshCallback(impl->window, nullptr);
    glfwSetWindowSizeCallback(impl->window, nullptr);
    glfwSetFramebufferSizeCallback(impl->window, nullptr);
    glfwSetWindowContentScaleCallback(impl->window, nullptr);
    glfwSetWindowIconifyCallback(impl->window, nullptr);
    glfwSetWindowUserPointer(impl->window, nullptr);

    for (GLFWcursor* cursor : impl->cursors) {
        if (cursor) {
            glfwDestroyCursor(cursor);
        }
    }
}

void GlfwInput::poll() {
    impl->applyCursor();
    impl->applyClipboard();
#ifdef __EMSCRIPTEN__
    impl->readClipboard();
#endif
#ifndef JST_OS_MAC
    if (glfwGetWindowAttrib(impl->window, GLFW_FOCUSED) == 0 || impl->pointerInside) {
        return;
    }
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(impl->window, &x, &y);
    impl->viewport.input().push(Render::InputQueue::Pointer{
        Extent2D<F32>{static_cast<F32>(x), static_cast<F32>(y)}});
#endif
}

void GlfwInput::setCursor(SurfaceCursor cursor) {
    if (impl->requestedCursor.exchange(cursor) != cursor) {
        WakeEvents();
    }
}

F64 GlfwInput::time() const {
    return glfwGetTime();
}

std::string GlfwInput::clipboardText() const {
    std::lock_guard lock(impl->clipboardMutex);
    return impl->clipboard;
}

void GlfwInput::setClipboardText(const std::string& text) {
    {
        std::lock_guard lock(impl->clipboardMutex);
        impl->clipboard = text;
        impl->pendingClipboard = text;
    }
    WakeEvents();
}

}  // namespace Jetstream::Viewport
