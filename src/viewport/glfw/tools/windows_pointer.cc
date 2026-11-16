#include "windows_pointer.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <commctrl.h>
#include <directmanipulation.h>
#include <wrl.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Jetstream::Viewport {

namespace {

constexpr F32 PointsPerWheelUnit = 10.0f;
constexpr F32 BaseDpi = 96.0f;
constexpr DWORD FreshTouchTicks = 250;
constexpr F64 FreshPollSeconds = 0.25;
constexpr ULONG_PTR PromotedMouseMask = 0xFFFFFF80;
constexpr ULONG_PTR PromotedTouchSignature = 0xFF515780;

F32 DpiScale(HWND window) {
    return std::max<F32>(1.0f, GetDpiForWindow(window) / BaseDpi);
}

bool FreshTouch(const POINTER_INFO& info) {
    return GetTickCount() - info.dwTime <= FreshTouchTicks;
}

struct ScrollSequence {
    std::function<void(const WindowsPointer::Scroll&)> emit;
    Extent2D<F32> previous = {0.0f, 0.0f};
    F32 scale = 1.0f;
    ScrollPhase phase = ScrollPhase::None;
    bool started = false;

    void send(ScrollPhase next, Extent2D<F32> delta = {0.0f, 0.0f}) {
        if (emit) {
            emit({delta, next, WindowsPointer::Modifiers()});
        }
    }

    void end() {
        if (started) {
            send(phase == ScrollPhase::Momentum ? ScrollPhase::MomentumEnded : ScrollPhase::Ended);
        }
        phase = ScrollPhase::None;
        started = false;
    }

    void running() {
        if (phase != ScrollPhase::Changed) {
            end();
            phase = ScrollPhase::Changed;
        }
    }

    void inertia() {
        if (phase == ScrollPhase::Changed && started) {
            send(ScrollPhase::Ended);
            phase = ScrollPhase::Momentum;
            send(ScrollPhase::Momentum);
        }
    }

    void move(F32 x, F32 y) {
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return;
        }
        const Extent2D<F32> delta = {(x - previous.x) / (PointsPerWheelUnit * scale),
                                     (y - previous.y) / (PointsPerWheelUnit * scale)};
        previous = {x, y};
        if (delta.x == 0.0f && delta.y == 0.0f) {
            return;
        }
        if (phase == ScrollPhase::None) {
            running();
        }
        send(started ? phase : ScrollPhase::Began, delta);
        started = true;
    }

    bool reset() {
        end();
        const bool translated = previous.x != 0.0f || previous.y != 0.0f;
        previous = {0.0f, 0.0f};
        return translated;
    }
};

using Microsoft::WRL::ComPtr;

class TouchpadHandler final : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IDirectManipulationViewportEventHandler, IDirectManipulationInteractionEventHandler> {
 public:
    ScrollSequence scroll;
    RECT bounds = {0, 0, 1, 1};
    bool updating = false;
    bool resetting = false;

    HRESULT STDMETHODCALLTYPE OnViewportStatusChanged(IDirectManipulationViewport* viewport,
        DIRECTMANIPULATION_STATUS current, DIRECTMANIPULATION_STATUS previous) override {
        if (current == previous) {
            return S_OK;
        }
        if (resetting) {
            if (current == DIRECTMANIPULATION_READY) {
                resetting = false;
            }
            return S_OK;
        }
        switch (current) {
            case DIRECTMANIPULATION_RUNNING:
                scroll.running();
                break;
            case DIRECTMANIPULATION_INERTIA:
                scroll.inertia();
                break;
            case DIRECTMANIPULATION_READY:
                if (scroll.reset()) {
                    resetting = true;
                    const HRESULT result = viewport->ZoomToRect(0.0f, 0.0f,
                        static_cast<float>(bounds.right), static_cast<float>(bounds.bottom), FALSE);
                    if (FAILED(result)) {
                        resetting = false;
                    }
                    return result;
                }
                break;
            case DIRECTMANIPULATION_DISABLED:
                scroll.end();
                updating = false;
                break;
            default:
                break;
        }
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnViewportUpdated(IDirectManipulationViewport*) override {
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnContentUpdated(IDirectManipulationViewport*,
                                               IDirectManipulationContent* content) override {
        if (resetting) {
            return S_OK;
        }
        float transform[6]{};
        const HRESULT result = content->GetContentTransform(transform, 6);
        if (SUCCEEDED(result)) {
            scroll.move(transform[4], transform[5]);
        }
        return result;
    }

    HRESULT STDMETHODCALLTYPE OnInteraction(IDirectManipulationViewport2*,
                                            DIRECTMANIPULATION_INTERACTION_TYPE interaction) override {
        if (interaction == DIRECTMANIPULATION_INTERACTION_BEGIN) {
            updating = true;
        } else if (interaction == DIRECTMANIPULATION_INTERACTION_END) {
            updating = false;
            scroll.end();
        }
        return S_OK;
    }
};

}  // namespace

struct WindowsPointer::State {
    HWND window = nullptr;
    ComPtr<IDirectManipulationManager> manager;
    ComPtr<IDirectManipulationUpdateManager> updates;
    ComPtr<IDirectManipulationViewport> viewport;
    ComPtr<TouchpadHandler> handler;
    DWORD cookie = 0;
    bool registered = false;
    bool installed = false;
    bool initializedCOM = false;
    bool ready = false;
    bool touchHandled = false;
    F64 polledAt = glfwGetTime();
    std::function<void(const Touch&)> onTouch;
    std::function<bool(const Extent2D<F32>&)> captionHit;
    std::unordered_map<UINT32, POINT> contacts;
    std::unordered_map<UINT32, HANDLE> nativeContacts;
    std::vector<POINTER_TOUCH_INFO> touchFrame;
    std::optional<std::pair<HANDLE, UINT32>> touchFrameId;

    bool initialize(GLFWwindow* handle, std::function<void(const Scroll&)> onScroll) {
        window = glfwGetWin32Window(handle);
        installed = window && SetWindowSubclass(window, Procedure, reinterpret_cast<UINT_PTR>(this),
                                                reinterpret_cast<DWORD_PTR>(this)) != FALSE;
        if (!installed) {
            return false;
        }
        const HRESULT com = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        initializedCOM = SUCCEEDED(com);
        if (FAILED(com) && com != RPC_E_CHANGED_MODE) {
            return false;
        }
        if (FAILED(CoCreateInstance(__uuidof(DirectManipulationManager), nullptr,
                CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&manager))) ||
            FAILED(manager->GetUpdateManager(IID_PPV_ARGS(&updates))) ||
            FAILED(manager->CreateViewport(nullptr, window, IID_PPV_ARGS(&viewport)))) {
            return false;
        }
        handler = Microsoft::WRL::Make<TouchpadHandler>();
        handler->scroll.emit = std::move(onScroll);
        GetClientRect(window, &handler->bounds);
        handler->bounds.right = std::max<LONG>(1, handler->bounds.right);
        handler->bounds.bottom = std::max<LONG>(1, handler->bounds.bottom);
        const auto configuration = DIRECTMANIPULATION_CONFIGURATION_INTERACTION |
            DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X |
            DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y |
            DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA |
            DIRECTMANIPULATION_CONFIGURATION_RAILS_X | DIRECTMANIPULATION_CONFIGURATION_RAILS_Y;
        if (FAILED(viewport->ActivateConfiguration(configuration)) ||
            FAILED(viewport->SetViewportOptions(DIRECTMANIPULATION_VIEWPORT_OPTIONS_MANUALUPDATE)) ||
            FAILED(viewport->SetViewportRect(&handler->bounds)) ||
            FAILED(viewport->AddEventHandler(window, handler.Get(), &cookie))) {
            return false;
        }
        registered = true;
        if (FAILED(manager->Activate(window)) || FAILED(viewport->Enable())) {
            return false;
        }
        return true;
    }

    ~State() {
        onTouch = {};
        if (installed) {
            RemoveWindowSubclass(window, Procedure, reinterpret_cast<UINT_PTR>(this));
        }
        if (handler) {
            handler->scroll.emit = {};
        }
        if (viewport) {
            viewport->Stop();
            if (registered) {
                viewport->RemoveEventHandler(cookie);
            }
            viewport->Abandon();
        }
        if (manager && window) {
            manager->Deactivate(window);
        }
        handler.Reset();
        viewport.Reset();
        updates.Reset();
        manager.Reset();
        if (initializedCOM) {
            CoUninitialize();
        }
    }

    void stop() {
        if (viewport) {
            viewport->Stop();
        }
        if (handler) {
            handler->scroll.end();
            handler->updating = false;
        }
    }

    bool ownsContact(UINT32 id) const {
        return contacts.contains(id);
    }

    void emitTouch(Touch event) {
        event.momentumAllowed = event.momentumAllowed && glfwGetTime() - polledAt <= FreshPollSeconds;
        if (onTouch) {
            onTouch(event);
        }
    }

    void cancelTouch() {
        contacts.clear();
        emitTouch({.type = Touch::Type::Cancel});
    }

    void trackNative(UINT32 pointer) {
        for (const auto& contact : touchFrame) {
            const auto& info = contact.pointerInfo;
            if (!(info.pointerFlags & (POINTER_FLAG_UP | POINTER_FLAG_CANCELED))) {
                nativeContacts.insert_or_assign(info.pointerId, info.sourceDevice);
            } else if (info.pointerId == pointer) {
                nativeContacts.erase(pointer);
            }
        }
    }

    bool touch(UINT32 pointer) {
        if (!onTouch) {
            return false;
        }
        POINTER_INPUT_TYPE type = PT_POINTER;
        if (!GetPointerType(pointer, &type)) {
            if (ownsContact(pointer)) {
                cancelTouch();
                return true;
            }
            return false;
        }
        if (type != PT_TOUCH) {
            return false;
        }

        UINT32 count = 0;
        GetPointerFrameTouchInfo(pointer, &count, nullptr);
        if (!count) {
            cancelTouch();
            return true;
        }
        touchFrame.resize(count);
        if (!GetPointerFrameTouchInfo(pointer, &count, touchFrame.data()) || !count) {
            cancelTouch();
            return true;
        }
        touchFrame.resize(count);

        std::erase_if(nativeContacts, [this](const auto& native) {
            return native.second == touchFrame.front().pointerInfo.sourceDevice &&
                   std::none_of(touchFrame.begin(), touchFrame.end(), [&](const auto& contact) {
                       return contact.pointerInfo.pointerId == native.first;
                   });
        });
        if (!nativeContacts.empty()) {
            trackNative(pointer);
            return false;
        }

        const auto& first = touchFrame.front().pointerInfo;
        const auto frame = std::pair{first.sourceDevice, first.frameId};
        if (touchFrameId == frame) {
            return true;
        }
        touchFrameId = frame;

        for (const auto& contact : touchFrame) {
            const auto& info = contact.pointerInfo;
            if ((info.pointerFlags & POINTER_FLAG_CANCELED) && ownsContact(info.pointerId)) {
                cancelTouch();
                return true;
            }
            if (info.pointerType != PT_TOUCH || !(info.pointerFlags & POINTER_FLAG_DOWN) ||
                (info.pointerFlags & POINTER_FLAG_CANCELED) || ownsContact(info.pointerId) ||
                info.hwndTarget != window) {
                continue;
            }
            if (SendMessageW(window, WM_NCHITTEST, 0,
                    MAKELPARAM(info.ptPixelLocation.x, info.ptPixelLocation.y)) != HTCLIENT) {
                cancelTouch();
                touchHandled = false;
                return true;
            }
            POINT point = info.ptPixelLocation;
            if (contacts.empty() && captionHit && ScreenToClient(window, &point) &&
                captionHit({static_cast<F32>(point.x), static_cast<F32>(point.y)})) {
                cancelTouch();
                touchHandled = false;
                trackNative(pointer);
                return false;
            }
        }

        const auto event = [scale = DpiScale(window)](const POINTER_INFO& info, Touch::Type type, POINT point) {
            return Touch{
                .type = type,
                .id = info.pointerId,
                .position = {static_cast<F32>(point.x), static_cast<F32>(point.y)},
                .timestamp = info.dwTime,
                .momentumAllowed = FreshTouch(info),
                .scale = scale,
            };
        };

        for (const auto& contact : touchFrame) {
            const auto& info = contact.pointerInfo;
            if (info.pointerType != PT_TOUCH || (info.pointerFlags & POINTER_FLAG_CANCELED) ||
                !(info.pointerFlags & (POINTER_FLAG_UPDATE | POINTER_FLAG_UP)) || !ownsContact(info.pointerId)) {
                continue;
            }
            POINT point = info.ptPixelLocation;
            if (!ScreenToClient(window, &point)) {
                cancelTouch();
                return true;
            }
            auto& previous = contacts.at(info.pointerId);
            if (point.x != previous.x || point.y != previous.y) {
                previous = point;
                emitTouch(event(info, Touch::Type::Motion, point));
            }
        }

        for (const auto& contact : touchFrame) {
            const auto& info = contact.pointerInfo;
            if (info.pointerType != PT_TOUCH || (info.pointerFlags & POINTER_FLAG_CANCELED) ||
                !(info.pointerFlags & POINTER_FLAG_DOWN) || ownsContact(info.pointerId) || info.hwndTarget != window) {
                continue;
            }
            POINT point = info.ptPixelLocation;
            if (!ScreenToClient(window, &point)) {
                cancelTouch();
                return true;
            }
            stop();
            contacts.emplace(info.pointerId, point);
            touchHandled = true;
            emitTouch(event(info, Touch::Type::Down, point));
        }

        bool fresh = true;
        for (const auto& contact : touchFrame) {
            const auto& info = contact.pointerInfo;
            if (!ownsContact(info.pointerId)) {
                continue;
            }
            fresh = fresh && FreshTouch(info);
            if (info.pointerFlags & POINTER_FLAG_UP) {
                emitTouch(event(info, Touch::Type::Up, {}));
                contacts.erase(info.pointerId);
            }
        }
        emitTouch({.type = Touch::Type::Frame, .momentumAllowed = fresh});
        return true;
    }

    bool touchMouse(UINT message) const {
        if (!onTouch || !touchHandled) {
            return false;
        }
        switch (message) {
            case WM_MOUSEMOVE:
            case WM_LBUTTONDOWN:
            case WM_LBUTTONUP:
            case WM_LBUTTONDBLCLK:
            case WM_RBUTTONDOWN:
            case WM_RBUTTONUP:
            case WM_RBUTTONDBLCLK:
            case WM_MBUTTONDOWN:
            case WM_MBUTTONUP:
            case WM_MBUTTONDBLCLK:
            case WM_XBUTTONDOWN:
            case WM_XBUTTONUP:
            case WM_XBUTTONDBLCLK:
            case WM_MOUSEWHEEL:
            case WM_MOUSEHWHEEL:
                return (static_cast<ULONG_PTR>(GetMessageExtraInfo()) & PromotedMouseMask) == PromotedTouchSignature;
            default:
                return false;
        }
    }

    static LRESULT CALLBACK Procedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam,
                                      UINT_PTR id, DWORD_PTR data) {
        auto& self = *reinterpret_cast<State*>(data);
        if (self.touchMouse(message)) {
            return 0;
        }
        switch (message) {
            case WM_POINTERDOWN:
            case WM_POINTERUPDATE:
            case WM_POINTERUP:
                if (self.touch(GET_POINTERID_WPARAM(wParam))) {
                    return 0;
                }
                break;
            case WM_POINTERCAPTURECHANGED:
                self.nativeContacts.erase(GET_POINTERID_WPARAM(wParam));
                if (self.ownsContact(GET_POINTERID_WPARAM(wParam))) {
                    self.cancelTouch();
                }
                break;
            case WM_NCPOINTERDOWN:
                self.cancelTouch();
                self.touchHandled = false;
                break;
            case DM_POINTERHITTEST: {
                if (!self.ready) {
                    break;
                }
                const UINT32 pointer = GET_POINTERID_WPARAM(wParam);
                POINTER_INPUT_TYPE type = PT_POINTER;
                if (GetPointerType(pointer, &type) && type == PT_TOUCHPAD) {
                    self.handler->scroll.scale = DpiScale(window);
                    if (SUCCEEDED(self.viewport->SetContact(pointer))) {
                        self.handler->updating = true;
                        return 0;
                    }
                }
                break;
            }
            case WM_SIZE: {
                if (!self.ready) {
                    break;
                }
                RECT bounds{};
                GetClientRect(window, &bounds);
                if (bounds.right > 0 && bounds.bottom > 0 &&
                    !EqualRect(&bounds, &self.handler->bounds)) {
                    self.stop();
                    self.handler->bounds = bounds;
                    self.viewport->SetViewportRect(&bounds);
                }
                break;
            }
            case WM_ACTIVATE:
                if (LOWORD(wParam) == WA_INACTIVE) {
                    self.stop();
                }
                break;
            case WM_CANCELMODE:
                self.cancelTouch();
                break;
            case WM_MOUSEWHEEL:
            case WM_MOUSEHWHEEL:
                if (self.handler && self.handler->scroll.started) {
                    self.stop();
                }
                break;
            case WM_NCDESTROY:
                self.cancelTouch();
                self.onTouch = {};
                self.stop();
                RemoveWindowSubclass(window, Procedure, id);
                self.installed = false;
                self.ready = false;
                break;
            default:
                break;
        }
        return DefSubclassProc(window, message, wParam, lParam);
    }
};

WindowsPointer::WindowsPointer(GLFWwindow* window, std::function<void(const Scroll&)> onScroll,
                               std::function<void(const Touch&)> onTouch,
                               std::function<bool(const Extent2D<F32>&)> captionHit)
    : state(std::make_unique<State>()) {
    state->onTouch = std::move(onTouch);
    state->captionHit = std::move(captionHit);
    state->ready = state->initialize(window, std::move(onScroll));
}

WindowsPointer::~WindowsPointer() = default;

bool WindowsPointer::available() const {
    return state->ready;
}

void WindowsPointer::focus(bool focused) {
    if (!focused) {
        state->cancelTouch();
        state->stop();
    }
}

void WindowsPointer::poll() {
    state->polledAt = glfwGetTime();
    if (animating()) {
        state->updates->Update(nullptr);
    }
}

bool WindowsPointer::animating() const {
    return state->ready && (state->handler->updating || state->handler->resetting);
}

KeyModifiers WindowsPointer::Modifiers() {
    const auto down = [](int key) { return (GetKeyState(key) & 0x8000) != 0; };
    return {.control = down(VK_CONTROL), .shift = down(VK_SHIFT), .alt = down(VK_MENU),
            .super = down(VK_LWIN) || down(VK_RWIN)};
}

}  // namespace Jetstream::Viewport
