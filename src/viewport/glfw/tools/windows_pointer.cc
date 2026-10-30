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
#include <utility>

namespace Jetstream::Viewport {

namespace {

constexpr F32 PointsPerWheelUnit = 10.0f;
constexpr F32 BaseDpi = 96.0f;

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

    bool initialize(GLFWwindow* handle, std::function<void(const Scroll&)> onScroll) {
        window = glfwGetWin32Window(handle);
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
        installed = SetWindowSubclass(window, Procedure, reinterpret_cast<UINT_PTR>(this),
                                      reinterpret_cast<DWORD_PTR>(this)) != FALSE;
        return installed;
    }

    ~State() {
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
        viewport->Stop();
        handler->scroll.end();
        handler->updating = false;
    }

    static LRESULT CALLBACK Procedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam,
                                      UINT_PTR id, DWORD_PTR data) {
        auto& self = *reinterpret_cast<State*>(data);
        switch (message) {
            case DM_POINTERHITTEST: {
                const UINT32 pointer = GET_POINTERID_WPARAM(wParam);
                POINTER_INPUT_TYPE type = PT_POINTER;
                if (GetPointerType(pointer, &type) && type == PT_TOUCHPAD) {
                    self.handler->scroll.scale = std::max<F32>(1.0f, GetDpiForWindow(window) / BaseDpi);
                    if (SUCCEEDED(self.viewport->SetContact(pointer))) {
                        self.handler->updating = true;
                        return 0;
                    }
                }
                break;
            }
            case WM_SIZE: {
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
            case WM_MOUSEWHEEL:
            case WM_MOUSEHWHEEL:
                if (self.handler->scroll.started) {
                    self.stop();
                }
                break;
            case WM_NCDESTROY:
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

WindowsPointer::WindowsPointer(GLFWwindow* window, std::function<void(const Scroll&)> onScroll)
    : state(std::make_unique<State>()) {
    state->ready = state->initialize(window, std::move(onScroll));
}

WindowsPointer::~WindowsPointer() = default;

bool WindowsPointer::available() const {
    return state->ready;
}

void WindowsPointer::poll() {
    if (state->ready && (state->handler->updating || state->handler->resetting)) {
        state->updates->Update(nullptr);
    }
}

KeyModifiers WindowsPointer::Modifiers() {
    const auto down = [](int key) { return (GetKeyState(key) & 0x8000) != 0; };
    return {.control = down(VK_CONTROL), .shift = down(VK_SHIFT), .alt = down(VK_MENU),
            .super = down(VK_LWIN) || down(VK_RWIN)};
}

}  // namespace Jetstream::Viewport
