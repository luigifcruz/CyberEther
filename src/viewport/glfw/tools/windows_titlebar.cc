#include "windows_titlebar.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <commctrl.h>
#include <dwmapi.h>
#include <uxtheme.h>
#include <windowsx.h>

#include <mutex>
#include <utility>

namespace Jetstream::Viewport {

namespace {

constexpr int TitlebarHeight = 40;
constexpr int CaptionButtonWidth = 46;
constexpr int CaptionButtonHeight = 34;

UINT AppearanceMessage() {
    static const UINT message = RegisterWindowMessageW(L"CyberEther.Titlebar.Appearance");
    return message;
}

void FrameChanged(HWND window) {
    SetWindowPos(window, nullptr, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
}

}  // namespace

struct WindowsTitlebar::State {
    HWND window = nullptr;
    HWND buttons = nullptr;
    GLFWwindow* handle = nullptr;
    std::function<void()> release;
    mutable std::mutex mutex;
    Render::TitlebarChrome chrome;
    bool pressed = false;
    bool requested = false;
    bool pendingDrag = false;
    bool pendingZoom = false;
    bool installed = false;
    bool dark = false;
    bool paintInitialized = false;
    int hovered = HTNOWHERE;
    int captionPressed = HTNOWHERE;
    POINT pressPoint{};

    COLORREF background() const {
        return dark ? RGB(0, 0, 0) : RGB(247, 247, 250);
    }

    int buttonWidth() const {
        return MulDiv(CaptionButtonWidth, static_cast<int>(GetDpiForWindow(window)), 96);
    }

    int buttonHit(POINT point) const {
        RECT client{};
        GetClientRect(window, &client);
        const int width = buttonWidth();
        const int start = client.right - 3 * width;
        if (point.y < 0 || point.y >= buttonHeight() || point.x < start || point.x >= client.right) {
            return HTNOWHERE;
        }
        const int hits[] = {HTMINBUTTON, HTMAXBUTTON, HTCLOSE};
        return hits[(point.x - start) / width];
    }

    void paintButtons() {
        PAINTSTRUCT paint{};
        HDC dc = BeginPaint(buttons, &paint);
        RECT client{};
        GetClientRect(buttons, &client);
        HDC bufferDC = nullptr;
        HPAINTBUFFER buffer = BeginBufferedPaint(dc, &client, BPBF_TOPDOWNDIB, nullptr, &bufferDC);
        HDC target = buffer ? bufferDC : dc;
        const int dpi = static_cast<int>(GetDpiForWindow(window));
        const int width = buttonWidth();
        const int half = MulDiv(5, dpi, 96);
        const int centerY = height() / 2;
        const int hits[] = {HTMINBUTTON, HTMAXBUTTON, HTCLOSE};
        for (int index = 0; index < 3; ++index) {
            const bool hot = hovered == hits[index];
            const COLORREF fill = hot ? (index == 2 ? RGB(196, 43, 28) :
                                          (dark ? RGB(40, 40, 40) : RGB(225, 225, 228))) : background();
            const COLORREF color = dark || (hot && index == 2) ? RGB(240, 240, 240) : RGB(24, 24, 24);
            RECT rect{index * width, 0, (index + 1) * width, buttonHeight()};
            HBRUSH brush = CreateSolidBrush(fill);
            FillRect(target, &rect, brush);
            DeleteObject(brush);
            HPEN pen = CreatePen(PS_SOLID, (dpi + 95) / 96, color);
            HGDIOBJ oldPen = SelectObject(target, pen);
            HGDIOBJ oldBrush = SelectObject(target, GetStockObject(HOLLOW_BRUSH));
            const int centerX = index * width + width / 2;
            if (index == 0) {
                MoveToEx(target, centerX - half, centerY, nullptr);
                LineTo(target, centerX + half, centerY);
            } else if (index == 1) {
                if (IsZoomed(window)) {
                    const int offset = MulDiv(2, dpi, 96);
                    Rectangle(target, centerX - half + offset, centerY - half,
                              centerX + half, centerY + half - offset);
                    HBRUSH cover = CreateSolidBrush(fill);
                    SelectObject(target, cover);
                    Rectangle(target, centerX - half, centerY - half + offset,
                              centerX + half - offset, centerY + half);
                    SelectObject(target, oldBrush);
                    DeleteObject(cover);
                } else {
                    Rectangle(target, centerX - half, centerY - half, centerX + half, centerY + half);
                }
            } else {
                MoveToEx(target, centerX - half, centerY - half, nullptr);
                LineTo(target, centerX + half, centerY + half);
                MoveToEx(target, centerX + half, centerY - half, nullptr);
                LineTo(target, centerX - half, centerY + half);
            }
            SelectObject(target, oldBrush);
            SelectObject(target, oldPen);
            DeleteObject(pen);
        }
        if (buffer) {
            BufferedPaintSetAlpha(buffer, nullptr, 255);
            EndBufferedPaint(buffer, TRUE);
        }
        EndPaint(buttons, &paint);
    }

    static LRESULT CALLBACK ButtonsProcedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam,
                                             UINT_PTR id, DWORD_PTR data) {
        auto& self = *reinterpret_cast<State*>(data);
        if (message == WM_NCHITTEST) {
            return HTTRANSPARENT;
        }
        if (message == WM_ERASEBKGND) {
            return 1;
        }
        if (message == WM_PAINT) {
            self.paintButtons();
            return 0;
        }
        if (message == WM_NCDESTROY) {
            RemoveWindowSubclass(window, ButtonsProcedure, id);
            self.buttons = nullptr;
        }
        return DefSubclassProc(window, message, wParam, lParam);
    }

    void layoutButtons() {
        if (!buttons) {
            return;
        }
        RECT client{};
        GetClientRect(window, &client);
        SetWindowPos(buttons, HWND_TOP, client.right - 3 * buttonWidth(), 0, 3 * buttonWidth(), buttonHeight(),
                     SWP_NOACTIVATE | (fullscreen() ? SWP_HIDEWINDOW : SWP_SHOWWINDOW));
        InvalidateRect(buttons, nullptr, FALSE);
    }

    bool fullscreen() const {
        return glfwGetWindowMonitor(handle) != nullptr;
    }

    int height() const {
        return MulDiv(TitlebarHeight, static_cast<int>(GetDpiForWindow(window)), 96);
    }

    int buttonHeight() const {
        return MulDiv(CaptionButtonHeight, static_cast<int>(GetDpiForWindow(window)), 96);
    }

    int resizeBorder() const {
        const UINT dpi = GetDpiForWindow(window);
        return GetSystemMetricsForDpi(SM_CYSIZEFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
    }

    void measure() {
        Render::TitlebarChrome next;
        if (installed && window) {
            next.integrated = true;
            if (!fullscreen()) {
                next.height = static_cast<F32>(height());
                next.controlsCenter = next.height * 0.5f;
                next.trailingInset = static_cast<F32>(3 * buttonWidth());
            }
        }
        std::lock_guard lock(mutex);
        chrome = next;
    }

    void extendFrame() {
        const MARGINS margins{0, 0, fullscreen() ? 0 : 1, 0};
        DwmExtendFrameIntoClientArea(window, &margins);
    }

    void cancelPress() {
        bool notify = false;
        {
            std::lock_guard lock(mutex);
            notify = requested;
            pressed = requested = pendingDrag = false;
        }
        if (notify && release) {
            release();
        }
    }

    static LRESULT CALLBACK Procedure(HWND window, UINT message, WPARAM wParam, LPARAM lParam,
                                      UINT_PTR id, DWORD_PTR data) {
        auto& self = *reinterpret_cast<State*>(data);
        if (message == WM_SYSCOMMAND) {
            return DefSubclassProc(window, message, wParam, lParam);
        }
        if (message == WM_NCDESTROY) {
            self.cancelPress();
            RemoveWindowSubclass(window, Procedure, id);
            self.window = nullptr;
            self.installed = false;
            self.measure();
            return DefSubclassProc(window, message, wParam, lParam);
        }
        if (message == AppearanceMessage()) {
            const BOOL dark = wParam != 0;
            self.dark = dark;
            DwmSetWindowAttribute(window, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark, sizeof(dark));
            const COLORREF background = self.background();
            const COLORREF foreground = dark ? RGB(255, 255, 255) : RGB(0, 0, 0);
            DwmSetWindowAttribute(window, DWMWA_CAPTION_COLOR, &background, sizeof(background));
            DwmSetWindowAttribute(window, DWMWA_TEXT_COLOR, &foreground, sizeof(foreground));
            if (self.buttons) {
                InvalidateRect(self.buttons, nullptr, FALSE);
            }
            RedrawWindow(window, nullptr, nullptr, RDW_INVALIDATE | RDW_FRAME);
            return 0;
        }
        if (message == WM_NCCALCSIZE && wParam && !self.fullscreen()) {
            auto& rect = reinterpret_cast<NCCALCSIZE_PARAMS*>(lParam)->rgrc[0];
            const LONG top = rect.top;
            DefSubclassProc(window, message, wParam, lParam);
            rect.top = top + (IsZoomed(window) ? self.resizeBorder() : 0);
            return 0;
        }
        if (!self.fullscreen() && (message == WM_NCLBUTTONDOWN || message == WM_NCLBUTTONDBLCLK) &&
            (wParam == HTMINBUTTON || wParam == HTMAXBUTTON || wParam == HTCLOSE)) {
            self.captionPressed = static_cast<int>(wParam);
            SetCapture(window);
            return 0;
        }
        if (message == WM_LBUTTONUP && self.captionPressed != HTNOWHERE) {
            const int pressed = std::exchange(self.captionPressed, HTNOWHERE);
            const int hit = self.buttonHit({GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)});
            ReleaseCapture();
            if (hit == pressed) {
                const WPARAM action = hit == HTCLOSE ? SC_CLOSE : hit == HTMINBUTTON ? SC_MINIMIZE :
                                      (IsZoomed(window) ? SC_RESTORE : SC_MAXIMIZE);
                PostMessageW(window, WM_SYSCOMMAND, action, 0);
            }
            return 0;
        }
        if (message == WM_CAPTURECHANGED) {
            self.captionPressed = HTNOWHERE;
        }
        if (message == WM_MOUSEMOVE && self.captionPressed != HTNOWHERE) {
            const int hovered = self.buttonHit({GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)});
            if (hovered != self.hovered) {
                self.hovered = hovered;
                if (self.buttons) {
                    InvalidateRect(self.buttons, nullptr, FALSE);
                }
            }
            return 0;
        }
        if (message == WM_LBUTTONDOWN || message == WM_LBUTTONDBLCLK) {
            self.cancelPress();
            std::lock_guard lock(self.mutex);
            self.pressed = true;
            self.pressPoint = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
            ClientToScreen(window, &self.pressPoint);
        } else if (message == WM_LBUTTONUP || message == WM_CANCELMODE ||
                   (message == WM_ACTIVATE && LOWORD(wParam) == WA_INACTIVE)) {
            self.cancelPress();
        }

        if (!self.fullscreen()) {
            if (message == WM_NCMOUSEMOVE || message == WM_NCMOUSELEAVE) {
                const int hovered = message == WM_NCMOUSELEAVE ? HTNOWHERE : static_cast<int>(wParam);
                if (self.hovered != hovered) {
                    self.hovered = hovered;
                    if (self.buttons) {
                        InvalidateRect(self.buttons, nullptr, FALSE);
                    }
                }
                if (message == WM_NCMOUSEMOVE) {
                    TRACKMOUSEEVENT track{sizeof(track), TME_NONCLIENT | TME_LEAVE, window, 0};
                    TrackMouseEvent(&track);
                }
            }
            if (message == WM_NCHITTEST) {
                POINT point{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
                ScreenToClient(window, &point);
                const bool resizing = !IsZoomed(window) && point.y < self.resizeBorder();
                const int hit = resizing ? HTNOWHERE : self.buttonHit(point);
                if (hit != HTNOWHERE) {
                    return hit;
                }
            }
            LRESULT result = 0;
            if (DwmDefWindowProc(window, message, wParam, lParam, &result)) {
                return result;
            }
            if (message == WM_NCHITTEST) {
                const LRESULT hit = DefSubclassProc(window, message, wParam, lParam);
                if (hit != HTCLIENT && hit != HTCAPTION) {
                    return hit;
                }
                POINT point{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
                ScreenToClient(window, &point);
                if (!IsZoomed(window) && point.y < self.resizeBorder()) {
                    RECT client{};
                    GetClientRect(window, &client);
                    if (point.x < self.resizeBorder()) {
                        return HTTOPLEFT;
                    }
                    if (point.x >= client.right - self.resizeBorder()) {
                        return HTTOPRIGHT;
                    }
                    return HTTOP;
                }
                return HTCLIENT;
            }
        }
        const LRESULT result = DefSubclassProc(window, message, wParam, lParam);
        if (message == WM_SIZE || message == WM_DPICHANGED || message == WM_DWMCOMPOSITIONCHANGED ||
            message == WM_STYLECHANGED || message == WM_ACTIVATE) {
            self.extendFrame();
            self.layoutButtons();
            self.measure();
        }
        return result;
    }
};

WindowsTitlebar::WindowsTitlebar(GLFWwindow* handle, std::function<void()> releaseLeftButton)
    : state(std::make_unique<State>()) {
    state->handle = handle;
    state->window = glfwGetWin32Window(handle);
    state->release = std::move(releaseLeftButton);
    if (!state->window || !SetWindowSubclass(state->window, State::Procedure,
                                            reinterpret_cast<UINT_PTR>(state.get()),
                                            reinterpret_cast<DWORD_PTR>(state.get()))) {
        return;
    }
    state->installed = true;
    state->paintInitialized = SUCCEEDED(BufferedPaintInit());
    state->buttons = CreateWindowExW(0, L"STATIC", nullptr, WS_CHILD | WS_VISIBLE,
                                    0, 0, 0, 0, state->window, nullptr, GetModuleHandleW(nullptr), nullptr);
    if (!state->buttons || !SetWindowSubclass(state->buttons, State::ButtonsProcedure,
                                             reinterpret_cast<UINT_PTR>(state.get()),
                                             reinterpret_cast<DWORD_PTR>(state.get()))) {
        if (state->buttons) {
            DestroyWindow(state->buttons);
        }
        state->buttons = nullptr;
        RemoveWindowSubclass(state->window, State::Procedure, reinterpret_cast<UINT_PTR>(state.get()));
        state->installed = false;
        return;
    }
    state->extendFrame();
    FrameChanged(state->window);
    state->layoutButtons();
    state->measure();
}

WindowsTitlebar::~WindowsTitlebar() {
    state->cancelPress();
    if (state->buttons) {
        DestroyWindow(state->buttons);
    }
    if (state->window && state->installed) {
        RemoveWindowSubclass(state->window, State::Procedure, reinterpret_cast<UINT_PTR>(state.get()));
        const MARGINS margins{};
        DwmExtendFrameIntoClientArea(state->window, &margins);
        FrameChanged(state->window);
    }
    if (state->paintInitialized) {
        BufferedPaintUnInit();
    }
}

void WindowsTitlebar::poll() {
    if (!state->window || !state->installed) {
        return;
    }
    bool drag = false;
    bool zoom = false;
    {
        std::lock_guard lock(state->mutex);
        drag = std::exchange(state->pendingDrag, false) && state->pressed;
        zoom = std::exchange(state->pendingZoom, false);
    }
    if (zoom && !state->fullscreen()) {
        state->cancelPress();
        ShowWindow(state->window, IsZoomed(state->window) ? SW_RESTORE : SW_MAXIMIZE);
    } else if (drag && !state->fullscreen() && (GetAsyncKeyState(VK_LBUTTON) & 0x8000)) {
        const POINT point = state->pressPoint;
        ReleaseCapture();
        SendMessageW(state->window, WM_SYSCOMMAND, SC_MOVE | HTCAPTION, MAKELPARAM(point.x, point.y));
        SendMessageW(state->window, WM_LBUTTONUP, 0, 0);
        state->cancelPress();
    }
    state->measure();
}

Render::TitlebarChrome WindowsTitlebar::chrome() const {
    std::lock_guard lock(state->mutex);
    return state->chrome;
}

void WindowsTitlebar::beginDrag() {
    std::lock_guard lock(state->mutex);
    if (state->pressed && !state->requested) {
        state->requested = state->pendingDrag = true;
    }
}

void WindowsTitlebar::zoom() {
    std::lock_guard lock(state->mutex);
    state->pendingZoom = true;
    state->pendingDrag = false;
}

bool WindowsTitlebar::SetAppearance(GLFWwindow* handle, bool dark) {
    HWND window = handle ? glfwGetWin32Window(handle) : nullptr;
    return window && PostMessageW(window, AppearanceMessage(), dark, 0);
}

}  // namespace Jetstream::Viewport
