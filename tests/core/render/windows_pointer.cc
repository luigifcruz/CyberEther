#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "jetstream/platform.hh"
#include "viewport/glfw/tools/glfw_input.hh"
#include "viewport/glfw/tools/windows_pointer.hh"
#include "viewport/glfw/tools/windows_titlebar.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <objbase.h>

namespace {

struct NativeFrame {
    std::vector<POINTER_TOUCH_INFO> contacts;
    UINT32 id = 0;
    DWORD time = 1000;
    LPARAM extra = 0;
    bool failType = false;
    bool failFrame = false;
    bool failManipulation = false;
};

NativeFrame* nativeFrame = nullptr;

BOOL WINAPI TestPointerType(UINT32 id, POINTER_INPUT_TYPE* type) {
    if (nativeFrame) {
        if (nativeFrame->failType) {
            return FALSE;
        }
        for (const auto& contact : nativeFrame->contacts) {
            if (contact.pointerInfo.pointerId == id) {
                *type = contact.pointerInfo.pointerType;
                return TRUE;
            }
        }
    }
    return GetPointerType(id, type);
}

BOOL WINAPI TestPointerFrame(UINT32 id, UINT32* count, POINTER_TOUCH_INFO* frame) {
    if (!nativeFrame) {
        return GetPointerFrameTouchInfo(id, count, frame);
    }
    if (nativeFrame->failFrame) {
        return FALSE;
    }
    const UINT32 size = static_cast<UINT32>(nativeFrame->contacts.size());
    if (!frame || *count < size) {
        *count = size;
        return frame == nullptr;
    }
    std::copy(nativeFrame->contacts.begin(), nativeFrame->contacts.end(), frame);
    *count = size;
    return TRUE;
}

DWORD WINAPI TestTickCount() { return nativeFrame ? nativeFrame->time : GetTickCount(); }
LPARAM WINAPI TestMessageExtra() { return nativeFrame ? nativeFrame->extra : GetMessageExtraInfo(); }

HRESULT WINAPI TestCreateInstance(REFCLSID clsid, LPUNKNOWN outer, DWORD context, REFIID iid, LPVOID* object) {
    if (nativeFrame && nativeFrame->failManipulation) {
        return E_FAIL;
    }
    return CoCreateInstance(clsid, outer, context, iid, object);
}

}  // namespace

#define GetPointerType TestPointerType
#define GetPointerFrameTouchInfo TestPointerFrame
#define GetTickCount TestTickCount
#define GetMessageExtraInfo TestMessageExtra
#define CoCreateInstance TestCreateInstance
#include "../../../src/viewport/glfw/tools/windows_pointer.cc"
#undef CoCreateInstance
#undef GetMessageExtraInfo
#undef GetTickCount
#undef GetPointerFrameTouchInfo
#undef GetPointerType

using namespace Jetstream;
using namespace Jetstream::Viewport;

namespace {

struct ScrollFixture {
    std::vector<WindowsPointer::Scroll> events;
    ScrollSequence sequence{.emit = [this](const auto& event) { events.push_back(event); }};
};

struct WindowFixture {
    GLFWwindow* window = nullptr;
    HWND native = nullptr;
    std::vector<WindowsPointer::Scroll> events;
    Extent2D<double> wheel = {0.0, 0.0};

    WindowFixture() {
        REQUIRE(glfwInit());
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(800, 600, "Pointer test", nullptr, nullptr);
        REQUIRE(window);
        native = glfwGetWin32Window(window);
        glfwSetWindowUserPointer(window, this);
        glfwSetScrollCallback(window, [](GLFWwindow* window, double x, double y) {
            auto& fixture = *static_cast<WindowFixture*>(glfwGetWindowUserPointer(window));
            fixture.wheel.x += x;
            fixture.wheel.y += y;
        });
    }

    ~WindowFixture() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

template<typename T>
std::vector<T> Select(const std::vector<Render::InputQueue::Event>& events) {
    std::vector<T> result;
    for (const auto& event : events) {
        if (const auto* value = std::get_if<T>(&event)) {
            result.push_back(*value);
        }
    }
    return result;
}

struct TouchFixture {
    GLFWwindow* window = nullptr;
    HWND native = nullptr;
    NativeFrame frame;
    Render::InputQueue input;
    TouchGesture gesture{input};
    std::vector<WindowsPointer::Touch> touches;
    std::unique_ptr<WindowsTitlebar> titlebar;
    std::unique_ptr<WindowsPointer> pointer;
    F32 captionStrip = 0.0f;
    U64 mouseButtons = 0;
    U64 mouseMoves = 0;
    F64 wheel = 0.0;

    explicit TouchFixture(bool failManipulation = false) {
        REQUIRE(glfwInit());
        glfwSetTime(1.0);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(800, 600, "Touch test", nullptr, nullptr);
        REQUIRE(window);
        native = glfwGetWin32Window(window);
        glfwSetWindowUserPointer(window, this);
        glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int, int, int) {
            ++static_cast<TouchFixture*>(glfwGetWindowUserPointer(window))->mouseButtons;
        });
        glfwSetCursorPosCallback(window, [](GLFWwindow* window, double, double) {
            ++static_cast<TouchFixture*>(glfwGetWindowUserPointer(window))->mouseMoves;
        });
        glfwSetScrollCallback(window, [](GLFWwindow* window, double, double y) {
            static_cast<TouchFixture*>(glfwGetWindowUserPointer(window))->wheel += y;
        });
        nativeFrame = &frame;
        frame.failManipulation = failManipulation;
        titlebar = std::make_unique<WindowsTitlebar>(window, [] {});
        pointer = std::make_unique<WindowsPointer>(window, [](const auto&) {},
            [this](const auto& event) {
                touches.push_back(event);
                gesture.handle(event, glfwGetTime());
            },
            [this](const auto& point) { return point.y < captionStrip; });
    }

    ~TouchFixture() {
        pointer.reset();
        titlebar.reset();
        nativeFrame = nullptr;
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    POINTER_TOUCH_INFO contact(UINT32 id, POINTER_FLAGS flags, LONG x = 100, LONG y = 120) const {
        POINTER_TOUCH_INFO info{};
        info.pointerInfo.pointerType = PT_TOUCH;
        info.pointerInfo.pointerId = id;
        info.pointerInfo.pointerFlags = flags;
        info.pointerInfo.hwndTarget = native;
        info.pointerInfo.ptPixelLocation = {x, y};
        ClientToScreen(native, &info.pointerInfo.ptPixelLocation);
        info.pointerInfo.dwTime = frame.time;
        return info;
    }

    LONG below(F32 points) const {
        return 120 + static_cast<LONG>(std::lround(points * DpiScale(native)));
    }

    void stamp() {
        ++frame.id;
        for (auto& contact : frame.contacts) {
            contact.pointerInfo.frameId = frame.id;
        }
    }

    void send(UINT message, std::vector<POINTER_TOUCH_INFO> contacts, UINT32 id = 1) {
        frame.contacts = std::move(contacts);
        stamp();
        SendMessageW(native, message, id, 0);
    }

    void clear() {
        input.drain();
        touches.clear();
    }
};

struct InputViewport : Generic {
    InputViewport() : Generic(Config{}) {}
    std::string id() const override { return "touch-test"; }
    std::string name() const override { return "Touch test"; }
    DeviceType device() const override { return DeviceType::None; }
    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    Result waitEvents() override { return Result::SUCCESS; }
    Result pollEvents() override { return Result::SUCCESS; }
    bool keepRunning() override { return true; }
};

}  // namespace

TEST_CASE_METHOD(ScrollFixture, "Windows touchpad preserves subpixel deltas and normalizes DPI",
                 "[core][render][windows][scroll]") {
    sequence.scale = 1.5f;
    sequence.running();
    sequence.move(0.375f, -0.75f);
    sequence.move(0.75f, -2.25f);
    sequence.move(0.75f, -2.25f);
    REQUIRE(events.size() == 2);
    CHECK(events[0].phase == ScrollPhase::Began);
    CHECK(events[0].delta.x == Catch::Approx(0.025f));
    CHECK(events[0].delta.y == Catch::Approx(-0.05f));
    CHECK(events[1].phase == ScrollPhase::Changed);
    CHECK(events[1].delta.x == Catch::Approx(0.025f));
    CHECK(events[1].delta.y == Catch::Approx(-0.1f));
}

TEST_CASE_METHOD(ScrollFixture, "Windows touchpad delivers release and momentum end without movement",
                 "[core][render][windows][scroll]") {
    sequence.running();
    sequence.move(0.0f, -30.0f);
    sequence.inertia();
    sequence.move(0.0f, -40.0f);
    CHECK(sequence.reset());
    REQUIRE(events.size() == 5);
    CHECK(events[0].phase == ScrollPhase::Began);
    CHECK(events[1].phase == ScrollPhase::Ended);
    CHECK(events[1].delta.y == 0.0f);
    CHECK(events[2].phase == ScrollPhase::Momentum);
    CHECK(events[3].phase == ScrollPhase::Momentum);
    CHECK(events[3].delta.y == -1.0f);
    CHECK(events[4].phase == ScrollPhase::MomentumEnded);
    CHECK(events[4].delta.y == 0.0f);
    sequence.end();
    CHECK(events.size() == 5);
}

TEST_CASE_METHOD(ScrollFixture, "A new Windows gesture interrupts inertia without a position jump",
                 "[core][render][windows][scroll]") {
    sequence.move(0.0f, -30.0f);
    sequence.inertia();
    sequence.move(0.0f, -45.0f);
    sequence.running();
    sequence.move(0.0f, -40.0f);
    REQUIRE(events.size() == 6);
    CHECK(events[4].phase == ScrollPhase::MomentumEnded);
    CHECK(events[5].phase == ScrollPhase::Began);
    CHECK(events[5].delta.y == 0.5f);
    sequence.end();
    REQUIRE(events.size() == 7);
    CHECK(events.back().phase == ScrollPhase::Ended);
}

TEST_CASE_METHOD(ScrollFixture, "Windows transform resets do not generate reverse scrolls or empty gestures",
                 "[core][render][windows][scroll]") {
    sequence.move(20.0f, -50.0f);
    CHECK(sequence.reset());
    sequence.running();
    sequence.move(0.0f, 0.0f);
    CHECK_FALSE(sequence.reset());
    REQUIRE(events.size() == 2);
    CHECK(events.back().phase == ScrollPhase::Ended);
    sequence.move(0.0f, -2.5f);
    REQUIRE(events.size() == 3);
    CHECK(events.back().phase == ScrollPhase::Began);
    CHECK(events.back().delta.y == -0.25f);
    sequence.move(std::numeric_limits<float>::quiet_NaN(), 0.0f);
    sequence.move(0.0f, -5.0f);
    REQUIRE(events.size() == 4);
    CHECK(events.back().delta.y == -0.25f);
}

TEST_CASE("Windows native status callbacks release scrolling when an interaction ends or is disabled",
          "[core][render][windows][scroll]") {
    std::vector<WindowsPointer::Scroll> events;
    auto handler = Microsoft::WRL::Make<TouchpadHandler>();
    handler->scroll.emit = [&](const auto& event) { events.push_back(event); };
    ComPtr<IDirectManipulationInteractionEventHandler> interaction;
    REQUIRE(SUCCEEDED(handler.As(&interaction)));
    interaction->OnInteraction(nullptr, DIRECTMANIPULATION_INTERACTION_BEGIN);
    CHECK(handler->updating);
    handler->OnViewportStatusChanged(nullptr, DIRECTMANIPULATION_RUNNING, DIRECTMANIPULATION_READY);
    handler->scroll.move(0.0f, -10.0f);
    handler->OnViewportStatusChanged(nullptr, DIRECTMANIPULATION_INERTIA, DIRECTMANIPULATION_RUNNING);
    interaction->OnInteraction(nullptr, DIRECTMANIPULATION_INTERACTION_END);
    REQUIRE(events.size() == 4);
    CHECK(events.back().phase == ScrollPhase::MomentumEnded);
    CHECK_FALSE(handler->updating);
    interaction->OnInteraction(nullptr, DIRECTMANIPULATION_INTERACTION_BEGIN);
    handler->OnViewportStatusChanged(nullptr, DIRECTMANIPULATION_RUNNING, DIRECTMANIPULATION_READY);
    handler->scroll.move(0.0f, -20.0f);
    handler->OnViewportStatusChanged(nullptr, DIRECTMANIPULATION_DISABLED, DIRECTMANIPULATION_RUNNING);
    CHECK(events.back().phase == ScrollPhase::Ended);
    CHECK_FALSE(handler->updating);
}

TEST_CASE_METHOD(WindowFixture, "Windows touchpad adapter initializes alongside the custom caption and preserves wheels",
                 "[core][render][windows][scroll][native]") {
    WindowsTitlebar titlebar(window, [] {});
    {
        WindowsPointer pointer(window, [this](const auto& event) { events.push_back(event); });
        REQUIRE(pointer.available());
        SendMessageW(native, WM_MOUSEWHEEL, MAKEWPARAM(0, WHEEL_DELTA), 0);
        SendMessageW(native, WM_MOUSEHWHEEL, MAKEWPARAM(0, -WHEEL_DELTA / 2), 0);
        CHECK(wheel.x == 0.5);
        CHECK(wheel.y == 1.0);
        SendMessageW(native, DM_POINTERHITTEST, 0, 0);
        SetWindowPos(native, nullptr, 0, 0, 900, 700, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
        pointer.poll();
        CHECK(events.empty());
        CHECK(titlebar.chrome().integrated);
    }
    SendMessageW(native, WM_MOUSEWHEEL, MAKEWPARAM(0, WHEEL_DELTA), 0);
    CHECK(wheel.y == 2.0);
    WindowsPointer replacement(window, [](const auto&) {});
    REQUIRE(replacement.available());
    replacement.poll();
}

TEST_CASE_METHOD(TouchFixture, "Windows GLFW touch owns the pointer until a physical mouse returns",
                 "[core][render][windows][touch][native]") {
    pointer.reset();
    InputViewport viewport;
    GlfwInput platform(window, viewport);
    viewport.input().drain();
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    auto queued = viewport.input().drain();
    auto positions = Select<Render::InputQueue::Pointer>(queued);
    REQUIRE(positions.size() == 1);
    CHECK(positions[0].position == Extent2D<F32>{100, 120});
    CHECK(Select<Render::InputQueue::Button>(queued).empty());
    SendMessageW(native, WM_MOUSEMOVE, 0, MAKELPARAM(200, 220));
    platform.poll();
    CHECK(viewport.input().drain().empty());
    frame.time += 100;
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    queued = viewport.input().drain();
    CHECK(Select<Render::InputQueue::Button>(queued).size() == 2);
    positions = Select<Render::InputQueue::Pointer>(queued);
    REQUIRE(positions.size() == 1);
    CHECK_FALSE(positions.back().position);
    frame.extra = static_cast<LPARAM>(0xFF515781);
    SendMessageW(native, WM_MOUSEMOVE, 0, MAKELPARAM(210, 230));
    SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(210, 230));
    SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(210, 230));
    platform.poll();
    CHECK(viewport.input().drain().empty());
    frame.extra = 0;
    SendMessageW(native, WM_MOUSEMOVE, 0, MAKELPARAM(250, 260));
    positions = Select<Render::InputQueue::Pointer>(viewport.input().drain());
    REQUIRE(positions.size() == 1);
    CHECK(positions[0].position == Extent2D<F32>{250, 260});
}

TEST_CASE_METHOD(TouchFixture, "Windows GLFW touch cannot steal a held physical mouse button",
                 "[core][render][windows][touch][native]") {
    pointer.reset();
    InputViewport viewport;
    GlfwInput platform(window, viewport);
    SendMessageW(native, WM_MOUSEMOVE, 0, MAKELPARAM(200, 220));
    SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(200, 220));
    REQUIRE(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    viewport.input().drain();
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    frame.time += 10;
    send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 100, 160)});
    frame.time += 10;
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    CHECK(viewport.input().drain().empty());
    CHECK(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(200, 220));
    const auto buttons = Select<Render::InputQueue::Button>(viewport.input().drain());
    REQUIRE(buttons.size() == 1);
    CHECK_FALSE(buttons[0].down);
    CHECK(buttons[0].button == MouseButton::Left);
}

TEST_CASE_METHOD(TouchFixture, "Windows touch preserves contact identity native times and client coordinates",
                 "[core][render][windows][touch][native]") {
    frame.time = 0xfffffff0u;
    send(WM_POINTERDOWN, {contact(7, POINTER_FLAG_DOWN, 125, 145)}, 7);
    REQUIRE(touches.size() == 2);
    CHECK(touches[0].type == WindowsPointer::Touch::Type::Down);
    CHECK(touches[0].id == 7);
    CHECK(touches[0].position == Extent2D<F32>{125, 145});
    CHECK(touches[0].timestamp == 0xfffffff0u);
    CHECK(touches[0].momentumAllowed);
    CHECK(touches[0].scale == DpiScale(native));
    CHECK(touches[1].type == WindowsPointer::Touch::Type::Frame);
    clear();
    frame.time = 4;
    send(WM_POINTERUPDATE, {contact(7, POINTER_FLAG_UPDATE, -15, 175)}, 7);
    REQUIRE(touches.size() == 2);
    CHECK(touches[0].type == WindowsPointer::Touch::Type::Motion);
    CHECK(touches[0].position == Extent2D<F32>{-15, 175});
    CHECK(touches[0].timestamp == 4);
    CHECK(touches[0].momentumAllowed);
    send(WM_POINTERUP, {contact(7, POINTER_FLAG_UP)}, 7);
    CHECK_FALSE(gesture.active());
}

TEST_CASE_METHOD(TouchFixture, "Windows touch emits one pinch per native frame without translation zoom",
                 "[core][render][windows][touch][native]") {
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN, 100, 120), contact(2, POINTER_FLAG_DOWN, 140, 120)});
    clear();
    send(WM_POINTERUPDATE, {contact(2, POINTER_FLAG_UPDATE, 160, 120), contact(1, POINTER_FLAG_UPDATE, 80, 120)}, 2);
    auto zooms = Select<Render::InputQueue::Zoom>(input.drain());
    REQUIRE(zooms.size() == 1);
    CHECK(zooms[0].factor == Catch::Approx(2.0f));
    CHECK(zooms[0].position == Extent2D<F32>{120, 120});
    SendMessageW(native, WM_POINTERUPDATE, 1, 0);
    CHECK(input.drain().empty());
    send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 90, 130), contact(2, POINTER_FLAG_UPDATE, 170, 130)});
    CHECK(Select<Render::InputQueue::Zoom>(input.drain()).empty());
    send(WM_POINTERUP, {contact(2, POINTER_FLAG_UP, 170, 130), contact(1, POINTER_FLAG_UPDATE, 50, 130)}, 2);
    zooms = Select<Render::InputQueue::Zoom>(input.drain());
    REQUIRE(zooms.size() == 1);
    CHECK(zooms[0].factor == Catch::Approx(1.5f));
    CHECK(zooms[0].position == Extent2D<F32>{120, 120});
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    CHECK(Select<Render::InputQueue::Button>(input.drain()).empty());
}

TEST_CASE_METHOD(TouchFixture, "Windows pinch contact arrival uses current frame positions in either packet order",
                 "[core][render][windows][touch][native]") {
    const bool newFirst = GENERATE(false, true);
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN, 100, 120)});
    clear();
    auto arrival = std::vector{contact(1, POINTER_FLAG_UPDATE, 104, 120), contact(2, POINTER_FLAG_DOWN, 140, 120)};
    if (newFirst) {
        std::reverse(arrival.begin(), arrival.end());
    }
    send(WM_POINTERDOWN, std::move(arrival), 2);
    auto queued = input.drain();
    CHECK(Select<Render::InputQueue::Zoom>(queued).empty());
    CHECK(Select<Render::InputQueue::Button>(queued).empty());
    const auto positions = Select<Render::InputQueue::Pointer>(queued);
    REQUIRE_FALSE(positions.empty());
    CHECK(positions.back().position == Extent2D<F32>{122, 120});
    send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 94, 120), contact(2, POINTER_FLAG_UPDATE, 150, 120)});
    const auto zooms = Select<Render::InputQueue::Zoom>(input.drain());
    REQUIRE(zooms.size() == 1);
    CHECK(zooms[0].factor == Catch::Approx(56.0f / 36.0f));
    CHECK(zooms[0].position == Extent2D<F32>{122, 120});
}

TEST_CASE_METHOD(TouchFixture, "Windows mixed caption and client touch frames cancel before a client release",
                 "[core][render][windows][touch][native]") {
    const bool captionFirst = GENERATE(false, true);
    const bool newFirst = GENERATE(false, true);
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    clear();
    frame.time += 50;
    frame.contacts = {contact(1, POINTER_FLAG_UP), contact(2, POINTER_FLAG_DOWN, 790, 15)};
    if (newFirst) {
        std::reverse(frame.contacts.begin(), frame.contacts.end());
    }
    stamp();
    const auto release = [&] { SendMessageW(native, WM_POINTERUP, 1, 0); };
    const auto caption = [&] { SendMessageW(native, WM_NCPOINTERDOWN, 2, 0); };
    if (captionFirst) {
        caption();
    } else {
        release();
    }
    CHECK(Select<Render::InputQueue::Button>(input.drain()).empty());
    if (captionFirst) {
        release();
    } else {
        caption();
    }
    CHECK_FALSE(gesture.active());
    CHECK(Select<Render::InputQueue::Button>(input.drain()).empty());
    gesture.poll(10.0);
    CHECK(input.drain().empty());
    frame.extra = static_cast<LPARAM>(0xFF515782);
    SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(200, 220));
    SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(200, 220));
    CHECK(mouseButtons == 2);
}

TEST_CASE_METHOD(TouchFixture, "Windows touch taps and holds use the shared deferred button contract",
                 "[core][render][windows][touch][native]") {
    const bool hold = GENERATE(false, true);
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    CHECK(Select<Render::InputQueue::Button>(input.drain()).empty());
    frame.time += hold ? 500 : 100;
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    const auto buttons = Select<Render::InputQueue::Button>(input.drain());
    REQUIRE(buttons.size() == 2);
    CHECK(buttons[0].button == (hold ? MouseButton::Right : MouseButton::Left));
    CHECK(buttons[0].down);
    CHECK_FALSE(buttons[1].down);
    CHECK(gesture.ownsPointer());
}

TEST_CASE_METHOD(TouchFixture, "Windows touch applies the final release position before classifying the gesture",
                 "[core][render][windows][touch][native]") {
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    clear();
    frame.time += 50;
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP, 100, below(40))});
    const auto queued = input.drain();
    CHECK(Select<Render::InputQueue::Button>(queued).empty());
    const auto wheels = Select<Render::InputQueue::Wheel>(queued);
    REQUIRE(wheels.size() == 2);
    CHECK(wheels[0].phase == ScrollPhase::Began);
    CHECK(wheels[0].delta.y == Catch::Approx((below(40) - 120) / (10.0f * DpiScale(native))));
    CHECK(wheels[1].phase == ScrollPhase::Ended);
    CHECK_FALSE(gesture.active());
}

TEST_CASE_METHOD(TouchFixture, "Windows touch frame deduplication does not conflate different devices",
                 "[core][render][windows][touch][native]") {
    auto first = contact(1, POINTER_FLAG_DOWN, 100, 120);
    first.pointerInfo.sourceDevice = reinterpret_cast<HANDLE>(1);
    send(WM_POINTERDOWN, {first});
    const auto frameId = frame.id;
    auto second = contact(2, POINTER_FLAG_DOWN, 140, 120);
    second.pointerInfo.sourceDevice = reinterpret_cast<HANDLE>(2);
    frame.id = frameId - 1;
    send(WM_POINTERDOWN, {second}, 2);
    CHECK(frame.id == frameId);
    REQUIRE(touches.size() == 4);
    CHECK(touches[2].type == WindowsPointer::Touch::Type::Down);
    CHECK(touches[2].id == 2);
}

TEST_CASE_METHOD(TouchFixture, "Windows unknown duplicate and foreign contacts cannot redirect a tap",
                 "[core][render][windows][touch][native]") {
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    clear();
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN, 300, 300)});
    auto foreign = contact(2, POINTER_FLAG_DOWN, 250, 250);
    foreign.pointerInfo.hwndTarget = nullptr;
    send(WM_POINTERDOWN, {foreign}, 2);
    send(WM_POINTERUPDATE, {contact(3, POINTER_FLAG_UPDATE, 250, 250)}, 3);
    send(WM_POINTERUP, {contact(3, POINTER_FLAG_UP)}, 3);
    CHECK(input.drain().empty());
    frame.time += 50;
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    CHECK(Select<Render::InputQueue::Button>(input.drain()).size() == 2);
}

TEST_CASE_METHOD(TouchFixture, "Windows focus loss cancels released touch momentum without a click",
                 "[core][render][windows][touch][native]") {
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    frame.time += 10;
    send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 100, below(20))});
    frame.time += 10;
    send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 100, below(40))});
    frame.time += 5;
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP, 100, below(40))});
    auto wheels = Select<Render::InputQueue::Wheel>(input.drain());
    REQUIRE(wheels.size() == 4);
    CHECK(wheels.back().phase == ScrollPhase::Momentum);
    pointer->focus(false);
    wheels = Select<Render::InputQueue::Wheel>(input.drain());
    REQUIRE(wheels.size() == 1);
    CHECK(wheels[0].phase == ScrollPhase::MomentumEnded);
    CHECK(wheels[0].cancel);
    gesture.poll(2.0);
    CHECK(input.drain().empty());
}

TEST_CASE_METHOD(TouchFixture, "Windows touch cancellation never clicks and ignores late contact packets",
                 "[core][render][windows][touch][native]") {
    const UINT cancellation = GENERATE(WM_CANCELMODE, WM_POINTERCAPTURECHANGED);
    const bool swipe = GENERATE(false, true);
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    if (swipe) {
        send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 100, below(40))});
    }
    clear();
    SendMessageW(native, cancellation, cancellation == WM_POINTERCAPTURECHANGED ? 1 : 0, 0);
    auto queued = input.drain();
    CHECK_FALSE(gesture.active());
    CHECK(Select<Render::InputQueue::Button>(queued).empty());
    const auto wheels = Select<Render::InputQueue::Wheel>(queued);
    REQUIRE(wheels.size() == (swipe ? 1 : 0));
    if (swipe) {
        CHECK(wheels[0].phase == ScrollPhase::Ended);
        CHECK(wheels[0].cancel);
    }
    clear();
    send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 100, below(60))});
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    gesture.poll(10.0);
    CHECK(input.drain().empty());
}

TEST_CASE_METHOD(TouchFixture, "Windows cancelled contacts and unavailable packet data cancel the gesture",
                 "[core][render][windows][touch][native]") {
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    clear();
    SECTION("native cancellation") {
        send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP | POINTER_FLAG_CANCELED)});
    }
    SECTION("lost native frame") {
        frame.failFrame = true;
        send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    }
    SECTION("lost pointer identity") {
        frame.failType = true;
        send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    }
    CHECK_FALSE(gesture.active());
    CHECK(Select<Render::InputQueue::Button>(input.drain()).empty());
}

TEST_CASE_METHOD(TouchFixture, "Windows stale touch frames cannot start released momentum",
                 "[core][render][windows][touch][native]") {
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    frame.time += 10;
    send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 100, below(20))});
    frame.time += 10;
    auto stale = contact(1, POINTER_FLAG_UPDATE, 100, below(40));
    frame.time += 251;
    send(WM_POINTERUPDATE, {stale});
    clear();
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP, 100, below(40))});
    const auto wheels = Select<Render::InputQueue::Wheel>(input.drain());
    REQUIRE(wheels.size() == 1);
    CHECK(wheels[0].phase == ScrollPhase::Ended);
    CHECK_FALSE(wheels[0].cancel);
    gesture.poll(2.0);
    CHECK(input.drain().empty());
}

TEST_CASE_METHOD(TouchFixture, "Windows touch filters promoted mouse input but preserves physical and pen input",
                 "[core][render][windows][touch][native]") {
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    frame.extra = static_cast<LPARAM>(0xFF515781);
    SendMessageW(native, WM_MOUSEMOVE, 0, MAKELPARAM(100, 120));
    SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(100, 120));
    SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(100, 120));
    SendMessageW(native, WM_MOUSEWHEEL, MAKEWPARAM(0, WHEEL_DELTA), 0);
    CHECK(mouseMoves == 0);
    CHECK(mouseButtons == 0);
    CHECK(wheel == 0.0);
    frame.extra = GENERATE(static_cast<LPARAM>(0), static_cast<LPARAM>(0xFF515701));
    SendMessageW(native, WM_MOUSEMOVE, 0, MAKELPARAM(200, 220));
    SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(200, 220));
    SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(200, 220));
    SendMessageW(native, WM_MOUSEWHEEL, MAKEWPARAM(0, WHEEL_DELTA), 0);
    CHECK(mouseMoves == 1);
    CHECK(mouseButtons == 2);
    CHECK(wheel == 1.0);
}

TEST_CASE_METHOD(TouchFixture, "Windows touch does not claim pen packets or native caption contacts",
                 "[core][render][windows][touch][native]") {
    auto pen = contact(1, POINTER_FLAG_DOWN);
    pen.pointerInfo.pointerType = PT_PEN;
    send(WM_POINTERDOWN, {pen});
    CHECK(touches.empty());
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN, 790, 15)});
    CHECK_FALSE(gesture.active());
    CHECK(Select<Render::InputQueue::Button>(input.drain()).empty());
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    CHECK(gesture.active());
    SendMessageW(native, WM_NCPOINTERDOWN, 2, 0);
    CHECK_FALSE(gesture.active());
    frame.extra = static_cast<LPARAM>(0xFF515782);
    SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(200, 220));
    SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(200, 220));
    CHECK(mouseButtons == 2);
}

TEST_CASE_METHOD(TouchFixture, "Windows titlebar touch stays native until every contact lifts",
                 "[core][render][windows][touch][native]") {
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    frame.time += 50;
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    clear();
    captionStrip = std::numeric_limits<F32>::max();
    send(WM_POINTERDOWN, {contact(2, POINTER_FLAG_DOWN)}, 2);
    captionStrip = 0.0f;
    REQUIRE(touches.size() == 1);
    CHECK(touches[0].type == WindowsPointer::Touch::Type::Cancel);
    CHECK_FALSE(gesture.active());
    clear();
    send(WM_POINTERDOWN, {contact(2, POINTER_FLAG_UPDATE), contact(3, POINTER_FLAG_DOWN, 100, 200)}, 3);
    send(WM_POINTERUPDATE, {contact(2, POINTER_FLAG_UPDATE, 140, 120), contact(3, POINTER_FLAG_UPDATE, 100, 240)}, 2);
    send(WM_POINTERUP, {contact(2, POINTER_FLAG_UP, 140, 120), contact(3, POINTER_FLAG_UPDATE, 100, 240)}, 2);
    frame.extra = static_cast<LPARAM>(0xFF515782);
    SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(140, 120));
    SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(140, 120));
    CHECK(mouseButtons == 2);
    send(WM_POINTERUP, {contact(3, POINTER_FLAG_UP, 100, 240)}, 3);
    CHECK(touches.empty());
    CHECK(input.drain().empty());
    send(WM_POINTERDOWN, {contact(4, POINTER_FLAG_DOWN)}, 4);
    CHECK(gesture.active());
}

TEST_CASE_METHOD(TouchFixture, "Windows titlebar touch without a client release cannot strand client touch",
                 "[core][render][windows][touch][native]") {
    captionStrip = std::numeric_limits<F32>::max();
    send(WM_POINTERDOWN, {contact(2, POINTER_FLAG_DOWN)}, 2);
    captionStrip = 0.0f;
    SECTION("capture loss") {
        SendMessageW(native, WM_POINTERCAPTURECHANGED, 2, 0);
    }
    SECTION("non-client release") {
    }
    clear();
    send(WM_POINTERDOWN, {contact(4, POINTER_FLAG_DOWN)}, 4);
    CHECK(gesture.active());
    REQUIRE_FALSE(touches.empty());
    CHECK(touches[0].type == WindowsPointer::Touch::Type::Down);
}

TEST_CASE_METHOD(TouchFixture, "Windows titlebar touch stays native while another device touches the client",
                 "[core][render][windows][touch][native]") {
    captionStrip = std::numeric_limits<F32>::max();
    auto held = contact(2, POINTER_FLAG_DOWN);
    held.pointerInfo.sourceDevice = reinterpret_cast<HANDLE>(1);
    send(WM_POINTERDOWN, {held}, 2);
    captionStrip = 0.0f;
    clear();
    auto other = contact(3, POINTER_FLAG_DOWN);
    other.pointerInfo.sourceDevice = reinterpret_cast<HANDLE>(2);
    send(WM_POINTERDOWN, {other}, 3);
    held.pointerInfo.pointerFlags = POINTER_FLAG_UPDATE;
    send(WM_POINTERUPDATE, {held}, 2);
    CHECK(touches.empty());
    CHECK_FALSE(gesture.active());
}

TEST_CASE("Windows touchscreen input does not depend on Direct Manipulation initialization",
          "[core][render][windows][touch][native]") {
    TouchFixture fixture(true);
    CHECK_FALSE(fixture.pointer->available());
    fixture.send(WM_POINTERDOWN, {fixture.contact(1, POINTER_FLAG_DOWN)});
    CHECK(fixture.gesture.active());
    fixture.frame.time += 50;
    fixture.send(WM_POINTERUP, {fixture.contact(1, POINTER_FLAG_UP)});
    CHECK(Select<Render::InputQueue::Button>(fixture.input.drain()).size() == 2);
}

TEST_CASE_METHOD(TouchFixture, "Windows pointer destruction detaches without touch callbacks",
                 "[core][render][windows][touch][native]") {
    const int phase = GENERATE(0, 1, 2);
    send(WM_POINTERDOWN, {contact(1, POINTER_FLAG_DOWN)});
    if (phase > 0) {
        frame.time += 10;
        send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 100, below(20))});
        frame.time += 10;
        send(WM_POINTERUPDATE, {contact(1, POINTER_FLAG_UPDATE, 100, below(40))});
    }
    if (phase == 2) {
        frame.time += 5;
        send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP, 100, below(40))});
    }
    clear();
    pointer.reset();
    CHECK(touches.empty());
    CHECK(input.drain().empty());
    send(WM_POINTERUP, {contact(1, POINTER_FLAG_UP)});
    SendMessageW(native, WM_CANCELMODE, 0, 0);
    CHECK(touches.empty());
    CHECK(Select<Render::InputQueue::Button>(input.drain()).empty());
}

TEST_CASE("Windows injected touch reaches the native adapter without mouse promotion",
          "[core][render][windows][touch][native][injection]") {
    std::string enabled;
    if (Platform::EnvironmentVariable("JETSTREAM_TEST_NATIVE_TOUCH", enabled) != Result::SUCCESS) {
        SKIP("Set JETSTREAM_TEST_NATIVE_TOUCH=1 to inject touch into a visible test window.");
    }
    TouchFixture fixture;
    nativeFrame = nullptr;
    glfwSetWindowPos(fixture.window, 20, 20);
    glfwSetWindowSize(fixture.window, 400, 400);
    glfwShowWindow(fixture.window);
    SetForegroundWindow(fixture.native);
    REQUIRE(InitializeTouchInjection(2, TOUCH_FEEDBACK_NONE));
    const auto pump = [&] {
        const F64 end = glfwGetTime() + 0.02;
        do {
            fixture.pointer->poll();
            glfwPollEvents();
            Sleep(1);
        } while (glfwGetTime() < end);
    };
    pump();
    fixture.clear();
    fixture.mouseButtons = 0;
    const auto contact = [&](UINT32 id, LONG x, LONG y) {
        POINTER_TOUCH_INFO info{};
        info.pointerInfo.pointerType = PT_TOUCH;
        info.pointerInfo.pointerId = id;
        info.pointerInfo.ptPixelLocation = {x, y};
        ClientToScreen(fixture.native, &info.pointerInfo.ptPixelLocation);
        if (WindowFromPoint(info.pointerInfo.ptPixelLocation) != fixture.native) {
            SKIP("The visible touch test window is obscured, so no touch is injected into another window.");
        }
        return info;
    };
    struct Injection {
        std::vector<POINTER_TOUCH_INFO> contacts;

        ~Injection() {
            for (auto& info : contacts) {
                info.pointerInfo.pointerFlags = POINTER_FLAG_UP | POINTER_FLAG_CANCELED;
            }
            if (!contacts.empty()) {
                InjectTouchInput(static_cast<UINT32>(contacts.size()), contacts.data());
            }
        }

        void send(POINTER_FLAGS flags) {
            for (auto& info : contacts) {
                info.pointerInfo.pointerFlags = flags;
            }
            REQUIRE(InjectTouchInput(static_cast<UINT32>(contacts.size()), contacts.data()));
            if (flags & POINTER_FLAG_UP) {
                contacts.clear();
            }
        }
    } injection;
    injection.contacts = {contact(1, 100, 120)};
    injection.send(POINTER_FLAG_DOWN | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT);
    pump();
    CHECK(fixture.gesture.active());
    CHECK(Select<Render::InputQueue::Button>(fixture.input.drain()).empty());
    injection.send(POINTER_FLAG_UP);
    pump();
    CHECK(Select<Render::InputQueue::Button>(fixture.input.drain()).size() == 2);
    CHECK(fixture.mouseButtons == 0);

    injection.contacts = {contact(1, 90, 150), contact(2, 150, 150)};
    injection.send(POINTER_FLAG_DOWN | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT);
    pump();
    fixture.clear();
    injection.contacts = {contact(1, 70, 150), contact(2, 170, 150)};
    injection.send(POINTER_FLAG_UPDATE | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT);
    pump();
    const auto zooms = Select<Render::InputQueue::Zoom>(fixture.input.drain());
    REQUIRE(zooms.size() == 1);
    CHECK(zooms[0].factor == Catch::Approx(100.0f / 60.0f));
    CHECK(zooms[0].position == Extent2D<F32>{120, 150});
    injection.send(POINTER_FLAG_UP);
    pump();
    CHECK(Select<Render::InputQueue::Button>(fixture.input.drain()).empty());
    CHECK(fixture.mouseButtons == 0);
}

int main(int argc, char** argv) {
    return Catch::Session().run(argc, argv);
}
