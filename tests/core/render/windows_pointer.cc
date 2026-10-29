#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <vector>

#include "../../../src/viewport/glfw/tools/windows_pointer.cc"
#include "viewport/glfw/tools/windows_titlebar.hh"

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

int main(int argc, char** argv) {
    return Catch::Session().run(argc, argv);
}
