#include "viewport/glfw/tools/windows_titlebar.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <windowsx.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

struct Fixture {
    GLFWwindow* handle = nullptr;
    HWND native = nullptr;
    int releases = 0;

    Fixture() {
        REQUIRE(glfwInit());
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        handle = glfwCreateWindow(800, 600, "Titlebar test", nullptr, nullptr);
        REQUIRE(handle);
        native = glfwGetWin32Window(handle);
    }

    ~Fixture() {
        glfwDestroyWindow(handle);
        glfwTerminate();
    }

    POINT origin() const {
        POINT point{};
        ClientToScreen(native, &point);
        return point;
    }

    LRESULT hit(int x, int y) const {
        const POINT point = origin();
        return SendMessageW(native, WM_NCHITTEST, 0, MAKELPARAM(point.x + x, point.y + y));
    }
};

}  // namespace

TEST_CASE_METHOD(Fixture, "Windows integrated titlebar reclaims the caption and restores the native frame",
                 "[core][render][windows][titlebar]") {
    const POINT before = origin();
    const LONG_PTR style = GetWindowLongPtrW(native, GWL_STYLE);
    {
        Jetstream::Viewport::WindowsTitlebar bar(handle, [&] { ++releases; });
        CHECK(bar.chrome().integrated);
        CHECK(bar.chrome().leadingInset == 0.0f);
        CHECK(bar.chrome().trailingInset > 0.0f);
        CHECK(bar.chrome().height == MulDiv(40, static_cast<int>(GetDpiForWindow(native)), 96));
        CHECK(origin().y < before.y);
        CHECK(GetWindowLongPtrW(native, GWL_STYLE) == style);
        CHECK(hit(400, 0) == HTTOP);
        CHECK(hit(1, 0) == HTTOPLEFT);
        CHECK(hit(400, 25) == HTCLIENT);
        CHECK(hit(400, 200) == HTCLIENT);
        RECT client{};
        GetClientRect(native, &client);
        const int buttonWidth = MulDiv(46, static_cast<int>(GetDpiForWindow(native)), 96);
        CHECK(hit(client.right - buttonWidth / 2, 20) == HTCLOSE);
        CHECK(hit(client.right - 3 * buttonWidth / 2, 20) == HTMAXBUTTON);
        CHECK(hit(client.right - 5 * buttonWidth / 2, 20) == HTMINBUTTON);
        const int buttonHeight = MulDiv(34, static_cast<int>(GetDpiForWindow(native)), 96);
        CHECK(hit(client.right - buttonWidth / 2, buttonHeight - 1) == HTCLOSE);
        CHECK(hit(client.right - 3 * buttonWidth / 2, buttonHeight - 1) == HTMAXBUTTON);
        CHECK(hit(client.right - buttonWidth / 2, buttonHeight) == HTCLIENT);
        CHECK(hit(client.right - 3 * buttonWidth / 2, buttonHeight) == HTCLIENT);
        RECT window{};
        GetWindowRect(native, &window);
        SetWindowPos(native, nullptr, -500, -200, 0, 0,
                     SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        CHECK(hit(400, 25) == HTCLIENT);
        CHECK(hit(client.right - buttonWidth / 2, 20) == HTCLOSE);
        SetWindowPos(native, nullptr, window.left, window.top, 0, 0,
                     SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        bar.beginDrag();
        bar.poll();
        CHECK(releases == 0);
    }
    CHECK(origin().y == before.y);
    CHECK(GetWindowLongPtrW(native, GWL_STYLE) == style);
}

TEST_CASE_METHOD(Fixture, "Windows titlebar cancels released presses and pending work on teardown",
                 "[core][render][windows][titlebar]") {
    {
        Jetstream::Viewport::WindowsTitlebar bar(handle, [&] { ++releases; });
        SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(400, 25));
        bar.beginDrag();
        SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(400, 25));
        bar.poll();
        CHECK(releases == 1);
        SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(400, 25));
        bar.beginDrag();
        bar.zoom();
    }
    CHECK(releases == 2);
    CHECK_FALSE(IsZoomed(native));
    CHECK(IsWindow(native));
    CHECK(Jetstream::Viewport::WindowsTitlebar::SetAppearance(handle, true));
    glfwPollEvents();
    CHECK(IsWindow(native));
    SendMessageW(native, WM_CLOSE, 0, 0);
    CHECK(glfwWindowShouldClose(handle));
}

TEST_CASE_METHOD(Fixture, "Windows titlebar survives destruction of its native window",
                 "[core][render][windows][titlebar]") {
    Jetstream::Viewport::WindowsTitlebar bar(handle, [&] { ++releases; });
    SendMessageW(native, WM_LBUTTONDOWN, MK_LBUTTON, MAKELPARAM(400, 25));
    bar.beginDrag();
    glfwDestroyWindow(handle);
    handle = nullptr;
    CHECK_FALSE(bar.chrome().integrated);
    CHECK(releases == 1);
    bar.poll();
}

TEST_CASE_METHOD(Fixture, "Windows caption buttons fit the compact header and cancel outside releases",
                 "[core][render][windows][titlebar]") {
    Jetstream::Viewport::WindowsTitlebar bar(handle, [] {});
    RECT client{};
    GetClientRect(native, &client);
    const int x = client.right - 10;
    const int y = MulDiv(34, static_cast<int>(GetDpiForWindow(native)), 96) - 1;
    REQUIRE(hit(x, y) == HTCLOSE);
    const POINT point = origin();
    SendMessageW(native, WM_NCLBUTTONDOWN, HTCLOSE, MAKELPARAM(point.x + x, point.y + y));
    SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(400, 200));
    glfwPollEvents();
    CHECK_FALSE(glfwWindowShouldClose(handle));
    SendMessageW(native, WM_NCLBUTTONDOWN, HTCLOSE, MAKELPARAM(point.x + x, point.y + y));
    SendMessageW(native, WM_LBUTTONUP, 0, MAKELPARAM(x, y));
    glfwPollEvents();
    CHECK(glfwWindowShouldClose(handle));
}

int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
