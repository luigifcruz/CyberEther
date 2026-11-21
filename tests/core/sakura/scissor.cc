#include <catch2/catch_test_macros.hpp>

#include "render/sakura/detail/helpers.hh"

using namespace Jetstream;

namespace {

void RequireScissor(const Rect& rect, const Extent2D<U64>& framebufferSize,
                    const Render::ScissorRect& expected) {
    CAPTURE(rect.x, rect.y, rect.width, rect.height, framebufferSize.x, framebufferSize.y);
    const auto actual = Sakura::RectToScissor(rect, framebufferSize);
    CHECK(actual.x == expected.x);
    CHECK(actual.y == expected.y);
    CHECK(actual.width == expected.width);
    CHECK(actual.height == expected.height);
}

}  // namespace

TEST_CASE("Scissor rounding includes fractional far-edge rows and columns", "[core][sakura][scissor]") {
    const Extent2D<U64> framebuffer = {100, 80};

    RequireScissor({10.5f, 20.25f, 1.0f, 1.0f}, framebuffer, {10, 20, 2, 2});
    RequireScissor({10.75f, 20.875f, 0.5f, 0.25f}, framebuffer, {10, 20, 2, 2});
    RequireScissor({10.25f, 20.25f, 0.25f, 0.5f}, framebuffer, {10, 20, 1, 1});
}

TEST_CASE("Scissor rounding keeps exact pixel boundaries", "[core][sakura][scissor]") {
    const Extent2D<U64> framebuffer = {100, 80};

    RequireScissor({10.0f, 20.0f, 30.0f, 40.0f}, framebuffer, {10, 20, 30, 40});
    RequireScissor({10.25f, 20.75f, 0.75f, 0.25f}, framebuffer, {10, 20, 1, 1});
    RequireScissor({10.0f, 20.0f, 5.5f, 0.25f}, framebuffer, {10, 20, 6, 1});
}

TEST_CASE("Scissor rounding clips fractional bounds at framebuffer edges", "[core][sakura][scissor]") {
    const Extent2D<U64> framebuffer = {100, 80};

    RequireScissor({-0.75f, -1.25f, 2.0f, 3.0f}, framebuffer, {0, 0, 2, 2});
    RequireScissor({-10.5f, 20.5f, 12.0f, 1.0f}, framebuffer, {0, 20, 2, 2});
    RequireScissor({10.5f, -20.5f, 1.0f, 22.0f}, framebuffer, {10, 0, 2, 2});
    RequireScissor({98.5f, 78.25f, 5.0f, 4.0f}, framebuffer, {98, 78, 2, 2});
    RequireScissor({99.75f, 79.75f, 0.5f, 0.5f}, framebuffer, {99, 79, 1, 1});
}

TEST_CASE("Scissor rounding leaves fully clipped axes empty", "[core][sakura][scissor]") {
    const Extent2D<U64> framebuffer = {100, 80};

    RequireScissor({-2.5f, 20.5f, 1.0f, 1.0f}, framebuffer, {0, 20, 0, 2});
    RequireScissor({100.5f, 20.5f, 1.0f, 1.0f}, framebuffer, {100, 20, 0, 2});
    RequireScissor({10.5f, -2.5f, 1.0f, 1.0f}, framebuffer, {10, 0, 2, 0});
    RequireScissor({10.5f, 80.5f, 1.0f, 1.0f}, framebuffer, {10, 80, 2, 0});
    RequireScissor({-1.25f, -2.5f, 1.25f, 2.5f}, framebuffer, {0, 0, 0, 0});
}

TEST_CASE("Scissor rounding preserves empty and inverted fractional axes independently",
          "[core][sakura][scissor]") {
    const Extent2D<U64> framebuffer = {100, 80};

    RequireScissor({10.5f, 20.5f, 0.0f, 0.0f}, framebuffer, {10, 20, 0, 0});
    RequireScissor({10.5f, 20.5f, -0.25f, -0.25f}, framebuffer, {10, 20, 0, 0});
    RequireScissor({10.5f, 20.5f, 0.0f, 1.0f}, framebuffer, {10, 20, 0, 2});
    RequireScissor({10.5f, 20.5f, 1.0f, 0.0f}, framebuffer, {10, 20, 2, 0});
    RequireScissor({10.5f, 20.5f, -0.25f, 1.0f}, framebuffer, {10, 20, 0, 2});
    RequireScissor({10.5f, 20.5f, 1.0f, -0.25f}, framebuffer, {10, 20, 2, 0});
}

TEST_CASE("Scissor rounding preserves zero framebuffer axes", "[core][sakura][scissor]") {
    const Rect rect = {10.5f, 20.5f, 1.0f, 1.0f};

    RequireScissor(rect, {0, 80}, {0, 20, 0, 2});
    RequireScissor(rect, {100, 0}, {10, 0, 2, 0});
    RequireScissor(rect, {0, 0}, {0, 0, 0, 0});
}
