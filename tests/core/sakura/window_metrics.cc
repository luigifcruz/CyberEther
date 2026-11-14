#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <jetstream/render/base/window_metrics.hh>

#include "render/sakura/metrics.hh"

#include <cmath>

using namespace Jetstream;
using Catch::Matchers::WithinAbs;

namespace {

using Metrics = Render::WindowMetrics;
using Policy = Render::WindowMetrics::Policy;

Metrics Retina(F32 interfaceScale, Policy policy = Policy::Interface) {
    return Metrics::Compute({1512.0f, 982.0f}, {3024.0f, 1964.0f}, {2.0f, 2.0f}, interfaceScale, policy);
}

Metrics Standard(F32 interfaceScale, Policy policy = Policy::Interface) {
    return Metrics::Compute({1920.0f, 1080.0f}, {1920.0f, 1080.0f}, {1.0f, 1.0f}, interfaceScale, policy);
}

Metrics X11(F32 interfaceScale, F32 contentScale = 1.5f) {
    return Metrics::Compute({1920.0f, 1080.0f}, {1920.0f, 1080.0f}, {contentScale, contentScale}, interfaceScale,
                            Policy::ContentScale);
}

}  // namespace

TEST_CASE("Retina displays keep the interface scale and double the framebuffer", "[core][sakura][window-metrics]") {
    const auto metrics = Retina(1.0f);

    REQUIRE(metrics.displaySize == Extent2D<F32>{1512.0f, 982.0f});
    REQUIRE(metrics.framebufferSize == Extent2D<F32>{3024.0f, 1964.0f});
    REQUIRE(metrics.framebufferScale == Extent2D<F32>{2.0f, 2.0f});
    REQUIRE(metrics.scalingFactor == 1.0f);
    REQUIRE(metrics.pixelRatio == 2.0f);
    REQUIRE(metrics.presentationSize() == Extent2D<U64>{3024, 1964});

    const Rect region = Sakura::DisplayMetrics::RootRegion(metrics, 28.0f);
    REQUIRE(region == Rect{0.0f, 28.0f, 1512.0f, 954.0f});

    const auto logical = Sakura::DisplayMetrics::LogicalSize(metrics, {region.width, region.height});
    REQUIRE(logical == Extent2D<F32>{1512.0f, 954.0f});
    REQUIRE(Sakura::DisplayMetrics::CanvasFramebuffer(metrics, logical) == Extent2D<U64>{3024, 1908});

    REQUIRE(Sakura::DisplayMetrics::Snap(metrics, {10.3f, 20.8f}) == Extent2D<F32>{10.5f, 21.0f});

    const auto scissor = metrics.scissor({10.25f, 20.0f, 100.0f, 50.0f});
    REQUIRE(scissor.x == 20);
    REQUIRE(scissor.y == 40);
    REQUIRE(scissor.width == 201);
    REQUIRE(scissor.height == 100);
}

TEST_CASE("Retina displays ignore the content scale under MoltenVK", "[core][sakura][window-metrics]") {
    REQUIRE(Retina(1.0f, Policy::ContentScale) == Retina(1.0f, Policy::Interface));
    REQUIRE(Retina(1.5f, Policy::ContentScale).scalingFactor == 1.5f);
}

TEST_CASE("Standard displays map logical units one to one", "[core][sakura][window-metrics]") {
    const auto metrics = Standard(1.0f);

    REQUIRE(metrics.framebufferScale == Extent2D<F32>{1.0f, 1.0f});
    REQUIRE(metrics.scalingFactor == 1.0f);
    REQUIRE(metrics.pixelRatio == 1.0f);

    const Rect region = Sakura::DisplayMetrics::RootRegion(metrics, 0.0f);
    REQUIRE(region == Rect{0.0f, 0.0f, 1920.0f, 1080.0f});
    const auto logical = Sakura::DisplayMetrics::LogicalSize(metrics, {region.width, region.height});
    REQUIRE(Sakura::DisplayMetrics::CanvasFramebuffer(metrics, logical) == Extent2D<U64>{1920, 1080});
    REQUIRE(Sakura::DisplayMetrics::Snap(metrics, {10.4f, 10.6f}) == Extent2D<F32>{10.0f, 11.0f});
    REQUIRE(Standard(1.0f, Policy::ContentScale) == metrics);
}

TEST_CASE("Vulkan on X11 multiplies the interface scale by the content scale", "[core][sakura][window-metrics]") {
    const auto metrics = X11(1.0f);

    REQUIRE(metrics.framebufferScale == Extent2D<F32>{1.0f, 1.0f});
    REQUIRE(metrics.scalingFactor == 1.5f);
    REQUIRE(metrics.pixelRatio == 1.5f);

    const Rect region = Sakura::DisplayMetrics::RootRegion(metrics, 28.0f);
    REQUIRE(region == Rect{0.0f, 42.0f, 1920.0f, 1038.0f});

    const auto logical = Sakura::DisplayMetrics::LogicalSize(metrics, {region.width, region.height});
    REQUIRE(logical == Extent2D<F32>{1280.0f, 692.0f});
    REQUIRE(Sakura::DisplayMetrics::CanvasFramebuffer(metrics, logical) == Extent2D<U64>{1920, 1038});

    const auto unscaled = Metrics::Compute({1920.0f, 1080.0f}, {1920.0f, 1080.0f}, {1.5f, 1.5f}, 1.0f,
                                           Policy::Interface);
    REQUIRE(unscaled.scalingFactor == 1.0f);

    REQUIRE(X11(1.0f, 1.0f).scalingFactor == 1.0f);
    REQUIRE(X11(1.0f, 0.75f).scalingFactor == 1.0f);
}

TEST_CASE("Minimized windows fall back to a unit framebuffer scale", "[core][sakura][window-metrics]") {
    const auto minimized = Metrics::Compute({0.0f, 0.0f}, {0.0f, 0.0f}, {2.0f, 2.0f}, 1.25f, Policy::Interface);

    REQUIRE(minimized.framebufferScale == Extent2D<F32>{1.0f, 1.0f});
    REQUIRE(minimized.scalingFactor == 1.25f);
    REQUIRE(minimized.pixelRatio == 1.25f);
    REQUIRE(minimized.presentationSize() == Extent2D<U64>{0, 0});
    REQUIRE(Sakura::DisplayMetrics::RootRegion(minimized, 28.0f).height == 0.0f);

    const auto collapsed = Metrics::Compute({800.0f, 0.0f}, {1600.0f, 0.0f}, {2.0f, 2.0f}, 1.0f, Policy::Interface);
    REQUIRE(collapsed.framebufferScale == Extent2D<F32>{1.0f, 1.0f});

    const auto scissor = minimized.scissor({0.0f, 0.0f, 100.0f, 100.0f});
    REQUIRE(scissor.width == 0);
    REQUIRE(scissor.height == 0);

    const auto minimizedX11 = Metrics::Compute({0.0f, 0.0f}, {0.0f, 0.0f}, {1.5f, 1.5f}, 1.0f, Policy::ContentScale);
    REQUIRE(minimizedX11.scalingFactor == 1.5f);
}

TEST_CASE("Moving between monitors recomputes every derived value", "[core][sakura][window-metrics]") {
    const auto retina = Retina(1.0f);
    const auto external = Metrics::Compute({1512.0f, 982.0f}, {1512.0f, 982.0f}, {1.0f, 1.0f}, 1.0f, Policy::Interface);

    REQUIRE(retina != external);
    REQUIRE(external.framebufferScale == Extent2D<F32>{1.0f, 1.0f});
    REQUIRE(external.pixelRatio == 1.0f);

    const Extent2D<F32> logical = {640.0f, 480.0f};
    REQUIRE(Sakura::DisplayMetrics::CanvasFramebuffer(retina, logical) == Extent2D<U64>{1280, 960});
    REQUIRE(Sakura::DisplayMetrics::CanvasFramebuffer(external, logical) == Extent2D<U64>{640, 480});
    REQUIRE(Sakura::DisplayMetrics::RootRegion(retina, 28.0f) == Sakura::DisplayMetrics::RootRegion(external, 28.0f));

    REQUIRE(Retina(1.0f) == retina);

    const auto x11Low = X11(1.0f, 1.0f);
    const auto x11High = X11(1.0f, 2.0f);
    REQUIRE(x11Low.scalingFactor == 1.0f);
    REQUIRE(x11High.scalingFactor == 2.0f);
    REQUIRE(Sakura::DisplayMetrics::CanvasFramebuffer(x11Low, logical) == Extent2D<U64>{640, 480});
    REQUIRE(Sakura::DisplayMetrics::CanvasFramebuffer(x11High, logical) == Extent2D<U64>{1280, 960});
}

TEST_CASE("Interface scale composes with every display configuration", "[core][sakura][window-metrics]") {
    const F32 scale = GENERATE(0.75f, 1.0f, 1.5f);
    CAPTURE(scale);

    const auto retina = Retina(scale);
    REQUIRE(retina.scalingFactor == scale);
    REQUIRE(retina.pixelRatio == scale * 2.0f);

    const auto standard = Standard(scale);
    REQUIRE(standard.scalingFactor == scale);
    REQUIRE(standard.pixelRatio == scale);

    const auto x11 = X11(scale);
    REQUIRE(x11.scalingFactor == scale * 1.5f);
    REQUIRE(x11.pixelRatio == scale * 1.5f);

    for (const auto& metrics : {retina, standard, x11}) {
        const Rect region = Sakura::DisplayMetrics::RootRegion(metrics, 28.0f);
        REQUIRE(region.y == std::trunc(28.0f * metrics.scalingFactor));
        REQUIRE(region.height == std::trunc(metrics.displaySize.y - 28.0f * metrics.scalingFactor));
        REQUIRE(Sakura::DisplayMetrics::RootClip(metrics, region) == region);

        const auto logical = Sakura::DisplayMetrics::LogicalSize(metrics, {region.width, region.height});
        REQUIRE_THAT(logical.x, WithinAbs(region.width / metrics.scalingFactor, 1e-3));

        const auto framebuffer = Sakura::DisplayMetrics::CanvasFramebuffer(metrics, logical);
        const auto presentation = metrics.presentationSize();
        REQUIRE(framebuffer.x == presentation.x);
        REQUIRE(framebuffer.y <= presentation.y);
    }
}

TEST_CASE("Root regions truncate fractional offsets like the host window did", "[core][sakura][window-metrics]") {
    const auto metrics = X11(0.75f);
    REQUIRE(metrics.scalingFactor == 1.125f);

    const Rect region = Sakura::DisplayMetrics::RootRegion(metrics, 28.0f);
    CHECK(region == Rect{0.0f, 31.0f, 1920.0f, 1048.0f});
    CHECK(region.bottom() == 1079.0f);
    CHECK(Sakura::DisplayMetrics::RootClip(metrics, region) == region);

    const auto odd = Metrics::Compute({1000.5f, 700.5f}, {2001.0f, 1401.0f}, {2.0f, 2.0f}, 1.0f, Policy::Interface);
    const Rect oddRegion = Sakura::DisplayMetrics::RootRegion(odd, 0.0f);
    CHECK(oddRegion == Rect{0.0f, 0.0f, 1000.0f, 700.0f});
    CHECK(Sakura::DisplayMetrics::RootClip(odd, {-5.0f, 10.4f, 2000.0f, 20.0f}) == Rect{0.0f, 10.0f, 1000.5f, 20.0f});
}

TEST_CASE("Fractional framebuffer scales truncate like the presentation backends", "[core][sakura][window-metrics]") {
    const auto metrics = Metrics::Compute({1001.0f, 701.0f}, {1502.0f, 1052.0f}, {1.5f, 1.5f}, 1.0f, Policy::Interface);

    REQUIRE_THAT(metrics.framebufferScale.x, WithinAbs(1502.0 / 1001.0, 1e-6));
    REQUIRE(metrics.presentationSize().x <= 1502);
    REQUIRE(metrics.presentationSize().x >= 1501);

    const auto scissor = metrics.scissor({-10.0f, -10.0f, 5000.0f, 5000.0f});
    REQUIRE(scissor.x == 0);
    REQUIRE(scissor.y == 0);
    REQUIRE(scissor.width == metrics.presentationSize().x);
    REQUIRE(scissor.height == metrics.presentationSize().y);

    const auto snapped = Sakura::DisplayMetrics::Snap(metrics, {100.0f, 100.0f});
    const F32 framebufferX = snapped.x * metrics.framebufferScale.x;
    REQUIRE_THAT(framebufferX, WithinAbs(std::round(framebufferX), 1e-3));
}

TEST_CASE("Surface scale, pixel ratio and logical height derive from the window metrics",
          "[core][sakura][window-metrics]") {
    for (const auto& metrics : {Retina(1.0f), Retina(1.5f), Standard(0.75f), X11(1.0f)}) {
        const F32 surfaceScale = Sakura::DisplayMetrics::SurfaceScale(metrics);
        REQUIRE(surfaceScale == metrics.pixelRatio * 0.5f);
        const Sakura::SurfaceResize resize = {.logicalSize = {1, 1}, .framebufferSize = {1, 1}, .scale = surfaceScale};
        REQUIRE(Sakura::DisplayMetrics::PixelRatio(resize) == metrics.pixelRatio);

        const auto framebuffer = Sakura::DisplayMetrics::CanvasFramebuffer(metrics, {640.0f, 480.0f});
        const auto display = Sakura::DisplayMetrics::FramebufferToDisplay(metrics, framebuffer);
        REQUIRE_THAT(display.x, WithinAbs(640.0f * metrics.scalingFactor, 1.0));
        REQUIRE_THAT(Sakura::DisplayMetrics::LogicalHeight(metrics, framebuffer.y), WithinAbs(480.0, 1.0));
    }

    const auto anisotropic = Metrics::Compute({800.0f, 600.0f}, {800.0f, 1200.0f}, {1.0f, 1.0f}, 1.0f,
                                              Policy::Interface);
    REQUIRE(Sakura::DisplayMetrics::LogicalHeight(anisotropic, 1200) == 600.0f);
    REQUIRE(Sakura::DisplayMetrics::PixelRatio({}) == 2.0f);
}
