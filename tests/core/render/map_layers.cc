#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/render/components/map_layer.hh"
#include "render/components/geomap_lines.hh"

using namespace Jetstream;
using namespace Jetstream::Render::Components;
using Catch::Approx;

namespace {
class TestLayer final : public MapLayer {
 public:
    TestLayer(std::string id, std::vector<std::string>& calls)
        : id(std::move(id)), calls(calls) {}

    Result create(Render::Window*, const MapContext&) override {
        calls.push_back(id + ":create");
        if (throwOnCreate) throw std::runtime_error("layer create failed");
        return createResult;
    }
    Result destroy(Render::Window*) override {
        calls.push_back(id + ":destroy");
        if (throwOnDestroy) throw std::runtime_error("layer destroy failed");
        return destroyResult;
    }
    Result surface(Render::Surface::Config& config) override {
        calls.push_back(id + ":surface");
        config.programs.push_back(nullptr);
        return surfaceResult;
    }
    Result present(const MapContext& context) override {
        calls.push_back(id + ":present");
        frame = &context;
        return Result::SUCCESS;
    }
    EventResult onEvent(const MouseEvent&, const MapContext&) override {
        calls.push_back(id + ":event");
        return eventResult;
    }

    Result createResult = Result::SUCCESS;
    Result destroyResult = Result::SUCCESS;
    Result surfaceResult = Result::SUCCESS;
    EventResult eventResult = EventResult::Ignored;
    bool throwOnCreate = false;
    bool throwOnDestroy = false;
    const MapContext* frame = nullptr;

 private:
    std::string id;
    std::vector<std::string>& calls;
};
}  // namespace

TEST_CASE("Map layers share one frame and preserve paint/lifecycle order", "[render][geomap]") {
    MapContext context;
    MapLayerStack stack;
    std::vector<std::string> calls;
    auto base = std::make_shared<TestLayer>("base", calls);
    auto overlay = std::make_shared<TestLayer>("overlay", calls);
    REQUIRE(stack.add(base) == Result::SUCCESS);
    REQUIRE(stack.add(overlay) == Result::SUCCESS);
    REQUIRE(stack.add(base) == Result::ERROR);
    REQUIRE(stack.add(nullptr) == Result::ERROR);
    REQUIRE(stack.present(context) == Result::ERROR);
    REQUIRE(stack.create(nullptr, context) == Result::SUCCESS);
    REQUIRE(stack.create(nullptr, context) == Result::ERROR);
    REQUIRE(stack.add(std::make_shared<TestLayer>("late", calls)) == Result::ERROR);

    Render::Surface::Config surface;
    REQUIRE(stack.surface(surface) == Result::SUCCESS);
    REQUIRE(surface.programs.size() == 2);
    REQUIRE(stack.present(context) == Result::SUCCESS);
    REQUIRE(base->frame == &context);
    REQUIRE(overlay->frame == &context);
    REQUIRE(stack.destroy(nullptr) == Result::SUCCESS);
    REQUIRE(calls == std::vector<std::string>{
        "base:create", "overlay:create", "base:surface", "overlay:surface",
        "base:present", "overlay:present", "overlay:destroy", "base:destroy",
    });
    REQUIRE(stack.destroy(nullptr) == Result::SUCCESS);
    REQUIRE(stack.create(nullptr, context) == Result::SUCCESS);
    REQUIRE(stack.destroy(nullptr) == Result::SUCCESS);
}

TEST_CASE("Map layer creation rolls back partial resources", "[render][geomap]") {
    MapContext context;
    MapLayerStack stack;
    std::vector<std::string> calls;
    auto base = std::make_shared<TestLayer>("base", calls);
    auto failed = std::make_shared<TestLayer>("failed", calls);
    auto later = std::make_shared<TestLayer>("later", calls);
    REQUIRE(stack.add(base) == Result::SUCCESS);
    REQUIRE(stack.add(failed) == Result::SUCCESS);
    REQUIRE(stack.add(later) == Result::SUCCESS);
    SECTION("returned error") {
        failed->createResult = Result::ERROR;
        REQUIRE(stack.create(nullptr, context) == Result::ERROR);
    }
    SECTION("exception") {
        failed->throwOnCreate = true;
        REQUIRE_THROWS_AS(stack.create(nullptr, context), std::runtime_error);
    }
    REQUIRE(calls == std::vector<std::string>{
        "base:create", "failed:create", "failed:destroy", "base:destroy",
    });
    REQUIRE(stack.destroy(nullptr) == Result::SUCCESS);
    REQUIRE(calls.size() == 4);
    failed->createResult = Result::SUCCESS;
    failed->throwOnCreate = false;
    REQUIRE(stack.create(nullptr, context) == Result::SUCCESS);
    REQUIRE(stack.destroy(nullptr) == Result::SUCCESS);
}

TEST_CASE("Map composition failure leaves the caller's surface intact", "[render][geomap]") {
    MapContext context;
    MapLayerStack stack;
    std::vector<std::string> calls;
    auto layer = std::make_shared<TestLayer>("layer", calls);
    layer->surfaceResult = Result::ERROR;
    layer->destroyResult = Result::ERROR;
    REQUIRE(stack.add(std::make_shared<TestLayer>("base", calls)) == Result::SUCCESS);
    REQUIRE(stack.add(layer) == Result::SUCCESS);
    REQUIRE(stack.create(nullptr, context) == Result::SUCCESS);
    Render::Surface::Config surface;
    REQUIRE(stack.surface(surface) == Result::ERROR);
    REQUIRE(surface.programs.empty());
    REQUIRE(stack.destroy(nullptr) == Result::ERROR);
    REQUIRE(calls.back() == "base:destroy");
}

TEST_CASE("Map layers accept successful reloads and clean up after teardown exceptions", "[render][geomap]") {
    MapContext context;
    MapLayerStack stack;
    std::vector<std::string> calls;
    auto base = std::make_shared<TestLayer>("base", calls);
    auto overlay = std::make_shared<TestLayer>("overlay", calls);
    base->createResult = Result::RELOAD;
    REQUIRE(stack.add(base) == Result::SUCCESS);
    REQUIRE(stack.add(overlay) == Result::SUCCESS);
    REQUIRE(stack.create(nullptr, context) == Result::SUCCESS);
    overlay->throwOnDestroy = true;
    REQUIRE_THROWS_AS(stack.destroy(nullptr), std::runtime_error);
    REQUIRE(calls.back() == "base:destroy");
    REQUIRE(stack.destroy(nullptr) == Result::SUCCESS);
    overlay->throwOnDestroy = false;
    REQUIRE(stack.create(nullptr, context) == Result::SUCCESS);
    REQUIRE(stack.destroy(nullptr) == Result::SUCCESS);
}

TEST_CASE("Map overlays receive events topmost first and can capture the pointer", "[render][geomap]") {
    MapContext context;
    MapLayerStack stack;
    std::vector<std::string> calls;
    auto base = std::make_shared<TestLayer>("base", calls);
    auto card = std::make_shared<TestLayer>("card", calls);
    REQUIRE(stack.add(base) == Result::SUCCESS);
    REQUIRE(stack.add(card) == Result::SUCCESS);
    REQUIRE(stack.create(nullptr, context) == Result::SUCCESS);
    calls.clear();
    REQUIRE_FALSE(stack.onEvent({.type = MouseEventType::Move}, context));
    REQUIRE(calls == std::vector<std::string>{"card:event", "base:event"});

    card->eventResult = MapLayer::EventResult::Capture;
    calls.clear();
    REQUIRE(stack.onEvent({.type = MouseEventType::Click}, context));
    card->eventResult = MapLayer::EventResult::Ignored;
    REQUIRE(stack.onEvent({.type = MouseEventType::Move}, context));
    REQUIRE(stack.onEvent({.type = MouseEventType::Release}, context));
    REQUIRE(calls == std::vector<std::string>{"card:event", "card:event", "card:event"});
    REQUIRE_FALSE(stack.onEvent({.type = MouseEventType::Move}, context));

    card->eventResult = MapLayer::EventResult::Capture;
    REQUIRE(stack.onEvent({.type = MouseEventType::Click}, context));
    calls.clear();
    REQUIRE_FALSE(stack.onEvent({.type = MouseEventType::Leave}, context));
    REQUIRE(calls == std::vector<std::string>{"card:event", "base:event"});
    card->eventResult = MapLayer::EventResult::Ignored;
    REQUIRE_FALSE(stack.onEvent({.type = MouseEventType::Move}, context));
    REQUIRE(stack.destroy(nullptr) == Result::SUCCESS);
}

TEST_CASE("Map GPU uniforms match the packed shader layout", "[render][geomap]") {
    using GpuUniforms = MapContext::GpuUniforms;
    REQUIRE(sizeof(GpuUniforms) == 144);
    REQUIRE(offsetof(GpuUniforms, surfaceScale) == 96);
    REQUIRE(offsetof(GpuUniforms, lineWidth) == 108);
    REQUIRE(offsetof(GpuUniforms, outlineStrength) == 136);
    REQUIRE(offsetof(GpuUniforms, lineOpacity) == 140);
}

TEST_CASE("Map sun model tracks the subsolar point", "[render][geomap]") {
    const auto j2000 = MapContext::SubsolarPoint(946728000.0);
    REQUIRE(j2000.y == Approx(-23.03).margin(0.1));
    REQUIRE(j2000.x == Approx(0.8).margin(0.3));

    const auto equinox = MapContext::SubsolarPoint(1774017960.0);
    REQUIRE(equinox.y == Approx(0.0).margin(0.1));
    REQUIRE(equinox.x == Approx(-39.6).margin(0.5));

    const auto solstice = MapContext::SubsolarPoint(1782030240.0);
    REQUIRE(solstice.y == Approx(23.44).margin(0.05));
    REQUIRE(solstice.x == Approx(54.4).margin(0.5));

    for (const F64 seconds : {946728000.0, 1774017960.0, 1782030240.0}) {
        const auto point = MapContext::SubsolarPoint(seconds);
        REQUIRE(point.x >= -180.0);
        REQUIRE(point.x <= 180.0);
        const auto direction = MapContext::SunDirection(seconds);
        REQUIRE(glm::length(direction) == Approx(1.0f).margin(1e-5f));
        const auto expected = MapContext::LonLatToSphere(
            static_cast<F32>(point.x), static_cast<F32>(point.y));
        REQUIRE(glm::distance(direction, expected) == Approx(0.0f).margin(1e-5f));
    }

    const auto morning = MapContext::SubsolarPoint(1774017960.0 - 6.0 * 3600.0);
    REQUIRE(morning.x == Approx(equinox.x + 90.0).margin(0.5));
}

TEST_CASE("Shared map camera matches the previous globe projection", "[render][geomap]") {
    for (F32 zoom : {-2.0f, 0.0f, 3.0f, 10.0f}) {
        for (F32 aspect : {0.5f, 1.0f, 2.0f}) {
            MapContext context;
            MapContext::Uniforms view;
            view.centerLon = 127.0f;
            view.centerLat = 36.0f;
            view.zoom = zoom;
            view.aspectRatio = aspect;
            REQUIRE(context.update(view) == Result::SUCCESS);

            // Original GeoMap::ComputeCamera, retained here as a regression
            // oracle rather than comparing the new implementation to itself.
            const F32 lat = glm::radians(view.centerLat);
            const F32 lon = glm::radians(view.centerLon);
            const glm::vec3 normal(std::cos(lat) * std::sin(lon), std::sin(lat),
                                   std::cos(lat) * std::cos(lon));
            const F32 height = (1.0f / 0.382683432365f - 1.0f) / std::pow(2.0f, zoom);
            const F32 distance = 1.0f + height;
            const glm::vec3 upProjection = glm::vec3(0, 1, 0) - normal * normal.y;
            const glm::vec3 up = upProjection / glm::length(upProjection);
            const glm::mat4 expected = glm::perspective(
                static_cast<F32>(45.0 * JST_PI / 180.0), aspect,
                std::max(1e-7f, height * 1e-3f), distance + 2.0f) *
                glm::lookAt(distance * normal, glm::vec3(0), up);
            for (int c = 0; c < 4; ++c) {
                for (int r = 0; r < 4; ++r) {
                    REQUIRE(context.camera.viewProjection[c][r] ==
                            Approx(expected[c][r]).margin(1e-6f));
                }
            }
            F32 x, y;
            REQUIRE(context.projectLonLat(view.centerLon, view.centerLat, x, y));
            REQUIRE(x == Approx(0.0f).margin(0.002f));
            REQUIRE(y == Approx(0.0f).margin(0.002f));
            REQUIRE_FALSE(context.projectLonLat(view.centerLon + 180, -view.centerLat, x, y));
        }
    }
}

TEST_CASE("Map projection validates views and handles wrapping and DPI", "[render][geomap]") {
    MapContext context;
    auto view = context.view;
    view.viewportWidth = 1000;
    view.viewportHeight = 500;
    view.surfaceScale = 2;
    view.aspectRatio = 0;
    view.centerLon = 540;
    REQUIRE(context.update(view) == Result::SUCCESS);
    REQUIRE(context.view.centerLon == -180);
    REQUIRE(context.view.aspectRatio == 2);
    REQUIRE(context.pixelSize().x == Approx(0.004f));
    REQUIRE(context.pixelSize().y == Approx(0.008f));
    F32 x1, y1, x2, y2;
    REQUIRE(context.projectLonLat(180, 0, x1, y1));
    REQUIRE(context.projectLonLat(-180, 0, x2, y2));
    REQUIRE(x1 == Approx(x2).margin(1e-6f));
    REQUIRE(y1 == Approx(y2).margin(1e-6f));
    REQUIRE_FALSE(context.projectLonLat(0, 91, x1, y1));
    REQUIRE_FALSE(context.projectLonLat(std::numeric_limits<F32>::quiet_NaN(), 0, x1, y1));
    const auto before = context.view;
    view.zoom = std::numeric_limits<F32>::infinity();
    REQUIRE(context.update(view) == Result::ERROR);
    REQUIRE(context.view == before);
    view = before;
    view.viewportWidth = 0;
    view.viewportHeight = 0;
    view.aspectRatio = 0;
    REQUIRE(context.update(view) == Result::SUCCESS);
    REQUIRE(std::isfinite(context.pixelSize().x));
    REQUIRE(std::isfinite(context.pixelSize().y));
}

TEST_CASE("Map navigation preserves fitted size and cancels globe dragging", "[render][geomap]") {
    MapContext context;
    MapNavigation navigation;
    REQUIRE(navigation.resize({SurfaceEventType::Resize, {1000, 500}, 2}, context) == Result::SUCCESS);
    REQUIRE(context.view.zoom == 1.0f);
    REQUIRE(context.view.surfaceScale == 2.0f);
    REQUIRE(navigation.mouse({.type = MouseEventType::Scroll, .position = {0.5f, 0.5f},
                              .scroll = {0, 2}}, context) == Result::SUCCESS);
    REQUIRE(context.view.detailZoom == Approx(0.3f));
    REQUIRE(context.view.centerLon == Approx(0).margin(1e-6f));
    REQUIRE(navigation.mouse({.type = MouseEventType::Click, .button = MouseButton::Left,
                              .position = {0.5f, 0.5f}}, context) == Result::SUCCESS);
    REQUIRE(navigation.mouse({.type = MouseEventType::Move, .position = {0.6f, 0.5f}}, context) == Result::SUCCESS);
    REQUIRE(context.view.centerLon < 0);
    const auto before = context.view;
    navigation.cancel();
    REQUIRE(navigation.mouse({.type = MouseEventType::Move, .position = {0.9f, 0.5f}}, context) == Result::SUCCESS);
    REQUIRE(context.view == before);

    for (const auto size : {Extent2D<U64>{1000, 500}, Extent2D<U64>{500, 1000}}) {
        REQUIRE(navigation.resize({SurfaceEventType::Resize, size, 1}, context) == Result::SUCCESS);
        REQUIRE(navigation.mouse({.type = MouseEventType::Scroll, .position = {0.5f, 0.5f},
                                  .scroll = {0, -1000}}, context) == Result::SUCCESS);
        const F32 distance = glm::length(glm::vec3(context.camera.cameraPos));
        const F32 radius = std::tan(std::asin(1.0f / distance)) / 0.414213562373f;
        REQUIRE(radius / std::min(context.view.aspectRatio, 1.0f) == Approx(0.70f));
    }
}

TEST_CASE("Map fitting frames geographic points with padding across view sizes", "[render][geomap][fit]") {
    const std::vector<Extent2D<F32>> points = {{-125, 36}, {-119, 40}, {-122, 38}};
    for (const auto size : {Extent2D<U64>{1000, 500}, Extent2D<U64>{500, 1000}}) {
        for (const F32 scale : {1.0f, 2.0f}) {
            MapContext context;
            MapNavigation navigation;
            REQUIRE(navigation.resize({SurfaceEventType::Resize, size, scale}, context) == Result::SUCCESS);
            const auto before = context.view;
            const auto fitted = context.fitView(points);
            REQUIRE(fitted.has_value());
            REQUIRE(context.view == before); // Fitting does not mutate the caller.
            REQUIRE(fitted->centerLon == Approx(-122));
            REQUIRE(fitted->centerLat == Approx(38));
            REQUIRE(fitted->zoom > before.zoom);
            REQUIRE(fitted->viewportWidth == before.viewportWidth);
            REQUIRE(fitted->viewportHeight == before.viewportHeight);
            REQUIRE(fitted->surfaceScale == scale);
            REQUIRE(context.update(*fitted) == Result::SUCCESS);
            F32 edge = 0;
            for (const auto& point : points) {
                F32 x, y;
                REQUIRE(context.projectLonLat(point.x, point.y, x, y));
                REQUIRE(std::abs(x) <= 0.8001f);
                REQUIRE(std::abs(y) <= 0.8001f);
                edge = std::max({edge, std::abs(x), std::abs(y)});
            }
            REQUIRE(edge == Approx(0.8f).margin(0.0001f));
            // Hovering or repeating the current resize must not undo the fit.
            REQUIRE(navigation.mouse({.type = MouseEventType::Move}, context) == Result::SUCCESS);
            REQUIRE(context.view.zoom == Approx(fitted->zoom));
            REQUIRE(navigation.resize({SurfaceEventType::Resize, size, scale}, context) == Result::SUCCESS);
            REQUIRE(context.view.zoom == Approx(fitted->zoom));
        }
    }
}

TEST_CASE("Map fitting handles the antimeridian, polar regions and small clusters", "[render][geomap][fit]") {
    MapContext context;
    for (const F32 latitude : {0.0f, 80.0f, -80.0f, 89.0f}) {
        const std::vector<Extent2D<F32>> points = {{179, latitude}, {-179, latitude}};
        const auto fitted = context.fitView(points, 0.2f, 5.0f);
        REQUIRE(fitted.has_value());
        REQUIRE(std::abs(fitted->centerLon) == Approx(180));
        REQUIRE(fitted->zoom <= 5);
        REQUIRE(context.update(*fitted) == Result::SUCCESS);
        for (const auto& point : points) {
            F32 x, y;
            REQUIRE(context.projectLonLat(point.x, point.y, x, y));
            REQUIRE(std::abs(x) <= 0.8001f);
            REQUIRE(std::abs(y) <= 0.8001f);
        }
    }
    const std::vector<Extent2D<F32>> single = {{-122, 38}};
    auto fitted = context.fitView(single, 0.2f, 5.0f);
    REQUIRE(fitted.has_value());
    REQUIRE(fitted->centerLon == Approx(-122));
    REQUIRE(fitted->centerLat == Approx(38));
    REQUIRE(fitted->zoom == 5);
    const std::vector<Extent2D<F32>> coincident = {{-122, 38}, {-122, 38}};
    REQUIRE(context.fitView(coincident, 0.2f, 5.0f) == fitted);
}

TEST_CASE("Map fitting ignores invalid fixes and leaves unfit spans unchanged", "[render][geomap][fit]") {
    MapContext context;
    const auto before = context.view;
    const F32 nan = std::numeric_limits<F32>::quiet_NaN();
    const F32 inf = std::numeric_limits<F32>::infinity();
    std::vector<Extent2D<F32>> points = {{nan, 0}, {0, inf}, {0, 91}, {inf, 0}};
    REQUIRE_FALSE(context.fitView({}).has_value());
    REQUIRE_FALSE(context.fitView(points).has_value());
    points.push_back({238, 38});
    const auto fitted = context.fitView(points);
    REQUIRE(fitted.has_value());
    REQUIRE(fitted->centerLon == Approx(-122));
    REQUIRE(fitted->centerLat == Approx(38));
    for (const F32 padding : {-1.0f, 1.0f, nan}) {
        REQUIRE_FALSE(context.fitView(points, padding).has_value());
    }
    for (const F32 maxZoom : {-9.0f, MapContext::MaxZoom + 1.0f, inf}) {
        REQUIRE_FALSE(context.fitView(points, 0.2f, maxZoom).has_value());
    }
    const std::vector<Extent2D<F32>> antipodes = {{0, 0}, {180, 0}};
    REQUIRE_FALSE(context.fitView(antipodes).has_value());
    REQUIRE(context.view == before);
}

TEST_CASE("Map camera enforces a shared precision-safe zoom limit", "[render][geomap]") {
    for (const auto center : {Extent2D<F32>{-122, 38}, Extent2D<F32>{127, 36},
                              Extent2D<F32>{10, 85}, Extent2D<F32>{179, -60},
                              Extent2D<F32>{180, 0}}) {
        MapContext context;
        auto view = context.view;
        view.centerLon = center.x;
        view.centerLat = center.y;
        view.viewportWidth = view.viewportHeight = 800;
        view.zoom = view.detailZoom = MapContext::MaxZoom;
        REQUIRE(context.update(view) == Result::SUCCESS);
        F32 x, y;
        REQUIRE(context.projectLonLat(center.x, center.y, x, y));
        REQUIRE(std::abs(x) * 400 < 0.5f);
        REQUIRE(std::abs(y) * 400 < 0.5f);
        const auto before = context.view;
        for (const F32 zoom : {MapContext::MaxZoom + 0.01f, 18.0f, 24.0f}) {
            view.zoom = zoom;
            REQUIRE(context.update(view) == Result::ERROR);
            REQUIRE(context.view == before);
        }
    }
    for (const auto size : {Extent2D<U64>{800, 600}, Extent2D<U64>{600, 800}}) {
        MapContext context;
        MapNavigation navigation;
        REQUIRE(navigation.resize({SurfaceEventType::Resize, size, 1}, context) == Result::SUCCESS);
        REQUIRE(navigation.mouse({.type = MouseEventType::Scroll, .position = {0.5f, 0.5f},
                                   .scroll = {0, 1000}}, context) == Result::SUCCESS);
        REQUIRE(context.view.zoom == Approx(MapContext::MaxZoom));
    }
}

TEST_CASE("Map line clipping retains arcs crossing the visible cap", "[render][geomap]") {
    MapContext context;
    auto view = context.view;
    view.centerLon = -53.195857f;
    view.centerLat = -9.682447f;
    view.zoom = 12;
    REQUIRE(context.update(view) == Result::SUCCESS);
    // Actual Natural Earth state border: neither endpoint is visible at zoom
    // 12, but its midpoint is directly below the camera.
    const auto originalA = MapContext::LonLatToSphere(-51.297716f, -9.789376f);
    const auto originalB = MapContext::LonLatToSphere(-55.092730f, -9.565101f);
    const glm::vec3 n(context.camera.targetNormal);
    const F32 threshold = context.camera.targetNormal.w;
    REQUIRE(glm::dot(originalA, n) < threshold);
    REQUIRE(glm::dot(originalB, n) < threshold);
    for (const bool reverse : {false, true}) {
        auto a = reverse ? originalB : originalA;
        auto b = reverse ? originalA : originalB;
        REQUIRE(GeoMapLines::clipLineToHorizon(a, b, n, threshold));
        REQUIRE(glm::length(a) == Approx(1).margin(1e-6f));
        REQUIRE(glm::length(b) == Approx(1).margin(1e-6f));
        REQUIRE(glm::dot(a, n) == Approx(threshold).margin(1e-6f));
        REQUIRE(glm::dot(b, n) == Approx(threshold).margin(1e-6f));
        const auto plane = glm::normalize(glm::cross(originalA, originalB));
        REQUIRE(glm::dot(a, plane) == Approx(0).margin(1e-6f));
        REQUIRE(glm::dot(b, plane) == Approx(0).margin(1e-6f));
    }
    auto a = -originalA;
    auto b = -originalB;
    REQUIRE_FALSE(GeoMapLines::clipLineToHorizon(a, b, n, threshold));

    a = n;
    b = originalB;
    REQUIRE(GeoMapLines::clipLineToHorizon(a, b, n, threshold));
    REQUIRE(a == n);
    REQUIRE(glm::dot(b, n) == Approx(threshold).margin(1e-6f));
}

TEST_CASE("Map line clipping handles horizon edges without projecting hidden arcs", "[render][geomap]") {
    for (const F32 centerLon : {0.0f, 179.0f, -179.0f}) {
        const auto n = MapContext::LonLatToSphere(centerLon, 0);
        const F32 threshold = std::cos(glm::radians(30.0f));
        auto point = [&](F32 lon) { return MapContext::LonLatToSphere(centerLon + lon, 0); };
        for (const auto endpoints : {Extent2D<F32>{-50, 50}, Extent2D<F32>{50, -50},
                                     Extent2D<F32>{0, 50}, Extent2D<F32>{50, 0}}) {
            auto a = point(endpoints.x);
            auto b = point(endpoints.y);
            REQUIRE(GeoMapLines::clipLineToHorizon(a, b, n, threshold));
            REQUIRE(glm::distance(a, point(std::clamp(endpoints.x, -30.0f, 30.0f))) < 1e-5f);
            REQUIRE(glm::distance(b, point(std::clamp(endpoints.y, -30.0f, 30.0f))) < 1e-5f);
        }
        for (const auto endpoints : {Extent2D<F32>{40, 80}, Extent2D<F32>{120, -120},
                                     Extent2D<F32>{-120, 120}, Extent2D<F32>{180, 180}}) {
            auto a = point(endpoints.x);
            auto b = point(endpoints.y);
            REQUIRE_FALSE(GeoMapLines::clipLineToHorizon(a, b, n, threshold));
        }
        auto a = point(-10);
        auto b = point(10);
        const auto originalA = a;
        const auto originalB = b;
        REQUIRE(GeoMapLines::clipLineToHorizon(a, b, n, threshold));
        REQUIRE(a == originalA);
        REQUIRE(b == originalB);
    }
}

TEST_CASE("Map dash ranges continue over short segments and the antimeridian", "[render][geomap]") {
    MapContext context;
    auto view = context.view;
    view.viewportWidth = view.viewportHeight = 512;
    REQUIRE(context.update(view) == Result::SUCCESS);
    const std::vector<F32> vertices = {
        -1, 23.5f, 0, 23.5f,
         0, 23.5f, 1, 23.5f,
         1, 23.5f, 2, 23.5f,
        -1, -23.5f, 0, -23.5f, // Another polyline resets its distance.
    };
    std::vector<F32> ranges(vertices.size() / 2);
    GeoMapLines::UpdateDashRanges(vertices, context, ranges);
    REQUIRE(ranges[0] == 0);
    REQUIRE(ranges[1] > 0);
    REQUIRE(ranges[1] < 8); // A single segment fits entirely inside one dash.
    REQUIRE(ranges[2] == ranges[1]);
    REQUIRE(ranges[3] > 8); // The connected next segment reaches the gap.
    REQUIRE(ranges[4] == ranges[3]);
    REQUIRE(ranges[5] > 16);
    REQUIRE(ranges[6] == 0);

    view.centerLon = 180;
    REQUIRE(context.update(view) == Result::SUCCESS);
    const std::vector<F32> seam = {179, 0, 180, 0, -180, 0, -179, 0};
    std::vector<F32> seamRanges(seam.size() / 2);
    GeoMapLines::UpdateDashRanges(seam, context, seamRanges);
    REQUIRE(seamRanges[1] > 0);
    REQUIRE(seamRanges[2] == seamRanges[1]);
    REQUIRE(seamRanges[3] > seamRanges[2]);

    const auto originalRanges = seamRanges;
    view.surfaceScale = 2;
    REQUIRE(context.update(view) == Result::SUCCESS);
    GeoMapLines::UpdateDashRanges(seam, context, seamRanges);
    for (U64 i = 0; i < seamRanges.size(); ++i) {
        REQUIRE(seamRanges[i] == Approx(originalRanges[i] / 2));
    }

    view.centerLon = 0;
    REQUIRE(context.update(view) == Result::SUCCESS);
    GeoMapLines::UpdateDashRanges(seam, context, seamRanges);
    REQUIRE(std::all_of(seamRanges.begin(), seamRanges.end(), [](F32 value) { return value == 0; }));
}
