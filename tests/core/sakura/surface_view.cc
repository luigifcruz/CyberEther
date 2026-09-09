#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/surface_view.hh>
#include <jetstream/render/sakura/surface.hh>

#include "harness.hh"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace Jetstream;

using SakuraTest::ResizeLog;

TEST_CASE("SurfaceView falls back to the available region when size is zero",
          "[core][sakura][surface_view]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    ResizeLog log;
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "fallback";
    config.size = {0.0f, 0.0f};
    config.onSize = [&log](const Sakura::SurfaceResize& resize) {
        log.record(resize);
    };

    REQUIRE(surface.update(config));
    ui.frame([&] {
        surface.render(ctx);
    });

    // Root window content area is 400x300 at scale 1.
    REQUIRE(log.count() == 1);
    REQUIRE(log.entries[0].logicalSize.x == 400);
    REQUIRE(log.entries[0].logicalSize.y == 300);
    REQUIRE(log.entries[0].framebufferSize.x == 400);
    REQUIRE(log.entries[0].framebufferSize.y == 300);
    REQUIRE(log.entries[0].scale == Catch::Approx(0.5f));
}

TEST_CASE("SurfaceView honors the explicit height override",
          "[core][sakura][surface_view]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    ResizeLog log;
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "override";
    config.size = {0.0f, 0.0f};
    config.height = 123.4f;
    config.onSize = [&log](const Sakura::SurfaceResize& resize) {
        log.record(resize);
    };

    REQUIRE(surface.update(config));
    ui.frame([&] {
        surface.render(ctx);
    });

    // Height override wins; width still falls back and truncates to units.
    REQUIRE(log.count() == 1);
    REQUIRE(log.entries[0].logicalSize.x == 400);
    REQUIRE(log.entries[0].logicalSize.y == 123);
    REQUIRE(log.entries[0].framebufferSize.y == 123);
}

TEST_CASE("SurfaceView ignores invalid height overrides",
          "[core][sakura][surface_view]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    for (const F32 invalid : {std::nanf(""), -3.0f}) {
        ResizeLog log;
        Sakura::SurfaceView surface;
        Sakura::SurfaceView::Config config;
        config.id = "invalid";
        config.size = {200.0f, 200.0f};
        config.height = invalid;
        config.onSize = [&log](const Sakura::SurfaceResize& resize) {
            log.record(resize);
        };

        REQUIRE(surface.update(config));
        ui.frame([&] {
            surface.render(ctx);
        });

        // Invalid heights clamp to zero: the render is a no-op, reports nothing.
        REQUIRE(log.count() == 0);
    }
}

TEST_CASE("SurfaceView emits resize only when the resolved size changes",
          "[core][sakura][surface_view][dedupe]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    ResizeLog log;
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "dedupe";
    config.size = {0.0f, 0.0f};
    config.height = 100.0f;
    config.onSize = [&log](const Sakura::SurfaceResize& resize) {
        log.record(resize);
    };

    REQUIRE(surface.update(config));

    ui.frame([&] {
        surface.render(ctx);
    });
    ui.frame([&] {
        surface.render(ctx);
    });
    REQUIRE(log.count() == 1);

    config.height = 200.0f;
    REQUIRE(surface.update(config));
    ui.frame([&] {
        surface.render(ctx);
    });
    REQUIRE(log.count() == 2);
    REQUIRE(log.entries[1].logicalSize.y == 200);

    ui.frame([&] {
        surface.render(ctx);
    });
    REQUIRE(log.count() == 2);
}

TEST_CASE("SurfaceView re-emits after a skipped frame",
          "[core][sakura][surface_view][dedupe]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    ResizeLog log;
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "frame-gap";
    config.size = {0.0f, 0.0f};
    config.height = 100.0f;
    config.onSize = [&log](const Sakura::SurfaceResize& resize) {
        log.record(resize);
    };

    REQUIRE(surface.update(config));

    ui.frame([&] {
        surface.render(ctx);
    });
    ui.frame([&] {
        surface.render(ctx);
    });
    REQUIRE(log.count() == 1);

    // The surface is not rendered during this frame (e.g. node hidden).
    ui.frame([] {});

    // Identical size, but the dedupe state was dropped for stale frames.
    ui.frame([&] {
        surface.render(ctx);
    });
    REQUIRE(log.count() == 2);
}

TEST_CASE("SurfaceView tolerates missing callbacks and textures",
          "[core][sakura][surface_view][robustness]") {
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();

    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "inert";

    REQUIRE(surface.update(config));

    // No texture, no callbacks: the render must be a safe no-op.
    ui.frame([&] {
        surface.render(ctx);
    });
}

TEST_CASE("SurfaceView reports framebuffer metrics for the window scaling factor",
          "[core][sakura][surface_view][scaling]") {
    SakuraTest::HeadlessUi ui(2.0f);
    const auto ctx = ui.sakura();

    ResizeLog log;
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "scaled";
    config.size = {0.0f, 0.0f};
    config.onSize = [&log](const Sakura::SurfaceResize& resize) {
        log.record(resize);
    };

    REQUIRE(surface.update(config));
    ui.frame([&] {
        surface.render(ctx);
    });

    // 400x300 device pixels = 200x150 logical at scale 2; scale = 2 * 0.5.
    REQUIRE(log.count() == 1);
    REQUIRE(log.entries[0].logicalSize.x == 200);
    REQUIRE(log.entries[0].logicalSize.y == 150);
    REQUIRE(log.entries[0].framebufferSize.x == 400);
    REQUIRE(log.entries[0].framebufferSize.y == 300);
    REQUIRE(log.entries[0].scale == Catch::Approx(1.0f));
}

TEST_CASE("SurfaceView captures both mouse buttons through release outside the surface",
          "[core][sakura][surface_view][capture]") {
    const auto button = GENERATE(ImGuiMouseButton_Left, ImGuiMouseButton_Right);
    SakuraTest::HeadlessUi ui;
    const auto ctx = ui.sakura();
    std::vector<MouseEvent> events;
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "capture";
    config.size = {200.0f, 100.0f};
    config.texture = 1;
    config.onMouse = [&](const MouseEvent& event) { events.push_back(event); };
    REQUIRE(surface.update(config));

    const auto frame = [&](const ImVec2& position, const bool down) {
        events.clear();
        ui.setMouse(position, button == ImGuiMouseButton_Left && down);
        ImGui::GetIO().MouseDown[ImGuiMouseButton_Right] = button == ImGuiMouseButton_Right && down;
        ui.frame([&] {
            ImGui::SetCursorScreenPos({20.0f, 20.0f});
            surface.render(ctx);
        });
    };
    const auto count = [&](const MouseEventType type) {
        return std::count_if(events.begin(), events.end(),
                             [type](const auto& event) { return event.type == type; });
    };

    frame({120.0f, 70.0f}, false);
    frame({120.0f, 70.0f}, false);
    frame({120.0f, 70.0f}, true);
    REQUIRE(count(MouseEventType::Click) == 1);

    frame({260.0f, 90.0f}, true);
    REQUIRE(count(MouseEventType::Move) == 1);
    REQUIRE(events.back().position.x == Catch::Approx(1.2f));
    frame({280.0f, 100.0f}, false);
    REQUIRE(count(MouseEventType::Release) == 1);
    const auto released = std::find_if(events.begin(), events.end(), [](const auto& event) {
        return event.type == MouseEventType::Release;
    });
    REQUIRE(released->button == (button == ImGuiMouseButton_Left ? MouseButton::Left : MouseButton::Right));
    REQUIRE(released->position.x == Catch::Approx(1.3f));
    REQUIRE(released->position.y == Catch::Approx(0.8f));

    frame({280.0f, 100.0f}, false);
    REQUIRE(events.empty());
    frame({120.0f, 70.0f}, false);
    REQUIRE(count(MouseEventType::Release) == 0);
    REQUIRE(count(MouseEventType::Click) == 0);
}
