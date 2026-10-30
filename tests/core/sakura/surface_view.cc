#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "render/sakura/surface_view.hh"
#include <jetstream/render/sakura/surface.hh>

#include "harness.hh"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace Jetstream;

using SakuraTest::ResizeLog;

namespace {

class Texture final : public Render::Texture {
 public:
    explicit Texture(U64 handle = 123) : Render::Texture(Config{.size = {200, 100}}), handle(handle) {}
    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    uint64_t raw() const override { return handle; }

 private:
    U64 handle;
};

constexpr F32 kWindowWidth = 400.0f;
constexpr F32 kWindowHeight = 300.0f;
constexpr Rect kRegion = {0.0f, 0.0f, kWindowWidth, kWindowHeight};

}  // namespace

TEST_CASE("SurfaceView visibility follows the displayed texture and draw clip",
          "[core][sakura][surface_view][visibility]") {
    SakuraTest::FakeWindow window;
    (void)window.frame();
    Sakura::Context ctx;
    ctx.render = &window;
    auto texture = std::make_shared<Texture>();
    auto other = std::make_shared<Texture>(456);
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config{.id = "clipped", .textureSource = texture, .size = {200, 100}};
    const Rect onWindow = {0.0f, 0.0f, 400.0f, 300.0f};
    const Rect offWindow = {10000.0f, 10000.0f, 400.0f, 300.0f};
    Rect region = offWindow;
    Rect clip = onWindow;
    bool visible = false;
    bool secondView = false;
    bool differentTexture = false;
    SECTION("A fully clipped image emits no layer") {}
    SECTION("A second visible view keeps the texture visible") { secondView = true; }
    SECTION("An expanded draw clip preserves off-window images") {
        clip = {0.0f, 0.0f, 20000.0f, 20000.0f};
        visible = true;
    }
    SECTION("A reduced draw clip suppresses an on-window image") {
        region = onWindow;
        clip = {1000.0f, 1000.0f, 100.0f, 100.0f};
    }
    SECTION("An overridden texture does not hide its unused textureSource") {
        config.resolveTexture = [other] { return std::shared_ptr<const Render::Texture>(other); };
        differentTexture = true;
    }
    surface.update(config);

    const U64 frame = window.frameIndex();
    const auto presented = surface.render(ctx, region, clip);
    REQUIRE(texture->visibleForPresentation(frame) == (visible || differentTexture));
    REQUIRE(presented.layer.has_value() == visible);
    if (visible) {
        REQUIRE(presented.layer->texture == texture);
        REQUIRE(presented.layer->clip == clip);
        REQUIRE(presented.rect == Rect{region.x, region.y, 200.0f, 100.0f});
    }
    if (secondView) {
        const auto again = surface.render(ctx, onWindow, onWindow);
        REQUIRE(texture->visibleForPresentation(frame));
        REQUIRE(again.layer.has_value());
    }
    REQUIRE(texture->visibleForPresentation(frame + 1));
}

TEST_CASE("SurfaceView composites premultiplied only when its config asks for it",
          "[core][sakura][surface_view]") {
    SakuraTest::FakeWindow window;
    auto texture = std::make_shared<Texture>();
    const Rect region = {0.0f, 0.0f, 400.0f, 300.0f};

    for (const bool premultiplied : {true, false}) {
        Sakura::Context ctx;
        ctx.render = &window;
        Sakura::SurfaceView surface;
        surface.update({.id = "blend", .textureSource = texture, .size = {200, 100}, .premultiplied = premultiplied});
        const auto presented = surface.render(ctx, region, region);
        REQUIRE(presented.layer.has_value());
        REQUIRE(presented.blend == (premultiplied ? Render::PresentList::Blend::Premultiplied
                                                  : Render::PresentList::Blend::Straight));
    }
}

TEST_CASE("SurfaceView falls back to the whole available window region when size is zero",
          "[core][sakura][surface_view]") {
    SakuraTest::FakeWindow ui;
    const auto ctx = ui.sakura();

    ResizeLog log;
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "fallback";
    config.size = {0.0f, 0.0f};
    config.onSize = [&log](const Sakura::SurfaceResize& resize) {
        log.record(resize);
    };

    surface.update(config);
    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });

    REQUIRE(log.count() == 1);
    REQUIRE(log.entries[0].logicalSize.x == kWindowWidth);
    REQUIRE(log.entries[0].logicalSize.y == kWindowHeight);
    REQUIRE(log.entries[0].framebufferSize.x == kWindowWidth);
    REQUIRE(log.entries[0].framebufferSize.y == kWindowHeight);
    REQUIRE(log.entries[0].scale == Catch::Approx(0.5f));
}

TEST_CASE("SurfaceView height override wins over the fallback height and truncates to whole units",
          "[core][sakura][surface_view]") {
    SakuraTest::FakeWindow ui;
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

    surface.update(config);
    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });

    REQUIRE(log.count() == 1);
    REQUIRE(log.entries[0].logicalSize.x == 400);
    REQUIRE(log.entries[0].logicalSize.y == 123);
    REQUIRE(log.entries[0].framebufferSize.y == 123);
}

TEST_CASE("SurfaceView treats invalid height overrides as zero and reports no size", "[core][sakura][surface_view]") {
    SakuraTest::FakeWindow ui;
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

        surface.update(config);
        ui.frame([&] {
            surface.render(ctx, kRegion, kRegion);
        });

        REQUIRE(log.count() == 0);
    }
}

TEST_CASE("SurfaceView emits resize only when the resolved size changes", "[core][sakura][surface_view][dedupe]") {
    SakuraTest::FakeWindow ui;
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

    surface.update(config);

    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });
    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });
    REQUIRE(log.count() == 1);

    config.height = 200.0f;
    surface.update(config);
    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });
    REQUIRE(log.count() == 2);
    REQUIRE(log.entries[1].logicalSize.y == 200);

    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });
    REQUIRE(log.count() == 2);
}

TEST_CASE("SurfaceView re-emits an unchanged size after a frame where it was not rendered",
          "[core][sakura][surface_view][dedupe]") {
    SakuraTest::FakeWindow ui;
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

    surface.update(config);

    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });
    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });
    REQUIRE(log.count() == 1);

    ui.frame([] {});

    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });
    REQUIRE(log.count() == 2);
}

TEST_CASE("SurfaceView renders as a safe no-op without callbacks or textures",
          "[core][sakura][surface_view][robustness]") {
    SakuraTest::FakeWindow ui;
    const auto ctx = ui.sakura();

    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "inert";

    surface.update(config);

    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });
}

TEST_CASE("SurfaceView reports halved logical size and framebuffer metrics for the window scaling factor",
          "[core][sakura][surface_view][scaling]") {
    SakuraTest::FakeWindow ui(2.0f);
    const auto ctx = ui.sakura();

    ResizeLog log;
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config;
    config.id = "scaled";
    config.size = {0.0f, 0.0f};
    config.onSize = [&log](const Sakura::SurfaceResize& resize) {
        log.record(resize);
    };

    surface.update(config);
    ui.frame([&] {
        surface.render(ctx, kRegion, kRegion);
    });

    REQUIRE(log.count() == 1);
    REQUIRE(log.entries[0].logicalSize.x == 200);
    REQUIRE(log.entries[0].logicalSize.y == 150);
    REQUIRE(log.entries[0].framebufferSize.x == 400);
    REQUIRE(log.entries[0].framebufferSize.y == 300);
    REQUIRE(log.entries[0].scale == Catch::Approx(1.0f));
}
