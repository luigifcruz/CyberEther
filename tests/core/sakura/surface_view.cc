#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/surface_view.hh>
#include <jetstream/render/sakura/surface.hh>
#include <jetstream/render/tools/imgui_blend_ext.h>

#include "harness.hh"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace Jetstream;

using SakuraTest::ResizeLog;

TEST_CASE("SurfaceView visibility follows the displayed texture and draw clip",
          "[core][sakura][surface_view][visibility]") {
    class Texture final : public Render::Texture {
     public:
        Texture() : Render::Texture(Config{.size = {200, 100}}) {}
        Result create() override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        uint64_t raw() const override { return 123; }
    };
    SakuraTest::HeadlessUi ui;
    auto texture = std::make_shared<Texture>();
    Sakura::SurfaceView surface;
    Sakura::SurfaceView::Config config{.id = "clipped", .textureSource = texture, .size = {200, 100}};
    bool visible = false;
    bool secondView = false;
    bool customClip = false;
    bool differentTexture = false;
    SECTION("A fully clipped image emits no geometry") {}
    SECTION("A second visible view keeps the texture visible") { secondView = true; }
    SECTION("An expanded draw clip preserves off-window images") { customClip = visible = true; }
    SECTION("A reduced draw clip suppresses an on-window image") { customClip = true; }
    SECTION("An overridden texture does not hide its unused textureSource") {
        config.onResolveTexture = [] { return U64(456); };
        differentTexture = true;
    }
    surface.update(config);
    ui.frame([&] {
        const auto frame = ImGui::GetFrameCount();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        auto* list = ImGui::GetWindowDrawList();
        const int indices = list->IdxBuffer.Size;
        if (customClip) {
            list->PushClipRect(visible ? ImVec2(0, 0) : ImVec2(1000, 1000),
                               visible ? ImVec2(20000, 20000) : ImVec2(1100, 1100), false);
        }
        if (!customClip || visible) ImGui::SetCursorScreenPos({10000, 10000});
        surface.render(ui.sakura());
        REQUIRE(texture->visibleForPresentation(frame) == (visible || differentTexture));
        REQUIRE((list->IdxBuffer.Size > indices) == visible);
        if (customClip) list->PopClipRect();
        ImGui::SetCursorScreenPos(origin);
        if (secondView) {
            surface.render(ui.sakura());
            REQUIRE(texture->visibleForPresentation(frame));
            REQUIRE(list->IdxBuffer.Size > indices);
        }
        REQUIRE(texture->visibleForPresentation(frame + 1));
    });
}

TEST_CASE("SurfaceView keeps rounded geometry with premultiplied edge coverage",
          "[core][sakura][surface_view][premultiplied]") {
    struct Geometry {
        std::vector<ImDrawVert> vertices;
        std::vector<ImDrawCallback> callbacks;
    };
    const ImDrawCallback blend = [](const ImDrawList*, const ImDrawCmd*) {};
    ImGui::RegisterPremultipliedAlphaCallback(blend);
    SakuraTest::HeadlessUi ui;
    const auto draw = [&](const bool premultiplied) {
        Geometry geometry;
        Sakura::SurfaceView surface;
        surface.update({.id = "rounded", .texture = 1, .size = {200, 100}, .rounding = 12.0f,
                        .premultiplied = premultiplied});
        ui.frame([&] {
            auto* list = ImGui::GetWindowDrawList();
            const int vertices = list->VtxBuffer.Size;
            const int commands = list->CmdBuffer.Size;
            ImGui::SetCursorScreenPos({20.0f, 20.0f});
            surface.render(ui.sakura());
            geometry.vertices.assign(list->VtxBuffer.begin() + vertices, list->VtxBuffer.end());
            for (int i = std::max(commands - 1, 0); i < list->CmdBuffer.Size; ++i) {
                if (list->CmdBuffer[i].UserCallback) geometry.callbacks.push_back(list->CmdBuffer[i].UserCallback);
            }
        });
        return geometry;
    };
    const auto straight = draw(false);
    const auto premultiplied = draw(true);
    ImGui::UnregisterPremultipliedAlphaCallback(blend);

    REQUIRE(straight.callbacks.empty());
    REQUIRE(premultiplied.callbacks == std::vector<ImDrawCallback>{blend, ImDrawCallback_ResetRenderState});
    REQUIRE(premultiplied.vertices.size() > 4);
    REQUIRE(premultiplied.vertices.size() == straight.vertices.size());
    bool fringe = false;
    for (std::size_t i = 0; i < premultiplied.vertices.size(); ++i) {
        const auto& vertex = premultiplied.vertices[i];
        REQUIRE(vertex.pos.x == straight.vertices[i].pos.x);
        REQUIRE(vertex.pos.y == straight.vertices[i].pos.y);
        const ImU32 alpha = (vertex.col >> IM_COL32_A_SHIFT) & 0xFF;
        REQUIRE(alpha == ((straight.vertices[i].col >> IM_COL32_A_SHIFT) & 0xFF));
        REQUIRE(vertex.col == IM_COL32(alpha, alpha, alpha, alpha));
        fringe |= alpha == 0;
    }
    REQUIRE(fringe);
}

TEST_CASE("Surface visibility accounts for finalized texture consumers",
          "[core][sakura][surface_view][visibility]") {
    class Window final : public SakuraTest::FakeWindow {
     public:
        using Render::Window::prepareImgui;
        using Render::Window::textureVisibleForPresentation;
    };
    class Texture final : public Render::Texture {
     public:
        Texture() : Render::Texture(Config{.size = {200, 100}}) {}
        Result create() override { return Result::SUCCESS; }
        Result destroy() override { return Result::SUCCESS; }
        uint64_t raw() const override { return 123; }
    };
    SakuraTest::HeadlessUi ui;
    Window window;
    Texture texture;
    bool hinted = true;
    bool sampled = false;
    bool unrelated = false;
    bool empty = false;
    bool callback = false;
    bool visible = false;
    SECTION("A hidden texture with no consumers skips graphics") {}
    SECTION("Textures without visibility hints remain visible") {
        hinted = false;
        visible = true;
    }
    SECTION("Another draw list sampling the texture keeps it visible") {
        sampled = visible = true;
    }
    SECTION("An unrelated texture does not keep a hidden surface visible") {
        unrelated = true;
    }
    SECTION("An empty texture command is not a sampling consumer") { empty = true; }
    SECTION("Custom draw callbacks conservatively keep surfaces visible") {
        callback = visible = true;
    }
    SECTION("A second visible view overrides a hidden view") { visible = true; }

    ui.frame([&] {
        if (hinted) texture.presentationHint(ImGui::GetFrameCount(), false);
        if (hinted && visible && !sampled && !callback) {
            texture.presentationHint(ImGui::GetFrameCount(), true);
            texture.presentationHint(ImGui::GetFrameCount(), false);
        }
        auto* list = ImGui::GetForegroundDrawList();
        if (sampled || unrelated) {
            list->AddImage(ImTextureRef(static_cast<ImTextureID>(sampled ? 123 : 456)),
                           {0, 0}, {100, 100});
        }
        if (empty) {
            list->PushTexture(ImTextureRef(static_cast<ImTextureID>(123)));
            list->AddDrawCmd();
            list->PopTexture();
        }
        if (callback) {
            list->AddCallback([](const ImDrawList*, const ImDrawCmd*) {}, nullptr);
        }
    });
    window.prepareImgui();
    REQUIRE(window.textureVisibleForPresentation(texture) == visible);

    // Neither sampled textures nor custom callbacks carry into the next frame.
    ui.frame([&] { texture.presentationHint(ImGui::GetFrameCount(), false); });
    window.prepareImgui();
    REQUIRE_FALSE(window.textureVisibleForPresentation(texture));
}

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
    config.onInput = [&](const InputEvent& event) {
        if (const auto* mouse = std::get_if<MouseEvent>(&event)) events.push_back(*mouse);
    };
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
