#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <memory>
#include <vector>

#include "jetstream/render/base/present.hh"

#include "harness.hh"
#include "render/devices/webgpu/present_bindings.hh"
#include "render/present_impl.hh"

using namespace Jetstream;
using Catch::Matchers::WithinAbs;

namespace {

using Blend = Render::PresentList::Blend;
using Layer = Render::PresentList::Layer;
using Policy = Render::WindowMetrics::Policy;

class TestTexture final : public Render::Texture {
 public:
    explicit TestTexture(U64 handle) : Render::Texture(Config{.size = {64, 64}}), handle(handle) {}

    TestTexture(U64 handle, const U8* pixels) : Render::Texture(Config{.size = {64, 64}, .buffer = pixels}),
                                                handle(handle) {}

    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    uint64_t raw() const override { return handle; }

    void reallocate() { markAllocated(); }

 private:
    U64 handle;
};

Render::WindowMetrics Metrics(F32 framebufferScale) {
    const Extent2D<F32> display = {800.0f, 600.0f};
    return Render::WindowMetrics::Compute(display, {display.x * framebufferScale, display.y * framebufferScale},
                                          {framebufferScale, framebufferScale}, 1.0f, Policy::Interface);
}

struct Rig {
    RenderTest::Window window;
    RenderTest::Viewport& viewport = window.viewport;
    std::shared_ptr<TestTexture> canvas = std::make_shared<TestTexture>(1);
    std::vector<U8> pixels = std::vector<U8>(64 * 64 * 4, 0);
    std::shared_ptr<TestTexture> plot = std::make_shared<TestTexture>(2, pixels.data());
    bool underlay = false;
    ColorRGBA<F32> clear = {0.0f, 0.0f, 0.0f, 1.0f};

    Rig() {
        viewport.display = {800.0f, 600.0f};
        REQUIRE(window.open() == Result::SUCCESS);
        REQUIRE(window.bind(std::static_pointer_cast<Render::WindowAttachment>(plot)) == Result::SUCCESS);
    }

    void upload() {
        pixels[0] = static_cast<U8>(pixels[0] + 1);
        REQUIRE(plot->fill() == Result::SUCCESS);
    }

    Result frame() {
        REQUIRE(window.begin() == Result::SUCCESS);
        window.frameClearColor(clear);
        std::vector<Layer> underlays;
        if (underlay) {
            underlays.push_back({.texture = plot, .rect = {10.0f, 10.0f, 100.0f, 50.0f}, .clip = {0.0f, 0.0f, 800.0f, 600.0f}});
        }
        const Layer layer = {.texture = canvas, .rect = {0.0f, 0.0f, 800.0f, 600.0f}, .clip = {0.0f, 0.0f, 800.0f, 600.0f}};
        window.present(Render::PresentList::Build(window.metrics(), underlays, layer, Blend::Premultiplied));
        return window.end();
    }

    U64 presents() const {
        return window.stats().presentedFrames;
    }
};

}  // namespace

TEST_CASE("Underlays present before the canvas with their own blend", "[core][render][present]") {
    const auto metrics = Metrics(1.0f);
    auto canvas = std::make_shared<TestTexture>(1);
    auto first = std::make_shared<TestTexture>(2);
    auto second = std::make_shared<TestTexture>(3);
    auto empty = std::make_shared<TestTexture>(0);

    const std::vector<Layer> underlays = {
        {.texture = first, .rect = {0.0f, 0.0f, 10.0f, 10.0f}, .clip = {0.0f, 0.0f, 10.0f, 10.0f}},
        {.texture = empty, .rect = {0.0f, 0.0f, 10.0f, 10.0f}, .clip = {0.0f, 0.0f, 10.0f, 10.0f}},
        {.texture = second, .rect = {5.0f, 5.0f, 10.0f, 10.0f}, .clip = {5.0f, 5.0f, 10.0f, 10.0f}},
    };
    const Layer layer = {.texture = canvas, .rect = {0.0f, 0.0f, 800.0f, 600.0f}, .clip = {0.0f, 0.0f, 800.0f, 600.0f}};

    const auto list = Render::PresentList::Build(metrics, underlays, layer, Blend::Premultiplied);
    REQUIRE(list.quads.size() == 3);
    CHECK(list.quads[0].texture == first);
    CHECK(list.quads[0].blend == Blend::Straight);
    CHECK(list.quads[1].texture == second);
    CHECK(list.quads[1].blend == Blend::Straight);
    CHECK(list.quads[2].texture == canvas);
    CHECK(list.quads[2].blend == Blend::Premultiplied);
    CHECK(list.quads[2].handle == 1);

    const auto straight = Render::PresentList::Build(metrics, {}, layer, Blend::Straight);
    REQUIRE(straight.quads.size() == 1);
    CHECK(straight.quads[0].blend == Blend::Straight);
}

TEST_CASE("Clips become scissors rounded outward and clamped to the framebuffer", "[core][render][present]") {
    auto canvas = std::make_shared<TestTexture>(1);
    const Layer layer = {.texture = canvas, .rect = {0.0f, 0.0f, 800.0f, 600.0f}, .clip = {10.6f, 20.4f, 100.7f, 50.2f}};

    const auto standard = Render::PresentList::Build(Metrics(1.0f), {}, layer, Blend::Premultiplied);
    REQUIRE(standard.quads.size() == 1);
    CHECK(standard.quads[0].scissor.x == 10);
    CHECK(standard.quads[0].scissor.y == 20);
    CHECK(standard.quads[0].scissor.width == 102);
    CHECK(standard.quads[0].scissor.height == 51);

    const auto retina = Render::PresentList::Build(Metrics(2.0f), {}, layer, Blend::Premultiplied);
    REQUIRE(retina.quads.size() == 1);
    CHECK(retina.quads[0].scissor.x == 21);
    CHECK(retina.quads[0].scissor.y == 40);
    CHECK(retina.quads[0].scissor.width == 202);
    CHECK(retina.quads[0].scissor.height == 102);

    const Layer overflowing = {.texture = canvas, .rect = layer.rect, .clip = {-50.0f, -50.0f, 2000.0f, 2000.0f}};
    const auto clamped = Render::PresentList::Build(Metrics(2.0f), {}, overflowing, Blend::Premultiplied);
    REQUIRE(clamped.quads.size() == 1);
    CHECK(clamped.quads[0].scissor.x == 0);
    CHECK(clamped.quads[0].scissor.y == 0);
    CHECK(clamped.quads[0].scissor.width == 1600);
    CHECK(clamped.quads[0].scissor.height == 1200);
}

TEST_CASE("Fully clipped layers are sampled but not drawn", "[core][render][present]") {
    auto canvas = std::make_shared<TestTexture>(1);
    auto hidden = std::make_shared<TestTexture>(2);
    auto unrelated = std::make_shared<TestTexture>(3);
    const std::vector<Layer> underlays = {
        {.texture = hidden, .rect = {0.0f, 0.0f, 10.0f, 10.0f}, .clip = {0.0f, 0.0f, 0.0f, 0.0f}},
    };
    const Layer layer = {.texture = canvas, .rect = {0.0f, 0.0f, 800.0f, 600.0f}, .clip = {0.0f, 0.0f, 800.0f, 600.0f}};

    const auto list = Render::PresentList::Build(Metrics(1.0f), underlays, layer, Blend::Premultiplied);
    CHECK(list.quads.size() == 1);
    CHECK(list.samples(*canvas));
    CHECK(list.samples(*hidden));
    CHECK_FALSE(list.samples(*unrelated));
}

TEST_CASE("Quad triangles follow each backend clip space", "[core][render][present]") {
    auto canvas = std::make_shared<TestTexture>(1);
    const Layer layer = {.texture = canvas, .rect = {200.0f, 150.0f, 400.0f, 300.0f}, .clip = {0.0f, 0.0f, 800.0f, 600.0f}};
    const auto list = Render::PresentList::Build(Metrics(1.0f), {}, layer, Blend::Premultiplied);
    REQUIRE(list.quads.size() == 1);

    const auto up = Render::PresentGeometry::Triangles(list.quads[0], {800.0f, 600.0f}, false);
    CHECK_THAT(up[0].x, WithinAbs(-0.5, 1e-6));
    CHECK_THAT(up[0].y, WithinAbs(0.5, 1e-6));
    CHECK(up[0].u == 0.0f);
    CHECK(up[0].v == 0.0f);
    CHECK_THAT(up[2].x, WithinAbs(0.5, 1e-6));
    CHECK_THAT(up[2].y, WithinAbs(-0.5, 1e-6));
    CHECK(up[2].u == 1.0f);
    CHECK(up[2].v == 1.0f);

    const auto down = Render::PresentGeometry::Triangles(list.quads[0], {800.0f, 600.0f}, true);
    CHECK_THAT(down[0].y, WithinAbs(-0.5, 1e-6));
    CHECK_THAT(down[2].y, WithinAbs(0.5, 1e-6));
}

TEST_CASE("Frame index advances per frame and gates presentation hints", "[core][render][present]") {
    Rig rig;
    CHECK(rig.window.frameIndex() == 0);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.window.frameIndex() == 1);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.window.frameIndex() == 2);

    TestTexture hidden(7);
    hidden.presentationHint(rig.window.frameIndex(), false);
    CHECK_FALSE(rig.window.textureVisibleForPresentation(hidden));
    hidden.presentationHint(rig.window.frameIndex(), true);
    CHECK(rig.window.textureVisibleForPresentation(hidden));

    TestTexture unhinted(8);
    CHECK(rig.window.textureVisibleForPresentation(unhinted));
}

TEST_CASE("Sampled textures stay visible for the frame that presents them", "[core][render][present]") {
    Rig rig;
    REQUIRE(rig.window.begin() == Result::SUCCESS);
    rig.plot->presentationHint(rig.window.frameIndex(), false);
    CHECK_FALSE(rig.window.textureVisibleForPresentation(*rig.plot));

    const Layer underlay = {.texture = rig.plot, .rect = {0.0f, 0.0f, 10.0f, 10.0f}, .clip = {0.0f, 0.0f, 10.0f, 10.0f}};
    rig.window.present(Render::PresentList::Build(rig.window.metrics(), {underlay}, std::nullopt, Blend::Premultiplied));
    CHECK(rig.window.textureVisibleForPresentation(*rig.plot));
    REQUIRE(rig.window.end() == Result::SUCCESS);

    REQUIRE(rig.window.begin() == Result::SUCCESS);
    rig.plot->presentationHint(rig.window.frameIndex(), false);
    CHECK_FALSE(rig.window.textureVisibleForPresentation(*rig.plot));
    REQUIRE(rig.window.end() == Result::SUCCESS);
}

TEST_CASE("Identical consecutive frames skip presentation", "[core][render][present][retained]") {
    Rig rig;
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 1);
    for (int frame = 0; frame < 10; ++frame) {
        REQUIRE(rig.frame() == Result::SUCCESS);
    }
    CHECK(rig.presents() == 1);
    CHECK(rig.window.stats().skippedPresents == 10);
    CHECK(rig.viewport.idleWaits == 9);
}

TEST_CASE("A texture upload with nothing else changed presents once", "[core][render][present][retained]") {
    Rig rig;
    rig.underlay = true;
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 1);

    const U64 revision = rig.plot->contentRevision();
    rig.upload();
    CHECK(rig.plot->contentRevision() == revision);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.plot->contentRevision() == revision + 1);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 2);

    rig.upload();
    REQUIRE(rig.plot->fillRow(0, 1) == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 3);
}

TEST_CASE("A recreated window starts with fresh input and presents its first frame",
          "[core][render][present][retained]") {
    using Queue = Render::InputQueue;
    Rig rig;
    rig.viewport.input().push(Queue::Key{KeyCode::B, true});
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.window.input().keyDown(KeyCode::B));
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.presents() == 1);

    rig.window.close();
    REQUIRE(rig.window.open() == Result::SUCCESS);
    CHECK_FALSE(rig.window.input().keyDown(KeyCode::B));
    for (U64 frame = 0; frame < 60; ++frame) {
        rig.viewport.platform().clock += 1.0 / 60.0;
        REQUIRE(rig.frame() == Result::SUCCESS);
        CHECK(RenderTest::Only<KeyEvent>(rig.window.input().events()).empty());
    }
    CHECK(rig.presents() == 1);
}

TEST_CASE("A canvas repaint presents once", "[core][render][present][retained]") {
    Rig rig;
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    rig.canvas->markContentChanged();
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 2);
}

TEST_CASE("A reallocated texture behind the same handle presents once", "[core][render][present][retained]") {
    Rig rig;
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    const U64 revision = rig.canvas->contentRevision();
    rig.canvas->reallocate();
    CHECK(rig.canvas->contentRevision() == revision);
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 2);
}

TEST_CASE("An underlay content change presents once and live underlays keep presenting",
          "[core][render][present][retained]") {
    Rig rig;
    rig.underlay = true;
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 1);

    rig.upload();
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 2);

    for (int frame = 0; frame < 5; ++frame) {
        rig.upload();
        REQUIRE(rig.frame() == Result::SUCCESS);
    }
    CHECK(rig.presents() == 7);

    rig.underlay = false;
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 8);
}

TEST_CASE("A resize and a scale change each present once", "[core][render][present][retained]") {
    Rig rig;
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 1);

    rig.viewport.display = {1024.0f, 768.0f};
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 2);

    rig.viewport.scale = 2.0f;
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 3);

    rig.window.setScale(1.5f);
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 4);
}

TEST_CASE("Redraw requests and clear color changes present once", "[core][render][present][retained]") {
    Rig rig;
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    rig.viewport.requestRedraw();
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 2);

    rig.clear = {1.0f, 1.0f, 1.0f, 1.0f};
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 3);
}

TEST_CASE("Viewports without retained presentation present every frame", "[core][render][present][retained]") {
    Rig rig;
    rig.viewport.retained = false;
    for (int frame = 0; frame < 4; ++frame) {
        REQUIRE(rig.frame() == Result::SUCCESS);
    }
    CHECK(rig.presents() == 4);
    CHECK(rig.window.stats().skippedPresents == 0);
}

TEST_CASE("Scissors clamp to the acquired drawable", "[core][render][present]") {
    using Render::PresentGeometry;
    const Render::ScissorRect inside = {10, 20, 100, 50};
    const auto same = PresentGeometry::Clamp(inside, {800, 600});
    CHECK(same.x == 10);
    CHECK(same.y == 20);
    CHECK(same.width == 100);
    CHECK(same.height == 50);

    const auto overflow = PresentGeometry::Clamp({700, 550, 300, 300}, {800, 600});
    CHECK(overflow.x == 700);
    CHECK(overflow.y == 550);
    CHECK(overflow.width == 100);
    CHECK(overflow.height == 50);

    const auto outside = PresentGeometry::Clamp({900, 700, 10, 10}, {800, 600});
    CHECK(outside.width == 0);
    CHECK(outside.height == 0);
}

TEST_CASE("A recreate acquisition recreates the swapchain and presents next frame", "[core][render][present][retained]") {
    Rig rig;
    REQUIRE(rig.frame() == Result::SUCCESS);
    rig.canvas->markContentChanged();
    rig.window.acquisitions.push_back(RenderTest::Window::Acquire::Recreate);
    CHECK(rig.frame() == Result::SKIP);
    CHECK(rig.window.recreates == 1);
    CHECK(rig.window.stats().recreatedFrames == 1);
    CHECK(rig.presents() == 1);

    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 2);
    CHECK(rig.window.recreates == 1);
}

TEST_CASE("A dropped drawable counts once and presents again next frame", "[core][render][present][retained]") {
    Rig rig;
    rig.window.acquisitions.push_back(RenderTest::Window::Acquire::Dropped);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.window.stats().droppedFrames == 1);
    CHECK(rig.presents() == 0);
    CHECK(rig.window.stats().skippedPresents == 0);
    CHECK(rig.viewport.idleWaits == 0);

    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 1);
    CHECK(rig.window.recreates == 0);
}

TEST_CASE("Present bindings follow texture identity and allocation", "[core][render][present]") {
    U64 created = 0;
    std::vector<int> released;
    Render::PresentBindingCache<int> cache([&](const Render::Texture&) { return static_cast<int>(++created); },
                                           [&](int binding) { released.push_back(binding); });

    auto texture = std::make_shared<TestTexture>(5);
    auto sibling = std::make_shared<TestTexture>(5);
    CHECK(cache.acquire(texture) == 1);
    CHECK(cache.acquire(texture) == 1);
    CHECK(cache.acquire(sibling) == 2);
    cache.advance();

    texture->reallocate();
    CHECK(cache.acquire(texture) == 3);
    CHECK(released == std::vector<int>{1});

    sibling.reset();
    cache.advance();
    CHECK(released == std::vector<int>{1, 2});
    CHECK(cache.size() == 1);

    cache.advance();
    cache.advance();
    CHECK(cache.size() == 0);
    CHECK(released == std::vector<int>{1, 2, 3});
}

TEST_CASE("A cancelled frame that recreates presents an unchanged scene once afterwards",
          "[core][render][present][retained]") {
    Rig rig;
    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 1);

    REQUIRE(rig.window.begin() == Result::SUCCESS);
    REQUIRE(rig.window.cancel() == Result::SUCCESS);
    CHECK(rig.window.cancels == 1);
    CHECK(rig.window.recreates == 1);

    REQUIRE(rig.frame() == Result::SUCCESS);
    REQUIRE(rig.frame() == Result::SUCCESS);
    CHECK(rig.presents() == 2);
}

TEST_CASE("Failed acquisitions after a redraw request present the unchanged scene next frame",
          "[core][render][present][retained]") {
    for (const auto acquisition : {RenderTest::Window::Acquire::Dropped, RenderTest::Window::Acquire::Recreate}) {
        Rig rig;
        REQUIRE(rig.frame() == Result::SUCCESS);
        REQUIRE(rig.frame() == Result::SUCCESS);
        CHECK(rig.presents() == 1);

        rig.viewport.requestRedraw();
        rig.window.acquisitions.push_back(acquisition);
        (void)rig.frame();
        CHECK(rig.presents() == 1);

        REQUIRE(rig.frame() == Result::SUCCESS);
        CHECK(rig.presents() == 2);
        REQUIRE(rig.frame() == Result::SUCCESS);
        CHECK(rig.presents() == 2);
    }
}

TEST_CASE("Failed or throwing frames force the unchanged scene to present next frame",
          "[core][render][present][retained]") {
    for (int mode = 0; mode < 3; ++mode) {
        CAPTURE(mode);
        Rig rig;
        REQUIRE(rig.frame() == Result::SUCCESS);
        REQUIRE(rig.frame() == Result::SUCCESS);
        CHECK(rig.presents() == 1);

        if (mode == 0) {
            rig.window.failBegin = Result::ERROR;
            CHECK(rig.window.begin() == Result::ERROR);
        } else if (mode == 1) {
            rig.window.failEnd = Result::ERROR;
            CHECK(rig.frame() == Result::ERROR);
        } else {
            rig.window.throwEnd = true;
            REQUIRE(rig.window.begin() == Result::SUCCESS);
            CHECK_THROWS(rig.window.end());
        }

        REQUIRE(rig.frame() == Result::SUCCESS);
        CHECK(rig.presents() == 2);
    }
}
