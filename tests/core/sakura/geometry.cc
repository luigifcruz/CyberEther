#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"

#include <jetstream/render/base/program.hh>
#include <jetstream/render/base/texture.hh>
#include <jetstream/render/components/curves.hh>
#include <jetstream/render/components/mesh.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/curves.hh>
#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>

#include "render/sakura/detail/corner_mask.hh"
#include "render/sakura/detail/drawable.hh"
#include "render/sakura/detail/helpers.hh"
#include "render/sakura/detail/occlusion.hh"
#include "render/sakura/detail/underlay.hh"
#include "render/sakura/surface_presenter.hh"
#include "render/sakura/metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

TEST_CASE("Wave row capacities cover subpixel periods and clipped sample phases", "[core][sakura][geometry][wave-metrics]") {
    auto ctx = MakeContext();
    ctx.framebufferSize = {3840, 2160};
    for (const F32 period : {0.25f, 0.5f, 1.0f, 12.0f, 63.0f}) {
        CAPTURE(period);
        const auto points = Sakura::WavePoints(-10.0f, 100.0f, 3840.0f, period, 2.0f, -10.0f, 3830.0f);
        REQUIRE(points.size() == static_cast<U64>(std::ceil(3840.0 * 12.0 / period)) + 1);
        CHECK(points.size() * Sakura::kWavePointVertices <= Sakura::WaveRowCapacity(ctx, period));
        CHECK(points.front().x == -10.0f);
        CHECK(points.front().y == 100.0f);
        CHECK(points[3].x == Catch::Approx(-10.0f + period * 0.25f));
        CHECK(points[3].y == 98.0f);
        CHECK(points[9].y == 102.0f);
        CHECK(points.back().x >= 3830.0f);
        CHECK(points.back().x <= 3830.0f + period / 12.0f);

        const auto clipped = Sakura::WavePoints(-10.0f, 100.0f, 1e9f, period, 2.0f, 100.1f, 110.2f);
        REQUIRE_FALSE(clipped.empty());
        CHECK(clipped.front().x <= 100.1f);
        CHECK(clipped.back().x >= 110.2f);
        for (const auto& point : clipped) {
            CHECK(std::find(points.begin(), points.end(), point) != points.end());
        }

        for (const F32 left : {-1000000.0f, -10.0f, 0.0f, 0.1f}) {
            const auto visible = Sakura::WavePoints(left, 100.0f, 1e9f, period, 2.0f, 0.0f, 3840.0f);
            REQUIRE_FALSE(visible.empty());
            CHECK(visible.size() * Sakura::kWavePointVertices <= Sakura::WaveRowCapacity(ctx, period));
        }
    }
}

TEST_CASE("Wave ranges clip before sampling and retain their endpoint brackets", "[core][sakura][geometry][wave-metrics]") {
    const auto whole = Sakura::WavePoints(10.0f, 0.0f, 100.25f, 12.0f, 1.0f, 10.0f, 110.25f);
    REQUIRE(whole.size() == 102);
    CHECK(whole.front().x == 10.0f);
    CHECK(whole.back().x == 111.0f);
    CHECK(Sakura::WavePoints(10.0f, 0.0f, 100.25f, 12.0f, 1.0f, -100.0f, 200.0f) == whole);
    CHECK(Sakura::WavePoints(10.0f, 0.0f, 100.25f, 12.0f, 1.0f, -100.0f, 9.9f).empty());
    CHECK(Sakura::WavePoints(10.0f, 0.0f, 100.25f, 12.0f, 1.0f, 110.5f, 200.0f).empty());
    CHECK(Sakura::WavePoints(10.0f, 0.0f, 100.25f, 12.0f, 1.0f, 20.0f, 10.0f).empty());
    CHECK(Sakura::WavePoints(10.0f, 0.0f, 0.0f, 12.0f, 1.0f, 10.0f, 20.0f).empty());
    CHECK(Sakura::WavePoints(10.0f, 0.0f, -1.0f, 12.0f, 1.0f, 10.0f, 20.0f).empty());
    const auto point = Sakura::WavePoints(10.0f, 0.0f, 100.25f, 12.0f, 1.0f, 20.0f, 20.0f);
    REQUIRE(point.size() == 1);
    CHECK(point.front() == whole[10]);
}

TEST_CASE("Wave helpers reject nonfinite inputs and bound extreme finite sampling", "[core][sakura][geometry][wave-metrics]") {
    auto ctx = MakeContext();
    ctx.framebufferSize = {3840, 2160};
    for (const F32 invalid : {std::numeric_limits<F32>::quiet_NaN(), std::numeric_limits<F32>::infinity(),
                              -std::numeric_limits<F32>::infinity()}) {
        for (U64 parameter = 0; parameter < 7; ++parameter) {
            CAPTURE(invalid, parameter);
            F32 args[] = {0.0f, 0.0f, 100.0f, 12.0f, 1.0f, 0.0f, 100.0f};
            args[parameter] = invalid;
            CHECK(Sakura::WavePoints(args[0], args[1], args[2], args[3], args[4], args[5], args[6]).empty());
        }
        CHECK(Sakura::WaveRowCapacity(ctx, invalid) == 0);
    }
    for (const F32 period : {0.0f, -1.0f}) {
        CHECK(Sakura::WavePoints(0.0f, 0.0f, 100.0f, period, 1.0f, 0.0f, 100.0f).empty());
        CHECK(Sakura::WaveRowCapacity(ctx, period) == 0);
    }
    for (const F32 period : {std::numeric_limits<F32>::denorm_min(), std::numeric_limits<F32>::min(), 1e-20f}) {
        CAPTURE(period);
        CHECK(Sakura::WavePoints(0.0f, 0.0f, 100.0f, period, 1.0f, 0.0f, 100.0f).empty());
        CHECK(Sakura::WaveRowCapacity(ctx, period) <= Sakura::kWaveMaxPoints * Sakura::kWavePointVertices);
    }

    const F32 huge = std::numeric_limits<F32>::max();
    CHECK(Sakura::WavePoints(-huge, 0.0f, huge, 12.0f, 1.0f, -10.0f, 0.0f).empty());
    CHECK(Sakura::WavePoints(huge, 0.0f, huge, huge, 1.0f, huge, huge).size() == 1);
    CHECK(Sakura::WavePoints(0.0f, huge, 100.0f, 12.0f, huge, 0.0f, 100.0f).empty());
    CHECK(Sakura::WavePoints(0.0f, 0.0f, huge, 12.0f, 1.0f, 0.0f, huge).empty());
    const auto distant = Sakura::WavePoints(-1e9f, 0.0f, 2e9f, 12.0f, 1.0f, 0.0f, 10.0f);
    REQUIRE(distant.size() == 11);
    CHECK(distant.front().x == 0.0f);
    CHECK(distant.back().x == 10.0f);
    CHECK(distant.front().y == Catch::Approx(-std::sin(2.0 * std::numbers::pi / 3.0)));

    ctx.framebufferSize.x = std::numeric_limits<U64>::max();
    CHECK(Sakura::WaveRowCapacity(ctx, 1.0f) == Sakura::kWaveMaxPoints * Sakura::kWavePointVertices);
    ctx.framebufferSize.x = 0;
    CHECK(Sakura::WaveRowCapacity(ctx, 12.0f) == 0);
    const F32 limit = static_cast<F32>(Sakura::kWaveMaxPoints - 1);
    CHECK(Sakura::WavePoints(0.0f, 0.0f, limit, 12.0f, 1.0f, 0.0f, limit).size() == Sakura::kWaveMaxPoints);
    CHECK(Sakura::WavePoints(0.0f, 0.0f, limit + 1.0f, 12.0f, 1.0f, 0.0f, limit + 1.0f).empty());
}

namespace {

namespace Components = Jetstream::Render::Components;

using Catch::Approx;

constexpr ColorRGBA<F32> kInk = {1.0f, 0.5f, 0.25f, 1.0f};

struct DeviceWindow : SakuraTest::FakeWindow {
    constexpr DeviceType device() const override {
#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
        return DeviceType::Vulkan;
#elif defined(JETSTREAM_RENDER_WEBGPU_AVAILABLE)
        return DeviceType::WebGPU;
#else
        return DeviceType::None;
#endif
    }
};

struct Rig {
    DeviceWindow window;
    Sakura::Palette palette;
    Sakura::Context ctx = MakeContext();
    Render::Surface::Config surface;
    std::vector<Sakura::Drawable*> drawables;

    Rig() {
        ctx.render = &window;
        ctx.palette = palette;
        ctx.attach = [this](const Sakura::Component&, Sakura::Drawable* drawable) {
            drawables.push_back(drawable);
            return drawable->attach(&ctx, surface);
        };
    }

    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;

    bool supported() const {
        return window.device() != DeviceType::None;
    }

    Result build(Sakura::Component& component) {
        return Sakura::Access::build(component, ctx);
    }

    Sakura::Drawable& drawable() {
        return *drawables.back();
    }

    Result upload() {
        return drawables.back()->upload();
    }

    Render::Program& program() {
        return *surface.programs.back();
    }
};

template<typename T>
std::span<T> Contents(const std::shared_ptr<Render::Buffer>& buffer) {
    return {static_cast<T*>(buffer->getConfig().buffer), static_cast<std::size_t>(buffer->size())};
}

std::span<F32> Uniforms(const Render::Program& program) {
    const auto& buffer = program.getConfig().buffers.front().first;
    return {static_cast<F32*>(buffer->getConfig().buffer), static_cast<std::size_t>(buffer->byteSize() / sizeof(F32))};
}

const Render::Vertex::Config& VertexLayout(const Render::Program& program) {
    return program.getConfig().draws.front()->getConfig().buffer->getConfig();
}

struct DrawableProbe : Sakura::Drawable {
    static Components::Mesh& mesh(Sakura::Drawable& drawable) {
        return *static_cast<Components::Mesh*>((drawable.*(&DrawableProbe::component)).get());
    }
};

struct MeshView {
    std::span<Extent2D<F32>> positions;
    std::span<ColorRGBA<F32>> colors;
    std::span<U32> indices;

    explicit MeshView(Rig& rig) {
        const auto& layout = VertexLayout(rig.program());
        positions = Contents<Extent2D<F32>>(layout.vertices[0].first);
        colors = Contents<ColorRGBA<F32>>(layout.vertices[1].first);
        indices = DrawableProbe::mesh(rig.drawable()).indices();
    }

    U64 vertices() const {
        U64 count = 0;
        while (count < colors.size() && colors[count].r > 0.0f) {
            ++count;
        }
        return count;
    }

    U64 triangles() const {
        U64 count = 0;
        while ((count + 1) * 3 <= indices.size()) {
            const U32 a = indices[count * 3 + 0];
            if (a == indices[count * 3 + 1] && a == indices[count * 3 + 2]) {
                break;
            }
            ++count;
        }
        return count;
    }

    std::vector<U32> head(U64 count) const {
        return {indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(count)};
    }
};

struct CurvesView {
    std::span<Extent2D<F32>> templateVertices;
    std::span<U32> templateIndices;
    std::span<Components::Curves::Instance> instances;

    explicit CurvesView(const Render::Program& program) {
        const auto& layout = VertexLayout(program);
        templateVertices = Contents<Extent2D<F32>>(layout.vertices[0].first);
        templateIndices = Contents<U32>(layout.indices);
        instances = Contents<Components::Curves::Instance>(layout.instances[0].first);
    }
};

class SizedTexture final : public Render::Texture {
 public:
    explicit SizedTexture(Extent2D<U64> size) : Texture(Config{.size = size}) {}

    Result create() override {
        return Result::SUCCESS;
    }

    Result destroy() override {
        return Result::SUCCESS;
    }

    uint64_t raw() const override {
        return 0;
    }
};

void RequireNear(Extent2D<F32> actual, Extent2D<F32> expected, F32 margin = 1e-4f) {
    REQUIRE(actual.x == Approx(expected.x).margin(margin));
    REQUIRE(actual.y == Approx(expected.y).margin(margin));
}

F32 Distance(Extent2D<F32> a, Extent2D<F32> b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

Sakura::Polyline::Instance Stroke(std::vector<Extent2D<F32>> points, F32 thickness = 1.0f) {
    return {.points = std::move(points), .color = kInk, .thickness = thickness};
}

Sakura::Polyline::Instance Fill(std::vector<Extent2D<F32>> points) {
    return {.points = std::move(points), .color = kInk, .filled = true};
}

std::vector<Extent2D<F32>> Square() {
    return {{0.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 10.0f}, {0.0f, 10.0f}};
}

Sakura::SurfaceResize Resize(U64 width, U64 height, F32 scale = 1.0f) {
    return {
        .logicalSize = {width, height},
        .framebufferSize = {static_cast<U64>(static_cast<F32>(width) * scale),
                            static_cast<U64>(static_cast<F32>(height) * scale)},
        .scale = scale,
    };
}

}  // namespace

TEST_CASE("Intersect returns the overlap of two rectangles and collapses disjoint ones to empty",
          "[core][sakura][geometry]") {
    constexpr Rect overlap = Sakura::Intersect({0.0f, 0.0f, 100.0f, 50.0f}, {40.0f, 20.0f, 100.0f, 100.0f});
    STATIC_REQUIRE(overlap.width == 60.0f);
    REQUIRE(overlap == Rect{40.0f, 20.0f, 60.0f, 30.0f});

    const Rect inside = Sakura::Intersect({0.0f, 0.0f, 100.0f, 100.0f}, {10.0f, 10.0f, 5.0f, 5.0f});
    REQUIRE(inside == Rect{10.0f, 10.0f, 5.0f, 5.0f});

    const Rect disjoint = Sakura::Intersect({0.0f, 0.0f, 10.0f, 10.0f}, {20.0f, 30.0f, 10.0f, 10.0f});
    REQUIRE(disjoint.empty());
    REQUIRE(disjoint.width == 0.0f);
    REQUIRE(disjoint.height == 0.0f);

    const Rect touching = Sakura::Intersect({0.0f, 0.0f, 10.0f, 10.0f}, {10.0f, 0.0f, 10.0f, 10.0f});
    REQUIRE(touching.empty());
}

TEST_CASE("PixelToNdc maps framebuffer corners to clip space and pins empty framebuffers to the top left",
          "[core][sakura][geometry]") {
    const Extent2D<U64> framebuffer = {800, 600};
    RequireNear(Sakura::PixelToNdc(framebuffer, 0.0f, 0.0f), {-1.0f, 1.0f});
    RequireNear(Sakura::PixelToNdc(framebuffer, 800.0f, 600.0f), {1.0f, -1.0f});
    RequireNear(Sakura::PixelToNdc(framebuffer, 400.0f, 300.0f), {0.0f, 0.0f});
    RequireNear(Sakura::PixelToNdc(framebuffer, 200.0f, 450.0f), {-0.5f, -0.5f});

    RequireNear(Sakura::PixelToNdc({0, 600}, 100.0f, 100.0f), {-1.0f, 1.0f});
    RequireNear(Sakura::PixelToNdc({800, 0}, 100.0f, 100.0f), {-1.0f, 1.0f});
}

TEST_CASE("RectToScissor covers every pixel a fractional rectangle touches", "[core][sakura][geometry]") {
    const auto scissor = Sakura::RectToScissor({0.5f, 2.25f, 1.0f, 1.5f}, {100, 80});
    REQUIRE(scissor.x == 0);
    REQUIRE(scissor.y == 2);
    REQUIRE(scissor.width == 2);
    REQUIRE(scissor.height == 2);
}

TEST_CASE("RectToScissor clamps rectangles to the framebuffer and rounds partial extents up",
          "[core][sakura][geometry]") {
    const Extent2D<U64> framebuffer = {100, 80};

    const auto exact = Sakura::RectToScissor({10.0f, 20.0f, 30.0f, 40.0f}, framebuffer);
    REQUIRE(exact.x == 10);
    REQUIRE(exact.y == 20);
    REQUIRE(exact.width == 30);
    REQUIRE(exact.height == 40);

    const auto partial = Sakura::RectToScissor({10.0f, 20.0f, 5.5f, 0.25f}, framebuffer);
    REQUIRE(partial.x == 10);
    REQUIRE(partial.y == 20);
    REQUIRE(partial.width == 6);
    REQUIRE(partial.height == 1);

    const auto overhang = Sakura::RectToScissor({-20.0f, -10.0f, 200.0f, 200.0f}, framebuffer);
    REQUIRE(overhang.x == 0);
    REQUIRE(overhang.y == 0);
    REQUIRE(overhang.width == 100);
    REQUIRE(overhang.height == 80);

    const auto outside = Sakura::RectToScissor({150.0f, 90.0f, 10.0f, 10.0f}, framebuffer);
    REQUIRE(outside.x == 100);
    REQUIRE(outside.y == 80);
    REQUIRE(outside.width == 0);
    REQUIRE(outside.height == 0);

    const auto inverted = Sakura::RectToScissor({50.0f, 40.0f, -10.0f, -10.0f}, framebuffer);
    REQUIRE(inverted.width == 0);
    REQUIRE(inverted.height == 0);
}

TEST_CASE("Bezier sampling starts and ends on the endpoints and clamps to at least one segment",
          "[core][sakura][geometry]") {
    const Extent2D<F32> p0 = {0.0f, 0.0f};
    const Extent2D<F32> p1 = {10.0f, 40.0f};
    const Extent2D<F32> p2 = {30.0f, -20.0f};
    const Extent2D<F32> p3 = {50.0f, 10.0f};

    const auto points = Sakura::Polyline::Bezier(p0, p1, p2, p3, 24);
    REQUIRE(points.size() == 25);
    REQUIRE(points.front() == p0);
    RequireNear(points.back(), p3);
    RequireNear(points[12], {0.125f * p0.x + 0.375f * p1.x + 0.375f * p2.x + 0.125f * p3.x,
                             0.125f * p0.y + 0.375f * p1.y + 0.375f * p2.y + 0.125f * p3.y});

    const auto clamped = Sakura::Polyline::Bezier(p0, p1, p2, p3, 0);
    REQUIRE(clamped.size() == 2);
    REQUIRE(clamped.front() == p0);
    RequireNear(clamped.back(), p3);
}

TEST_CASE("Bezier sampling of an evenly spaced straight control polygon is uniform",
          "[core][sakura][geometry]") {
    const auto points = Sakura::Polyline::Bezier({0.0f, 5.0f}, {10.0f, 5.0f}, {20.0f, 5.0f}, {30.0f, 5.0f}, 6);
    REQUIRE(points.size() == 7);
    for (U64 i = 0; i < points.size(); ++i) {
        RequireNear(points[i], {5.0f * static_cast<F32>(i), 5.0f});
    }
}

TEST_CASE("Circle sampling places points on the radius without repeating the start",
          "[core][sakura][geometry]") {
    const Extent2D<F32> center = {50.0f, -20.0f};
    const auto points = Sakura::Polyline::Circle(center, 8.0f, 12);
    REQUIRE(points.size() == 12);
    RequireNear(points.front(), {58.0f, -20.0f});
    RequireNear(points[3], {50.0f, -12.0f});
    RequireNear(points[6], {42.0f, -20.0f});
    for (U64 i = 0; i < points.size(); ++i) {
        REQUIRE(Distance(points[i], center) == Approx(8.0f).margin(1e-4f));
        const auto& next = points[(i + 1) % points.size()];
        REQUIRE(Distance(points[i], next) == Approx(2.0f * 8.0f * std::sin(std::numbers::pi_v<F32> / 12.0f))
                                                 .margin(1e-4f));
    }

    REQUIRE(Sakura::Polyline::Circle(center, 8.0f, 0).size() == 3);
    REQUIRE(Sakura::Polyline::Circle(center, 8.0f, 2).size() == 3);
}

TEST_CASE("Corner masks cover each rectangle corner with a filled quarter arc fan",
          "[core][sakura][geometry]") {
    const Rect rect = {100.0f, 50.0f, 200.0f, 80.0f};
    const ColorRGBA<F32> top = {0.1f, 0.2f, 0.3f, 1.0f};
    const ColorRGBA<F32> bottom = {0.4f, 0.5f, 0.6f, 1.0f};
    const auto masks = Sakura::Polyline::CornerMasks(rect, 10.0f, top, bottom, 6);
    REQUIRE(masks.size() == 4);

    const Extent2D<F32> corners[] = {{100.0f, 50.0f}, {300.0f, 50.0f}, {300.0f, 130.0f}, {100.0f, 130.0f}};
    const Extent2D<F32> centers[] = {{110.0f, 60.0f}, {290.0f, 60.0f}, {290.0f, 120.0f}, {110.0f, 120.0f}};
    const Extent2D<F32> starts[] = {{110.0f, 50.0f}, {300.0f, 60.0f}, {290.0f, 130.0f}, {100.0f, 120.0f}};
    const Extent2D<F32> ends[] = {{100.0f, 60.0f}, {290.0f, 50.0f}, {300.0f, 120.0f}, {110.0f, 130.0f}};

    for (U64 corner = 0; corner < 4; ++corner) {
        const auto& mask = masks[corner];
        REQUIRE(mask.filled);
        REQUIRE_FALSE(mask.closed);
        REQUIRE(mask.visible);
        REQUIRE(mask.color == (corner < 2 ? top : bottom));
        REQUIRE(mask.points.size() == 8);
        REQUIRE(mask.points.front() == corners[corner]);
        RequireNear(mask.points[1], starts[corner]);
        RequireNear(mask.points.back(), ends[corner]);
        for (U64 i = 1; i < mask.points.size(); ++i) {
            REQUIRE(Distance(mask.points[i], centers[corner]) == Approx(10.0f).margin(1e-4f));
            REQUIRE(mask.points[i].x >= rect.x - 1e-4f);
            REQUIRE(mask.points[i].x <= rect.right() + 1e-4f);
            REQUIRE(mask.points[i].y >= rect.y - 1e-4f);
            REQUIRE(mask.points[i].y <= rect.bottom() + 1e-4f);
        }
    }
}

TEST_CASE("Corner mask polygons wind so their fringe faces out of the rectangle",
          "[core][sakura][geometry]") {
    const Rect rect = {0.0f, 0.0f, 100.0f, 60.0f};
    const auto masks = Sakura::Polyline::CornerMasks(rect, 12.0f, kInk, kInk, 8);
    REQUIRE(masks.size() == 4);
    for (const auto& mask : masks) {
        const auto& a = mask.points[0];
        const auto& b = mask.points[1];
        const F32 length = Distance(a, b);
        const Extent2D<F32> normal = {(b.y - a.y) / length, -(b.x - a.x) / length};
        const Extent2D<F32> probe = {(a.x + b.x) * 0.5f + normal.x, (a.y + b.y) * 0.5f + normal.y};
        REQUIRE_FALSE(rect.contains(probe));
    }
}

TEST_CASE("Corner mask radius is clamped to half the shorter side and vanishes when not positive",
          "[core][sakura][geometry]") {
    const Rect rect = {0.0f, 0.0f, 40.0f, 10.0f};
    const auto masks = Sakura::Polyline::CornerMasks(rect, 50.0f, kInk, kInk, 4);
    REQUIRE(masks.size() == 4);
    RequireNear(masks[0].points[1], {5.0f, 0.0f});
    RequireNear(masks[0].points.back(), {0.0f, 5.0f});
    RequireNear(masks[2].points[1], {35.0f, 10.0f});

    REQUIRE(Sakura::Polyline::CornerMasks(rect, 0.0f, kInk, kInk).empty());
    REQUIRE(Sakura::Polyline::CornerMasks(rect, -4.0f, kInk, kInk).empty());
    REQUIRE(Sakura::Polyline::CornerMasks({0.0f, 0.0f, 0.0f, 10.0f}, 4.0f, kInk, kInk).empty());

    const auto single = Sakura::Polyline::CornerMasks(rect, 2.0f, kInk, kInk, 0);
    REQUIRE(single.size() == 4);
    REQUIRE(single[0].points.size() == 3);
}

TEST_CASE("Thin strokes emit a solid center and two transparent fringe vertices per point",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline line;
    line.update({.instances = {Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}, {20.0f, 0.0f}})}});
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE(rig.program().enabled());

    const MeshView mesh(rig);
    REQUIRE(mesh.vertices() == 9);
    REQUIRE(mesh.triangles() == 8);
    for (U64 point = 0; point < 3; ++point) {
        const F32 x = 10.0f * static_cast<F32>(point);
        RequireNear(mesh.positions[point * 3 + 0], {x, 0.0f});
        RequireNear(mesh.positions[point * 3 + 1], {x, -1.0f});
        RequireNear(mesh.positions[point * 3 + 2], {x, 1.0f});
        REQUIRE(mesh.colors[point * 3 + 0] == kInk);
        REQUIRE(mesh.colors[point * 3 + 1].a == 0.0f);
        REQUIRE(mesh.colors[point * 3 + 1].r == kInk.r);
        REQUIRE(mesh.colors[point * 3 + 2].a == 0.0f);
    }
    REQUIRE(mesh.head(24) == std::vector<U32>{3, 0, 2, 2, 5, 3, 4, 1, 0, 0, 3, 4,
                                              6, 3, 5, 5, 8, 6, 7, 4, 3, 3, 6, 7});
}

TEST_CASE("Thin stroke joints are mitered so the fringe keeps its width around corners",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline line;
    line.update({.instances = {Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 10.0f}})}});
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);

    const MeshView mesh(rig);
    REQUIRE(mesh.vertices() == 9);
    RequireNear(mesh.positions[4], {11.0f, -1.0f});
    RequireNear(mesh.positions[5], {9.0f, 1.0f});
    RequireNear(mesh.positions[7], {11.0f, 10.0f});
    RequireNear(mesh.positions[8], {9.0f, 10.0f});
}

TEST_CASE("Sharp hairpin joints keep the miter within ten times the fringe",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    const std::vector<Extent2D<F32>> points = {{0.0f, 0.0f}, {10.0f, 0.0f}, {0.0f, 0.1f}};
    Sakura::Polyline line;
    line.update({.instances = {Stroke(points)}});
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);

    const MeshView mesh(rig);
    REQUIRE(mesh.vertices() == 9);
    RequireNear(mesh.positions[4], {10.5f, -0.0025f}, 1e-3f);
    RequireNear(mesh.positions[5], {9.5f, 0.0025f}, 1e-3f);
    for (U64 point = 0; point < points.size(); ++point) {
        REQUIRE(Distance(mesh.positions[point * 3 + 1], points[point]) <= 10.0f);
        REQUIRE(Distance(mesh.positions[point * 3 + 2], points[point]) <= 10.0f);
    }
}

TEST_CASE("Strokes below the minimum thickness choose thin or thick geometry from the clamped thickness",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline line;
    line.update({.instances = {Stroke({{0.0f, 0.0f}, {20.0f, 0.0f}}, 0.3f)}, .fringe = 0.5f});
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE(MeshView(rig).vertices() == 8);
}

TEST_CASE("Thick strokes emit a solid core between transparent fringes",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline line;
    line.update({.instances = {Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}}, 4.0f)}});
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);

    const MeshView mesh(rig);
    REQUIRE(mesh.vertices() == 8);
    REQUIRE(mesh.triangles() == 6);
    const F32 offsets[] = {-2.5f, -1.5f, 1.5f, 2.5f};
    const F32 alphas[] = {0.0f, 1.0f, 1.0f, 0.0f};
    for (U64 point = 0; point < 2; ++point) {
        for (U64 side = 0; side < 4; ++side) {
            RequireNear(mesh.positions[point * 4 + side], {10.0f * static_cast<F32>(point), offsets[side]});
            REQUIRE(mesh.colors[point * 4 + side].a == alphas[side]);
        }
    }
    REQUIRE(mesh.head(18) == std::vector<U32>{5, 1, 2, 2, 6, 5, 5, 1, 0, 0, 4, 5, 6, 2, 3, 3, 7, 6});
}

TEST_CASE("Closed strokes wrap the last segment back onto the first vertex",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    auto outline = Stroke(Square());
    outline.closed = true;
    Sakura::Polyline line;
    line.update({.instances = {outline}});
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);

    const MeshView mesh(rig);
    REQUIRE(mesh.vertices() == 12);
    REQUIRE(mesh.triangles() == 16);
    RequireNear(mesh.positions[1], {-1.0f, -1.0f});
    RequireNear(mesh.positions[2], {1.0f, 1.0f});
    RequireNear(mesh.positions[7], {11.0f, 11.0f});
    const auto indices = mesh.head(48);
    REQUIRE(std::vector<U32>(indices.end() - 12, indices.end()) ==
            std::vector<U32>{0, 9, 11, 11, 2, 0, 1, 10, 9, 9, 0, 1});
}

TEST_CASE("Filled polygons fan the interior and ring it with a half fringe",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline shape;
    shape.update({.instances = {Fill(Square())}});
    REQUIRE(rig.build(shape) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);

    const MeshView mesh(rig);
    REQUIRE(mesh.vertices() == 8);
    REQUIRE(mesh.triangles() == 10);
    const Extent2D<F32> inner[] = {{0.5f, 0.5f}, {9.5f, 0.5f}, {9.5f, 9.5f}, {0.5f, 9.5f}};
    const Extent2D<F32> outer[] = {{-0.5f, -0.5f}, {10.5f, -0.5f}, {10.5f, 10.5f}, {-0.5f, 10.5f}};
    for (U64 point = 0; point < 4; ++point) {
        RequireNear(mesh.positions[point * 2 + 0], inner[point]);
        RequireNear(mesh.positions[point * 2 + 1], outer[point]);
        REQUIRE(mesh.colors[point * 2 + 0] == kInk);
        REQUIRE(mesh.colors[point * 2 + 1].a == 0.0f);
    }
    REQUIRE(mesh.head(12) == std::vector<U32>{0, 2, 4, 0, 4, 6, 0, 6, 7, 7, 1, 0});
}

TEST_CASE("Stroke fringe and minimum thickness follow the polyline scale and fringe",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    const std::vector<Extent2D<F32>> segment = {{0.0f, 0.0f}, {10.0f, 0.0f}};

    SECTION("Thin strokes at scale two use a half unit fringe") {
        Sakura::Polyline line;
        line.update({.instances = {Stroke(segment, 0.25f)}, .scale = 2.0f});
        REQUIRE(rig.build(line) == Result::SUCCESS);
        REQUIRE(rig.upload() == Result::SUCCESS);
        const MeshView mesh(rig);
        REQUIRE(mesh.vertices() == 6);
        RequireNear(mesh.positions[1], {0.0f, -0.5f});
        RequireNear(mesh.positions[2], {0.0f, 0.5f});
    }

    SECTION("Thick strokes at scale two shrink the fringe but keep the core") {
        Sakura::Polyline line;
        line.update({.instances = {Stroke(segment, 1.0f)}, .scale = 2.0f});
        REQUIRE(rig.build(line) == Result::SUCCESS);
        REQUIRE(rig.upload() == Result::SUCCESS);
        const MeshView mesh(rig);
        REQUIRE(mesh.vertices() == 8);
        RequireNear(mesh.positions[0], {0.0f, -0.75f});
        RequireNear(mesh.positions[1], {0.0f, -0.25f});
        RequireNear(mesh.positions[2], {0.0f, 0.25f});
        RequireNear(mesh.positions[3], {0.0f, 0.75f});
    }

    SECTION("Sub pixel thick strokes are widened to one pixel") {
        Sakura::Polyline line;
        line.update({.instances = {Stroke(segment, 0.75f)}, .fringe = 0.5f});
        REQUIRE(rig.build(line) == Result::SUCCESS);
        REQUIRE(rig.upload() == Result::SUCCESS);
        const MeshView mesh(rig);
        REQUIRE(mesh.vertices() == 8);
        RequireNear(mesh.positions[0], {0.0f, -0.75f});
        RequireNear(mesh.positions[1], {0.0f, -0.25f});
    }

    SECTION("Wider fringes keep thin strokes thin") {
        Sakura::Polyline line;
        line.update({.instances = {Stroke(segment, 1.5f)}, .fringe = 2.0f});
        REQUIRE(rig.build(line) == Result::SUCCESS);
        REQUIRE(rig.upload() == Result::SUCCESS);
        const MeshView mesh(rig);
        REQUIRE(mesh.vertices() == 6);
        RequireNear(mesh.positions[1], {0.0f, -2.0f});
    }
}

TEST_CASE("Hidden, transparent, and degenerate polyline instances emit no geometry",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    auto hidden = Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}});
    hidden.visible = false;
    auto clear = Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}});
    clear.color.a = 0.0f;
    auto clearFill = Fill(Square());
    clearFill.color.a = 0.0f;

    Sakura::Polyline line;
    line.update({.instances = {
                     hidden,
                     clear,
                     clearFill,
                     Stroke({{5.0f, 5.0f}}),
                     Stroke({}),
                     Fill({{0.0f, 0.0f}, {10.0f, 0.0f}}),
                 }});
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE_FALSE(rig.program().enabled());

    const MeshView mesh(rig);
    REQUIRE(mesh.vertices() == 0);
    REQUIRE(mesh.triangles() == 0);
}

TEST_CASE("Polyline instances that overflow the capacity are dropped along with every later instance",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline line;
    line.update({
        .instances = {
            Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}}),
            Stroke({{0.0f, 20.0f}, {10.0f, 20.0f}}, 4.0f),
            Stroke({{0.0f, 40.0f}, {10.0f, 40.0f}}),
        },
        .capacity = 12,
    });
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);

    const MeshView mesh(rig);
    REQUIRE(mesh.positions.size() == 12);
    REQUIRE(mesh.vertices() == 6);
    REQUIRE(mesh.triangles() == 4);
    for (U64 vertex = 0; vertex < 6; ++vertex) {
        REQUIRE(std::abs(mesh.positions[vertex].y) <= 1.0f);
    }
}

TEST_CASE("Polyline uploads combine origin, translation, scale, clip, and inherited clip",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline line;
    Sakura::Polyline::Config config = {
        .instances = {Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}})},
        .clip = Rect{0.0f, 0.0f, 100.0f, 100.0f},
        .origin = {10.0f, 20.0f},
        .scale = 2.0f,
    };
    line.update(config);
    REQUIRE(rig.build(line) == Result::SUCCESS);
    rig.drawable().translation = {5.0f, 5.0f};
    rig.drawable().inherited = Rect{50.0f, 50.0f, 100.0f, 100.0f};
    REQUIRE(rig.upload() == Result::SUCCESS);

    const auto uniforms = Uniforms(rig.program());
    REQUIRE(uniforms.size() == 8);
    REQUIRE(std::vector<F32>(uniforms.begin(), uniforms.end()) ==
            std::vector<F32>{800.0f, 600.0f, 15.0f, 25.0f, 2.0f, 0.0f, 0.0f, 0.0f});
    const auto scissor = rig.program().getConfig().scissorRect;
    REQUIRE(scissor.has_value());
    REQUIRE(scissor->x == 55);
    REQUIRE(scissor->y == 55);
    REQUIRE(scissor->width == 50);
    REQUIRE(scissor->height == 50);

    rig.drawable().inherited.reset();
    config.clip.reset();
    line.update(config);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE_FALSE(rig.program().getConfig().scissorRect.has_value());

    rig.drawable().inherited = Rect{-40.0f, -40.0f, 60.0f, 60.0f};
    REQUIRE(rig.upload() == Result::SUCCESS);
    const auto inherited = rig.program().getConfig().scissorRect;
    REQUIRE(inherited.has_value());
    REQUIRE(inherited->x == 0);
    REQUIRE(inherited->y == 0);
    REQUIRE(inherited->width == 25);
    REQUIRE(inherited->height == 25);
}

TEST_CASE("Polyline uploads copy geometry only after it changes",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline line;
    Sakura::Polyline::Config config = {.instances = {Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}})}};
    line.update(config);
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);

    const MeshView mesh(rig);
    const Extent2D<F32> sentinel = {-7.0f, -7.0f};
    mesh.positions[0] = sentinel;

    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE(mesh.positions[0] == sentinel);

    config.clip = Rect{0.0f, 0.0f, 20.0f, 20.0f};
    config.origin = {3.0f, 4.0f};
    line.update(config);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE(mesh.positions[0] == sentinel);

    config.instances.front().points.front() = {2.0f, 0.0f};
    line.update(config);
    REQUIRE(rig.upload() == Result::SUCCESS);
    RequireNear(mesh.positions[0], {2.0f, 0.0f});
}

TEST_CASE("Polyline updates repaint only when geometry, clip, or origin change",
          "[core][sakura][geometry]") {
    Sakura::Polyline line;
    Sakura::Polyline::Config config = {.instances = {Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}})}};
    line.update(config);
    Frames frames(line);
    REQUIRE(frames.run().rebuilt);
    REQUIRE(frames.run().clean());

    REQUIRE(frames.run([&] { line.update(config); }).clean());

    config.instances.front().color.g = 0.75f;
    REQUIRE(RepaintOnly(frames.run([&] { line.update(config); })));

    config.clip = Rect{0.0f, 0.0f, 50.0f, 50.0f};
    REQUIRE(RepaintOnly(frames.run([&] { line.update(config); })));

    config.origin = {1.0f, 1.0f};
    REQUIRE(RepaintOnly(frames.run([&] { line.update(config); })));

    config.scale = 3.0f;
    REQUIRE(RepaintOnly(frames.run([&] { line.update(config); })));

    config.fringe = 0.5f;
    REQUIRE(RepaintOnly(frames.run([&] { line.update(config); })));

    REQUIRE(frames.run().clean());
}

TEST_CASE("Hiding attached polylines and curves disables their programs",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline line;
    line.update({.instances = {Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}})}});
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);
    auto& mesh = rig.program();
    REQUIRE(mesh.enabled());

    Sakura::Curves curves;
    curves.update({.instances = {{.p3 = {10.0f, 10.0f}, .segments = 8}}});
    REQUIRE(rig.build(curves) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);
    auto& curve = rig.program();
    REQUIRE(curve.enabled());

    line.setVisible(false);
    curves.setVisible(false);
    REQUIRE(Sakura::Access::paint(line) == Result::SUCCESS);
    REQUIRE(Sakura::Access::paint(curves) == Result::SUCCESS);
    REQUIRE_FALSE(mesh.enabled());
    REQUIRE_FALSE(curve.enabled());
}

TEST_CASE("Changing polyline capacity after attachment rebuilds a mesh of the new size",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Sakura::Polyline line;
    Sakura::Polyline::Config config = {.instances = {Stroke({{0.0f, 0.0f}, {10.0f, 0.0f}})}, .capacity = 64};
    line.update(config);
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE(MeshView(rig).positions.size() == 64);
    REQUIRE_FALSE(Sakura::Access::resourceDirty(line));

    line.update(config);
    REQUIRE_FALSE(Sakura::Access::resourceDirty(line));

    config.capacity = 128;
    line.update(config);
    REQUIRE(Sakura::Access::resourceDirty(line));
    REQUIRE(rig.build(line) == Result::SUCCESS);
    REQUIRE_FALSE(Sakura::Access::resourceDirty(line));
    REQUIRE(rig.surface.programs.size() == 2);
    REQUIRE(MeshView(rig).positions.size() == 128);

    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE(MeshView(rig).vertices() == 6);
}

TEST_CASE("Attached boxes rebuild only when their instances outgrow the capacity",
          "[core][sakura][geometry][retention]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound shape resources require Vulkan or WebGPU support.");
    }
    const auto paintOnly = [](const Sakura::Component& component) {
        return !Sakura::Access::resourceDirty(component) && Sakura::Access::paintDirty(component);
    };

    SECTION("Automatic capacity grows with the instance count and never shrinks") {
        Sakura::Box box;
        Sakura::Box::Config config = {.instances = {{.rect = {0.0f, 0.0f, 10.0f, 10.0f}}}};
        box.update(config);
        REQUIRE(rig.build(box) == Result::SUCCESS);
        REQUIRE(Sakura::Access::paint(box) == Result::SUCCESS);
        REQUIRE_FALSE(Sakura::Access::resourceDirty(box));
        REQUIRE_FALSE(Sakura::Access::paintDirty(box));
        const U64 programs = rig.surface.programs.size();

        config.instances.front().backgroundColor = kInk;
        box.update(config);
        CHECK(paintOnly(box));
        REQUIRE(Sakura::Access::paint(box) == Result::SUCCESS);

        config.instances.push_back({.rect = {20.0f, 0.0f, 10.0f, 10.0f}});
        box.update(config);
        CHECK(Sakura::Access::resourceDirty(box));
        REQUIRE(rig.build(box) == Result::SUCCESS);
        CHECK_FALSE(Sakura::Access::resourceDirty(box));
        CHECK(rig.surface.programs.size() > programs);
        REQUIRE(Sakura::Access::paint(box) == Result::SUCCESS);

        config.instances.pop_back();
        box.update(config);
        CHECK(paintOnly(box));
        REQUIRE(Sakura::Access::paint(box) == Result::SUCCESS);

        config.instances.push_back({.rect = {40.0f, 0.0f, 10.0f, 10.0f}});
        box.update(config);
        CHECK(paintOnly(box));
    }

    SECTION("Fixed capacity rebuilds only when the capacity itself changes") {
        Sakura::Box box;
        Sakura::Box::Config config = {.instances = {{.rect = {0.0f, 0.0f, 10.0f, 10.0f}}}, .capacity = 4};
        box.update(config);
        REQUIRE(rig.build(box) == Result::SUCCESS);
        REQUIRE(Sakura::Access::paint(box) == Result::SUCCESS);

        for (U64 index = 1; index < 6; ++index) {
            config.instances.push_back({.rect = {20.0f * static_cast<F32>(index), 0.0f, 10.0f, 10.0f}});
            box.update(config);
            CAPTURE(index);
            CHECK(paintOnly(box));
            REQUIRE(Sakura::Access::paint(box) == Result::SUCCESS);
        }

        config.capacity = 8;
        box.update(config);
        CHECK(Sakura::Access::resourceDirty(box));
        REQUIRE(rig.build(box) == Result::SUCCESS);
        CHECK_FALSE(Sakura::Access::resourceDirty(box));
    }
}

TEST_CASE("Attached labels rebuild only when their instance or character capacity changes",
          "[core][sakura][geometry][retention]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound text resources require Vulkan or WebGPU support.");
    }
    MonoFontWindow fonts;
    REQUIRE(rig.window.addFont("default_mono", fonts.font("default_mono")) == Result::SUCCESS);

    Sakura::Label label;
    Sakura::Label::Config config = {
        .instances = {{.rect = {0.0f, 0.0f, 80.0f, 16.0f}, .str = "Gain", .color = kInk}},
        .maxCharacters = 16,
    };
    label.update(config);
    REQUIRE(rig.build(label) == Result::SUCCESS);
    REQUIRE_FALSE(Sakura::Access::resourceDirty(label));
    const U64 programs = rig.surface.programs.size();

    config.instances.front().str = "Bandwidth";
    config.instances.front().color.g = 0.75f;
    label.update(config);
    CHECK_FALSE(Sakura::Access::resourceDirty(label));

    config.instances.push_back({.rect = {0.0f, 20.0f, 80.0f, 16.0f}, .str = "Rate", .color = kInk});
    label.update(config);
    CHECK(Sakura::Access::resourceDirty(label));
    REQUIRE(rig.build(label) == Result::SUCCESS);
    CHECK_FALSE(Sakura::Access::resourceDirty(label));
    CHECK(rig.surface.programs.size() == programs + 1);

    config.instances.pop_back();
    label.update(config);
    CHECK_FALSE(Sakura::Access::resourceDirty(label));

    config.instances.push_back({.rect = {0.0f, 40.0f, 80.0f, 16.0f}, .str = "Span", .color = kInk});
    label.update(config);
    CHECK_FALSE(Sakura::Access::resourceDirty(label));

    config.maxCharacters = 32;
    label.update(config);
    CHECK(Sakura::Access::resourceDirty(label));
    REQUIRE(rig.build(label) == Result::SUCCESS);
    CHECK_FALSE(Sakura::Access::resourceDirty(label));

    config.capacity = 8;
    label.update(config);
    CHECK(Sakura::Access::resourceDirty(label));
    REQUIRE(rig.build(label) == Result::SUCCESS);
    CHECK_FALSE(Sakura::Access::resourceDirty(label));

    for (U64 index = 2; index < 10; ++index) {
        config.instances.push_back({.rect = {0.0f, 20.0f * static_cast<F32>(index), 80.0f, 16.0f},
                                    .str = "Row",
                                    .color = kInk});
    }
    label.update(config);
    CHECK_FALSE(Sakura::Access::resourceDirty(label));
}

TEST_CASE("Attached images rebuild when the texture is swapped and repaint for visual changes",
          "[core][sakura][geometry][retention]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound image resources require Vulkan or WebGPU support.");
    }
    const auto first = std::make_shared<SizedTexture>(Extent2D<U64>{64, 64});
    const auto second = std::make_shared<SizedTexture>(Extent2D<U64>{128, 128});
    Sakura::Image image;
    Sakura::Image::Config config = {.texture = first, .rect = {0.0f, 0.0f, 64.0f, 64.0f}};
    image.update(config);
    REQUIRE(rig.build(image) == Result::SUCCESS);
    REQUIRE(Sakura::Access::paint(image) == Result::SUCCESS);
    REQUIRE_FALSE(Sakura::Access::resourceDirty(image));
    const U64 programs = rig.surface.programs.size();
    REQUIRE(programs > 0);

    image.update(config);
    CHECK_FALSE(Sakura::Access::resourceDirty(image));
    CHECK_FALSE(Sakura::Access::paintDirty(image));

    config.opacity = 0.5f;
    config.rect = {10.0f, 10.0f, 64.0f, 64.0f};
    image.update(config);
    CHECK_FALSE(Sakura::Access::resourceDirty(image));
    CHECK(Sakura::Access::paintDirty(image));
    REQUIRE(Sakura::Access::paint(image) == Result::SUCCESS);

    config.texture = second;
    image.update(config);
    CHECK(Sakura::Access::resourceDirty(image));
    REQUIRE(rig.build(image) == Result::SUCCESS);
    CHECK_FALSE(Sakura::Access::resourceDirty(image));
    CHECK(rig.surface.programs.size() == programs + 1);
}

TEST_CASE("Corner mask updates fill four corners at exact capacity and clip to the visible rectangle",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    const ColorRGBA<F32> backdrop = {1.0f, 0.5f, 0.25f, 1.0f};
    const ColorRGBA<F32> header = {1.0f, 0.25f, 0.5f, 1.0f};
    rig.palette = {{"backdrop", backdrop}, {"header", header}};
    rig.ctx.backdropColorKey = "backdrop";
    rig.ctx.backdropTopColorKey = "header";
    rig.ctx.pixelRatio = 2.0f;

    const Rect rect = {100.0f, 100.0f, 200.0f, 100.0f};
    Sakura::Polyline mask;
    Sakura::UpdateCornerMask(rig.ctx, mask, rect, {0.0f, 0.0f, 250.0f, 600.0f});
    REQUIRE(rig.build(mask) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE(rig.program().enabled());

    const MeshView mesh(rig);
    REQUIRE(mesh.positions.size() == Sakura::kCornerMaskCapacity);
    REQUIRE(mesh.vertices() == Sakura::kCornerMaskCapacity);
    REQUIRE(mesh.triangles() == 4 * (Sakura::kCornerMaskSegments + 2 * (Sakura::kCornerMaskSegments + 2)));
    const U64 half = Sakura::kCornerMaskCapacity / 2;
    for (U64 vertex = 0; vertex < Sakura::kCornerMaskCapacity; vertex += 2) {
        const auto& expected = vertex < half ? header : backdrop;
        REQUIRE(mesh.colors[vertex] == expected);
    }
    RequireNear(mesh.positions[0], {100.0f, 100.0f}, 1.0f);
    REQUIRE(Distance(mesh.positions[2], {116.0f, 116.0f}) == Approx(16.0f).margin(0.5f));

    const auto scissor = rig.program().getConfig().scissorRect;
    REQUIRE(scissor.has_value());
    REQUIRE(scissor->x == 100);
    REQUIRE(scissor->y == 100);
    REQUIRE(scissor->width == 150);
    REQUIRE(scissor->height == 100);
}

TEST_CASE("Corner masks fall back to the backdrop color and clear without one",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    const ColorRGBA<F32> backdrop = {1.0f, 0.5f, 0.25f, 1.0f};
    rig.palette = {{"backdrop", backdrop}};
    rig.ctx.backdropColorKey = "backdrop";

    const Rect rect = {10.0f, 10.0f, 100.0f, 60.0f};
    Sakura::Polyline mask;
    Sakura::UpdateCornerMask(rig.ctx, mask, rect, kScreen);
    REQUIRE(rig.build(mask) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);
    const MeshView mesh(rig);
    for (U64 vertex = 0; vertex < Sakura::kCornerMaskCapacity; vertex += 2) {
        REQUIRE(mesh.colors[vertex] == backdrop);
    }

    rig.ctx.backdropColorKey.clear();
    Sakura::UpdateCornerMask(rig.ctx, mask, rect, kScreen);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE_FALSE(rig.program().enabled());

    rig.ctx.backdropColorKey = "backdrop";
    Sakura::UpdateCornerMask(rig.ctx, mask, {10.0f, 10.0f, 0.0f, 60.0f}, kScreen);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE_FALSE(rig.program().enabled());
}

TEST_CASE("Curve uploads clamp segment counts and instance totals to the attached capacity",
          "[core][sakura][geometry]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound curve resources require Vulkan or WebGPU support.");
    }
    const Sakura::Curves::Instance first = {
        .p0 = {1.0f, 2.0f},
        .p1 = {3.0f, 4.0f},
        .p2 = {5.0f, 6.0f},
        .p3 = {7.0f, 8.0f},
        .color = kInk,
        .thickness = 2.5f,
        .segments = 0,
    };
    Sakura::Curves::Instance second = first;
    second.segments = 1000;
    Sakura::Curves::Instance third = first;
    third.segments = 4;

    Sakura::Curves curves;
    Sakura::Curves::Config config = {
        .instances = {first, second, third},
        .origin = {3.0f, 4.0f},
        .scale = 1.5f,
        .clip = Rect{10.0f, 10.0f, 20.0f, 20.0f},
        .fringe = 0.75f,
        .capacity = 2,
        .maxSegments = 16,
    };
    curves.update(config);
    REQUIRE(rig.build(curves) == Result::SUCCESS);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE(rig.program().enabled());

    const CurvesView view(rig.program());
    REQUIRE(view.instances.size() == 2);
    REQUIRE(view.instances[0].p0 == first.p0);
    REQUIRE(view.instances[0].p1 == first.p1);
    REQUIRE(view.instances[0].p2 == first.p2);
    REQUIRE(view.instances[0].p3 == first.p3);
    REQUIRE(view.instances[0].color == kInk);
    REQUIRE(view.instances[0].thickness == 2.5f);
    REQUIRE(view.instances[0].segments == 1.0f);
    REQUIRE(view.instances[1].segments == 16.0f);

    const auto uniforms = Uniforms(rig.program());
    REQUIRE(std::vector<F32>(uniforms.begin(), uniforms.end()) ==
            std::vector<F32>{800.0f, 600.0f, 3.0f, 4.0f, 1.5f, 0.75f, 16.0f, 0.0f});
    const auto scissor = rig.program().getConfig().scissorRect;
    REQUIRE(scissor.has_value());
    REQUIRE(scissor->x == 10);
    REQUIRE(scissor->width == 20);

    view.instances[0].thickness = 99.0f;
    config.origin = {6.0f, 7.0f};
    curves.update(config);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE(view.instances[0].thickness == 99.0f);
    REQUIRE(Uniforms(rig.program())[2] == 6.0f);

    config.instances = {third};
    curves.update(config);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE(view.instances[0].thickness == 2.5f);
    REQUIRE(view.instances[0].segments == 4.0f);

    config.instances.clear();
    curves.update(config);
    REQUIRE(rig.upload() == Result::SUCCESS);
    REQUIRE_FALSE(rig.program().enabled());
}

TEST_CASE("Curve updates repaint on visible changes and rebuild only when attached capacity changes",
          "[core][sakura][geometry]") {
    SECTION("Detached curves repaint for every visible field") {
        Sakura::Curves curves;
        Sakura::Curves::Config config = {.instances = {{.p3 = {10.0f, 10.0f}, .segments = 8}}};
        curves.update(config);
        Frames frames(curves);
        REQUIRE(frames.run().rebuilt);
        REQUIRE(frames.run().clean());
        REQUIRE(frames.run([&] { curves.update(config); }).clean());

        config.instances.front().thickness = 2.0f;
        REQUIRE(RepaintOnly(frames.run([&] { curves.update(config); })));
        config.origin = {1.0f, 2.0f};
        REQUIRE(RepaintOnly(frames.run([&] { curves.update(config); })));
        config.scale = 2.0f;
        REQUIRE(RepaintOnly(frames.run([&] { curves.update(config); })));
        config.clip = Rect{0.0f, 0.0f, 5.0f, 5.0f};
        REQUIRE(RepaintOnly(frames.run([&] { curves.update(config); })));
        config.fringe = 2.0f;
        REQUIRE(RepaintOnly(frames.run([&] { curves.update(config); })));
    }

    SECTION("Attached curves rebuild when capacity or segment limits change") {
        Rig rig;
        if (!rig.supported()) {
            SKIP("Unbound curve resources require Vulkan or WebGPU support.");
        }
        Sakura::Curves curves;
        Sakura::Curves::Config config = {.instances = {{.segments = 8}}, .capacity = 4, .maxSegments = 8};
        curves.update(config);
        REQUIRE(rig.build(curves) == Result::SUCCESS);
        REQUIRE_FALSE(Sakura::Access::resourceDirty(curves));

        config.capacity = 6;
        curves.update(config);
        REQUIRE(Sakura::Access::resourceDirty(curves));
        REQUIRE(rig.build(curves) == Result::SUCCESS);
        REQUIRE(CurvesView(rig.program()).instances.size() == 6);

        config.maxSegments = 32;
        curves.update(config);
        REQUIRE(Sakura::Access::resourceDirty(curves));
        REQUIRE(rig.build(curves) == Result::SUCCESS);
        REQUIRE(CurvesView(rig.program()).templateVertices.size() == 33 * 4);
    }
}

TEST_CASE("Mesh configuration clamps capacities to at least one whole triangle",
          "[core][sakura][geometry]") {
    const Components::Mesh empty({.vertexCapacity = 0, .indexCapacity = 10});
    REQUIRE(empty.getConfig().vertexCapacity == 3);
    REQUIRE(empty.getConfig().indexCapacity == 9);

    const Components::Mesh tiny({.vertexCapacity = 5, .indexCapacity = 1});
    REQUIRE(tiny.getConfig().vertexCapacity == 5);
    REQUIRE(tiny.getConfig().indexCapacity == 3);

    const Components::Mesh exact({.vertexCapacity = 64, .indexCapacity = 288});
    REQUIRE(exact.getConfig().indexCapacity == 288);
}

TEST_CASE("Mesh programs draw only while whole triangles are pending and the mesh is enabled",
          "[core][sakura][geometry]") {
    DeviceWindow window;
    if (window.device() == DeviceType::None) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Components::Mesh mesh({.vertexCapacity = 8, .indexCapacity = 12});
    REQUIRE(mesh.create(&window) == Result::SUCCESS);
    Render::Surface::Config surface;
    REQUIRE(mesh.surface(surface) == Result::SUCCESS);
    REQUIRE(surface.programs.size() == 1);
    const auto& program = *surface.programs.front();

    REQUIRE(mesh.positions().size() == 8);
    REQUIRE(mesh.colors().size() == 8);
    REQUIRE(mesh.indices().size() == 12);
    REQUIRE_FALSE(program.enabled());

    REQUIRE(mesh.update(3, 2) == Result::SUCCESS);
    REQUIRE_FALSE(program.enabled());
    REQUIRE(mesh.update(3, 4) == Result::SUCCESS);
    REQUIRE(program.enabled());
    mesh.enabled(false);
    REQUIRE_FALSE(program.enabled());
    REQUIRE(mesh.update(100, 100) == Result::SUCCESS);
    REQUIRE_FALSE(program.enabled());
    mesh.enabled(true);
    REQUIRE(program.enabled());
    REQUIRE(mesh.update(0, 0) == Result::SUCCESS);
    REQUIRE_FALSE(program.enabled());
}

TEST_CASE("Mesh transforms track the framebuffer size, origin, scale, and scissor",
          "[core][sakura][geometry]") {
    DeviceWindow window;
    if (window.device() == DeviceType::None) {
        SKIP("Unbound mesh resources require Vulkan or WebGPU support.");
    }
    Components::Mesh mesh({.framebufferSize = {320.0f, 240.0f}});
    REQUIRE(mesh.create(&window) == Result::SUCCESS);
    Render::Surface::Config surface;
    REQUIRE(mesh.surface(surface) == Result::SUCCESS);
    const auto& program = *surface.programs.front();

    const auto initial = Uniforms(program);
    REQUIRE(std::vector<F32>(initial.begin(), initial.end()) ==
            std::vector<F32>{320.0f, 240.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});

    REQUIRE(mesh.updateTransform({1024.0f, 768.0f}, {12.0f, -4.0f}, 1.5f) == Result::SUCCESS);
    REQUIRE(mesh.getConfig().framebufferSize == Extent2D<F32>{1024.0f, 768.0f});
    const auto moved = Uniforms(program);
    REQUIRE(std::vector<F32>(moved.begin(), moved.end()) ==
            std::vector<F32>{1024.0f, 768.0f, 12.0f, -4.0f, 1.5f, 0.0f, 0.0f, 0.0f});

    REQUIRE(mesh.updateScissorRect(Render::ScissorRect{1, 2, 3, 4}) == Result::SUCCESS);
    REQUIRE(program.getConfig().scissorRect.has_value());
    REQUIRE(program.getConfig().scissorRect->width == 3);
    REQUIRE(mesh.updateScissorRect(std::nullopt) == Result::SUCCESS);
    REQUIRE_FALSE(program.getConfig().scissorRect.has_value());
}

TEST_CASE("Curve components clamp their limits and build a four sided template per segment",
          "[core][sakura][geometry]") {
    const Components::Curves empty({.capacity = 0, .maxSegments = 0});
    REQUIRE(empty.getConfig().capacity == 1);
    REQUIRE(empty.getConfig().maxSegments == 1);

    DeviceWindow window;
    if (window.device() == DeviceType::None) {
        SKIP("Unbound curve resources require Vulkan or WebGPU support.");
    }
    Components::Curves curves({.capacity = 4, .maxSegments = 3});
    REQUIRE(curves.create(&window) == Result::SUCCESS);
    Render::Surface::Config surface;
    REQUIRE(curves.surface(surface) == Result::SUCCESS);
    const auto& program = *surface.programs.front();

    const CurvesView view(program);
    REQUIRE(view.templateVertices.size() == 16);
    REQUIRE(view.templateIndices.size() == 54);
    REQUIRE(view.instances.size() == 4);
    REQUIRE(curves.instances().size() == 4);

    const F32 sides[] = {2.0f, 1.0f, -1.0f, -2.0f};
    for (U64 point = 0; point < 4; ++point) {
        for (U64 side = 0; side < 4; ++side) {
            REQUIRE(view.templateVertices[point * 4 + side] == Extent2D<F32>{static_cast<F32>(point), sides[side]});
        }
    }
    REQUIRE(std::vector<U32>(view.templateIndices.begin() + 18, view.templateIndices.begin() + 36) ==
            std::vector<U32>{9, 5, 6, 6, 10, 9, 9, 5, 4, 4, 8, 9, 10, 6, 7, 7, 11, 10});
    for (const auto index : view.templateIndices) {
        REQUIRE(index < 16);
    }
    for (const auto& instance : view.instances) {
        REQUIRE(instance.color.a == 0.0f);
        REQUIRE(instance.segments == 1.0f);
    }
}

TEST_CASE("Curve components draw only with pending instances and track transform uniforms",
          "[core][sakura][geometry]") {
    DeviceWindow window;
    if (window.device() == DeviceType::None) {
        SKIP("Unbound curve resources require Vulkan or WebGPU support.");
    }
    Components::Curves curves({.capacity = 2, .maxSegments = 4, .framebufferSize = {640.0f, 480.0f}});
    REQUIRE(curves.create(&window) == Result::SUCCESS);
    Render::Surface::Config surface;
    REQUIRE(curves.surface(surface) == Result::SUCCESS);
    const auto& program = *surface.programs.front();

    REQUIRE_FALSE(program.enabled());
    REQUIRE(curves.update(5) == Result::SUCCESS);
    REQUIRE(program.enabled());
    curves.enabled(false);
    REQUIRE_FALSE(program.enabled());
    curves.enabled(true);
    REQUIRE(program.enabled());
    REQUIRE(curves.update(0) == Result::SUCCESS);
    REQUIRE_FALSE(program.enabled());

    const auto initial = Uniforms(program);
    REQUIRE(std::vector<F32>(initial.begin(), initial.end()) ==
            std::vector<F32>{640.0f, 480.0f, 0.0f, 0.0f, 1.0f, 1.0f, 4.0f, 0.0f});
    REQUIRE(curves.updateTransform({800.0f, 600.0f}, {2.0f, 3.0f}, 0.5f, 1.25f) == Result::SUCCESS);
    REQUIRE(curves.getConfig().framebufferSize == Extent2D<F32>{800.0f, 600.0f});
    const auto moved = Uniforms(program);
    REQUIRE(std::vector<F32>(moved.begin(), moved.end()) ==
            std::vector<F32>{800.0f, 600.0f, 2.0f, 3.0f, 0.5f, 1.25f, 4.0f, 0.0f});
}

TEST_CASE("Surface presenters emit each distinct resize once and ignore scale jitter",
          "[core][sakura][geometry]") {
    Sakura::Private::SurfacePresenter presenter;
    SakuraTest::ResizeLog log;
    const auto record = [&](const Sakura::SurfaceResize& resize) { log.record(resize); };

    presenter.beginFrame(0);
    presenter.emitResize(Resize(200, 100), record);
    REQUIRE(log.count() == 1);
    presenter.emitResize(Resize(200, 100), record);
    REQUIRE(log.count() == 1);

    auto jitter = Resize(200, 100);
    jitter.scale += 1e-7f;
    presenter.emitResize(jitter, record);
    REQUIRE(log.count() == 1);

    presenter.emitResize(Resize(200, 100, 2.0f), record);
    REQUIRE(log.count() == 2);
    REQUIRE(log.entries.back().framebufferSize == Extent2D<U64>{400, 200});

    presenter.emitResize(std::nullopt, record);
    REQUIRE(log.count() == 2);

    presenter.emitResize(Resize(200, 100), record);
    REQUIRE(log.count() == 3);
    REQUIRE(log.entries.back() == Resize(200, 100));

    Sakura::Private::SurfacePresenter silent;
    silent.emitResize(Resize(10, 10), {});
    REQUIRE_FALSE(silent.lastResize.has_value());
}

TEST_CASE("Surface presenters re-emit the last resize after a skipped frame or a reset",
          "[core][sakura][geometry]") {
    Sakura::Private::SurfacePresenter presenter;
    SakuraTest::ResizeLog log;
    const auto record = [&](const Sakura::SurfaceResize& resize) { log.record(resize); };

    presenter.beginFrame(5);
    presenter.emitResize(Resize(64, 32), record);
    presenter.beginFrame(5);
    presenter.emitResize(Resize(64, 32), record);
    presenter.beginFrame(6);
    presenter.emitResize(Resize(64, 32), record);
    REQUIRE(log.count() == 1);

    presenter.beginFrame(8);
    presenter.emitResize(Resize(64, 32), record);
    REQUIRE(log.count() == 2);

    presenter.resetResize();
    presenter.beginFrame(9);
    presenter.emitResize(Resize(64, 32), record);
    REQUIRE(log.count() == 3);
}

TEST_CASE("Surface presenters settle once the texture matches the requested framebuffer",
          "[core][sakura][geometry]") {
    Sakura::Private::SurfacePresenter presenter;
    REQUIRE(presenter.settle(nullptr, std::nullopt, 0));
    REQUIRE(presenter.settled);

    const SizedTexture matching({400, 200});
    Sakura::Private::SurfacePresenter matched;
    REQUIRE(matched.settle(&matching, Resize(200, 100, 2.0f), 10));

    const SizedTexture stale({100, 100});
    Sakura::Private::SurfacePresenter waiting;
    REQUIRE_FALSE(waiting.settle(&stale, Resize(200, 100, 2.0f), 10));
    REQUIRE(waiting.settleStart == 10);
    REQUIRE_FALSE(waiting.settle(&stale, Resize(200, 100, 2.0f), 11));
    REQUIRE(waiting.settle(&matching, Resize(200, 100, 2.0f), 12));
    REQUIRE(waiting.settle(&stale, Resize(200, 100, 2.0f), 13));

    waiting.resetSettle();
    REQUIRE_FALSE(waiting.settle(&stale, Resize(200, 100, 2.0f), 14));
    REQUIRE(waiting.settleStart == 14);
}

TEST_CASE("Surface presenters give up waiting after the settle frame budget",
          "[core][sakura][geometry]") {
    const SizedTexture stale({100, 100});
    const int budget = Sakura::Private::SurfacePresenter::kSettleFrames;

    Sakura::Private::SurfacePresenter presenter;
    REQUIRE_FALSE(presenter.settle(&stale, Resize(200, 100), 100));
    REQUIRE_FALSE(presenter.settle(&stale, Resize(200, 100), 100 + budget - 1));
    REQUIRE(presenter.settle(&stale, Resize(200, 100), 100 + budget));

    Sakura::Private::SurfacePresenter unsized;
    REQUIRE_FALSE(unsized.settle(&stale, std::nullopt, 0));
    REQUIRE_FALSE(unsized.settle(&stale, std::nullopt, budget - 1));
    REQUIRE(unsized.settle(&stale, std::nullopt, budget));
}

TEST_CASE("Canvases punch live images only on erase capable backends with an opaque background",
          "[core][sakura][geometry]") {
    REQUIRE(Sakura::UnderlaysAvailable(true, {0.0f, 0.0f, 0.0f, 1.0f}));
    REQUIRE_FALSE(Sakura::UnderlaysAvailable(false, {0.0f, 0.0f, 0.0f, 1.0f}));
    REQUIRE_FALSE(Sakura::UnderlaysAvailable(true, {0.0f, 0.0f, 0.0f, 0.5f}));
}

TEST_CASE("Live image underlays follow the translated image rect and its scissor",
          "[core][sakura][geometry]") {
    const auto texture = std::make_shared<SizedTexture>(Extent2D<U64>{200, 100});
    const auto underlay = Sakura::MakeUnderlay(texture, {30.0f, 40.0f}, {10.0f, 20.0f, 200.0f, 100.0f},
                                                 Render::ScissorRect{40, 60, 150, 80}, {800, 600});
    REQUIRE(underlay.has_value());
    REQUIRE(underlay->texture == texture);
    REQUIRE(underlay->rect == Rect{40.0f, 60.0f, 200.0f, 100.0f});
    REQUIRE(underlay->clip == Rect{40.0f, 60.0f, 150.0f, 80.0f});
    REQUIRE(underlay->frame == Extent2D<U64>{800, 600});

    const Render::ScissorRect full = {0, 0, 10, 10};
    REQUIRE_FALSE(Sakura::MakeUnderlay(nullptr, {}, {0.0f, 0.0f, 10.0f, 10.0f}, full, {10, 10}));
    REQUIRE_FALSE(Sakura::MakeUnderlay(texture, {}, {0.0f, 0.0f, 0.0f, 10.0f}, full, {10, 10}));
    REQUIRE_FALSE(Sakura::MakeUnderlay(texture, {}, {0.0f, 0.0f, 10.0f, 10.0f}, Render::ScissorRect{0, 0, 0, 10},
                                         {10, 10}));
}

TEST_CASE("Underlay geometry and scissor map through the extents the GPU used while a resize is pending",
          "[core][sakura][geometry]") {
    const Rect screen = {0.0f, 0.0f, 200.0f, 100.0f};
    const Extent2D<U64> requested = {200, 100};
    const Extent2D<U64> current = {100, 100};
    const Rect image = Sakura::MapToScreen({40.0f, 0.0f, 100.0f, 100.0f}, screen, requested);
    const Rect scissor = Sakura::MapToScreen({50.0f, 0.0f, 50.0f, 100.0f}, screen, current);
    REQUIRE(image == Rect{40.0f, 0.0f, 100.0f, 100.0f});
    REQUIRE(scissor == Rect{100.0f, 0.0f, 100.0f, 100.0f});
    REQUIRE(Sakura::Intersect(image, scissor) == Rect{100.0f, 0.0f, 40.0f, 100.0f});
}

TEST_CASE("Opaque covers hide a target only inside their rounded body and clip", "[core][sakura][geometry]") {
    const Rect card = {0.0f, 0.0f, 100.0f, 100.0f};
    const ColorRGBA<F32> opaque = {0.1f, 0.1f, 0.1f, 1.0f};
    REQUIRE(Sakura::OpaqueCovers(card, 8.0f, opaque, std::nullopt, {10.0f, 10.0f, 80.0f, 80.0f}));
    REQUIRE_FALSE(Sakura::OpaqueCovers(card, 8.0f, opaque, std::nullopt, {2.0f, 2.0f, 20.0f, 20.0f}));
    REQUIRE_FALSE(Sakura::OpaqueCovers(card, 0.0f, {0.1f, 0.1f, 0.1f, 0.9f}, std::nullopt,
                                         {10.0f, 10.0f, 10.0f, 10.0f}));
    REQUIRE_FALSE(Sakura::OpaqueCovers(card, 0.0f, opaque, Rect{0.0f, 0.0f, 50.0f, 100.0f},
                                         {40.0f, 10.0f, 20.0f, 10.0f}));
    REQUIRE(Sakura::OpaqueCovers(card, 0.0f, opaque, Rect{0.0f, 0.0f, 50.0f, 100.0f}, {10.0f, 10.0f, 20.0f, 10.0f}));
}

TEST_CASE("Transparent bordered boxes hollow out past their stroke and rounded corners",
          "[core][sakura][geometry][occlusion]") {
    const Rect tile = {0.0f, 0.0f, 400.0f, 300.0f};
    const ColorRGBA<F32> clear = {0.0f, 0.0f, 0.0f, 0.0f};
    REQUIRE(Sakura::HollowInterior(tile, 4.0f, 2.0f, 0.0f, clear) == Rect{5.0f, 5.0f, 390.0f, 290.0f});
    REQUIRE(Sakura::HollowInterior(tile, 12.0f, 2.0f, 0.0f, clear) == Rect{12.0f, 12.0f, 376.0f, 276.0f});
    REQUIRE(Sakura::HollowInterior(tile, 4.0f, 0.0f, 0.0f, clear).empty());
    REQUIRE(Sakura::HollowInterior(tile, 4.0f, 2.0f, 6.0f, clear).empty());
    REQUIRE(Sakura::HollowInterior(tile, 4.0f, 2.0f, 0.0f, {0.0f, 0.0f, 0.0f, 0.01f}).empty());
}

TEST_CASE("Opaque interiors exclude rounded corners, translucent strokes, clips, and small boxes",
          "[core][sakura][geometry][occlusion]") {
    const Rect card = {100.0f, 100.0f, 400.0f, 300.0f};
    const ColorRGBA<F32> opaque = {0.2f, 0.2f, 0.2f, 1.0f};
    REQUIRE(Sakura::OpaqueInterior(card, 8.0f, 1.0f, 0.0f, opaque, std::nullopt) ==
            Rect{109.0f, 109.0f, 382.0f, 282.0f});
    REQUIRE(Sakura::OpaqueInterior(card, 0.0f, 3.0f, 0.0f, opaque, std::nullopt) ==
            Rect{104.0f, 104.0f, 392.0f, 292.0f});
    REQUIRE(Sakura::OpaqueInterior(card, 0.0f, 0.0f, 0.0f, opaque, Rect{0.0f, 0.0f, 300.0f, 1000.0f}) ==
            Rect{101.0f, 101.0f, 199.0f, 298.0f});
    REQUIRE(Sakura::OpaqueInterior(card, 0.0f, 0.0f, 0.0f, {0.2f, 0.2f, 0.2f, 0.99f}, std::nullopt).empty());
    REQUIRE(Sakura::OpaqueInterior(card, 0.0f, 0.0f, 4.0f, opaque, std::nullopt).empty());
    REQUIRE(Sakura::OpaqueInterior({0.0f, 0.0f, 40.0f, 40.0f}, 0.0f, 0.0f, 0.0f, opaque, std::nullopt).empty());
}

TEST_CASE("The largest covered region inside a box becomes its hole", "[core][sakura][geometry][occlusion]") {
    const Rect shape = {0.0f, 0.0f, 1000.0f, 800.0f};
    const Rect own = {10.0f, 10.0f, 100.0f, 100.0f};
    const std::vector<Rect> occluders = {
        {500.0f, 400.0f, 1000.0f, 1000.0f},
        {-50.0f, -50.0f, 600.0f, 300.0f},
        {2000.0f, 0.0f, 500.0f, 500.0f},
    };
    REQUIRE(Sakura::LargestHole(shape, own, occluders) == Rect{500.0f, 400.0f, 500.0f, 400.0f});
    REQUIRE(Sakura::LargestHole(shape, own, {}) == own);
    REQUIRE(Sakura::LargestHole(shape, {0.0f, 0.0f, 10.0f, 10.0f}, {{0.0f, 0.0f, 20.0f, 20.0f}}).empty());
}

TEST_CASE("Holes map into the local space of the shape with Y pointing up", "[core][sakura][geometry][occlusion]") {
    const Rect shape = {100.0f, 200.0f, 400.0f, 200.0f};
    REQUIRE(Sakura::LocalHole(shape, shape) == Rect{-0.5f, -0.5f, 1.0f, 1.0f});
    REQUIRE(Sakura::LocalHole(shape, {100.0f, 200.0f, 200.0f, 50.0f}) == Rect{-0.5f, 0.25f, 0.5f, 0.25f});
    REQUIRE(Sakura::LocalHole(shape, {}).empty());
}

TEST_CASE("Box instances past capacity never occlude or cover what lies beneath",
          "[core][sakura][geometry][occlusion]") {
    Rig rig;
    if (!rig.supported()) {
        SKIP("Unbound shape resources require Vulkan or WebGPU support.");
    }
    const ColorRGBA<F32> opaque = {0.2f, 0.2f, 0.2f, 1.0f};
    Sakura::Box box;
    box.update({
        .instances = {
            {.rect = {0.0f, 0.0f, 200.0f, 200.0f}, .backgroundColor = opaque},
            {.rect = {300.0f, 0.0f, 200.0f, 200.0f}, .backgroundColor = opaque},
        },
        .capacity = 1,
    });
    REQUIRE(rig.build(box) == Result::SUCCESS);
    REQUIRE(Sakura::Access::paint(box) == Result::SUCCESS);

    std::vector<Rect> rects;
    rig.drawable().opaqueRects(rects);
    REQUIRE(rects == std::vector<Rect>{{1.0f, 1.0f, 198.0f, 198.0f}});
    CHECK(rig.drawable().covers({10.0f, 10.0f, 50.0f, 50.0f}));
    CHECK_FALSE(rig.drawable().covers({310.0f, 10.0f, 50.0f, 50.0f}));
}

TEST_CASE("Overlapping live images keep punching only when opaque paint separates them",
          "[core][sakura][geometry]") {
    const std::vector<Sakura::LiveRegion> regions = {
        {.order = 2, .rect = {0.0f, 0.0f, 100.0f, 100.0f}},
        {.order = 5, .rect = {50.0f, 50.0f, 100.0f, 100.0f}},
        {.order = 9, .rect = {300.0f, 0.0f, 50.0f, 50.0f}},
    };

    std::vector<std::pair<U64, U64>> asked;
    const auto separated = Sakura::UnderlayEligibility(regions, [&](U64 earlier, U64 later, const Rect& overlap) {
        asked.push_back({earlier, later});
        REQUIRE(overlap == Rect{50.0f, 50.0f, 50.0f, 50.0f});
        return true;
    });
    REQUIRE(separated == std::vector<bool>{true, true, true});
    REQUIRE(asked == std::vector<std::pair<U64, U64>>{{2, 5}});

    const auto exposed = Sakura::UnderlayEligibility(regions, [](U64, U64, const Rect&) { return false; });
    REQUIRE(exposed == std::vector<bool>{true, false, true});

    const std::vector<Sakura::LiveRegion> chain = {
        {.order = 0, .rect = {0.0f, 0.0f, 100.0f, 100.0f}},
        {.order = 1, .rect = {50.0f, 0.0f, 100.0f, 100.0f}},
        {.order = 2, .rect = {120.0f, 0.0f, 100.0f, 100.0f}},
    };
    const auto skipped = Sakura::UnderlayEligibility(chain, [](U64, U64, const Rect&) { return false; });
    REQUIRE(skipped == std::vector<bool>{true, false, true});
}

TEST_CASE("Underlays map from canvas pixels onto the canvas screen rect", "[core][sakura][geometry]") {
    const Rect screen = {100.0f, 50.0f, 400.0f, 300.0f};
    REQUIRE(Sakura::MapToScreen({0.0f, 0.0f, 800.0f, 600.0f}, screen, {800, 600}) == screen);
    REQUIRE(Sakura::MapToScreen({200.0f, 300.0f, 100.0f, 60.0f}, screen, {800, 600}) ==
            Rect{200.0f, 200.0f, 50.0f, 30.0f});
    REQUIRE(Sakura::MapToScreen({200.0f, 300.0f, 100.0f, 60.0f}, screen, {0, 600}).empty());
}
