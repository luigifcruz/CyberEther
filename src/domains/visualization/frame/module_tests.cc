#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "jetstream/domains/visualization/frame/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/scheduler_context.hh"
#include "jetstream/testing.hh"

#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct FrameImplAccess : Modules::FrameImpl {
    static auto viewMember() {
        return &FrameImplAccess::view;
    }
};

struct FrameViewFixture {
    std::shared_ptr<Module> module;
    Modules::FrameImpl* frame = nullptr;

    void create(const Extent2D<U64>& size, const std::string& fit) {
        Tensor input;
        REQUIRE(input.create(DeviceType::CPU, DataType::F32, {size.y, size.x}) ==
                Result::SUCCESS);
        TensorMap inputs;
        inputs["frame"].requested("test", "frame");
        inputs["frame"].tensor = input;

        REQUIRE(Registry::BuildModule("frame", DeviceType::CPU, RuntimeType::NATIVE,
                                      "generic", module) == Result::SUCCESS);
        Modules::Frame config;
        config.fit = fit;
        REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);
        frame = module->getImpl<Modules::FrameImpl>();
        REQUIRE(frame);
        present();
    }

    void present() {
        auto* presenter = module->getImpl<Scheduler::Context>();
        REQUIRE(presenter);
        REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    }

    void zoomTo(const F32 zoom) {
        module->surface()->pushMouseEvent({
            .type = MouseEventType::Scroll,
            .position = {0.5f, 0.5f},
            .scroll = {0.0f, std::log(zoom) / Modules::kFrameZoomSpeed},
        });
        present();
        REQUIRE((frame->*FrameImplAccess::viewMember()).zoom == Catch::Approx(zoom));
    }

    void pan(const Extent2D<F32>& delta) {
        MouseEvent event{};
        event.type = MouseEventType::Click;
        event.button = MouseButton::Left;
        event.position = {0.5f, 0.5f};
        module->surface()->pushMouseEvent(event);
        event.type = MouseEventType::Move;
        event.position = {0.5f + delta.x, 0.5f + delta.y};
        module->surface()->pushMouseEvent(event);
        event.type = MouseEventType::Release;
        module->surface()->pushMouseEvent(event);
        present();
    }

    void requireCenter(const F32 x, const F32 y) {
        const auto& view = frame->*FrameImplAccess::viewMember();
        REQUIRE(view.center.x == Catch::Approx(x).margin(1e-6f));
        REQUIRE(view.center.y == Catch::Approx(y).margin(1e-6f));
    }

    ~FrameViewFixture() {
        if (module) {
            CHECK(module->destroy() == Result::SUCCESS);
        }
    }
};

void RequireFrameValidationError(const Registry::ModuleRegistration& impl,
                                 const DataType dtype,
                                 const Shape& shape,
                                 const bool broadcast = false,
                                 const Modules::Frame& config = {}) {
    Tensor input;
    if (broadcast) {
        REQUIRE(input.create(impl.device, dtype, Shape(shape.size(), 1)) == Result::SUCCESS);
        REQUIRE(input.broadcastTo(shape) == Result::SUCCESS);
    } else {
        REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["frame"].requested("test", "frame");
    inputs["frame"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("frame", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
}

}  // namespace

TEST_CASE("Frame module accepts valid F32 frames", "[modules][frame]") {
    auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("frame", impl.device, impl.runtime, impl.provider);

            Tensor scalar;
            REQUIRE(scalar.create(DeviceType::CPU, DataType::F32, {16, 32}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", scalar);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Modules::Frame config;
            config.colormap = "turbo";
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);

            config.fit = "cover";
            config.smooth = true;
            config.autoRange = false;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor rgb;
            REQUIRE(rgb.create(DeviceType::CPU, DataType::F32, {16, 32, 3}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", rgb);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor scalarChannel;
            REQUIRE(scalarChannel.create(DeviceType::CPU, DataType::F32, {16, 32, 1}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", scalarChannel);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor rgba;
            REQUIRE(rgba.create(DeviceType::CPU, DataType::F32, {16, 32, 4}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", rgba);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Frame module rejects invalid inputs", "[modules][frame][validation]") {
    auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("dtype must be F32") {
                RequireFrameValidationError(impl, DataType::U8, {16, 32});
            }

            SECTION("rank must be two or three") {
                RequireFrameValidationError(impl, DataType::F32, {32});
                RequireFrameValidationError(impl, DataType::F32, {2, 2, 2, 2});
            }

            SECTION("channels must be one, three, or four") {
                RequireFrameValidationError(impl, DataType::F32, {16, 32, 2});
            }

            SECTION("fit must be contain, cover, or stretch") {
                Modules::Frame config;
                config.fit = "tile";
                RequireFrameValidationError(impl, DataType::F32, {16, 32}, false, config);
            }

            SECTION("colormap must be known") {
                Modules::Frame config;
                config.colormap = "rainbow";
                RequireFrameValidationError(impl, DataType::F32, {16, 32}, false, config);
            }

        }
    }
}

TEST_CASE("Frame module rejects unsupported rendering size",
          "[modules][frame][validation][size]") {
    const auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const U64 maxElementCount = std::min({
                static_cast<U64>(std::numeric_limits<I32>::max()),
                static_cast<U64>(std::numeric_limits<std::size_t>::max()) / sizeof(F32),
                static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(F32),
            });
            const Shape shape = {
                1,
                maxElementCount + 1,
            };
            RequireFrameValidationError(impl, DataType::F32, shape, true);
        }
    }
}

TEST_CASE("Frame module supports repeated configurations",
           "[modules][frame][state]") {
    auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("frame", impl.device, impl.runtime, impl.provider);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {8, 8}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", input);

            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);

            Modules::Frame config;
            config.colormap = "turbo";
            REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}

TEST_CASE_METHOD(FrameViewFixture, "Frame Cover panning reaches both image edges",
                 "[modules][frame][interaction]") {
    struct Bounds {
        F32 zoom;
        F32 croppedMin;
        F32 croppedMax;
        F32 fullMin;
        F32 fullMax;
    };
    // A 2:1 image in a square plot crops half of one axis. At each zoom,
    // these centers align the original image edges with the viewport edges.
    const Bounds bounds[] = {
        {1.0f, 0.0f, 1.0f, 0.5f, 0.5f},
        {2.0f, -0.25f, 1.25f, 0.25f, 0.75f},
        {64.0f, -0.4921875f, 1.4921875f, 0.0078125f, 0.9921875f},
    };

    for (const bool wide : {true, false}) {
        for (const auto& bound : bounds) {
            DYNAMIC_SECTION((wide ? "Wide" : "Tall") << " image at " << bound.zoom << "x") {
                create(wide ? Extent2D<U64>{32, 16} : Extent2D<U64>{16, 32}, "cover");
                zoomTo(bound.zoom);
                pan({200.0f, 200.0f});
                requireCenter(wide ? bound.croppedMin : bound.fullMin,
                              wide ? bound.fullMin : bound.croppedMin);
                pan({-200.0f, -200.0f});
                requireCenter(wide ? bound.croppedMax : bound.fullMax,
                              wide ? bound.fullMax : bound.croppedMax);
            }
        }
    }
}

TEST_CASE_METHOD(FrameViewFixture, "Frame Contain and Stretch keep panning within the image",
                 "[modules][frame][interaction]") {
    for (const std::string fit : {"contain", "stretch"}) {
        for (const bool wide : {true, false}) {
            DYNAMIC_SECTION(fit << " with a " << (wide ? "wide" : "tall") << " image") {
                create(wide ? Extent2D<U64>{32, 16} : Extent2D<U64>{16, 32}, fit);
                pan({100.0f, 100.0f});
                requireCenter(0.5f, 0.5f);
                pan({-100.0f, -100.0f});
                requireCenter(0.5f, 0.5f);

                zoomTo(2.0f);
                const bool centeredX = fit == "contain" && !wide;
                const bool centeredY = fit == "contain" && wide;
                pan({100.0f, 100.0f});
                requireCenter(centeredX ? 0.5f : 0.25f, centeredY ? 0.5f : 0.25f);
                pan({-100.0f, -100.0f});
                requireCenter(centeredX ? 0.5f : 0.75f, centeredY ? 0.5f : 0.75f);
            }
        }
    }
}

TEST_CASE_METHOD(FrameViewFixture, "Frame refreshes pan bounds before input after geometry changes",
                 "[modules][frame][interaction]") {
    create({32, 16}, "cover");
    zoomTo(2.0f);
    pan({-100.0f, -100.0f});
    requireCenter(1.25f, 0.75f);

    SECTION("resize reclamps an idle view and updates the next drag") {
        module->surface()->pushSurfaceEvent({
            .type = SurfaceEventType::Resize,
            .size = {1024, 512},
        });
        present();
        requireCenter(0.75f, 0.75f);

        module->surface()->pushSurfaceEvent({
            .type = SurfaceEventType::Resize,
            .size = {512, 1024},
        });
        pan({-100.0f, -100.0f});
        requireCenter(2.25f, 0.75f);
    }

    SECTION("fit changes reclamp an idle view and update the next drag") {
        REQUIRE(module->reconfigure({{"fit", std::string("stretch")}}) == Result::SUCCESS);
        present();
        requireCenter(0.75f, 0.75f);

        REQUIRE(module->reconfigure({{"fit", std::string("contain")}}) == Result::SUCCESS);
        present();
        requireCenter(0.75f, 0.5f);

        REQUIRE(module->reconfigure({{"fit", std::string("cover")}}) == Result::SUCCESS);
        pan({100.0f, 100.0f});
        requireCenter(-0.25f, 0.25f);
    }
}

TEST_CASE_METHOD(FrameViewFixture, "Frame zoom selections use the release position and finish cleanly",
                 "[modules][frame][interaction]") {
    create({16, 16}, "stretch");
    zoomTo(2.0f);
    auto& view = frame->*FrameImplAccess::viewMember();
    MouseEvent event{};
    event.type = MouseEventType::Click;
    event.button = MouseButton::Right;
    event.position = {0.25f, 0.25f};
    module->surface()->pushMouseEvent(event);
    present();
    REQUIRE(view.selecting);

    F32 expectedZoom = 4.0f;
    F32 expectedCenterX = 0.5f;
    SECTION("release inside without a preceding move") {
        event.position = {0.75f, 0.75f};
    }
    SECTION("release outside clamps the box to the plot") {
        event.position = {1.25f, 0.75f};
        expectedZoom = 8.0f / 3.0f;
        expectedCenterX = 0.5625f;
    }
    SECTION("right click resets the view") {
        expectedZoom = 1.0f;
    }
    event.type = MouseEventType::Release;
    module->surface()->pushMouseEvent(event);
    present();
    REQUIRE_FALSE(view.selecting);
    REQUIRE(view.zoom == Catch::Approx(expectedZoom));
    requireCenter(expectedCenterX, 0.5f);

    event.type = MouseEventType::Move;
    event.position = {0.8f, 0.1f};
    module->surface()->pushMouseEvent(event);
    present();
    REQUIRE_FALSE(view.selecting);
    REQUIRE(view.zoom == Catch::Approx(expectedZoom));
    requireCenter(expectedCenterX, 0.5f);
}

TEST_CASE_METHOD(FrameViewFixture, "Frame selections exclude letterbox space in either drag direction",
                 "[modules][frame][interaction]") {
    for (const bool wide : {true, false}) {
        for (const bool reverse : {false, true}) {
            DYNAMIC_SECTION((wide ? "Wide" : "Tall") << " image, reverse drag: " << reverse) {
                create(wide ? Extent2D<U64>{64, 16} : Extent2D<U64>{16, 64}, "contain");
                Extent2D<F32> start = wide ? Extent2D<F32>{0.4f, 0.1f} : Extent2D<F32>{0.1f, 0.4f};
                Extent2D<F32> end = wide ? Extent2D<F32>{0.6f, 0.9f} : Extent2D<F32>{0.9f, 0.6f};
                if (reverse) {
                    std::swap(start, end);
                }
                MouseEvent event{};
                event.type = MouseEventType::Click;
                event.button = MouseButton::Right;
                event.position = start;
                module->surface()->pushMouseEvent(event);
                event.type = MouseEventType::Move;
                event.position = end;
                module->surface()->pushMouseEvent(event);
                present();
                event.type = MouseEventType::Release;
                module->surface()->pushMouseEvent(event);
                present();

                // The image occupies only a quarter of the plot on its short
                // axis. Empty space must not enlarge the selected region.
                const auto& view = frame->*FrameImplAccess::viewMember();
                REQUIRE_FALSE(view.selecting);
                REQUIRE(view.zoom == Catch::Approx(4.0f));
                requireCenter(0.5f, 0.5f);
            }
        }
    }
}

TEST_CASE_METHOD(FrameViewFixture, "Frame selection bounds follow the zoomed and panned image",
                 "[modules][frame][interaction]") {
    for (const bool wide : {true, false}) {
        DYNAMIC_SECTION((wide ? "Wide" : "Tall") << " image") {
            create(wide ? Extent2D<U64>{64, 16} : Extent2D<U64>{16, 64}, "contain");
            zoomTo(2.0f);
            pan({-100.0f, -100.0f});

            MouseEvent event{};
            event.type = MouseEventType::Click;
            event.button = MouseButton::Right;
            event.position = wide ? Extent2D<F32>{0.4f, 0.5f} : Extent2D<F32>{0.5f, 0.4f};
            module->surface()->pushMouseEvent(event);
            event.type = MouseEventType::Release;
            event.position = wide ? Extent2D<F32>{0.6f, 1.2f} : Extent2D<F32>{1.2f, 0.6f};
            module->surface()->pushMouseEvent(event);
            present();

            const auto& view = frame->*FrameImplAccess::viewMember();
            REQUIRE_FALSE(view.selecting);
            REQUIRE(view.zoom == Catch::Approx(8.0f));
            requireCenter(wide ? 0.75f : 0.5625f, wide ? 0.5625f : 0.75f);
        }
    }
}
