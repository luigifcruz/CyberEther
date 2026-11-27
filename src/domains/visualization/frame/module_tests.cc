#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

#include "jetstream/domains/visualization/frame/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/parser.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/scheduler_context.hh"
#include "jetstream/testing.hh"

#include "module_impl.hh"
#include "surface_fixture.hh"

using namespace Jetstream;

namespace {

struct FrameImplAccess : Modules::FrameImpl {
    static auto viewMember() {
        return &FrameImplAccess::view;
    }

    static auto displayRangeMember() {
        return &FrameImplAccess::displayRange;
    }

    static auto renderSurfaceMember() {
        return &FrameImplAccess::renderSurface;
    }

    static auto textMember() {
        return &FrameImplAccess::text;
    }

    static auto boxesMember() {
        return &FrameImplAccess::boxes;
    }

    static auto imageToPlotMember() {
        return &FrameImplAccess::imageToPlot;
    }

    static auto plotToImageMember() {
        return &FrameImplAccess::plotToImage;
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
        REQUIRE(PresentFrame(*presenter) == Result::SUCCESS);
    }

    void zoomTo(const F32 zoom) {
        module->surface()->pushInputEvent(MouseEvent{
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
        module->surface()->pushInputEvent(event);
        event.type = MouseEventType::Move;
        event.position = {0.5f + delta.x, 0.5f + delta.y};
        module->surface()->pushInputEvent(event);
        event.type = MouseEventType::Release;
        module->surface()->pushInputEvent(event);
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

TEST_CASE("Frame presents captured pixels with their captured auto range", "[modules][frame]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2, 2}) == Result::SUCCESS);
    TensorMap inputs;
    inputs["frame"].requested("test", "frame");
    inputs["frame"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("frame", DeviceType::CPU, RuntimeType::NATIVE,
                                  "generic", module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::Frame{}, inputs) == Result::SUCCESS);
    auto* frame = module->getImpl<Modules::FrameImpl>();
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(frame);
    REQUIRE(presenter);

    Runtime runtime("test", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
    std::unordered_set<std::string> skipped;
    std::unordered_set<std::string> stalled;
    std::unordered_set<std::string> failed;
    const auto compute = [&](const std::array<F32, 4>& values) {
        std::copy(values.begin(), values.end(), input.data<F32>());
        REQUIRE(runtime.compute({}, skipped, stalled, failed) == Result::SUCCESS);
        REQUIRE(failed.empty());
    };
    const auto range = [&] {
        return (frame->*FrameImplAccess::displayRangeMember())();
    };

    compute({0.25f, 0.5f, 0.75f, 1.0f});
    REQUIRE(presenter->presentCapture() == Result::SUCCESS);
    compute({-4.0f, 0.0f, 4.0f, 8.0f});
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(range() == std::pair<F32, F32>{0.25f, 1.0f});

    REQUIRE(presenter->presentCapture() == Result::SUCCESS);
    REQUIRE(presenter->presentSubmit() == Result::SUCCESS);
    REQUIRE(range() == std::pair<F32, F32>{-4.0f, 8.0f});

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
TEST_CASE("Frame redraws its retained surface only on input and new data",
          "[modules][frame][present][retained]") {
    Tensor input(DeviceType::CPU, DataType::F32, {4, 4});
    TensorMap inputs;
    inputs["frame"].tensor = input;
    auto window = std::make_shared<SurfaceTestWindow>();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("frame", DeviceType::CPU, RuntimeType::NATIVE,
                                  "generic", module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::Frame{}, inputs, window) == Result::SUCCESS);
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(presenter);
    REQUIRE(presenter->presentInitialize() == Result::SUCCESS);
    Runtime runtime("test", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);

    auto* frame = module->getImpl<Modules::FrameImpl>();
    REQUIRE(frame);
    auto& surface = *(frame->*FrameImplAccess::renderSurfaceMember());

    std::unordered_set<std::string> skipped, stalled, failed;
    F32 level = 0.1f;
    const auto compute = [&] {
        std::fill_n(input.data<F32>(), input.size(), level);
        level += 0.1f;
        REQUIRE(runtime.compute({}, skipped, stalled, failed) == Result::SUCCESS);
    };

    compute();
    REQUIRE(PresentFrame(*presenter) == Result::SUCCESS);
    SurfaceProbe::consumeInvalidation(surface);
    REQUIRE_FALSE(PresentRedraws(*presenter, surface));

    compute();
    REQUIRE(PresentRedraws(*presenter, surface));
    REQUIRE_FALSE(PresentRedraws(*presenter, surface));

    module->surface()->pushSurfaceEvent({.type = SurfaceEventType::Resize, .size = {640, 480}});
    REQUIRE(PresentRedraws(*presenter, surface));
    REQUIRE_FALSE(PresentRedraws(*presenter, surface));

    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Scroll, .position = {0.5f, 0.5f}, .scroll = {0.0f, 1.0f},
    });
    REQUIRE(PresentRedraws(*presenter, surface));
    REQUIRE_FALSE(PresentRedraws(*presenter, surface));

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Frame header shows only the zoom while attached and hides the readout",
          "[modules][frame][present][attached]") {
    Tensor input(DeviceType::CPU, DataType::F32, {4, 4});
    TensorMap inputs;
    inputs["frame"].tensor = input;
    auto window = std::make_shared<SurfaceTestWindow>();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("frame", DeviceType::CPU, RuntimeType::NATIVE,
                                  "generic", module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::Frame{}, inputs, window) == Result::SUCCESS);
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(presenter);
    REQUIRE(presenter->presentInitialize() == Result::SUCCESS);
    Runtime runtime("test", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);

    auto* frame = module->getImpl<Modules::FrameImpl>();
    REQUIRE(frame);
    const auto& text = *(frame->*FrameImplAccess::textMember());

    std::unordered_set<std::string> skipped, stalled, failed;
    std::fill_n(input.data<F32>(), input.size(), 0.5f);
    REQUIRE(runtime.compute({}, skipped, stalled, failed) == Result::SUCCESS);
    module->surface()->pushInputEvent(MouseEvent{.type = MouseEventType::Move, .position = {0.5f, 0.5f}});
    REQUIRE(PresentFrame(*presenter) == Result::SUCCESS);
    const std::string frameLabel = text.get("header").fill;
    REQUIRE(frameLabel.starts_with("4 x 4"));
    REQUIRE(text.get("readout").fill.starts_with("X "));

    module->surface()->pushSurfaceEvent({.type = SurfaceEventType::Resize,
                                         .size = {512, 512},
                                         .placement = SurfacePlacementType::Attached});
    REQUIRE(PresentFrame(*presenter) == Result::SUCCESS);
    REQUIRE(text.get("header").fill == " ");
    REQUIRE(text.get("readout").fill == " ");

    module->surface()->pushInputEvent(MouseEvent{
        .type = MouseEventType::Scroll,
        .position = {0.5f, 0.5f},
        .scroll = {0.0f, std::log(2.0f) / Modules::kFrameZoomSpeed},
    });
    REQUIRE(PresentFrame(*presenter) == Result::SUCCESS);
    REQUIRE(text.get("header").fill == "ZOOM 2.0x");
    REQUIRE(text.get("readout").fill == " ");

    module->surface()->pushSurfaceEvent({.type = SurfaceEventType::Resize,
                                         .size = {512, 512},
                                         .placement = SurfacePlacementType::Detached});
    REQUIRE(PresentFrame(*presenter) == Result::SUCCESS);
    REQUIRE(text.get("header").fill == frameLabel + "   ZOOM 2.0x");
    REQUIRE(text.get("readout").fill.starts_with("X "));

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}
#endif

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
            config.interpolation = "bilinear";
            config.autoRange = false;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);

            config.interpolation = "bicubic";
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
    module->surface()->pushInputEvent(event);
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
    module->surface()->pushInputEvent(event);
    present();
    REQUIRE_FALSE(view.selecting);
    REQUIRE(view.zoom == Catch::Approx(expectedZoom));
    requireCenter(expectedCenterX, 0.5f);

    event.type = MouseEventType::Move;
    event.position = {0.8f, 0.1f};
    module->surface()->pushInputEvent(event);
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
                module->surface()->pushInputEvent(event);
                event.type = MouseEventType::Move;
                event.position = end;
                module->surface()->pushInputEvent(event);
                present();
                event.type = MouseEventType::Release;
                module->surface()->pushInputEvent(event);
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
            module->surface()->pushInputEvent(event);
            event.type = MouseEventType::Release;
            event.position = wide ? Extent2D<F32>{0.6f, 1.2f} : Extent2D<F32>{1.2f, 0.6f};
            module->surface()->pushInputEvent(event);
            present();

            const auto& view = frame->*FrameImplAccess::viewMember();
            REQUIRE_FALSE(view.selecting);
            REQUIRE(view.zoom == Catch::Approx(8.0f));
            requireCenter(wide ? 0.75f : 0.5625f, wide ? 0.5625f : 0.75f);
        }
    }
}

namespace {

Parser::Map BoxEntry(const std::any& x,
                     const std::any& y,
                     const std::any& width,
                     const std::any& height) {
    Parser::Map entry;
    entry["x"] = x;
    entry["y"] = y;
    entry["width"] = width;
    entry["height"] = height;
    return entry;
}

}  // namespace

TEST_CASE("Frame parses box attributes and skips malformed entries", "[modules][frame][boxes]") {
    U64 rejected = 0;

    SECTION("Missing attribute and empty list draw nothing") {
        REQUIRE(Modules::ParseFrameBoxes(std::any{}, rejected).empty());
        REQUIRE(rejected == 0);
        REQUIRE(Modules::ParseFrameBoxes(Parser::Sequence{}, rejected).empty());
        REQUIRE(rejected == 0);
    }

    SECTION("Numbers of any width, labels, and colors") {
        auto first = BoxEntry(static_cast<U64>(10), static_cast<I64>(-4), 2.5,
                              static_cast<F32>(8.0f));
        first["label"] = std::string("pulse");
        first["color"] = std::vector<F64>{1.0, 0.5, 2.0};
        auto second = BoxEntry(0.0, 0.0, 1.0, 1.0);
        second["color"] = std::string("#FF000080");
        auto third = BoxEntry(1.0, 2.0, 3.0, 4.0);
        third["color"] = Parser::Sequence{static_cast<U64>(0), 1.0, static_cast<U64>(0), 0.25};

        const auto boxes = Modules::ParseFrameBoxes(Parser::Sequence{first, second, third},
                                                    rejected);
        REQUIRE(rejected == 0);
        REQUIRE(boxes.size() == 3);
        REQUIRE(boxes[0].x == 10.0f);
        REQUIRE(boxes[0].y == -4.0f);
        REQUIRE(boxes[0].width == 2.5f);
        REQUIRE(boxes[0].height == 8.0f);
        REQUIRE(boxes[0].label == "pulse");
        REQUIRE(boxes[0].color == ColorRGBA<F32>{1.0f, 0.5f, 1.0f, 1.0f});
        REQUIRE(boxes[1].label.empty());
        REQUIRE(boxes[1].color.r == 1.0f);
        REQUIRE(boxes[1].color.g == 0.0f);
        REQUIRE(boxes[1].color.a == Catch::Approx(128.0f / 255.0f));
        REQUIRE(boxes[2].color == ColorRGBA<F32>{0.0f, 1.0f, 0.0f, 0.25f});
    }

    SECTION("A single map is one box") {
        const auto boxes = Modules::ParseFrameBoxes(BoxEntry(1.0, 2.0, 3.0, 4.0), rejected);
        REQUIRE(rejected == 0);
        REQUIRE(boxes.size() == 1);
    }

    SECTION("Malformed entries are rejected") {
        auto missing = BoxEntry(1.0, 2.0, 3.0, 4.0);
        missing.erase("height");
        auto negative = BoxEntry(1.0, 2.0, -3.0, 4.0);
        auto nonFinite = BoxEntry(std::numeric_limits<F64>::quiet_NaN(), 2.0, 3.0, 4.0);
        auto text = BoxEntry(std::string("1"), 2.0, 3.0, 4.0);
        auto badColor = BoxEntry(1.0, 2.0, 3.0, 4.0);
        badColor["color"] = std::vector<F64>{1.0, 0.0};
        auto badLabel = BoxEntry(1.0, 2.0, 3.0, 4.0);
        badLabel["label"] = 3.0;

        const auto boxes = Modules::ParseFrameBoxes(Parser::Sequence{
            missing, negative, nonFinite, text, badColor, badLabel, 7.0,
            BoxEntry(0.0, 0.0, 1.0, 1.0),
        }, rejected);
        REQUIRE(rejected == 7);
        REQUIRE(boxes.size() == 1);
        REQUIRE(Modules::ParseFrameBoxes(std::string("box"), rejected).empty());
        REQUIRE(rejected == 1);
    }

    SECTION("Labels are truncated and the box count is capped") {
        auto entry = BoxEntry(0.0, 0.0, 1.0, 1.0);
        entry["label"] = std::string(200, 'a');
        Parser::Sequence many(Modules::kFrameMaxBoxes + 3, entry);
        const auto boxes = Modules::ParseFrameBoxes(many, rejected);
        REQUIRE(boxes.size() == Modules::kFrameMaxBoxes);
        REQUIRE(rejected == 3);
        REQUIRE(boxes[0].label.size() == Modules::kFrameMaxBoxLabelCharacters);
    }
}

TEST_CASE("Frame publishes boxes from the input attribute with the captured frame",
          "[modules][frame][boxes]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {8, 16}) == Result::SUCCESS);
    TensorMap inputs;
    inputs["frame"].requested("test", "frame");
    inputs["frame"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("frame", DeviceType::CPU, RuntimeType::NATIVE,
                                  "generic", module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::Frame{}, inputs) == Result::SUCCESS);
    auto* frame = module->getImpl<Modules::FrameImpl>();
    auto* presenter = module->getImpl<Scheduler::Context>();
    REQUIRE(frame);
    REQUIRE(presenter);
    const auto& boxes = frame->*FrameImplAccess::boxesMember();

    Runtime runtime("test", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
    std::unordered_set<std::string> skipped, stalled, failed;
    const auto compute = [&] {
        REQUIRE(runtime.compute({}, skipped, stalled, failed) == Result::SUCCESS);
        REQUIRE(failed.empty());
    };
    const std::string key(Modules::kFrameBoxesAttribute);

    auto entry = BoxEntry(2.0, 3.0, 4.0, 2.0);
    entry["label"] = std::string("carrier");
    REQUIRE(input.setAttribute(key, Parser::Sequence{entry}) == Result::SUCCESS);
    compute();
    REQUIRE(boxes.empty());
    REQUIRE(presenter->presentCapture() == Result::SUCCESS);
    REQUIRE(boxes.size() == 1);
    REQUIRE(boxes[0].x == 2.0f);
    REQUIRE(boxes[0].label == "carrier");

    REQUIRE(input.setAttribute(key, Parser::Sequence{}) == Result::SUCCESS);
    compute();
    REQUIRE(presenter->presentCapture() == Result::SUCCESS);
    REQUIRE(boxes.empty());

    REQUIRE(input.setAttribute(key, Parser::Sequence{entry, entry}) == Result::SUCCESS);
    compute();
    REQUIRE(presenter->presentCapture() == Result::SUCCESS);
    REQUIRE(boxes.size() == 2);

    REQUIRE(input.removeAttribute(key) == Result::SUCCESS);
    compute();
    REQUIRE(presenter->presentCapture() == Result::SUCCESS);
    REQUIRE(boxes.empty());

    Parser::Map edit;
    edit["showBoxes"] = false;
    REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
    REQUIRE(input.setAttribute(key, Parser::Sequence{entry}) == Result::SUCCESS);
    compute();
    REQUIRE(presenter->presentCapture() == Result::SUCCESS);
    REQUIRE(boxes.empty());

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE_METHOD(FrameViewFixture, "Frame box placement inverts the zoomed and panned view",
                 "[modules][frame][boxes][interaction]") {
    for (const auto* fit : {"contain", "cover", "stretch"}) {
        DYNAMIC_SECTION("Fit " << fit) {
            create({64, 16}, fit);
            zoomTo(3.0f);
            pan({0.1f, -0.05f});

            for (const auto& image : {Extent2D<F32>{0.0f, 0.0f}, Extent2D<F32>{0.3f, 0.7f},
                                      Extent2D<F32>{1.0f, 1.0f}}) {
                DYNAMIC_SECTION("Image " << image.x << ", " << image.y) {
                    const auto plot = (frame->*FrameImplAccess::imageToPlotMember())(image);
                    const auto back = (frame->*FrameImplAccess::plotToImageMember())(plot);
                    REQUIRE(back.x == Catch::Approx(image.x).margin(1e-5f));
                    REQUIRE(back.y == Catch::Approx(image.y).margin(1e-5f));
                }
            }
        }
    }
}
