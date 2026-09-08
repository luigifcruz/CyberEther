#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <any>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <string>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/instance.hh"
#include "jetstream/scheduler_context.hh"
#include "jetstream/domains/dsp/adsb/module.hh"
#include "jetstream/render/base/program.hh"

#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct AdsbImplAccess : Modules::AdsbImpl {
    static auto inputMember() {
        return &AdsbImplAccess::input;
    }

    static auto aircraftMapMember() { return &AdsbImplAccess::aircraftMap; }
    static auto aircraftTableDirtyMember() { return &AdsbImplAccess::aircraftTableDirty; }
    static auto mapMember() { return &AdsbImplAccess::geoMapComponent; }
    static auto mapStateMember() { return &AdsbImplAccess::mapState; }
    static auto mapAutoFitMember() { return &AdsbImplAccess::mapAutoFit; }
};

TensorMap AdsbInput(const Tensor& input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;
    return inputs;
}

void RequireAdsbValidationError(const Registry::ModuleRegistration& impl,
                                const Tensor& input) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("adsb", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);

    Result result = Result::SUCCESS;
    REQUIRE_NOTHROW(result = module->create("test", Modules::Adsb{},
                                            AdsbInput(input)));
    REQUIRE(result == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());

    const auto* adsb = module->getImpl<Modules::AdsbImpl>();
    REQUIRE(adsb != nullptr);
    REQUIRE((adsb->*AdsbImplAccess::inputMember()).empty());
    REQUIRE((adsb->*AdsbImplAccess::aircraftMapMember()).empty());
    REQUIRE(adsb->getAircraftTable() == "No aircraft detected.");
}

}  // namespace

TEST_CASE("ADS-B startup fit waits for positions and frames the collected aircraft only once",
          "[modules][adsb][render][fit]") {
    Modules::AdsbMapAutoFit autoFit;
    Render::Components::MapContext context;
    REQUIRE_FALSE(autoFit.update({}, context, {}, 0).has_value());
    std::vector<Modules::AdsbMapAircraft> aircraft = {{.latitude = 37, .longitude = -124}};
    constexpr U64 start = 10000;
    constexpr U64 delay = Modules::AdsbMapAutoFit::CollectionMilliseconds;
    REQUIRE_FALSE(autoFit.update(aircraft, context, {}, start).has_value());
    aircraft.push_back({.latitude = 39, .longitude = -120});
    const std::array<MouseEvent, 2> passiveEvents = {{
        {.type = MouseEventType::Move, .position = {0.5f, 0.5f}},
        {.type = MouseEventType::Click, .button = MouseButton::Right, .position = {0.5f, 0.5f}},
    }};
    REQUIRE_FALSE(autoFit.update(aircraft, context, passiveEvents, start + delay - 1).has_value());
    const auto fitted = autoFit.update(aircraft, context, {}, start + delay);
    REQUIRE(fitted.has_value());
    REQUIRE(fitted->centerLon == Catch::Approx(-122));
    REQUIRE(fitted->centerLat == Catch::Approx(38));
    REQUIRE(fitted->zoom > 0);
    REQUIRE(fitted->zoom <= 5);
    REQUIRE(context.update(*fitted) == Result::SUCCESS);
    for (const auto& ac : aircraft) {
        F32 x, y;
        REQUIRE(context.projectLonLat(ac.longitude, ac.latitude, x, y));
        REQUIRE(std::abs(x) <= 0.8001f);
        REQUIRE(std::abs(y) <= 0.8001f);
    }
    aircraft.push_back({.latitude = 0, .longitude = 0});
    REQUIRE_FALSE(autoFit.update(aircraft, context, {}, start + 2 * delay).has_value());
    // The policy survives a recreated map and a new stream of fixes.
    Render::Components::MapContext recreated;
    REQUIRE(recreated.update(*fitted) == Result::SUCCESS);
    REQUIRE_FALSE(autoFit.update({}, recreated, {}, start + 3 * delay).has_value());
    REQUIRE_FALSE(autoFit.update(aircraft, recreated, {}, start + 4 * delay).has_value());
    REQUIRE(recreated.view == *fitted);
}

TEST_CASE("ADS-B startup fit ignores expired and invalid positions and restarts an empty collection",
          "[modules][adsb][render][fit]") {
    Modules::AdsbMapAutoFit autoFit;
    Render::Components::MapContext context;
    const F32 nan = std::numeric_limits<F32>::quiet_NaN();
    std::vector<Modules::AdsbMapAircraft> aircraft = {
        {.latitude = 0, .longitude = 0, .positionAgeSeconds = 121},
        {.latitude = nan, .longitude = 0}, {.latitude = 91, .longitude = 0},
        {.latitude = 0, .longitude = nan}, {.positionAgeSeconds = nan},
    };
    constexpr U64 delay = Modules::AdsbMapAutoFit::CollectionMilliseconds;
    REQUIRE_FALSE(autoFit.update(aircraft, context, {}, 0).has_value());
    REQUIRE_FALSE(autoFit.update(aircraft, context, {}, delay).has_value());
    aircraft.push_back({.latitude = 38, .longitude = -122, .positionAgeSeconds = 0});
    REQUIRE_FALSE(autoFit.update(aircraft, context, {}, 2 * delay).has_value());
    REQUIRE_FALSE(autoFit.update({}, context, {}, 3 * delay).has_value());
    REQUIRE_FALSE(autoFit.update(aircraft, context, {}, 4 * delay).has_value());
    const auto fitted = autoFit.update(aircraft, context, {}, 5 * delay);
    REQUIRE(fitted.has_value());
    REQUIRE(fitted->centerLon == Catch::Approx(-122));
    REQUIRE(fitted->centerLat == Catch::Approx(38));
    REQUIRE(fitted->zoom == 5);
}

TEST_CASE("ADS-B startup fit yields to map navigation before or during collection",
          "[modules][adsb][render][fit]") {
    const std::array<Modules::AdsbMapAircraft, 1> aircraft{{{.latitude = 38, .longitude = -122}}};
    const std::array<MouseEvent, 2> interactions = {{
        {.type = MouseEventType::Click, .button = MouseButton::Left, .position = {0.5f, 0.5f}},
        {.type = MouseEventType::Scroll, .position = {0.5f, 0.5f}, .scroll = {0, 1}},
    }};
    constexpr U64 delay = Modules::AdsbMapAutoFit::CollectionMilliseconds;
    for (const auto& event : interactions) {
        for (const bool collecting : {false, true}) {
            Modules::AdsbMapAutoFit autoFit;
            Render::Components::MapContext context;
            if (collecting) REQUIRE_FALSE(autoFit.update(aircraft, context, {}, 0).has_value());
            REQUIRE_FALSE(autoFit.update({}, context, {&event, 1}, delay).has_value());
            REQUIRE_FALSE(autoFit.update(aircraft, context, {}, 2 * delay).has_value());
            REQUIRE_FALSE(autoFit.update(aircraft, context, {}, 3 * delay).has_value());
        }
    }
}

TEST_CASE("ADS-B radar history uses bounded geographic fixes across the antimeridian",
          "[modules][adsb][render]") {
    Modules::AdsbMapAircraft ac;
    ac.longitude = -170;
    ac.track = {{0, 170}, {0, -176}, {0, -173}, {0, -170}};
    Render::Components::MapContext context;
    auto view = context.view;
    view.centerLon = 180;
    view.zoom = 2;
    REQUIRE(context.update(view) == Result::SUCCESS);
    auto dots = Modules::AdsbHistoryDots(ac, context);
    REQUIRE(dots.size() == 3);
    F32 x, y;
    REQUIRE(context.projectLonLat(-173, 0, x, y));
    REQUIRE(dots[0].x == x);
    REQUIRE(dots[0].y == y);
    const auto pixel = context.pixelSize();
    for (U64 i = 1; i < dots.size(); ++i) {
        REQUIRE(std::hypot((dots[i].x - dots[i - 1].x) / pixel.x,
                           (dots[i].y - dots[i - 1].y) / pixel.y) >= 7 * Modules::AdsbTrackingScale);
    }
    ac.track.clear();
    REQUIRE(Modules::AdsbHistoryDots(ac, context).empty());
    for (int i = 0; i < 50; ++i) ac.track.emplace_back(0, -180 + i * 0.2);
    dots = Modules::AdsbHistoryDots(ac, context);
    REQUIRE_FALSE(dots.empty());
    REQUIRE(dots.size() <= 6);
    ac.positionAgeSeconds = 121;
    REQUIRE(Modules::AdsbHistoryDots(ac, context).empty());
}

TEST_CASE("ADS-B radar projection uses current camera and hides back-side aircraft",
          "[modules][adsb][render]") {
    Render::Components::MapContext context;
    const std::vector<Modules::AdsbMapAircraft> aircraft = {
        {.icao = 1, .longitude = 180}, {.icao = 2, .longitude = 0},
    };
    auto targets = Modules::AdsbVisibleTargets(aircraft, context);
    REQUIRE(targets.size() == 1);
    REQUIRE(aircraft[targets[0].index].icao == 2);
    auto view = context.view;
    view.centerLon = 180;
    REQUIRE(context.update(view) == Result::SUCCESS);
    targets = Modules::AdsbVisibleTargets(aircraft, context);
    REQUIRE(targets.size() == 1);
    REQUIRE(aircraft[targets[0].index].icao == 1);
    REQUIRE(Modules::AdsbVisibleTargets({}, context).empty());
}

TEST_CASE("ADS-B radar headings align with geographic motion through the globe camera",
          "[modules][adsb][render][heading]") {
    struct ViewCase {
        Extent2D<F32> center;
        Extent2D<F32> position;
        F32 zoom;
    };
    const std::array<ViewCase, 5> views = {{
        {{0, 60}, {60, 60}, 0},
        {{0, -60}, {-60, -60}, 0},
        {{170, 60}, {-179, 65}, 0},
        {{179, 85}, {-179, 85}, 2},
        {{-122, 38}, {-122, 38}, Render::Components::MapContext::MaxZoom},
    }};
    for (const auto& viewCase : views) {
        for (const auto size : {Extent2D<F32>{800, 600}, Extent2D<F32>{600, 800}}) {
            for (const F32 scale : {1.0f, 2.0f}) {
                Render::Components::MapContext context;
                auto view = context.view;
                view.centerLon = viewCase.center.x;
                view.centerLat = viewCase.center.y;
                view.zoom = viewCase.zoom;
                view.viewportWidth = size.x;
                view.viewportHeight = size.y;
                view.aspectRatio = size.x / size.y;
                view.surfaceScale = scale;
                REQUIRE(context.update(view) == Result::SUCCESS);

                Modules::AdsbMapAircraft ac;
                ac.longitude = viewCase.position.x;
                ac.latitude = viewCase.position.y;
                ac.hasVelocity = true;
                for (const F32 heading : {0.0f, 45.0f, 90.0f, 180.0f, 270.0f}) {
                    CAPTURE(viewCase.center, viewCase.position, viewCase.zoom, size, scale, heading);
                    ac.heading = heading;
                    const auto targets = Modules::AdsbVisibleTargets({&ac, 1}, context);
                    REQUIRE(targets.size() == 1);
                    REQUIRE(targets.front().heading.has_value());

                    // Independent oracle: project nearby geographic fixes on a
                    // great-circle course, instead of differentiating the camera.
                    const F64 lon = ac.longitude * JST_PI / 180.0;
                    const F64 lat = ac.latitude * JST_PI / 180.0;
                    const F64 bearing = heading * JST_PI / 180.0;
                    auto projectFix = [&](F64 distance) {
                        const F64 nextLat = std::asin(std::sin(lat) * std::cos(distance) +
                            std::cos(lat) * std::sin(distance) * std::cos(bearing));
                        const F64 nextLon = lon + std::atan2(
                            std::sin(bearing) * std::sin(distance) * std::cos(lat),
                            std::cos(distance) - std::sin(lat) * std::sin(nextLat));
                        Extent2D<F32> ndc;
                        REQUIRE(context.projectLonLat(
                            static_cast<F32>(nextLon * 180.0 / JST_PI),
                            static_cast<F32>(nextLat * 180.0 / JST_PI), ndc.x, ndc.y));
                        return ndc;
                    };
                    const auto before = projectFix(-1e-4);
                    const auto after = projectFix(1e-4);
                    const auto pixel = context.pixelSize();
                    const F32 x = (after.x - before.x) / pixel.x;
                    const F32 y = (after.y - before.y) / pixel.y;
                    const F32 length = std::hypot(x, y);
                    REQUIRE(length > 0);
                    const F32 projected = *targets.front().heading;
                    REQUIRE(std::sin(projected) == Catch::Approx(x / length).margin(0.003f));
                    REQUIRE(std::cos(projected) == Catch::Approx(y / length).margin(0.003f));
                }
            }
        }
    }
}

TEST_CASE("ADS-B radar targets have no directional heading without valid velocity",
          "[modules][adsb][render][heading]") {
    Render::Components::MapContext context;
    Modules::AdsbMapAircraft ac;
    auto targets = Modules::AdsbVisibleTargets({&ac, 1}, context);
    REQUIRE(targets.size() == 1);
    REQUIRE_FALSE(targets.front().heading.has_value());

    ac.hasVelocity = true;
    for (const F32 heading : {std::numeric_limits<F32>::quiet_NaN(),
                              std::numeric_limits<F32>::infinity()}) {
        ac.heading = heading;
        targets = Modules::AdsbVisibleTargets({&ac, 1}, context);
        REQUIRE(targets.size() == 1);
        REQUIRE_FALSE(targets.front().heading.has_value());
    }
}

TEST_CASE("ADS-B radar data blocks distinguish reported and unavailable fields", "[modules][adsb][render]") {
    Modules::AdsbMapAircraft ac;
    ac.icao = 0xABC123;
    ac.callsign = "ual123";
    ac.altitude = 8000;
    ac.groundSpeed = 240;
    ac.positionAgeSeconds = 0;
    REQUIRE(Modules::FormatAdsbDataBlock(ac) == "UAL123\n--- --");
    ac.hasAltitude = ac.hasVelocity = true;
    REQUIRE(Modules::FormatAdsbDataBlock(ac) == "UAL123\n080 24");
    ac.altitude = 0;
    ac.groundSpeed = 0;
    REQUIRE(Modules::FormatAdsbDataBlock(ac) == "UAL123\n000 00");
    ac.altitude = std::numeric_limits<F32>::quiet_NaN();
    ac.groundSpeed = std::numeric_limits<F32>::infinity();
    REQUIRE(Modules::FormatAdsbDataBlock(ac) == "UAL123\n--- --");
    ac.positionAgeSeconds = 15;
    REQUIRE(Modules::FormatAdsbDataBlock(ac).find("STALE 15s") != std::string::npos);
    REQUIRE(Modules::AdsbTargetVisible(ac));
    ac.positionAgeSeconds = 121;
    REQUIRE_FALSE(Modules::AdsbTargetVisible(ac));
    ac.positionAgeSeconds.reset();
    ac.callsign.clear();
    REQUIRE(Modules::FormatAdsbDataBlock(ac) == "ICAO ABC123\n--- --\nAGE ?");
    ac.callsign = "AB\n123";
    REQUIRE(Modules::FormatAdsbDataBlock(ac) == "ICAO ABC123\n--- --\nAGE ?");
}

TEST_CASE("ADS-B old or hidden records cannot crowd live targets out of the display", "[modules][adsb][render]") {
    std::vector<Modules::AdsbMapAircraft> aircraft(Modules::AdsbMapState::MaxAircraft + 1);
    for (auto& ac : aircraft) ac.positionAgeSeconds = 121;
    aircraft.back().positionAgeSeconds = 0;
    Render::Components::MapContext context;
    const auto targets = Modules::AdsbVisibleTargets(aircraft, context);
    REQUIRE(targets.size() == 1);
    REQUIRE(targets[0].index == Modules::AdsbMapState::MaxAircraft);
    for (auto& ac : aircraft) ac.positionAgeSeconds = 0;
    REQUIRE(Modules::AdsbVisibleTargets(aircraft, context).size() == Modules::AdsbMapState::MaxAircraft);
}

TEST_CASE("ADS-B data blocks avoid occupied labels and stay inside the viewport", "[modules][adsb][render]") {
    const auto first = Modules::PlaceAdsbDataBlock({150, 150}, {80, 36}, {300, 300}, {});
    REQUIRE(first.box.x == 150 + 24 * Modules::AdsbTrackingScale);
    REQUIRE(first.box.y == 150 - 24 * Modules::AdsbTrackingScale - 36);
    const std::array<Modules::AdsbLabelBox, 1> occupied{first.box};
    const auto second = Modules::PlaceAdsbDataBlock({150, 150}, {80, 36}, {300, 300}, occupied);
    REQUIRE(second.direction != first.direction);
    REQUIRE((second.box.y >= first.box.y + first.box.height ||
             second.box.x >= first.box.x + first.box.width ||
             second.box.y + second.box.height <= first.box.y ||
             second.box.x + second.box.width <= first.box.x));
    const auto stable = Modules::PlaceAdsbDataBlock({150, 150}, {80, 36}, {300, 300}, {}, 5);
    REQUIRE(stable.direction == 5);
    for (const auto anchor : {Extent2D<F32>{0, 0}, Extent2D<F32>{300, 300}}) {
        const auto placement = Modules::PlaceAdsbDataBlock(anchor, {80, 36}, {300, 300}, {});
        REQUIRE(placement.box.x >= 6);
        REQUIRE(placement.box.y >= 6);
        REQUIRE(placement.box.x + placement.box.width <= 294);
        REQUIRE(placement.box.y + placement.box.height <= 294);
    }
}

// Opt-in because it creates a real graphics window. Run with the [.gpu] filter
// on a desktop with Metal/Vulkan support; ordinary CI remains display-free.
TEST_CASE("ADS-B layered surface presents, resizes, clears and recreates",
          "[.gpu][modules][adsb][render]") {
#if defined(JETSTREAM_VIEWPORT_GLFW_AVAILABLE) && \
    (defined(JETSTREAM_RENDER_METAL_AVAILABLE) || defined(JETSTREAM_RENDER_VULKAN_AVAILABLE))
    struct Guard {
        std::shared_ptr<Instance> instance = std::make_shared<Instance>();
        std::shared_ptr<Module> module;
        bool created = false;
        bool started = false;
        ~Guard() {
            try {
                if (module && module->state() == Module::State::CREATED) module->destroy();
                if (started) instance->stop();
                if (created) instance->destroy();
            } catch (...) {}
        }
    } guard;
    REQUIRE(guard.instance->create({.size = {800, 600}}) == Result::SUCCESS);
    guard.created = true;
    std::shared_ptr<Render::Window> window;
    REQUIRE(guard.instance->renderGet(window) == Result::SUCCESS);
    REQUIRE(Registry::BuildModule("adsb", DeviceType::CPU, RuntimeType::NATIVE,
                                   "generic", guard.module) == Result::SUCCESS);
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {8192}) == Result::SUCCESS);
    REQUIRE(guard.module->create("test", Modules::Adsb{}, AdsbInput(input), window) == Result::SUCCESS);
    const auto scheduler = guard.module->context()->scheduler();
    REQUIRE(scheduler->presentInitialize() == Result::SUCCESS);
    auto* adsb = guard.module->getImpl<Modules::AdsbImpl>();
    REQUIRE(adsb != nullptr);
    auto& map = adsb->*AdsbImplAccess::mapMember();
    auto& state = adsb->*AdsbImplAccess::mapStateMember();
    auto& aircraft = adsb->*AdsbImplAccess::aircraftMapMember();
    {
        Render::Surface::Config composition;
        REQUIRE(map->surface(composition) == Result::SUCCESS);
        const Render::Buffer* solidDash = nullptr;
        const Render::Buffer* markerQuad = nullptr;
        std::set<const Render::Buffer*> dashedBuffers;
        U64 solidCategories = 0;
        U64 markersAndStars = 0;
        for (const auto& program : composition.programs) {
            const auto& config = program->getConfig();
            for (const auto& draw : config.draws) {
                const auto& vertex = draw->getConfig().buffer->getConfig();
                if (vertex.instances.size() == 2 &&
                    vertex.instances[0].second == 4 && vertex.instances[1].second == 2) {
                    REQUIRE(config.buffers.front().first->byteSize() == sizeof(Render::Components::MapContext::GpuUniforms));
                    const auto* camera = static_cast<const Render::Components::MapContext::GpuUniforms*>(
                        config.buffers.front().first->getConfig().buffer);
                    const auto& ranges = vertex.instances[1].first;
                    REQUIRE(ranges->size() >= vertex.instances[0].first->size() / 2);
                    if (camera->lineStyle > 0.5f) {
                        dashedBuffers.insert(ranges.get());
                    } else {
                        if (!solidDash) solidDash = ranges.get();
                        REQUIRE(ranges.get() == solidDash);
                        ++solidCategories;
                    }
                }
                if (!vertex.indices && vertex.vertices.size() == 1 &&
                    vertex.vertices[0].second == 2 && vertex.instances.size() == 1 &&
                    vertex.instances[0].second == 4) {
                    const auto* quad = vertex.vertices[0].first.get();
                    if (!markerQuad) markerQuad = quad;
                    REQUIRE(quad == markerQuad);
                    ++markersAndStars;
                }
            }
        }
        REQUIRE(solidCategories > 1);
        REQUIRE(dashedBuffers.size() == 2); // Tropics and polar circles only.
        REQUIRE_FALSE(dashedBuffers.contains(solidDash));
        REQUIRE(markersAndStars == 5);
    }
    // No compute runs in this fixture; mutation is synchronized with present.
    aircraft[0xABC123] = {
        .icao = 0xABC123, .callsign = "POC123", .altitude = 32000, .speed = 240,
        .heading = 45, .latitude = 38, .longitude = -122,
        .hasCallsign = true, .hasAltitude = true, .hasVelocity = true, .hasPosition = true,
        .lastPositionTimestamp = static_cast<U64>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()),
        .track = {{37.7, -122.5}, {37.9, -122.3}, {38, -122}},
    };
    // Prime the collection clock without sleeping; present() must apply the
    // fitted view from the real decoder snapshot after processing the resize.
    const Modules::AdsbMapAircraft startup{.latitude = 38, .longitude = -122};
    auto& autoFit = adsb->*AdsbImplAccess::mapAutoFitMember();
    const U64 start = aircraft[0xABC123].lastPositionTimestamp;
    REQUIRE_FALSE(autoFit.update({&startup, 1}, map->getContext(), {},
        start - Modules::AdsbMapAutoFit::CollectionMilliseconds).has_value());
    guard.module->surface()->pushSurfaceEvent({SurfaceEventType::Resize, {800, 600}, 1});
    guard.module->surface()->pushMouseEvent({.type = MouseEventType::Move,
                                            .position = {0.5f, 0.5f}});
    REQUIRE(guard.instance->start() == Result::SUCCESS);
    guard.started = true;
    auto frame = [&] {
        REQUIRE(guard.instance->poll(false) == Result::SUCCESS);
        REQUIRE(guard.instance->present([&] { return scheduler->presentSubmit(); }) == Result::SUCCESS);
        REQUIRE(window->synchronize() == Result::SUCCESS);
    };
    frame();
    REQUIRE(map->getUniforms().centerLon == Catch::Approx(-122));
    REQUIRE(map->getUniforms().centerLat == Catch::Approx(38));
    REQUIRE(map->getUniforms().zoom == 5);
    const auto initialFit = map->getUniforms();
    aircraft[0xDEF456] = {.icao = 0xDEF456, .latitude = -38, .longitude = 58,
                         .hasPosition = true};
    aircraft[0x123456] = {.icao = 0x123456};  // No position: not renderable.
    frame();
    REQUIRE(map->getUniforms() == initialFit);
    REQUIRE(state->aircraft.size() == 2);
    const auto targets = Modules::AdsbVisibleTargets(state->aircraft, map->getContext());
    REQUIRE(targets.size() == 1);
    REQUIRE(state->aircraft[targets[0].index].icao == 0xABC123);
    const auto manifests = guard.module->surface()->manifests();
    REQUIRE(manifests.size() == 1);
    REQUIRE(manifests[0].surface->raw() != 0);
    REQUIRE(manifests[0].size.x == 800);
    REQUIRE(manifests[0].size.y == 600);

    // Aircraft are passive overlays: a drag starting on a target navigates the map.
    const auto beforeDrag = map->getUniforms();
    guard.module->surface()->pushMouseEvent({.type = MouseEventType::Click,
        .button = MouseButton::Left, .position = {0.5f, 0.5f}});
    guard.module->surface()->pushMouseEvent({.type = MouseEventType::Move,
        .position = {0.6f, 0.5f}});
    guard.module->surface()->pushMouseEvent({.type = MouseEventType::Release,
        .button = MouseButton::Left, .position = {0.6f, 0.5f}});
    frame();
    REQUIRE(map->getUniforms().centerLon != beforeDrag.centerLon);
    const auto beforeRightClick = map->getUniforms();
    guard.module->surface()->pushMouseEvent({.type = MouseEventType::Click,
        .button = MouseButton::Right, .position = {0.5f, 0.5f}});
    guard.module->surface()->pushMouseEvent({.type = MouseEventType::Release,
        .button = MouseButton::Right, .position = {0.5f, 0.5f}});
    frame();
    REQUIRE(map->getUniforms() == beforeRightClick);

    aircraft.clear();
    guard.module->surface()->pushMouseEvent({.type = MouseEventType::Leave});
    guard.module->surface()->pushSurfaceEvent({SurfaceEventType::Resize, {600, 800}, 2});
    frame();
    REQUIRE(state->aircraft.empty());
    REQUIRE_FALSE(map->getContext().cursor.has_value());
    REQUIRE(map->getUniforms().surfaceScale == 2);
    REQUIRE(guard.module->surface()->manifests()[0].size.x == 600);
    REQUIRE(guard.module->surface()->manifests()[0].size.y == 800);
    const auto savedView = map->getUniforms();
    REQUIRE(guard.module->destroy() == Result::SUCCESS);
    REQUIRE(guard.module->create("test", Modules::Adsb{}, AdsbInput(input), window) == Result::SUCCESS);
    REQUIRE(scheduler->presentInitialize() == Result::SUCCESS);
    REQUIRE(map->getUniforms() == savedView);
    REQUIRE_FALSE(autoFit.update({&startup, 1}, map->getContext(), {}, start + 10000).has_value());
    REQUIRE_FALSE(autoFit.update({&startup, 1}, map->getContext(), {}, start + 20000).has_value());
    frame();
#else
    SKIP("Requires a desktop Metal or Vulkan renderer.");
#endif
}

TEST_CASE("ADS-B - Silence Input",
          "[modules][adsb][silence]") {
    auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("adsb", impl.device, impl.runtime,
                           impl.provider);

            Modules::Adsb config;
            ctx.setConfig(config);

            const U64 bufferSize = 240;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                  {bufferSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("frequency", F32{1090e6f}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleRate", F32{2e6f}) ==
                    Result::SUCCESS);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

        }
    }
}

TEST_CASE("ADS-B - Random Noise Input",
          "[modules][adsb][noise]") {
    auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("adsb", impl.device, impl.runtime,
                           impl.provider);

            Modules::Adsb config;
            ctx.setConfig(config);

            // Create low-level noise input.
            const U64 bufferSize = 65536;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                const F32 r = static_cast<F32>(i % 7) * 0.001f - 0.003f;
                const F32 q = static_cast<F32>(i % 11) * 0.001f - 0.005f;
                input.at<CF32>(i) = CF32(r, q);
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

        }
    }
}

TEST_CASE("ADS-B - Invalid Input DType",
          "[modules][adsb][validation]") {
    auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            Tensor input;
            REQUIRE(input.create(impl.device, DataType::F32,
                                  {8192}) == Result::SUCCESS);
            RequireAdsbValidationError(impl, input);
        }
    }
}

TEST_CASE("ADS-B - Malformed Metadata",
          "[modules][adsb][validation][metadata]") {
    const auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            for (const std::string& key : {std::string("frequency"),
                                           std::string("sampleRate")}) {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::CF32,
                                     {8192}) == Result::SUCCESS);
                const F64 value = key == "frequency" ? 1090e6 : 2e6;
                REQUIRE(input.setAttribute(key, value) == Result::SUCCESS);
                RequireAdsbValidationError(impl, input);
            }
        }
    }
}

TEST_CASE("ADS-B - Input Size Boundaries",
          "[modules][adsb][validation][size]") {
    const auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            SECTION("libmodes minimum") {
                Tensor input;
                REQUIRE(input.create(reinterpret_cast<void*>(0x1000),
                                     impl.device,
                                     DataType::CF32,
                                     {239}) == Result::SUCCESS);
                REQUIRE(input.contiguous());
                RequireAdsbValidationError(impl, input);
            }

            SECTION("libmodes sample-count representation") {
                constexpr U64 inputSize =
                    static_cast<U64>(std::numeric_limits<U32>::max()) + 1;
                Tensor input;
                REQUIRE(input.create(reinterpret_cast<void*>(0x1000),
                                     impl.device,
                                     DataType::CF32,
                                     {inputSize}) == Result::SUCCESS);
                REQUIRE(input.contiguous());
                REQUIRE(input.size() == inputSize);
                RequireAdsbValidationError(impl, input);
            }
        }
    }
}

TEST_CASE("ADS-B - Metadata Validation Preserves Live State",
          "[modules][adsb][validation][metadata][rollback]") {
    const auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            Tensor input;
            REQUIRE(input.create(impl.device, DataType::CF32, {8192}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("frequency", F32{1090e6f}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleRate", F32{2e6f}) ==
                    Result::SUCCESS);

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("adsb", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", Modules::Adsb{}, AdsbInput(input)) ==
                    Result::SUCCESS);

            auto* adsb = module->getImpl<Modules::AdsbImpl>();
            REQUIRE(adsb != nullptr);
            const Tensor& activeInput = adsb->*AdsbImplAccess::inputMember();
            const Index inputId = activeInput.id();
            REQUIRE(module->outputs().empty());
            REQUIRE(adsb->getAircraftTable() == "No aircraft detected.");
            auto& aircraft = adsb->*AdsbImplAccess::aircraftMapMember();
            auto& tableDirty = adsb->*AdsbImplAccess::aircraftTableDirtyMember();
            aircraft[0xABC123] = {
                .icao = 0xABC123, .callsign = "TEST123", .latitude = 38, .longitude = -122,
                .hasCallsign = true, .hasPosition = true,
            };
            tableDirty = true;
            adsb->updateAircraftTable();
            const auto table = adsb->getAircraftTable();
            REQUIRE(table.find("ABC123\tTEST123\t-\t-\t-\t38.0000\t-122.0000\n") != std::string::npos);
            REQUIRE_FALSE(tableDirty);
            adsb->updateAircraftTable();
            REQUIRE(adsb->getAircraftTable() == table);

            REQUIRE(input.setAttribute("sampleRate", F64{2e6}) ==
                    Result::SUCCESS);
            REQUIRE(adsb->validate() == Result::ERROR);
            REQUIRE(module->state() == Module::State::CREATED);
            REQUIRE(activeInput.id() == inputId);
            REQUIRE(aircraft.size() == 1);
            REQUIRE(aircraft.at(0xABC123).callsign == "TEST123");
            REQUIRE(aircraft.at(0xABC123).hasPosition);
            REQUIRE(adsb->getAircraftTable() == table);

            REQUIRE(input.setAttribute("sampleRate", F32{4e6f}) ==
                    Result::SUCCESS);
            REQUIRE(adsb->validate() == Result::SUCCESS);
            REQUIRE(std::any_cast<F32>(activeInput.attribute("sampleRate")) ==
                    4e6f);
            REQUIRE(adsb->getAircraftTable() == table);
            aircraft.clear();
            tableDirty = true;
            adsb->updateAircraftTable();
            REQUIRE(adsb->getAircraftTable() == "No aircraft detected.");
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
