#include <catch2/catch_test_macros.hpp>

#include "jetstream/surface.hh"

using namespace Jetstream;

TEST_CASE("Surface interaction reports placement-only changes as view changes",
          "[core][surface][interaction]") {
    SurfaceInteractionState state;
    state.viewSize = {640, 480};
    state.scale = 1.0f;
    state.placement = SurfacePlacementType::Detached;

    SurfaceEvent event;
    event.type = SurfaceEventType::Resize;
    event.size = state.viewSize;
    event.scale = state.scale;
    event.placement = SurfacePlacementType::Attached;

    const auto result = ProcessSurfaceInteraction(state, {event}, {});

    REQUIRE(result.viewChanged);
    REQUIRE(result.placement == SurfacePlacementType::Attached);
    REQUIRE(result.viewSize == state.viewSize);
    REQUIRE(result.scale == state.scale);
}

TEST_CASE("Surface interaction ignores unchanged surface events",
          "[core][surface][interaction]") {
    SurfaceInteractionState state;
    state.viewSize = {640, 480};
    state.scale = 1.0f;
    state.placement = SurfacePlacementType::Attached;

    SurfaceEvent event;
    event.type = SurfaceEventType::Resize;
    event.size = state.viewSize;
    event.scale = state.scale;
    event.placement = state.placement;

    const auto result = ProcessSurfaceInteraction(state, {event}, {});

    REQUIRE_FALSE(result.viewChanged);
    REQUIRE(result.placement == SurfacePlacementType::Attached);
}

TEST_CASE("Surface interaction reports size and scale changes as view changes",
          "[core][surface][interaction]") {
    SurfaceInteractionState state;
    state.viewSize = {640, 480};
    state.scale = 1.0f;

    SurfaceEvent event;
    event.type = SurfaceEventType::Resize;
    event.size = {1280, 720};
    event.scale = 2.0f;

    const auto result = ProcessSurfaceInteraction(state, {event}, {});

    REQUIRE(result.viewChanged);
    REQUIRE(result.viewSize == event.size);
    REQUIRE(result.scale == event.scale);
}

TEST_CASE("Surface interaction restores unchanged placement on plain resize",
          "[core][surface][interaction]") {
    SurfaceInteractionState state;
    state.viewSize = {640, 480};
    state.scale = 1.0f;
    state.placement = SurfacePlacementType::Detached;

    SurfaceEvent event;
    event.type = SurfaceEventType::Resize;
    event.size = {800, 600};
    event.scale = 1.0f;
    event.placement = SurfacePlacementType::Detached;

    const auto result = ProcessSurfaceInteraction(state, {event}, {});

    REQUIRE(result.viewChanged);
    REQUIRE(result.placement == SurfacePlacementType::Detached);
}
