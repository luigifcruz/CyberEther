#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "jetstream/surface.hh"

using namespace Jetstream;

TEST_CASE("Surface input buffer preserves mixed event order and drains once",
          "[core][surface][input]") {
    EventBuffer buffer;
    buffer.pushInput(KeyEvent{KeyEventType::Press, KeyCode::LeftShift, {.shift = true}});
    buffer.pushInput(MouseEvent{.type = MouseEventType::Click, .position = {0.25f, 0.75f},
                                .modifiers = {.shift = true}});
    buffer.pushInput(KeyEvent{KeyEventType::Release, KeyCode::LeftShift, {}});
    buffer.pushSurface({.type = SurfaceEventType::Resize, .size = {640, 480}});

    const auto events = buffer.consumeInputEvents();
    REQUIRE(events.size() == 3);
    REQUIRE(std::get<KeyEvent>(events[0]).type == KeyEventType::Press);
    REQUIRE(std::get<MouseEvent>(events[1]).position.x == 0.25f);
    REQUIRE(std::get<MouseEvent>(events[1]).modifiers.shift);
    REQUIRE(std::get<KeyEvent>(events[2]).type == KeyEventType::Release);
    REQUIRE(buffer.consumeInputEvents().empty());
    REQUIRE(buffer.consumeSurfaceEvents().size() == 1);
    buffer.pushInput(FocusEvent{false});
    REQUIRE(buffer.consumeInputEvents().size() == 1);
}

TEST_CASE("Keyboard input leaves cursor and mouse navigation unchanged",
          "[core][surface][input][interaction]") {
    SurfaceInteractionState state;
    state.zoom = 2.0f;
    state.cursorNormalized = {0.3f, 0.7f};
    state = ProcessSurfaceInteraction(state, {}, {
        FocusEvent{true},
        KeyEvent{KeyEventType::Press, KeyCode::A, {}},
    });
    REQUIRE_FALSE(state.cursorMoved);
    REQUIRE_FALSE(state.viewChanged);
    REQUIRE(state.cursorNormalized.x == 0.3f);
    REQUIRE(state.cursorNormalized.y == 0.7f);

    state = ProcessSurfaceInteraction(state, {}, {
        MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Left, .position = {0.5f, 0.5f}},
        KeyEvent{KeyEventType::Press, KeyCode::LeftShift, {.shift = true}},
        MouseEvent{.type = MouseEventType::Move, .position = {0.7f, 0.8f}},
    });
    REQUIRE(state.dragging);
    REQUIRE(state.offset == Catch::Approx(-0.1f));
    REQUIRE(state.cursorNormalized.x == 0.7f);
    REQUIRE(state.cursorNormalized.y == 0.8f);

    state = ProcessSurfaceInteraction(state, {}, {
        FocusEvent{false},
        MouseEvent{.type = MouseEventType::Move, .position = {0.9f, 0.8f}},
    });
    REQUIRE_FALSE(state.dragging);
    REQUIRE(state.offset == Catch::Approx(-0.1f));
}

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
