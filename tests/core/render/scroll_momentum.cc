#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "jetstream/render/base/input.hh"

using namespace Jetstream;

namespace {

using Queue = Render::InputQueue;

Queue::Wheel Finger(ScrollPhase phase, U32 time, F32 y = -1.0f) {
    return {.delta = {0.0f, y}, .phase = phase, .precise = true, .explicitRelease = true,
            .synthesizeMomentum = true, .timestamp = time};
}

std::vector<MouseEvent> Scrolls(const Render::InputState& state) {
    std::vector<MouseEvent> result;
    for (const auto& event : state.events()) {
        if (const auto* mouse = std::get_if<MouseEvent>(&event); mouse && mouse->type == MouseEventType::Scroll) {
            result.push_back(*mouse);
        }
    }
    return result;
}

void Fling(Render::InputState& state) {
    state.advance(1.0, {Queue::Pointer{Extent2D<F32>{50.0f, 70.0f}},
        Finger(ScrollPhase::Began, 100), Finger(ScrollPhase::Changed, 110),
        Finger(ScrollPhase::Changed, 120), Finger(ScrollPhase::Ended, 125, 0.0f)});
}

}  // namespace

TEST_CASE("Finger movement stays direct and inactivity never starts momentum", "[core][render][scroll-momentum]") {
    Render::InputState state;
    state.advance(1.0, {Finger(ScrollPhase::Began, 100), Finger(ScrollPhase::Changed, 110)});
    const auto events = Scrolls(state);
    REQUIRE(events.size() == 2);
    CHECK(events[0].scroll.y == -1.0f);
    CHECK(events[1].scroll.y == -1.0f);
    CHECK_FALSE(state.hasPending());
    state.advance(4.0, {});
    CHECK(Scrolls(state).empty());
    CHECK_FALSE(state.hasPending());
    state.advance(4.01, {Finger(ScrollPhase::Ended, 3100, 0.0f)});
    REQUIRE(Scrolls(state).size() == 1);
    CHECK(Scrolls(state)[0].phase == ScrollPhase::Ended);
    CHECK_FALSE(state.hasPending());
}

TEST_CASE("Explicit release starts decaying momentum and eventually allows idle", "[core][render][scroll-momentum]") {
    Render::InputState state;
    Fling(state);
    REQUIRE(state.hasPending());
    const auto released = Scrolls(state);
    REQUIRE(released.size() == 5);
    CHECK(released[3].phase == ScrollPhase::Ended);
    CHECK(released[4].phase == ScrollPhase::Momentum);
    CHECK(released[4].scroll.y == 0.0f);
    F32 previous = -100.0f;
    bool ended = false;
    for (U32 i = 1; i <= 240 && state.hasPending(); ++i) {
        state.advance(1.0 + i / 120.0, {});
        for (const auto& event : Scrolls(state)) {
            CHECK(event.position == Extent2D<F32>{50.0f, 70.0f});
            if (event.phase == ScrollPhase::Momentum) {
                CHECK(event.scroll.y > previous);
                CHECK(event.scroll.y < 0.0f);
                previous = event.scroll.y;
            } else {
                CHECK(event.phase == ScrollPhase::MomentumEnded);
                ended = true;
            }
        }
    }
    CHECK(ended);
    CHECK_FALSE(state.hasPending());
    state.advance(5.0, {});
    CHECK(Scrolls(state).empty());
}

TEST_CASE("Momentum distance is independent of rendering rate and input batching", "[core][render][scroll-momentum]") {
    const auto travel = [](int hz, bool batched) {
        Render::InputState state;
        if (batched) {
            Fling(state);
        } else {
            state.advance(0.975, {Finger(ScrollPhase::Began, 100)});
            state.advance(0.985, {Finger(ScrollPhase::Changed, 110)});
            state.advance(0.995, {Finger(ScrollPhase::Changed, 120)});
            state.advance(1.0, {Finger(ScrollPhase::Ended, 125, 0.0f)});
        }
        F32 distance = 0.0f;
        for (int i = 1; i < hz * 3 && state.hasPending(); ++i) {
            state.advance(1.0 + static_cast<F64>(i) / hz, {});
            for (const auto& event : Scrolls(state)) {
                distance += event.scroll.y;
            }
        }
        CHECK_FALSE(state.hasPending());
        return distance;
    };
    const F32 reference = travel(60, false);
    CHECK(reference < -5.0f);
    CHECK(travel(60, true) == Catch::Approx(reference).margin(0.001f));
    CHECK(travel(144, true) == Catch::Approx(reference).margin(0.001f));
    CHECK(travel(240, false) == Catch::Approx(reference).margin(0.001f));
}

TEST_CASE("New input cancels synthesized momentum before any further travel", "[core][render][scroll-momentum]") {
    Render::InputState state;
    Fling(state);
    std::vector<Queue::Event> events;
    SECTION("button") { events = {Queue::Button{MouseButton::Left, true}}; }
    SECTION("keyboard") { events = {Queue::Key{KeyCode::Escape, true}}; }
    SECTION("focus loss") { events = {Queue::Focus{false}}; }
    SECTION("pointer leave") { events = {Queue::Hover{false}}; }
    SECTION("hold gesture") { events = {Queue::Wheel{.phase = ScrollPhase::MomentumEnded, .precise = true}}; }
    SECTION("another wheel") { events = {Queue::Wheel{.delta = {0.0f, 0.125f}}}; }
    SECTION("new finger gesture") { events = {Finger(ScrollPhase::Began, 150)}; }
    state.advance(1.01, events);
    CHECK_FALSE(state.hasPending());
    for (const auto& event : Scrolls(state)) {
        CHECK(event.phase != ScrollPhase::Momentum);
    }
}

TEST_CASE("Native cancellations end each synthesized stream exactly once", "[core][render][scroll-momentum]") {
    const auto cancel = [](ScrollPhase phase) {
        return Queue::Wheel{.phase = phase, .precise = true, .explicitRelease = true};
    };
    const auto count = [](const Render::InputState& state, ScrollPhase phase) {
        U64 result = 0;
        for (const auto& event : Scrolls(state)) {
            result += event.phase == phase ? 1 : 0;
        }
        return result;
    };
    Render::InputState state;
    SECTION("during the gesture") {
        state.advance(1.0, {Finger(ScrollPhase::Began, 100), Finger(ScrollPhase::Changed, 110)});
        state.advance(1.01, {cancel(ScrollPhase::Ended)});
        CHECK(Scrolls(state).size() == 1);
        CHECK(count(state, ScrollPhase::Ended) == 1);
    }
    SECTION("while coasting") {
        Fling(state);
        REQUIRE(state.hasPending());
        state.advance(1.01, {cancel(ScrollPhase::MomentumEnded)});
        CHECK(Scrolls(state).size() == 1);
        CHECK(count(state, ScrollPhase::MomentumEnded) == 1);
    }
    SECTION("after the fling already ended") {
        Fling(state);
        for (U32 i = 1; i <= 480 && state.hasPending(); ++i) {
            state.advance(1.0 + i / 120.0, {});
        }
        REQUIRE_FALSE(state.hasPending());
        state.advance(10.0, {cancel(ScrollPhase::MomentumEnded)});
        CHECK(Scrolls(state).empty());
    }
    CHECK_FALSE(state.hasPending());
}

TEST_CASE("A native cancellation releases a gesture that resumed after a key press", "[core][render][scroll-momentum]") {
    Render::InputState state;
    state.advance(1.0, {Finger(ScrollPhase::Began, 100)});
    state.advance(1.005, {Queue::Key{KeyCode::Escape, true}});
    REQUIRE(Scrolls(state).size() == 1);
    CHECK(Scrolls(state)[0].phase == ScrollPhase::Ended);
    state.advance(1.01, {Finger(ScrollPhase::Changed, 110)});
    REQUIRE(Scrolls(state).size() == 1);
    CHECK(Scrolls(state)[0].phase == ScrollPhase::Changed);
    state.advance(1.02, {Queue::Wheel{.phase = ScrollPhase::Ended, .precise = true, .explicitRelease = true}});
    REQUIRE(Scrolls(state).size() == 1);
    CHECK(Scrolls(state)[0].phase == ScrollPhase::Ended);
    CHECK_FALSE(state.hasPending());
}

TEST_CASE("Native momentum and unclassified precision input never synthesize additional inertia",
          "[core][render][scroll-momentum]") {
    Render::InputState state;
    state.advance(1.0, {
        Queue::Wheel{.delta = {0.0f, -1.0f}, .phase = ScrollPhase::Began, .precise = true},
        Queue::Wheel{.delta = {0.0f, -1.0f}, .phase = ScrollPhase::Changed, .precise = true},
        Queue::Wheel{.phase = ScrollPhase::Ended, .precise = true},
        Queue::Wheel{.delta = {0.0f, -0.5f}, .phase = ScrollPhase::Momentum, .precise = true},
        Queue::Wheel{.phase = ScrollPhase::MomentumEnded, .precise = true},
        Queue::Wheel{.delta = {0.0f, -0.125f}, .precise = true},
    });
    REQUIRE(Scrolls(state).size() == 6);
    CHECK_FALSE(state.hasPending());
    state.advance(1.02, {});
    CHECK(Scrolls(state).empty());
}

TEST_CASE("Native scroll timestamps wrap and direction reversals discard old velocity",
          "[core][render][scroll-momentum]") {
    Render::ScrollMomentum momentum;
    momentum.sample(0xfffffff0u, {0.0f, -1.0f});
    momentum.sample(0xfffffffau, {0.0f, -1.0f});
    momentum.sample(4, {0.0f, -1.0f});
    REQUIRE(momentum.release(8));
    CHECK(momentum.advance(0.01).y < 0.0f);
    momentum.reset();
    momentum.sample(10, {0.0f, -10.0f});
    momentum.sample(20, {0.0f, -10.0f});
    momentum.sample(30, {0.0f, 0.25f});
    REQUIRE(momentum.release(35));
    const F32 distance = momentum.advance(0.01).y;
    CHECK(distance > 0.0f);
    CHECK(distance < 0.25f);
    momentum.advance(1.0);
    CHECK_FALSE(momentum.active());
}

TEST_CASE("Native scroll positions do not restore hover after a leave", "[core][render][scroll-momentum]") {
    Render::InputState state;
    Fling(state);
    state.advance(1.01, {Queue::Hover{false}, Queue::Wheel{
        .phase = ScrollPhase::MomentumEnded, .precise = true, .position = Extent2D<F32>{50.0f, 70.0f}}});
    CHECK_FALSE(state.pointer().has_value());
    CHECK_FALSE(state.hasPending());
    const auto events = Scrolls(state);
    REQUIRE_FALSE(events.empty());
    CHECK(events.back().positioned);
    CHECK(events.back().position == Extent2D<F32>{50.0f, 70.0f});
}

TEST_CASE("Delayed explicit releases do not replay stale momentum", "[core][render][scroll-momentum]") {
    Render::InputState state;
    SECTION("native dispatch stalls before release") {
        state.advance(1.0, {Finger(ScrollPhase::Began, 100)});
        state.advance(1.01, {Finger(ScrollPhase::Changed, 110)});
        state.advance(1.02, {Finger(ScrollPhase::Changed, 120)});
        state.advance(2.02, {Finger(ScrollPhase::Ended, 125, 0.0f)});
    }
    SECTION("a complete gesture waits in the producer queue") {
        Queue queue;
        for (auto event : {Finger(ScrollPhase::Began, 100), Finger(ScrollPhase::Changed, 110),
                           Finger(ScrollPhase::Changed, 120), Finger(ScrollPhase::Ended, 125, 0.0f)}) {
            event.receivedAt = std::chrono::steady_clock::now() - std::chrono::seconds(1);
            queue.push(event);
        }
        state.advance(2.02, queue.drain());
    }
    SECTION("a batched gesture waits behind trickled button input") {
        state.advance(1.0, {Queue::Button{MouseButton::Left, true},
            Finger(ScrollPhase::Began, 100), Finger(ScrollPhase::Changed, 110),
            Finger(ScrollPhase::Changed, 120), Finger(ScrollPhase::Ended, 125, 0.0f)});
        REQUIRE(state.hasPending());
        state.advance(2.02, {});
    }
    CHECK_FALSE(state.hasPending());
    REQUIRE_FALSE(Scrolls(state).empty());
    CHECK(Scrolls(state).back().phase == ScrollPhase::Ended);
    state.advance(2.03, {});
    CHECK(Scrolls(state).empty());
}

TEST_CASE("Fresh finger input after a long idle still flings across native timestamp wrap",
          "[core][render][scroll-momentum]") {
    Render::InputState state;
    state.advance(1.0, {Finger(ScrollPhase::Began, 0xfffffff0u)});
    state.advance(1.01, {Finger(ScrollPhase::Changed, 0xfffffffau)});
    state.advance(1.02, {Finger(ScrollPhase::Changed, 4)});
    state.advance(1.025, {Finger(ScrollPhase::Ended, 9, 0.0f)});
    REQUIRE(state.hasPending());
    state.advance(10.0, {Finger(ScrollPhase::Began, 8984), Finger(ScrollPhase::Changed, 8994),
                          Finger(ScrollPhase::Changed, 9004), Finger(ScrollPhase::Ended, 9009, 0.0f)});
    CHECK(state.hasPending());
    state.advance(10.01, {});
    REQUIRE(Scrolls(state).size() == 1);
    CHECK(Scrolls(state).front().scroll.y < 0.0f);
}

TEST_CASE("Input queue stamps native momentum packets at receipt", "[core][render][scroll-momentum]") {
    Queue queue;
    const auto before = std::chrono::steady_clock::now();
    queue.push(Finger(ScrollPhase::Began, 100));
    const auto after = std::chrono::steady_clock::now();
    const auto events = queue.drain();
    REQUIRE(events.size() == 1);
    const auto& wheel = std::get<Queue::Wheel>(events.front());
    REQUIRE(wheel.receivedAt);
    CHECK(*wheel.receivedAt >= before);
    CHECK(*wheel.receivedAt <= after);
}
