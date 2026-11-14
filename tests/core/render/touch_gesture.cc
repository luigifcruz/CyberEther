#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "viewport/tools/touch_gesture.hh"

using namespace Jetstream;

namespace {
using Queue = Render::InputQueue;
using Gesture = Viewport::TouchGesture;
using Type = Gesture::Event::Type;

template<typename T>
std::vector<T> Select(const std::vector<Queue::Event>& events) {
    std::vector<T> result;
    for (const auto& event : events) {
        if (const auto* value = std::get_if<T>(&event)) result.push_back(*value);
    }
    return result;
}

struct Touch {
    Queue queue;
    Gesture gesture{queue};
    void send(Type type, I64 id = 1, Extent2D<F32> position = {}, U32 timestamp = 0,
              F64 now = 1.0, bool allowed = true, bool fresh = true) {
        gesture.handle({type, id, position, timestamp, fresh}, now, allowed);
    }
};
}

TEST_CASE("Touch taps defer activation and preserve double click delivery", "[core][render][touch]") {
    Touch touch;
    Render::InputState state;
    F64 clock = 1.0;
    std::vector<MouseEvent> buttons;
    for (int tap = 0; tap < 2; ++tap) {
        touch.send(Type::Down, 1, {30, 40});
        const auto down = touch.queue.drain();
        CHECK(Select<Queue::Button>(down).empty());
        state.advance(clock += 0.01, down);
        touch.send(Type::Up, 1, {}, 100, 1.1);
        state.advance(clock += 0.01, touch.queue.drain());
        for (int frame = 0; frame < 5; ++frame) {
            for (const auto& event : state.events()) {
                if (const auto* mouse = std::get_if<MouseEvent>(&event);
                    mouse && (mouse->type == MouseEventType::Click || mouse->type == MouseEventType::Release)) {
                    buttons.push_back(*mouse);
                }
            }
            if (!state.hasPending()) break;
            state.advance(clock += 0.01, {});
        }
        CHECK_FALSE(state.buttons()[0]);
        CHECK_FALSE(state.pointer());
    }
    REQUIRE(buttons.size() == 4);
    CHECK(buttons[0].clicks == 1);
    CHECK(buttons[2].clicks == 2);
    for (const auto& event : buttons) {
        CHECK(event.button == MouseButton::Left);
        CHECK(event.position == Extent2D<F32>{30, 40});
    }
}

TEST_CASE("Touch long press activates once at the threshold without a left click", "[core][render][touch]") {
    Touch touch;
    touch.send(Type::Down, 1, {30, 40});
    touch.queue.drain();
    touch.gesture.poll(1.499);
    CHECK(touch.queue.drain().empty());
    touch.gesture.poll(1.5);
    auto buttons = Select<Queue::Button>(touch.queue.drain());
    REQUIRE(buttons.size() == 2);
    CHECK(buttons[0].button == MouseButton::Right);
    CHECK(buttons[0].down);
    CHECK_FALSE(buttons[1].down);
    touch.gesture.poll(2.0);
    touch.send(Type::Up, 1, {}, 2000, 2.0);
    CHECK(Select<Queue::Button>(touch.queue.drain()).empty());
}

TEST_CASE("Delayed touch delivery preserves the native tap versus hold duration", "[core][render][touch]") {
    Touch touch;
    MouseButton expected = MouseButton::Left;
    SECTION("a quick release queued during a stalled pump remains a tap") {
        touch.send(Type::Down, 1, {30, 40}, 100, 1.0);
        touch.send(Type::Up, 1, {}, 200, 2.0, true, false);
    }
    SECTION("a complete long press delivered in one batch remains a hold") {
        touch.send(Type::Down, 1, {30, 40}, 100, 2.0, true, false);
        touch.send(Type::Up, 1, {}, 700, 2.0, true, false);
        expected = MouseButton::Right;
    }
    const auto buttons = Select<Queue::Button>(touch.queue.drain());
    REQUIRE(buttons.size() == 2);
    CHECK(buttons[0].button == expected);
    CHECK(buttons[0].down);
    CHECK(buttons[1].button == expected);
    CHECK_FALSE(buttons[1].down);
}

TEST_CASE("Touch release classifies the exact hold threshold across timestamp wrap", "[core][render][touch]") {
    const U32 start = GENERATE(100u, 0xffffff00u);
    const U32 duration = GENERATE(499u, 500u, 501u);
    Touch touch;
    touch.send(Type::Down, 1, {30, 40}, start, 2.0);
    touch.send(Type::Up, 1, {}, start + duration, 2.0);
    const auto buttons = Select<Queue::Button>(touch.queue.drain());
    REQUIRE(buttons.size() == 2);
    CHECK(buttons[0].button == (duration < 500 ? MouseButton::Left : MouseButton::Right));
    CHECK(buttons[0].down);
    CHECK_FALSE(buttons[1].down);
}

TEST_CASE("Touch cancellation clears pending holds and ignores late contact packets", "[core][render][touch]") {
    Touch touch;
    touch.send(Type::Down);
    touch.queue.drain();
    touch.send(Type::Cancel);
    CHECK_FALSE(touch.gesture.active());
    auto pointers = Select<Queue::Pointer>(touch.queue.drain());
    REQUIRE(pointers.size() == 1);
    CHECK_FALSE(pointers[0].position);
    touch.gesture.poll(5.0);
    touch.send(Type::Motion, 1, {40, 40});
    touch.send(Type::Up);
    touch.send(Type::Cancel);
    CHECK(touch.queue.drain().empty());
}

TEST_CASE("Touch swipes report logical deltas in scroll units and coast across timestamp wrap", "[core][render][touch]") {
    Touch touch;
    touch.send(Type::Down, 1, {10, 20});
    touch.queue.drain();
    touch.send(Type::Motion, 1, {18, 20}, 0xfffffff0u);
    CHECK(touch.queue.drain().empty());
    touch.send(Type::Motion, 1, {19, 20}, 0xfffffffau);
    touch.send(Type::Motion, 1, {22, 24}, 4);
    touch.send(Type::Up, 1, {}, 9, 1.1);
    const auto events = touch.queue.drain();
    const auto wheels = Select<Queue::Wheel>(events);
    REQUIRE(wheels.size() == 4);
    CHECK(wheels[0].phase == ScrollPhase::Began);
    CHECK(wheels[0].delta.x == Catch::Approx(0.9f));
    CHECK(wheels[0].delta.y == 0.0f);
    CHECK(wheels[1].delta.x == Catch::Approx(0.3f));
    CHECK(wheels[1].delta.y == Catch::Approx(0.4f));
    CHECK(wheels[2].phase == ScrollPhase::Ended);
    CHECK(wheels[3].phase == ScrollPhase::Momentum);
    for (const auto& wheel : wheels) {
        CHECK(wheel.position == Extent2D<F32>{10, 20});
        CHECK(wheel.touch);
        CHECK(wheel.precise);
        CHECK_FALSE(wheel.cancel);
    }
    const auto pointers = Select<Queue::Pointer>(events);
    REQUIRE_FALSE(pointers.empty());
    CHECK_FALSE(pointers.back().position);
}

TEST_CASE("Touch rejects a physical drag until every ignored contact has lifted", "[core][render][touch]") {
    Touch touch;
    touch.send(Type::Down, 1, {}, 0, 1.0, false);
    touch.send(Type::Down, 2, {50, 0});
    touch.send(Type::Motion, 1, {20, 0});
    touch.gesture.poll(2.0);
    CHECK_FALSE(touch.gesture.active());
    CHECK_FALSE(touch.gesture.ownsPointer());
    touch.send(Type::Up, 1);
    touch.send(Type::Up, 2);
    CHECK(touch.queue.drain().empty());
    touch.send(Type::Down);
    CHECK(touch.gesture.active());
    touch.gesture.resumePointer();
    CHECK(touch.gesture.ownsPointer());
    touch.send(Type::Up);
    CHECK(touch.gesture.ownsPointer());
    touch.gesture.resumePointer();
    CHECK_FALSE(touch.gesture.ownsPointer());
}

TEST_CASE("Pinch batches both contacts and pins the initial midpoint", "[core][render][touch]") {
    Touch touch;
    touch.send(Type::Down, 1, {10, 20});
    touch.send(Type::Down, 2, {30, 20});
    touch.queue.drain();
    touch.send(Type::Motion, 1, {0, 20});
    touch.send(Type::Motion, 2, {40, 20});
    CHECK(touch.queue.drain().empty());
    touch.send(Type::Frame);
    auto zooms = Select<Queue::Zoom>(touch.queue.drain());
    REQUIRE(zooms.size() == 1);
    CHECK(zooms[0].factor == Catch::Approx(2));
    CHECK(zooms[0].position == Extent2D<F32>{20, 20});
    touch.send(Type::Motion, 1, {20, 30});
    touch.send(Type::Motion, 2, {60, 30});
    touch.send(Type::Frame);
    CHECK(touch.queue.drain().empty());
    touch.send(Type::Motion, 2, {100, 30});
    touch.send(Type::Up, 2);
    zooms = Select<Queue::Zoom>(touch.queue.drain());
    REQUIRE(zooms.size() == 1);
    CHECK(zooms[0].factor == Catch::Approx(2));
    CHECK(zooms[0].position == Extent2D<F32>{20, 20});
    touch.send(Type::Down, 3, {200, 0});
    touch.send(Type::Motion, 1, {0, 0});
    touch.send(Type::Motion, 3, {300, 0});
    touch.send(Type::Frame);
    touch.send(Type::Up, 1);
    touch.send(Type::Up, 3);
    const auto remaining = touch.queue.drain();
    CHECK(Select<Queue::Zoom>(remaining).empty());
    CHECK(Select<Queue::Button>(remaining).empty());
    CHECK(Select<Queue::Wheel>(remaining).empty());
    CHECK_FALSE(touch.gesture.active());
}

TEST_CASE("Pinch rebases coincident contacts without a zoom jump", "[core][render][touch]") {
    Touch touch;
    touch.send(Type::Down, 1, {});
    touch.send(Type::Down, 2, {});
    touch.queue.drain();
    touch.send(Type::Motion, 2, {10, 0});
    touch.send(Type::Frame);
    CHECK(touch.queue.drain().empty());
    touch.send(Type::Motion, 2, {20, 0});
    touch.send(Type::Frame);
    auto zooms = Select<Queue::Zoom>(touch.queue.drain());
    REQUIRE(zooms.size() == 1);
    CHECK(zooms[0].factor == Catch::Approx(2));
}

TEST_CASE("A second contact cancels swipe ownership before pinch", "[core][render][touch]") {
    Touch touch;
    touch.send(Type::Down);
    touch.send(Type::Motion, 1, {20, 0}, 100);
    touch.queue.drain();
    touch.send(Type::Down, 2, {40, 0}, 110);
    const auto wheels = Select<Queue::Wheel>(touch.queue.drain());
    REQUIRE(wheels.size() == 1);
    CHECK(wheels[0].phase == ScrollPhase::Ended);
    CHECK(wheels[0].cancel);
    touch.gesture.poll(2.0);
    CHECK(touch.queue.drain().empty());
}

TEST_CASE("Touch cancellation terminates an active swipe or its released momentum", "[core][render][touch]") {
    const bool released = GENERATE(false, true);
    Touch touch;
    touch.send(Type::Down);
    touch.send(Type::Motion, 1, {20, 0}, 100);
    touch.send(Type::Motion, 1, {40, 0}, 110);
    if (released) touch.send(Type::Up, 1, {}, 115, 1.1);
    touch.queue.drain();
    touch.send(Type::Cancel);
    const auto wheels = Select<Queue::Wheel>(touch.queue.drain());
    REQUIRE(wheels.size() == 1);
    CHECK(wheels[0].phase == (released ? ScrollPhase::MomentumEnded : ScrollPhase::Ended));
    CHECK(wheels[0].cancel);
    touch.send(Type::Cancel);
    CHECK(touch.queue.drain().empty());
    touch.send(Type::Down, 3, {100, 100}, 200, 2.0);
    touch.send(Type::Up, 3, {}, 210, 2.01);
    CHECK(Select<Queue::Button>(touch.queue.drain()).size() == 2);
}

TEST_CASE("Queued pinch zoom preserves its anchor and does not synthesize button presses", "[core][render][touch]") {
    Render::InputState state;
    state.advance(1.0, {Queue::Zoom{{30, 40}, 2}, Queue::Zoom{{30, 40}, 0.5f}});
    std::vector<MouseEvent> zooms;
    for (const auto& event : state.events()) {
        if (const auto* mouse = std::get_if<MouseEvent>(&event)) {
            CHECK(mouse->type != MouseEventType::Click);
            if (mouse->type == MouseEventType::Zoom) zooms.push_back(*mouse);
        }
    }
    REQUIRE(zooms.size() == 2);
    CHECK(zooms[0].position == Extent2D<F32>{30, 40});
    CHECK(zooms[1].position == zooms[0].position);
    CHECK(zooms[0].zoom * zooms[1].zoom == Catch::Approx(1));
    CHECK_FALSE(state.hasPending());
    state.advance(1.01, {Queue::Zoom{{30, 40}, 0}, Queue::Zoom{{30, 40}, -1}, Queue::Zoom{{30, 40}, 1}});
    CHECK(state.events().empty());
}

TEST_CASE("Unknown and duplicate contacts cannot change a pending tap", "[core][render][touch]") {
    Touch touch;
    touch.send(Type::Down, 1, {10, 20});
    touch.queue.drain();
    touch.send(Type::Down, 1, {100, 200});
    touch.send(Type::Motion, 9, {300, 0});
    touch.send(Type::Up, 9);
    touch.send(Type::Frame);
    CHECK(touch.queue.drain().empty());
    CHECK(touch.gesture.active());
    touch.send(Type::Up, 1, {}, 100, 1.1);
    CHECK(Select<Queue::Button>(touch.queue.drain()).size() == 2);
}

TEST_CASE("A stale native packet prevents fling but preserves the real release", "[core][render][touch]") {
    const auto stale = GENERATE(Type::Down, Type::Motion, Type::Frame, Type::Up);
    Touch touch;
    touch.send(Type::Down, 1, {}, 100, 1.0, true, stale != Type::Down);
    touch.send(Type::Motion, 1, {20, 0}, 110, 1.01, true, stale != Type::Motion);
    touch.send(Type::Motion, 1, {40, 0}, 120);
    touch.send(Type::Frame, 1, {}, 0, 1.02, true, stale != Type::Frame);
    touch.send(Type::Up, 1, {}, 125, 1.025, true, stale != Type::Up);
    auto wheels = Select<Queue::Wheel>(touch.queue.drain());
    REQUIRE(wheels.size() == 3);
    CHECK(wheels.back().phase == ScrollPhase::Ended);
    CHECK_FALSE(wheels.back().cancel);
    touch.gesture.poll(1.03);
    CHECK(Select<Queue::Wheel>(touch.queue.drain()).empty());
}

TEST_CASE("Released touch momentum decays on poll and ends exactly once", "[core][render][touch]") {
    Touch touch;
    touch.send(Type::Down, 1, {}, 100, 1.0);
    touch.send(Type::Motion, 1, {0, 20}, 110, 1.01);
    touch.send(Type::Motion, 1, {0, 40}, 120, 1.02);
    touch.send(Type::Up, 1, {}, 125, 1.025);
    auto wheels = Select<Queue::Wheel>(touch.queue.drain());
    REQUIRE(wheels.back().phase == ScrollPhase::Momentum);
    F32 previous = 1e9f;
    F32 travel = 0.0f;
    U64 ended = 0;
    for (U32 i = 1; i <= 1000 && ended == 0; ++i) {
        touch.gesture.poll(1.025 + i * 0.003);
        for (const auto& wheel : Select<Queue::Wheel>(touch.queue.drain())) {
            CHECK(wheel.touch);
            CHECK(wheel.position == Extent2D<F32>{0, 0});
            if (wheel.phase == ScrollPhase::Momentum) {
                CHECK(wheel.delta.y > 0.0f);
                CHECK(wheel.delta.y <= previous);
                previous = wheel.delta.y;
                travel += wheel.delta.y;
            } else {
                CHECK(wheel.phase == ScrollPhase::MomentumEnded);
                CHECK_FALSE(wheel.cancel);
                ++ended;
            }
        }
    }
    CHECK(ended == 1);
    CHECK(travel > 4.0f);
    touch.gesture.poll(10.0);
    touch.send(Type::Cancel);
    CHECK(Select<Queue::Wheel>(touch.queue.drain()).empty());
}

TEST_CASE("A new contact or a returning mouse stops released touch momentum", "[core][render][touch]") {
    const bool contact = GENERATE(true, false);
    Touch touch;
    touch.send(Type::Down, 1, {}, 100, 1.0);
    touch.send(Type::Motion, 1, {0, 20}, 110, 1.01);
    touch.send(Type::Motion, 1, {0, 40}, 120, 1.02);
    touch.send(Type::Up, 1, {}, 125, 1.025);
    touch.queue.drain();
    if (contact) {
        touch.send(Type::Down, 2, {5, 5}, 130, 1.03);
    } else {
        touch.gesture.resumePointer();
    }
    const auto wheels = Select<Queue::Wheel>(touch.queue.drain());
    REQUIRE(wheels.size() == 1);
    CHECK(wheels[0].phase == ScrollPhase::MomentumEnded);
    CHECK(wheels[0].cancel);
    touch.gesture.poll(1.05);
    CHECK(Select<Queue::Wheel>(touch.queue.drain()).empty());
}
