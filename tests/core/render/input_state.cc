#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <jetstream/render/base/input.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace Jetstream;
using Catch::Matchers::WithinAbs;

namespace {

using Queue = Render::InputQueue;

template <typename Type>
std::vector<Type> Only(const std::vector<Render::InputState::Event>& events) {
    std::vector<Type> filtered;
    for (const auto& event : events) {
        if (const auto* typed = std::get_if<Type>(&event)) {
            filtered.push_back(*typed);
        }
    }
    return filtered;
}

struct Feed {
    Render::InputState state;
    F64 clock = 10.0;

    explicit Feed(bool macOS = false) : state(Render::InputState::Config{.macOS = macOS}) {}

    void frame(const std::vector<Queue::Event>& events, F64 step = 1.0 / 60.0) {
        clock += step;
        state.advance(clock, events);
    }

    std::vector<Render::InputState::Event> settle(const std::vector<Queue::Event>& events, U64* frames = nullptr) {
        std::vector<Render::InputState::Event> collected;
        U64 count = 0;
        frame(events);
        while (true) {
            ++count;
            collected.insert(collected.end(), state.events().begin(), state.events().end());
            if (!state.hasPending()) {
                break;
            }
            frame({});
        }
        if (frames) {
            *frames = count;
        }
        return collected;
    }
};

std::vector<MouseEvent> Buttons(const std::vector<Render::InputState::Event>& events) {
    auto mouse = Only<MouseEvent>(events);
    std::erase_if(mouse, [](const MouseEvent& event) {
        return event.type == MouseEventType::Move || event.type == MouseEventType::Leave;
    });
    return mouse;
}

std::vector<MouseEvent> Mouse(const Render::InputState& state) {
    return Buttons(state.events());
}

std::vector<KeyEvent> Keys(const Render::InputState& state) {
    return Only<KeyEvent>(state.events());
}

std::vector<char32_t> Text(const Render::InputState& state) {
    std::vector<char32_t> codepoints;
    for (const auto& text : Only<TextEvent>(state.events())) {
        codepoints.push_back(text.codepoint);
    }
    return codepoints;
}

U8 ClickAt(Feed& feed, Extent2D<F32> position, F64 step) {
    feed.frame({Queue::Pointer{position}, Queue::Button{MouseButton::Left, true}}, step);
    const auto events = Mouse(feed.state);
    REQUIRE(events.size() == 1);
    REQUIRE(events[0].type == MouseEventType::Click);
    const U8 clicks = events[0].clicks;
    feed.frame({Queue::Button{MouseButton::Left, false}});
    return clicks;
}

}  // namespace

TEST_CASE("Clicks within the double click time and distance count up", "[core][render][input_state]") {
    Feed feed;

    CHECK(ClickAt(feed, {100.0f, 100.0f}, 1.0) == 1);
    CHECK(ClickAt(feed, {102.0f, 103.0f}, 0.1) == 2);
    CHECK(ClickAt(feed, {101.0f, 101.0f}, 0.1) == 3);
    CHECK(ClickAt(feed, {101.0f, 101.0f}, 0.35) == 1);
    CHECK(ClickAt(feed, {110.0f, 101.0f}, 0.1) == 1);
    CHECK(ClickAt(feed, {115.0f, 101.0f}, 0.1) == 2);
}

TEST_CASE("A press and release inside one frame still reports every click in order", "[core][render][input_state]") {
    Feed feed;
    feed.frame({Queue::Pointer{Extent2D<F32>{20.0f, 30.0f}}});

    U64 frames = 0;
    const auto events = Buttons(feed.settle({
        Queue::Button{MouseButton::Left, true},
        Queue::Button{MouseButton::Left, false},
        Queue::Button{MouseButton::Left, true},
        Queue::Button{MouseButton::Left, false},
    }, &frames));

    CHECK(frames == 4);
    REQUIRE(events.size() == 4);
    CHECK(events[0].type == MouseEventType::Click);
    CHECK(events[0].clicks == 1);
    CHECK(events[0].position == Extent2D<F32>{20.0f, 30.0f});
    CHECK(events[1].type == MouseEventType::Release);
    CHECK(events[2].type == MouseEventType::Click);
    CHECK(events[2].clicks == 2);
    CHECK(events[3].type == MouseEventType::Release);
    CHECK_FALSE(feed.state.buttons()[0]);
}

TEST_CASE("Duplicate button and key transitions are ignored", "[core][render][input_state]") {
    Feed feed;
    const auto events = feed.settle({
        Queue::Button{MouseButton::Middle, true},
        Queue::Button{MouseButton::Middle, true},
        Queue::Key{KeyCode::A, true},
        Queue::Key{KeyCode::A, true},
        Queue::Key{KeyCode::Unknown, true},
        Queue::Wheel{{0.0f, 0.0f}},
    });

    const auto mouse = Buttons(events);
    REQUIRE(mouse.size() == 1);
    CHECK(mouse[0].button == MouseButton::Middle);
    CHECK(feed.state.buttons()[2]);
    CHECK(Only<KeyEvent>(events).size() == 1);
    CHECK(feed.state.keyDown(KeyCode::A));
    CHECK_FALSE(feed.state.keyDown(KeyCode::Unknown));
}

TEST_CASE("Held keys repeat after the delay at the repeat rate", "[core][render][input_state]") {
    Feed feed;
    feed.frame({Queue::Modifiers{{.shift = true}}, Queue::Key{KeyCode::Backspace, true}}, 0.02);
    REQUIRE(Keys(feed.state).size() == 1);
    CHECK_FALSE(Keys(feed.state)[0].repeat);

    std::vector<U64> repeats;
    for (U64 frame = 1; frame <= 25; ++frame) {
        feed.frame({}, 0.02);
        for (const auto& key : Keys(feed.state)) {
            CHECK(key.repeat);
            CHECK(key.type == KeyEventType::Press);
            CHECK(key.key == KeyCode::Backspace);
            CHECK(key.modifiers.shift);
            repeats.push_back(frame);
        }
    }
    CHECK(repeats == std::vector<U64>{14, 17, 19, 22, 24});

    feed.frame({Queue::Key{KeyCode::Backspace, false}}, 0.02);
    const auto released = Keys(feed.state);
    REQUIRE(released.size() == 1);
    CHECK(released[0].type == KeyEventType::Release);
    feed.frame({}, 0.5);
    CHECK(Keys(feed.state).empty());
}

TEST_CASE("Control and left click becomes a right click on macOS only", "[core][render][input_state]") {
    const bool macOS = GENERATE(false, true);
    Feed feed(macOS);

    feed.frame({Queue::Modifiers{{.control = true}}, Queue::Button{MouseButton::Left, true}});
    auto events = Mouse(feed.state);
    REQUIRE(events.size() == 1);
    CHECK(events[0].button == (macOS ? MouseButton::Right : MouseButton::Left));
    CHECK(events[0].modifiers.control);
    CHECK(feed.state.buttons()[0] == !macOS);
    CHECK(feed.state.buttons()[1] == macOS);

    feed.frame({Queue::Modifiers{}, Queue::Button{MouseButton::Left, false}});
    events = Mouse(feed.state);
    REQUIRE(events.size() == 1);
    CHECK(events[0].type == MouseEventType::Release);
    CHECK(events[0].button == (macOS ? MouseButton::Right : MouseButton::Left));
    CHECK_FALSE(feed.state.buttons()[0]);
    CHECK_FALSE(feed.state.buttons()[1]);

    feed.frame({Queue::Button{MouseButton::Left, true}});
    events = Mouse(feed.state);
    REQUIRE(events.size() == 1);
    CHECK(events[0].button == MouseButton::Left);

    events = Buttons(feed.settle({Queue::Modifiers{{.super = true}}, Queue::Button{MouseButton::Left, false},
                                  Queue::Button{MouseButton::Left, true}}));
    REQUIRE(events.size() == 2);
    CHECK(events[1].button == MouseButton::Left);
}

TEST_CASE("Invalid button indices are ignored before alias handling", "[core][render][input_state]") {
    const bool macOS = GENERATE(false, true);
    const auto invalid = GENERATE(static_cast<MouseButton>(3), static_cast<MouseButton>(255));
    Feed feed(macOS);
    feed.frame({Queue::Modifiers{{.control = true}}, Queue::Button{invalid, true}, Queue::Button{invalid, false},
                Queue::Button{MouseButton::Left, true}});
    auto clicks = Mouse(feed.state);
    REQUIRE(clicks.size() == 1);
    CHECK(clicks[0].button == (macOS ? MouseButton::Right : MouseButton::Left));
    CHECK_FALSE(feed.state.hasPending());

    feed.frame({Queue::Button{invalid, true}, Queue::Button{invalid, false}, Queue::Modifiers{},
                Queue::Button{MouseButton::Left, false}});
    clicks = Mouse(feed.state);
    REQUIRE(clicks.size() == 1);
    CHECK(clicks[0].type == MouseEventType::Release);
    CHECK(clicks[0].button == (macOS ? MouseButton::Right : MouseButton::Left));
    CHECK(feed.state.buttons() == std::array<bool, 3>{});
    CHECK_FALSE(feed.state.hasPending());
}

TEST_CASE("Losing focus releases held state and keeps the window inert until focus returns",
          "[core][render][input_state]") {
    Feed feed;
    feed.frame({
        Queue::Pointer{Extent2D<F32>{5.0f, 5.0f}},
        Queue::Modifiers{{.alt = true}},
        Queue::Key{KeyCode::LeftAlt, true},
        Queue::Button{MouseButton::Right, true},
    });
    CHECK(feed.state.windowFocused());
    CHECK(feed.state.keyDown(KeyCode::LeftAlt));

    feed.frame({Queue::Focus{false}});
    CHECK_FALSE(feed.state.windowFocused());
    const auto lost = Only<FocusEvent>(feed.state.events());
    REQUIRE(lost.size() == 1);
    CHECK_FALSE(lost[0].focused);
    CHECK(feed.state.events().size() == 1);
    CHECK_FALSE(feed.state.keyDown(KeyCode::LeftAlt));
    CHECK_FALSE(feed.state.buttons()[1]);
    CHECK_FALSE(feed.state.modifiers().alt);
    CHECK_FALSE(feed.state.pointer().has_value());

    for (U64 frame = 0; frame < 60; ++frame) {
        feed.frame({
            Queue::Pointer{Extent2D<F32>{20.0f + static_cast<F32>(frame), 30.0f}},
            Queue::Wheel{{0.0f, 1.0f}},
            Queue::Key{KeyCode::A, frame % 2 == 0},
            Queue::Text{'a'},
        });
        CHECK_FALSE(feed.state.windowFocused());
        CHECK(feed.state.events().empty());
        CHECK_FALSE(feed.state.pointer().has_value());
    }

    feed.frame({Queue::Focus{false}, Queue::Focus{true}});
    CHECK(feed.state.windowFocused());
    const auto regained = Only<FocusEvent>(feed.state.events());
    REQUIRE(regained.size() == 1);
    CHECK(regained[0].focused);
}

TEST_CASE("The click that refocuses the window lands where the pointer moved while inactive",
          "[core][render][input_state]") {
    Feed feed;
    feed.frame({Queue::Focus{false}});
    feed.frame({Queue::Pointer{Extent2D<F32>{40.0f, 50.0f}}});
    CHECK_FALSE(feed.state.pointer().has_value());

    feed.frame({Queue::Focus{true}, Queue::Button{MouseButton::Left, true}});
    REQUIRE(feed.state.pointer().has_value());
    CHECK(*feed.state.pointer() == Extent2D<F32>{40.0f, 50.0f});
    const auto clicks = Mouse(feed.state);
    REQUIRE(clicks.size() == 1);
    CHECK(clicks[0].type == MouseEventType::Click);
    CHECK(clicks[0].positioned);
    CHECK(clicks[0].position == Extent2D<F32>{40.0f, 50.0f});
}

TEST_CASE("Focus restores a cached pointer only while it is inside", "[core][render][input_state]") {
    const bool outside = GENERATE(false, true);
    Feed feed;
    feed.frame({Queue::Pointer{Extent2D<F32>{40.0f, 50.0f}}});
    feed.frame({Queue::Focus{false}});
    CHECK_FALSE(feed.state.pointer().has_value());
    if (outside) {
        feed.frame({Queue::Hover{false}});
    }

    feed.frame({Queue::Focus{true}, Queue::Button{MouseButton::Left, true}});
    CHECK(feed.state.pointer().has_value() == !outside);
    const auto clicks = Mouse(feed.state);
    REQUIRE(clicks.size() == 1);
    CHECK(clicks[0].positioned == !outside);
    if (!outside) {
        CHECK(clicks[0].position == Extent2D<F32>{40.0f, 50.0f});
    }

    feed.settle({Queue::Button{MouseButton::Left, false}, Queue::Hover{true}});
    REQUIRE(feed.state.pointer().has_value());
    CHECK(*feed.state.pointer() == Extent2D<F32>{40.0f, 50.0f});
}

TEST_CASE("Hover exit without an enter clears the pointer and preserves its cache", "[core][render][input_state]") {
    const bool focused = GENERATE(false, true);
    Feed feed;
    feed.frame({Queue::Pointer{Extent2D<F32>{12.0f, 34.0f}}});
    if (!focused) {
        feed.frame({Queue::Focus{false}});
    }
    feed.frame({Queue::Hover{false}});
    if (!focused) {
        feed.frame({Queue::Focus{true}});
    }
    CHECK_FALSE(feed.state.pointer().has_value());

    feed.frame({Queue::Hover{true}});
    REQUIRE(feed.state.pointer().has_value());
    CHECK(*feed.state.pointer() == Extent2D<F32>{12.0f, 34.0f});
    feed.frame({Queue::Pointer{}, Queue::Hover{false}, Queue::Hover{true}});
    REQUIRE(feed.state.pointer().has_value());
    CHECK(*feed.state.pointer() == Extent2D<F32>{12.0f, 34.0f});
}

TEST_CASE("Focus loss forgets click history and click counts saturate", "[core][render][input_state]") {
    Feed feed;
    CHECK(ClickAt(feed, {10.0f, 10.0f}, 0.01) == 1);
    CHECK(ClickAt(feed, {10.0f, 10.0f}, 0.01) == 2);
    feed.frame({Queue::Focus{false}});
    feed.frame({Queue::Focus{true}});
    CHECK(ClickAt(feed, {10.0f, 10.0f}, 0.01) == 1);

    U8 clicks = 0;
    for (U64 index = 0; index < 300; ++index) {
        clicks = ClickAt(feed, {10.0f, 10.0f}, 0.01);
    }
    CHECK(clicks == std::numeric_limits<U8>::max());
}

TEST_CASE("Losing and regaining focus in one frame releases held keys and buttons",
          "[core][render][input_state]") {
    Feed feed;
    feed.frame({
        Queue::Pointer{Extent2D<F32>{5.0f, 5.0f}},
        Queue::Key{KeyCode::B, true},
        Queue::Button{MouseButton::Left, true},
    });
    REQUIRE(feed.state.keyDown(KeyCode::B));
    REQUIRE(feed.state.buttons()[0]);

    feed.frame({Queue::Focus{false}, Queue::Key{KeyCode::B, false}, Queue::Focus{true}});
    CHECK(feed.state.windowFocused());
    CHECK_FALSE(feed.state.keyDown(KeyCode::B));
    CHECK_FALSE(feed.state.buttons()[0]);

    for (U64 frame = 0; frame < 60; ++frame) {
        feed.frame({});
        CHECK(Keys(feed.state).empty());
    }

    feed.frame({Queue::Pointer{Extent2D<F32>{5.0f, 5.0f}}, Queue::Button{MouseButton::Left, true}});
    const auto clicks = Mouse(feed.state);
    REQUIRE(clicks.size() == 1);
    CHECK(clicks[0].type == MouseEventType::Click);
    CHECK(clicks[0].clicks == 1);
}

TEST_CASE("Nonfinite pointer positions read as a missing pointer", "[core][render][input_state]") {
    const F32 invalid = GENERATE(std::numeric_limits<F32>::quiet_NaN(), std::numeric_limits<F32>::infinity());
    const bool focused = GENERATE(true, false);
    Feed feed;
    feed.frame({Queue::Hover{true}, Queue::Pointer{Extent2D<F32>{10.0f, 20.0f}}});
    if (!focused) {
        feed.frame({Queue::Focus{false}});
    }
    feed.frame({Queue::Pointer{Extent2D<F32>{invalid, 20.0f}}});
    CHECK_FALSE(feed.state.pointer().has_value());
    if (!focused) {
        feed.frame({Queue::Focus{true}});
    }
    feed.frame({Queue::Hover{false}});
    feed.frame({Queue::Hover{true}});
    if (feed.state.pointer().has_value()) {
        CHECK(std::isfinite(feed.state.pointer()->x));
        CHECK(std::isfinite(feed.state.pointer()->y));
    }
}

TEST_CASE("Every input state carries its own source generation", "[core][render][input_state]") {
    const Render::InputState first;
    const Render::InputState second;
    CHECK(first.generation() != 0);
    CHECK(second.generation() != 0);
    CHECK(first.generation() != second.generation());
}

TEST_CASE("Pointer positions floor and survive leaving and entering the window", "[core][render][input_state]") {
    Feed feed;
    feed.frame({Queue::Hover{true}, Queue::Pointer{Extent2D<F32>{10.7f, 20.2f}}});
    REQUIRE(feed.state.pointer().has_value());
    CHECK(*feed.state.pointer() == Extent2D<F32>{10.0f, 20.0f});

    feed.frame({Queue::Hover{false}});
    CHECK_FALSE(feed.state.pointer().has_value());

    feed.frame({Queue::Hover{true}});
    REQUIRE(feed.state.pointer().has_value());
    CHECK(*feed.state.pointer() == Extent2D<F32>{10.0f, 20.0f});

    feed.frame({Queue::Pointer{}});
    CHECK_FALSE(feed.state.pointer().has_value());

    feed.frame({Queue::Wheel{{1.0f, -1.0f}}});
    const auto events = Mouse(feed.state);
    REQUIRE(events.size() == 1);
    CHECK(events[0].position == Extent2D<F32>{0.0f, 0.0f});
}

TEST_CASE("UTF-8 text decodes into UTF-32 codepoints in order", "[core][render][input_state]") {
    Queue queue;
    queue.pushText("a\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80");
    queue.pushText("\xFF" "b" "\xE2\x82");
    queue.pushText(std::string{"c\0d", 3});
    queue.push(Queue::Text{0});
    queue.push(Queue::Text{0xD800});
    queue.push(Queue::Text{0xDFFF});
    queue.push(Queue::Text{0x110000});
    queue.push(Queue::Text{0x10FFFF});

    Feed feed;
    feed.frame(queue.drain());
    CHECK(Text(feed.state) == std::vector<char32_t>{U'a', 0xE9, 0x20AC, 0x1F600, 0xFFFD, U'b', 0xFFFD,
                                                  U'c', U'd', 0xFFFD, 0xFFFD, 0xFFFD, 0x10FFFF});
    CHECK(queue.drain().empty());

    CHECK(Queue::DecodeUtf8("\xC0\x80") == std::vector<char32_t>{0xFFFD});
    CHECK(Queue::DecodeUtf8("\xED\xA0\x80") == std::vector<char32_t>{0xFFFD});
    CHECK(Queue::DecodeUtf8("\xF4\x90\x80\x80") == std::vector<char32_t>{0xFFFD});
}

TEST_CASE("Delta time follows the frame clock", "[core][render][input_state]") {
    Render::InputState state;

    state.advance(5.0, {});
    CHECK_THAT(state.deltaTime(), WithinAbs(1.0 / 60.0, 1e-6));
    CHECK(state.frame() == 1);

    F64 clock = 5.0;
    for (U64 frame = 0; frame < 120; ++frame) {
        clock += 1.0 / 30.0;
        state.advance(clock, {});
    }
    CHECK_THAT(state.deltaTime(), WithinAbs(1.0 / 30.0, 1e-5));
    CHECK_THAT(state.time(), WithinAbs(1.0 / 60.0 + 120.0 / 30.0, 1e-3));

    for (U64 frame = 0; frame < 30; ++frame) {
        clock += 1.0 / 60.0;
        state.advance(clock, {});
    }
    CHECK_THAT(state.deltaTime(), WithinAbs(1.0 / 60.0, 1e-5));

    state.advance(clock - 1.0, {});
    CHECK(state.deltaTime() > 0.0f);
    CHECK(state.deltaTime() < 1e-4f);
    CHECK(state.frame() == 152);
}

TEST_CASE("Pointer moves are ordered events with their own position and modifiers", "[core][render][input_state]") {
    Feed feed;
    const auto all = Only<MouseEvent>(feed.settle({
        Queue::Pointer{Extent2D<F32>{10.4f, 10.6f}},
        Queue::Button{MouseButton::Left, true},
        Queue::Modifiers{{.shift = true}},
        Queue::Pointer{Extent2D<F32>{90.0f, 90.0f}},
        Queue::Pointer{Extent2D<F32>{90.2f, 90.7f}},
        Queue::Button{MouseButton::Left, false},
        Queue::Pointer{},
    }));
    REQUIRE(all.size() == 5);
    CHECK(all[4].type == MouseEventType::Leave);
    CHECK_FALSE(all[4].positioned);
    const std::vector<MouseEvent> events(all.begin(), all.begin() + 4);
    REQUIRE(events.size() == 4);
    CHECK(events[0].type == MouseEventType::Move);
    CHECK(events[0].position == Extent2D<F32>{10.0f, 10.0f});
    CHECK_FALSE(events[0].modifiers.shift);
    CHECK(events[1].type == MouseEventType::Click);
    CHECK(events[1].position == Extent2D<F32>{10.0f, 10.0f});
    CHECK(events[2].type == MouseEventType::Move);
    CHECK(events[2].position == Extent2D<F32>{90.0f, 90.0f});
    CHECK(events[2].modifiers.shift);
    CHECK(events[3].type == MouseEventType::Release);
    CHECK(events[3].position == Extent2D<F32>{90.0f, 90.0f});
    CHECK(events[3].modifiers.shift);
    CHECK_FALSE(feed.state.pointer().has_value());
}

TEST_CASE("Releasing and pressing a held key in one batch restarts the repeat delay", "[core][render][input_state]") {
    Feed feed;
    feed.frame({Queue::Key{KeyCode::Backspace, true}}, 0.02);
    U64 repeats = 0;
    for (int frame = 0; frame < 20; ++frame) {
        feed.frame({}, 0.02);
        repeats += Keys(feed.state).size();
    }
    REQUIRE(repeats > 0);

    feed.frame({Queue::Key{KeyCode::Backspace, false}, Queue::Key{KeyCode::Backspace, true}}, 0.02);
    auto keys = Keys(feed.state);
    REQUIRE(keys.size() == 1);
    CHECK(keys[0].type == KeyEventType::Release);
    CHECK(feed.state.hasPending());

    feed.frame({}, 0.02);
    keys = Keys(feed.state);
    REQUIRE(keys.size() == 1);
    CHECK(keys[0].type == KeyEventType::Press);
    CHECK_FALSE(keys[0].repeat);

    std::vector<U64> repeated;
    for (U64 frame = 1; frame <= 14; ++frame) {
        feed.frame({}, 0.02);
        if (!Keys(feed.state).empty()) {
            repeated.push_back(frame);
        }
    }
    CHECK(repeated == std::vector<U64>{14});
}
