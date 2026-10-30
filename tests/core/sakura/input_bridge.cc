#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "harness.hh"
#include "scene.hh"
#include "render/sakura/input.hh"
#include "render/sakura/detail/input_router.hh"
#include "domains/visualization/frame/module_impl.hh"
#include "domains/visualization/signal_view/waterfall_3d.hh"

#include <jetstream/render/base/input.hh>
#include <jetstream/render/sakura/base.hh>
#include <jetstream/render/sakura/clipboard.hh>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

using namespace Jetstream;

namespace {

using Queue = Render::InputQueue;

using RenderTest::Only;

struct FrameInputProbe : Modules::FrameImpl {
    Result validate() override { return Result::SUCCESS; }
    Result define() override { return Result::SUCCESS; }
    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    Result reconfigure() override { return Result::SUCCESS; }
    using Modules::FrameImpl::processInputEvents;
    using Modules::FrameImpl::view;
};

template<typename Tag, typename Tag::Type member>
struct InputMemberAccess {
    friend typename Tag::Type InputMember(Tag) { return member; }
};

struct WaterfallInputMember {
    using Type = void (Modules::SignalViewWaterfall3D::*)(std::vector<InputEvent>&&);
    friend Type InputMember(WaterfallInputMember);
};

struct WaterfallCameraMember {
    using Type = Modules::detail::Waterfall3DCamera Modules::SignalViewWaterfall3D::*;
    friend Type InputMember(WaterfallCameraMember);
};

template struct InputMemberAccess<WaterfallInputMember, &Modules::SignalViewWaterfall3D::processInputEvents>;
template struct InputMemberAccess<WaterfallCameraMember, &Modules::SignalViewWaterfall3D::cameraGoal>;

struct Feed {
    SakuraTest::FakeWindow window;
    const Render::InputState& state = window.input();

    void macOS(bool enabled) {
        window.viewport.macOS = enabled;
    }

    void frame(const std::vector<Queue::Event>& events) {
        REQUIRE(window.inject(events) == Result::SUCCESS);
    }

    SurfaceCursor cursor() {
        return window.viewport.platform().cursor;
    }

    std::vector<Sakura::FrameEvent> settle(const std::vector<Queue::Event>& events, Sakura::InputFrame& frame) {
        std::vector<Sakura::FrameEvent> collected;
        this->frame(events);
        while (true) {
            Sakura::Private::SampleInput(frame, state);
            collected.insert(collected.end(), frame.events.begin(), frame.events.end());
            if (!state.hasPending()) {
                break;
            }
            this->frame({});
        }
        return collected;
    }
};

struct Routed {
    std::vector<Sakura::FrameEvent> events;
    Sakura::InputRouter::Output last;
};

Routed Route(Feed& feed, Sakura::InputFrame& frame, Sakura::InputRouter& router,
             const std::vector<Queue::Event>& events, const Jetstream::Rect& screen) {
    Routed routed;
    feed.frame(events);
    while (true) {
        Sakura::Private::SampleInput(frame, feed.state);
        routed.last = router.route(frame, screen);
        routed.events.insert(routed.events.end(), routed.last.events.begin(), routed.last.events.end());
        if (!feed.state.hasPending()) {
            break;
        }
        feed.frame({});
    }
    return routed;
}

}  // namespace

TEST_CASE("The input bridge keeps native Control and Command names and modifiers on every platform",
          "[core][sakura][input][bridge]") {
    const bool macOS = GENERATE(false, true);
    Feed feed;
    feed.macOS(macOS);
    Sakura::InputFrame frame;

    feed.frame({
        Queue::Modifiers{{.control = true}},
        Queue::Key{KeyCode::LeftCtrl, true},
        Queue::Modifiers{{.control = true, .super = true}},
        Queue::Key{KeyCode::RightSuper, true},
        Queue::Modifiers{{.control = true, .shift = true, .alt = true, .super = true}},
    });
    Sakura::Private::SampleInput(frame, feed.state);

    const auto keys = Only<KeyEvent>(frame.events);
    REQUIRE(keys.size() == 2);
    CHECK(keys[0].key == KeyCode::LeftCtrl);
    CHECK(keys[0].modifiers.control);
    CHECK_FALSE(keys[0].modifiers.super);
    CHECK(keys[1].key == KeyCode::RightSuper);
    CHECK(keys[1].modifiers.control);
    CHECK(keys[1].modifiers.super);
    CHECK(frame.modifiers.control);
    CHECK(frame.modifiers.shift);
    CHECK(frame.modifiers.alt);
    CHECK(frame.modifiers.super);

    feed.frame({Queue::Wheel{{-2.0f, 1.0f}}});
    Sakura::Private::SampleInput(frame, feed.state);
    const auto mouse = Only<MouseEvent>(frame.events);
    REQUIRE(mouse.size() == 1);
    CHECK(mouse[0].type == MouseEventType::Scroll);
    CHECK(mouse[0].scroll == Extent2D<F32>{-2.0f, 1.0f});
    CHECK(mouse[0].modifiers.control);
    CHECK(mouse[0].modifiers.shift);
    CHECK(mouse[0].modifiers.alt);
    CHECK(mouse[0].modifiers.super);
}

TEST_CASE("The input bridge samples ordered input with native modifiers once per frame",
          "[core][sakura][input][bridge]") {
    Feed feed;
    feed.macOS(true);
    Sakura::Context ctx;
    ctx.render = &feed.window;
    Sakura::InputFrame frame;

    const auto pressed = feed.settle({
        Queue::Modifiers{{.super = true}},
        Queue::Key{KeyCode::LeftSuper, true},
        Queue::Key{KeyCode::S, true},
        Queue::Text{U'x'},
        Queue::Wheel{{0.0f, 1.0f}},
        Queue::Pointer{Extent2D<F32>{30.0f, 40.0f}},
        Queue::Button{MouseButton::Left, true},
    }, frame);
    Sakura::Private::ApplyCursor(ctx, SurfaceCursor::Hand);
    CHECK(feed.cursor() == SurfaceCursor::Default);

    const U64 serial = frame.serial;
    CHECK(serial > 1);
    CHECK(frame.modifiers.super);
    CHECK_FALSE(frame.modifiers.control);
    CHECK(frame.pointer == Extent2D<F32>{30.0f, 40.0f});
    CHECK(frame.buttons[0]);

    auto mouse = Only<MouseEvent>(pressed);
    std::erase_if(mouse, [](const MouseEvent& event) { return event.type == MouseEventType::Move; });
    REQUIRE(mouse.size() == 2);
    const auto click = std::find_if(mouse.begin(), mouse.end(), [](const auto& event) {
        return event.type == MouseEventType::Click;
    });
    const auto scroll = std::find_if(mouse.begin(), mouse.end(), [](const auto& event) {
        return event.type == MouseEventType::Scroll;
    });
    REQUIRE(click != mouse.end());
    REQUIRE(scroll != mouse.end());
    CHECK(click->button == MouseButton::Left);
    CHECK(click->clicks == 1);
    CHECK(click->position == Extent2D<F32>{30.0f, 40.0f});
    CHECK(click->modifiers.super);
    CHECK(scroll->scroll == Extent2D<F32>{0.0f, 1.0f});

    const auto keys = Only<KeyEvent>(pressed);
    REQUIRE(keys.size() == 2);
    CHECK(keys[0].key == KeyCode::LeftSuper);
    CHECK(keys[1].key == KeyCode::S);
    CHECK(keys[1].type == KeyEventType::Press);
    CHECK(keys[1].modifiers.super);
    CHECK_FALSE(keys[1].repeat);

    const auto text = Only<TextEvent>(pressed);
    REQUIRE(text.size() == 1);
    CHECK(text[0].codepoint == U'x');

    const auto released = feed.settle({
        Queue::Key{KeyCode::S, false},
        Queue::Key{KeyCode::LeftSuper, false},
        Queue::Modifiers{},
        Queue::Button{MouseButton::Left, false},
    }, frame);
    CHECK(feed.cursor() == SurfaceCursor::Hand);
    Sakura::Private::ApplyCursor(ctx, SurfaceCursor::Default);

    CHECK(frame.serial > serial);
    CHECK_FALSE(frame.modifiers.super);
    CHECK_FALSE(frame.buttons[0]);
    const auto releases = Only<MouseEvent>(released);
    REQUIRE(releases.size() == 1);
    CHECK(releases[0].type == MouseEventType::Release);
    CHECK(Only<KeyEvent>(released).size() == 2);
    CHECK(Only<TextEvent>(released).empty());

    feed.frame({});
    CHECK(feed.cursor() == SurfaceCursor::Default);
}

TEST_CASE("The input bridge keeps each wheel delta, key and character in order with the modifiers held at the time",
          "[core][sakura][input][bridge]") {
    Feed feed;
    Sakura::InputFrame frame;

    const auto events = feed.settle({
        Queue::Wheel{{0.0f, 1.0f}},
        Queue::Modifiers{{.shift = true}},
        Queue::Wheel{{0.0f, -1.0f}},
        Queue::Text{U'a'},
        Queue::Key{KeyCode::B, true},
        Queue::Modifiers{},
        Queue::Key{KeyCode::B, false},
    }, frame);

    REQUIRE(events.size() == 5);
    const auto* first = std::get_if<MouseEvent>(&events[0]);
    const auto* second = std::get_if<MouseEvent>(&events[1]);
    const auto* text = std::get_if<TextEvent>(&events[2]);
    const auto* press = std::get_if<KeyEvent>(&events[3]);
    const auto* release = std::get_if<KeyEvent>(&events[4]);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(text);
    REQUIRE(press);
    REQUIRE(release);
    CHECK(first->scroll == Extent2D<F32>{0.0f, 1.0f});
    CHECK_FALSE(first->modifiers.shift);
    CHECK(second->scroll == Extent2D<F32>{0.0f, -1.0f});
    CHECK(second->modifiers.shift);
    CHECK(text->codepoint == U'a');
    CHECK(text->modifiers.shift);
    CHECK(press->key == KeyCode::B);
    CHECK(press->type == KeyEventType::Press);
    CHECK(press->modifiers.shift);
    CHECK(release->type == KeyEventType::Release);
    CHECK_FALSE(release->modifiers.shift);
    CHECK_FALSE(frame.modifiers.shift);
}

TEST_CASE("The input bridge samples a frame once however often it is asked",
          "[core][sakura][input][bridge]") {
    Feed feed;
    Sakura::InputFrame frame;

    feed.frame({Queue::Text{U'x'}});
    Sakura::Private::SampleInput(frame, feed.state);
    Sakura::Private::SampleInput(frame, feed.state);
    CHECK(frame.serial == 1);
    CHECK(Only<TextEvent>(frame.events).size() == 1);
    CHECK(frame.time == feed.state.time());
    CHECK(frame.deltaTime == feed.state.deltaTime());

    Sakura::InputRouter router;
    const auto routed = router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f});
    CHECK(Only<TextEvent>(routed.events).size() == 1);
    CHECK(router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f}).events.empty());

    feed.frame({});
    Sakura::Private::SampleInput(frame, feed.state);
    CHECK(frame.serial == 2);
    CHECK(frame.events.empty());
}

TEST_CASE("The input bridge reports focus changes and stays inert while unfocused",
          "[core][sakura][input][bridge]") {
    Feed feed;
    Sakura::InputFrame frame;

    feed.frame({
        Queue::Pointer{Extent2D<F32>{10.0f, 10.0f}},
        Queue::Button{MouseButton::Left, true},
        Queue::Modifiers{{.shift = true}},
    });
    Sakura::Private::SampleInput(frame, feed.state);
    CHECK(frame.windowFocused);
    CHECK(frame.buttons[0]);

    feed.frame({Queue::Focus{false}});
    Sakura::Private::SampleInput(frame, feed.state);
    CHECK_FALSE(frame.windowFocused);
    CHECK_FALSE(frame.buttons[0]);
    CHECK_FALSE(frame.pointer.has_value());
    CHECK_FALSE(frame.modifiers.shift);
    const auto lost = Only<FocusEvent>(frame.events);
    REQUIRE(lost.size() == 1);
    CHECK_FALSE(lost[0].focused);

    feed.frame({Queue::Pointer{Extent2D<F32>{20.0f, 20.0f}}, Queue::Button{MouseButton::Left, true}});
    Sakura::Private::SampleInput(frame, feed.state);
    CHECK_FALSE(frame.windowFocused);
    CHECK_FALSE(frame.pointer.has_value());
    CHECK(frame.events.empty());

    feed.frame({Queue::Focus{true}});
    Sakura::Private::SampleInput(frame, feed.state);
    CHECK(frame.windowFocused);
    const auto regained = Only<FocusEvent>(frame.events);
    REQUIRE(regained.size() == 1);
    CHECK(regained[0].focused);
}

TEST_CASE("Sakura clipboard follows the window viewport and is empty without one",
          "[core][sakura][input][bridge]") {
    SakuraTest::FakeWindow window;
    Sakura::SetClipboardText(&window, "alpha");
    CHECK(window.viewport.platform().clipboard == "alpha");
    window.viewport.platform().clipboard = "system";
    CHECK(Sakura::GetClipboardText(&window) == "system");

    window.attached = false;
    CHECK(Sakura::GetClipboardText(&window).empty());
    Sakura::SetClipboardText(&window, "beta");
    CHECK(Sakura::GetClipboardText(&window).empty());
    CHECK(window.viewport.platform().clipboard == "system");

    Sakura::SetClipboardText(nullptr, "ignored");
    CHECK(Sakura::GetClipboardText(nullptr).empty());
}

TEST_CASE("Input routing validates event types and button indices before accessing capture",
          "[core][sakura][surface][bridge]") {
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    frame.pointer = Extent2D<F32>{25.0f, 25.0f};
    for (const auto type : {MouseEventType::Click, MouseEventType::Release, MouseEventType::Move,
                            MouseEventType::Scroll, MouseEventType::Enter, MouseEventType::Leave,
                            static_cast<MouseEventType>(255)}) {
        frame.advance(0.1f);
        frame.events = {MouseEvent{.type = type, .button = static_cast<MouseButton>(255),
                                  .position = {25.0f, 25.0f}}};
        const auto output = router.route(frame, {0.0f, 0.0f, 100.0f, 100.0f});
        CHECK_FALSE(output.active);
        CHECK_FALSE(output.middle);
        for (const auto& mouse : Only<MouseEvent>(output.events)) {
            CHECK(mouse.type != MouseEventType::Click);
            CHECK(mouse.type != MouseEventType::Release);
        }
    }
}

TEST_CASE("All cancellation boundaries balance three buttons and keys without unfocused actions",
          "[core][sakura][surface][bridge]") {
    const int boundary = GENERATE(0, 1, 2, 3, 4, 5, 6);
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    Jetstream::Rect rect{0.0f, 0.0f, 100.0f, 100.0f};
    frame.advance(0.1f);
    frame.pointer = Extent2D<F32>{25.0f, 25.0f};
    frame.buttons.fill(true);
    frame.events = {KeyEvent{KeyEventType::Press, KeyCode::A, {}},
                    MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Left, .position = {25, 25}},
                    MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right, .position = {25, 25}},
                    MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Middle, .position = {25, 25}}};
    const auto pressed = router.route(frame, rect);
    REQUIRE(pressed.active);
    REQUIRE(pressed.middle);
    frame.advance(0.1f);
    if (boundary == 0) frame.windowFocused = false;
    if (boundary == 1) frame.events.push_back(MouseEvent{.type = MouseEventType::Cancel});
    if (boundary == 2) rect.width = 0.0f;
    if (boundary == 3) rect.x = std::numeric_limits<F32>::quiet_NaN();
    if (boundary == 4) rect.height = std::numeric_limits<F32>::infinity();
    if (boundary == 5) frame.buttons[2] = false;
    if (boundary == 6) frame.events.push_back(FocusEvent{false});
    const auto cancelled = router.route(frame, rect);
    CHECK_FALSE(cancelled.active);
    CHECK_FALSE(cancelled.middle);
    CHECK_FALSE(cancelled.hovered);
    const auto keys = Only<KeyEvent>(cancelled.events);
    REQUIRE(keys.size() == 1);
    CHECK(keys[0].type == KeyEventType::Release);
    const auto mouse = Only<MouseEvent>(cancelled.events);
    REQUIRE(mouse.size() == 4);
    for (U64 index = 0; index < 3; ++index) {
        CHECK(mouse[index].type == MouseEventType::Release);
        CHECK(mouse[index].button == static_cast<MouseButton>(index));
        CHECK(mouse[index].cancel);
        REQUIRE(SurfaceMouseEvent(InputEvent{mouse[index]}));
        CHECK(SurfaceMouseEvent(InputEvent{mouse[index]})->type == MouseEventType::Cancel);
    }
    CHECK(mouse[3].type == MouseEventType::Cancel);
    frame.advance(0.1f);
    frame.windowFocused = false;
    frame.events = {MouseEvent{.type = MouseEventType::Click, .position = {25, 25}},
                    MouseEvent{.type = MouseEventType::Scroll, .position = {25, 25}, .scroll = {0, 1}},
                    KeyEvent{KeyEventType::Press, KeyCode::A, {}}, TextEvent{U'x', {}}};
    const auto unfocused = router.route(frame, {0, 0, 100, 100});
    CHECK(unfocused.events.empty());
    CHECK_FALSE(unfocused.active);
}

TEST_CASE("Input routing identifies manual frames and explicit source lifetimes independently of serial",
          "[core][sakura][surface][bridge]") {
    const bool explicitSource = GENERATE(false, true);
    Sakura::InputFrame first;
    Sakura::InputFrame second;
    if (explicitSource) {
        first.source = &first;
        second.source = &second;
    }
    first.advance(0.1f);
    second.advance(0.1f);
    first.pointer = Extent2D<F32>{25, 25};
    first.buttons.fill(true);
    first.events = {TextEvent{U'a', {}}, KeyEvent{KeyEventType::Press, KeyCode::A, {}},
                    MouseEvent{.type = MouseEventType::Click, .position = {25, 25}}};
    second.events = {TextEvent{U'b', {}}};
    Sakura::InputRouter router;
    const Jetstream::Rect rect{0, 0, 100, 100};
    CHECK(Only<TextEvent>(router.route(first, rect).events).size() == 1);
    CHECK(router.route(first, rect).events.empty());
    const auto transitioned = router.route(second, rect);
    REQUIRE(Only<TextEvent>(transitioned.events).size() == 1);
    CHECK(Only<TextEvent>(transitioned.events)[0].codepoint == U'b');
    CHECK_FALSE(transitioned.active);
    REQUIRE(Only<KeyEvent>(transitioned.events).size() == 1);
    CHECK(Only<KeyEvent>(transitioned.events)[0].type == KeyEventType::Release);
    REQUIRE(Only<MouseEvent>(transitioned.events).size() == 2);
    CHECK(Only<MouseEvent>(transitioned.events)[0].cancel);
    CHECK(router.route(second, rect).events.empty());
    ++second.sourceGeneration;
    CHECK(Only<TextEvent>(router.route(second, rect).events).size() == 1);
}

TEST_CASE("Cancelled releases leave frame zoom and the waterfall camera unchanged",
          "[core][sakura][surface][bridge][consumer-cancel]") {
    FrameInputProbe frame;
    frame.view.zoom = 4.0f;
    frame.view.center = {0.4f, 0.6f};
    frame.processInputEvents({MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right,
                                          .position = {0.25f, 0.25f}}});
    REQUIRE(frame.view.selecting);
    frame.processInputEvents({MouseEvent{.type = MouseEventType::Release, .button = MouseButton::Right,
                                          .position = {0.25f, 0.25f}, .cancel = true}});
    CHECK_FALSE(frame.view.selecting);
    CHECK(frame.view.zoom == 4.0f);
    CHECK(frame.view.center == Extent2D<F32>{0.4f, 0.6f});

    SurfaceInteractionState interaction;
    interaction.zoom = 4.0f;
    interaction.offset = 0.2f;
    interaction.dragging = true;
    interaction = ProcessSurfaceInteraction(interaction, {}, {
        MouseEvent{.type = MouseEventType::Release, .button = MouseButton::Right, .cancel = true}});
    CHECK_FALSE(interaction.dragging);
    CHECK(interaction.zoom == 4.0f);
    CHECK(interaction.offset == 0.2f);

    Modules::SignalViewWaterfall3D waterfall;
    auto& camera = waterfall.*InputMember(WaterfallCameraMember{});
    camera.azimuth = 1.2f;
    camera.distance = 2.5f;
    camera.target = {0.2f, 0.3f, 0.4f};
    const auto before = camera;
    const auto process = InputMember(WaterfallInputMember{});
    (waterfall.*process)({MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right,
                                    .position = {0.25f, 0.25f}}});
    (waterfall.*process)({MouseEvent{.type = MouseEventType::Release, .button = MouseButton::Right,
                                    .position = {0.25f, 0.25f}, .cancel = true}});
    (waterfall.*process)({MouseEvent{.type = MouseEventType::Move, .position = {0.75f, 0.75f}}});
    CHECK(camera.azimuth == before.azimuth);
    CHECK(camera.elevation == before.elevation);
    CHECK(camera.distance == before.distance);
    CHECK(camera.target == before.target);

    Modules::detail::SignalViewSplitInteraction splitter;
    splitter.dragging = true;
    bool commit = true;
    const auto layout = Modules::detail::CalculateSignalViewPanels({1, 1}, {100, 100}, 0.5f);
    CHECK(splitter.process(MouseEvent{.type = MouseEventType::Release, .cancel = true},
                            layout, {100, 100}, 1.0f, true, commit));
    CHECK_FALSE(splitter.dragging);
    CHECK_FALSE(commit);
}

TEST_CASE("SignalView marker clearing followed by a cancelled left release preserves the clear",
          "[core][sakura][surface][bridge][signal-cancel]") {
    const bool enabled = GENERATE(false, true);
    Modules::SignalView config;
    config.markers = {0.0f};
    Modules::SignalViewCanvas canvas;
    canvas.reset(config);
    canvas.axis = std::make_shared<Render::Components::Axis>(Render::Components::Axis::Config{});
    const Modules::SignalViewCanvas::Context context{config, {}, 256};
    U64 requests = 0;
    const Modules::SignalViewCanvas::Edits edits{
        .pending = [&] { return requests > 0; },
        .enabled = [&](const std::string&) { return enabled; },
        .request = [&](const Parser::Map&) { ++requests; return Result::SUCCESS; },
    };
    canvas.processInputEvents({MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Left,
                                         .position = {0.5f, 0.25f}}}, context, edits);
    REQUIRE(canvas.markerDrag.index == 0);
    canvas.processInputEvents({
        MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right,
                   .position = {0.5f, 0.25f}, .modifiers = {.shift = true}},
        MouseEvent{.type = MouseEventType::Release, .button = MouseButton::Left,
                   .position = {0.5f, 0.25f}, .cancel = true},
        MouseEvent{.type = MouseEventType::Move, .position = {0.5f, 0.25f}},
    }, context, edits);
    CHECK(canvas.markerPositions.empty());
    CHECK_FALSE(canvas.markerDrag.index);
    CHECK(config.markers == std::vector<F32>{0.0f});
    CHECK(requests == (enabled ? 1 : 0));
}

TEST_CASE("Retained scrollbar cancellation stops thumb dragging without another scroll callback",
          "[core][sakura][input][widget-cancel]") {
    const bool horizontal = GENERATE(false, true);
    SakuraTest::Scene scene;
    F32 value = 0.0f;
    U64 changes = 0;
    Sakura::Scroller scroll;
    scroll.update({.contentWidth = horizontal ? 1000.0f : 200.0f,
                   .contentHeight = horizontal ? 200.0f : 1000.0f, .thickness = 10, .margin = 0,
                   .onScrollY = [&](F32 next) { value = next; ++changes; },
                   .onScrollX = [&](F32 next) { value = next; ++changes; }});
    SakuraTest::Host host(scroll);
    host.place(scene.ctx, {0, 0, 200, 200});
    const Extent2D<F32> start = horizontal ? Extent2D<F32>{25, 195} : Extent2D<F32>{195, 25};
    const Extent2D<F32> drag = horizontal ? Extent2D<F32>{70, 195} : Extent2D<F32>{195, 70};
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Click, .position = start}));
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Move, .position = drag}));
    REQUIRE(value > 0.0f);
    const F32 before = value;
    const U64 callbacks = changes;
    host.send(MouseEvent{.type = MouseEventType::Cancel});
    host.send(MouseEvent{.type = MouseEventType::Move, .position = {180, 180}});
    host.send(MouseEvent{.type = MouseEventType::Release, .position = {180, 180}});
    CHECK(value == before);
    CHECK(changes == callbacks);
}

TEST_CASE("Retained text selection cancellation does not edit or resume on later motion",
          "[core][sakura][input][widget-cancel]") {
    SakuraTest::Scene scene;
    U64 edits = 0;
    U64 selections = 0;
    Sakura::TextGrid grid;
    grid.update({.value = "abcdefghijklmnopqrstuvwxyz", .editable = true,
                 .fontSize = 15, .lineNumbers = false, .scrollbar = false, .padding = Sakura::Padding{},
                 .onChange = [&](const auto&) { ++edits; },
                 .onSelect = [&](auto, auto) { ++selections; }});
    SakuraTest::Host host(grid);
    host.place(scene.ctx, {0, 0, 256, 100});
    host.send(MouseEvent{.type = MouseEventType::Click, .position = {25, 30}});
    host.send(MouseEvent{.type = MouseEventType::Move, .position = {80, 30}});
    const auto cursor = grid.cursor();
    REQUIRE(cursor.column > 0);
    const U64 callbacks = selections;
    host.send(MouseEvent{.type = MouseEventType::Cancel});
    host.send(MouseEvent{.type = MouseEventType::Move, .position = {170, 30}});
    host.send(MouseEvent{.type = MouseEventType::Release, .position = {170, 30}});
    CHECK(grid.cursor() == cursor);
    CHECK(edits == 0);
    CHECK(selections == callbacks);
}

TEST_CASE("Retained console resize cancellation discards its preview and ignores later release",
          "[core][sakura][input][widget-cancel]") {
    SakuraTest::Scene scene;
    U64 changes = 0;
    Sakura::CodeEditor editor;
    editor.update({.height = 256, .consoleVisible = true, .onChange = [&](const auto&) { ++changes; }});
    SakuraTest::Host host(editor);
    const auto layout = [&] { host.place(scene.ctx, {0, 0, 256, 256}); };
    layout();
    const auto bars = SakuraTest::Find<Sakura::StatusBar>(editor);
    REQUIRE(bars.size() == 1);
    const auto bar = SakuraTest::Peek::Frame(*bars.front());
    const Extent2D<F32> toggle{bar.right() - 15.0f, bar.y + bar.height * 0.5f};
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Click, .position = toggle}));
    host.send(MouseEvent{.type = MouseEventType::Release, .position = toggle});
    layout();
    const auto consoles = SakuraTest::Find<Sakura::TextView>(editor);
    REQUIRE(consoles.size() == 1);
    const auto before = SakuraTest::Peek::Frame(*consoles.front());
    const Extent2D<F32> start{128, before.y - 6.0f};
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Click, .position = start}));
    REQUIRE(host.send(MouseEvent{.type = MouseEventType::Move, .position = {128, start.y + 30}}));
    layout();
    CHECK(SakuraTest::Peek::Frame(*consoles.front()) != before);
    host.send(MouseEvent{.type = MouseEventType::Cancel});
    layout();
    CHECK(SakuraTest::Peek::Frame(*consoles.front()) == before);
    host.send(MouseEvent{.type = MouseEventType::Release, .position = {128, start.y + 30}});
    layout();
    CHECK(SakuraTest::Peek::Frame(*consoles.front()) == before);
    CHECK(changes == 0);
}

TEST_CASE("Queued pointer positions retain their order through sampling and routing",
          "[core][sakura][surface][bridge]") {
    Feed feed;
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    const Jetstream::Rect screen = {0.0f, 0.0f, 100.0f, 100.0f};
    const auto pressed = Route(feed, frame, router, {
        Queue::Pointer{Extent2D<F32>{25.0f, 25.0f}},
        Queue::Button{MouseButton::Left, true},
        Queue::Key{KeyCode::A, true},
        Queue::Text{'x'},
        Queue::Wheel{{0.0f, 1.0f}},
        Queue::Pointer{Extent2D<F32>{250.0f, 25.0f}},
        Queue::Wheel{{0.0f, 2.0f}},
    }, screen);
    CHECK(pressed.last.active);
    CHECK_FALSE(pressed.last.hovered);
    CHECK_FALSE(pressed.last.outsideClick);
    REQUIRE(pressed.events.size() == 6);
    CHECK(std::get<MouseEvent>(pressed.events[0]).type == MouseEventType::Move);
    CHECK(std::get<MouseEvent>(pressed.events[1]).type == MouseEventType::Click);
    CHECK(std::get<MouseEvent>(pressed.events[1]).position == Extent2D<F32>{0.25f, 0.25f});
    CHECK(std::holds_alternative<KeyEvent>(pressed.events[2]));
    CHECK(std::holds_alternative<TextEvent>(pressed.events[3]));
    CHECK(std::get<MouseEvent>(pressed.events[4]).type == MouseEventType::Scroll);
    CHECK(std::get<MouseEvent>(pressed.events[4]).position == Extent2D<F32>{0.25f, 0.25f});
    CHECK(std::get<MouseEvent>(pressed.events[5]).type == MouseEventType::Move);
    CHECK(std::get<MouseEvent>(pressed.events[5]).position == Extent2D<F32>{2.5f, 0.25f});

    const auto released = Route(feed, frame, router, {
        Queue::Key{KeyCode::A, false},
        Queue::Pointer{Extent2D<F32>{300.0f, 50.0f}},
        Queue::Button{MouseButton::Left, false},
    }, screen);
    CHECK_FALSE(released.last.active);
    const auto mouse = Only<MouseEvent>(released.events);
    REQUIRE(mouse.size() == 2);
    CHECK(mouse[0].type == MouseEventType::Move);
    CHECK(mouse[1].type == MouseEventType::Release);
    CHECK(mouse[1].position == Extent2D<F32>{3.0f, 0.5f});
    CHECK_FALSE(mouse[1].cancel);
}

TEST_CASE("Pointer leave and reentry preserve a subsequent click",
          "[core][sakura][surface][bridge][pointer-reentry]") {
    Feed feed;
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    const Jetstream::Rect screen = {0.0f, 0.0f, 100.0f, 100.0f};
    Route(feed, frame, router, {Queue::Hover{true}, Queue::Pointer{Extent2D<F32>{25.0f, 25.0f}}}, screen);
    const auto routed = Route(feed, frame, router, {
        Queue::Hover{false},
        Queue::Pointer{Extent2D<F32>{25.0f, 25.0f}},
        Queue::Button{MouseButton::Left, true},
    }, screen);
    CHECK(routed.last.active);
    CHECK(routed.last.hovered);
    const auto mouse = Only<MouseEvent>(routed.events);
    REQUIRE_FALSE(mouse.empty());
    CHECK(mouse.back().type == MouseEventType::Click);
    CHECK(mouse.back().position == Extent2D<F32>{0.25f, 0.25f});
}

TEST_CASE("Cancellation preserves later key text and pointer events in the same batch",
          "[core][sakura][surface][bridge][cancel-batch]") {
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    frame.advance(0.1f);
    frame.pointer = Extent2D<F32>{25, 25};
    frame.buttons[0] = true;
    frame.events = {KeyEvent{KeyEventType::Press, KeyCode::A, {}},
                    MouseEvent{.type = MouseEventType::Click, .position = {25, 25}}};
    REQUIRE(router.route(frame, {0, 0, 100, 100}).active);
    frame.advance(0.1f);
    frame.buttons[0] = false;
    frame.buttons[1] = true;
    frame.events = {MouseEvent{.type = MouseEventType::Cancel},
                    KeyEvent{KeyEventType::Press, KeyCode::B, {}}, TextEvent{U'x', {}},
                    MouseEvent{.type = MouseEventType::Move, .position = {25, 25}},
                    MouseEvent{.type = MouseEventType::Click, .button = MouseButton::Right, .position = {25, 25}}};
    const auto output = router.route(frame, {0, 0, 100, 100});
    CHECK(output.active);
    REQUIRE(output.events.size() == 7);
    CHECK(std::get<KeyEvent>(output.events[0]).type == KeyEventType::Release);
    CHECK(std::get<MouseEvent>(output.events[1]).cancel);
    CHECK(std::get<MouseEvent>(output.events[2]).type == MouseEventType::Cancel);
    CHECK(std::get<KeyEvent>(output.events[3]).key == KeyCode::B);
    CHECK(std::get<KeyEvent>(output.events[3]).type == KeyEventType::Press);
    CHECK(std::get<TextEvent>(output.events[4]).codepoint == U'x');
    CHECK(std::get<MouseEvent>(output.events[5]).type == MouseEventType::Move);
    CHECK(std::get<MouseEvent>(output.events[6]).button == MouseButton::Right);

    Feed feed;
    Sakura::InputFrame sampled;
    Sakura::InputRouter sampledRouter;
    const Jetstream::Rect screen = {0.0f, 0.0f, 100.0f, 100.0f};
    Route(feed, sampled, sampledRouter, {Queue::Hover{true}, Queue::Pointer{Extent2D<F32>{25.0f, 25.0f}}}, screen);
    const auto routed = Route(feed, sampled, sampledRouter, {
        Queue::Hover{false},
        Queue::Key{KeyCode::B, true},
        Queue::Text{'x'},
    }, screen);
    REQUIRE(routed.events.size() == 2);
    CHECK(std::get<KeyEvent>(routed.events[0]).type == KeyEventType::Press);
    CHECK(std::get<TextEvent>(routed.events[1]).codepoint == U'x');
    const auto released = Route(feed, sampled, sampledRouter, {Queue::Key{KeyCode::B, false}}, screen);
    REQUIRE(Only<KeyEvent>(released.events).size() == 1);
    CHECK(Only<KeyEvent>(released.events)[0].type == KeyEventType::Release);
}

TEST_CASE("Missing and nonfinite pointers cancel every owned button",
          "[core][sakura][surface][bridge][flat-invalid]") {
    const F32 invalid = GENERATE(-std::numeric_limits<F32>::max(),
                                 std::numeric_limits<F32>::quiet_NaN(),
                                 std::numeric_limits<F32>::infinity());
    const bool vertical = GENERATE(false, true);
    Feed feed;
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    const Jetstream::Rect screen = {0.0f, 0.0f, 100.0f, 100.0f};
    const auto pressed = Route(feed, frame, router, {
        Queue::Hover{true},
        Queue::Pointer{Extent2D<F32>{25.0f, 25.0f}},
        Queue::Button{MouseButton::Left, true},
        Queue::Button{MouseButton::Right, true},
        Queue::Button{MouseButton::Middle, true},
    }, screen);
    REQUIRE(pressed.last.active);
    REQUIRE(pressed.last.middle);

    const bool missing = invalid == -std::numeric_limits<F32>::max();
    const Queue::Event lost = missing ? Queue::Event{Queue::Pointer{}}
                                      : Queue::Event{Queue::Pointer{vertical ? Extent2D<F32>{25.0f, invalid}
                                                                             : Extent2D<F32>{invalid, 25.0f}}};
    const auto cancelled = Route(feed, frame, router, {lost}, screen);
    CHECK_FALSE(cancelled.last.active);
    CHECK_FALSE(cancelled.last.middle);
    CHECK_FALSE(cancelled.last.hovered);
    const auto mouse = Only<MouseEvent>(cancelled.events);
    REQUIRE(mouse.size() == 4);
    for (U64 index = 0; index < 3; ++index) {
        CHECK(mouse[index].type == MouseEventType::Release);
        CHECK(mouse[index].button == static_cast<MouseButton>(index));
        CHECK(mouse[index].cancel);
    }
    CHECK(mouse[3].type == MouseEventType::Cancel);
    const auto recovered = Route(feed, frame, router, {}, screen);
    CHECK_FALSE(recovered.last.active);
    CHECK_FALSE(recovered.last.middle);

    Sakura::InputFrame manual;
    Sakura::InputRouter manualRouter;
    manual.pointer = Extent2D<F32>{25, 25};
    manual.buttons[0] = true;
    manual.events = {MouseEvent{.type = MouseEventType::Click, .position = {25, 25}}};
    REQUIRE(manualRouter.route(manual, {0, 0, 100, 100}).active);
    manual.advance(0.1f);
    if (invalid == -std::numeric_limits<F32>::max()) manual.pointer.reset();
    else manual.pointer = vertical ? Extent2D<F32>{25, invalid} : Extent2D<F32>{invalid, 25};
    const auto manualCancelled = manualRouter.route(manual, {0, 0, 100, 100});
    CHECK_FALSE(manualCancelled.active);
    REQUIRE(Only<MouseEvent>(manualCancelled.events).size() == 2);
    CHECK(Only<MouseEvent>(manualCancelled.events)[0].cancel);
}

TEST_CASE("Sampling stamps the source generation and a missing source cancels held input",
          "[core][sakura][input][bridge]") {
    Feed feed;
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    const Jetstream::Rect screen = {0.0f, 0.0f, 100.0f, 100.0f};
    const auto pressed = Route(feed, frame, router, {
        Queue::Hover{true},
        Queue::Pointer{Extent2D<F32>{25.0f, 25.0f}},
        Queue::Button{MouseButton::Left, true},
    }, screen);
    REQUIRE(pressed.last.active);
    CHECK(frame.sourceGeneration == feed.state.generation());

    Sakura::Private::ResetInput(frame);
    CHECK(frame.source == nullptr);
    CHECK(frame.sourceGeneration == 0);
    CHECK_FALSE(frame.windowFocused);
    CHECK_FALSE(frame.pointer.has_value());
    CHECK(frame.buttons == std::array<bool, 3>{false, false, false});
    const auto cancelled = router.route(frame, screen);
    CHECK_FALSE(cancelled.active);
    const auto mouse = Only<MouseEvent>(cancelled.events);
    REQUIRE_FALSE(mouse.empty());
    CHECK(mouse.front().type == MouseEventType::Release);
    CHECK(mouse.front().cancel);

    Sakura::Private::SampleInput(frame, feed.state);
    CHECK(frame.source == &feed.state);
    CHECK(frame.sourceGeneration == feed.state.generation());
}
