#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/components/popup.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/sakura/components/scroll_area.hh>
#include <jetstream/render/sakura/components/surface_image.hh>

#include "harness.hh"
#include "scene.hh"
#include "render/sakura/detail/input_router.hh"
#include "render/sakura/detail/root_component.hh"
#include "render/sakura/detail/touch_scroll.hh"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

using namespace Jetstream;

namespace {

struct RegionLayer : Sakura::Layer {
    Sakura::Component* content;
    Rect region;

    RegionLayer(Sakura::Component& content, Rect region) : content(&content), region(region) {
        update({.content = &content});
    }

    bool covers(const Extent2D<F32>& point) const override {
        return region.contains(point.x, point.y);
    }

    void layout(const Sakura::Context& ctx) override {
        if (content) {
            layoutChild(ctx, *content, region);
        }
    }
};

struct InputSink : Sakura::Component {
    std::vector<MouseEvent> events;

    bool mouseEvent(const MouseEvent& event) override {
        if (event.type == MouseEventType::Click || event.type == MouseEventType::Release) {
            events.push_back(event);
        }
        return true;
    }
};

struct FocusEditor : Sakura::Component {
    U64 blurs = 0;

    void focus() { requestFocus(); }
    bool focused() const { return hasFocus(); }
    bool editsText() const override { return true; }
    void focusChanged(bool focused) override { blurs += !focused; }
    Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32>) override { return {100, 20}; }
};

struct EditorContent : Sakura::Component {
    FocusEditor editor;

    EditorContent() { addChild(editor); }
    Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32>) override { return {100, 20}; }
    void layout(const Sakura::Context& ctx) override { layoutChild(ctx, editor, frame()); }
};

struct InputRoot : Sakura::RootComponent {
    bool editing() const { return textEditorFocused(); }

    bool send(MouseEventType type, F32 x, F32 y, MouseButton button = MouseButton::Left) {
        return mouseEvent({.type = type, .button = button, .position = {x, y}});
    }

    void dispatch(const Sakura::InputRouter::Output& output) {
        for (const auto& event : output.events) {
            if (const auto* raw = std::get_if<MouseEvent>(&event)) {
                auto mouse = *raw;
                mouse.position = {mouse.position.x * 800.0f, mouse.position.y * 300.0f};
                mouseEvent(mouse);
            }
        }
    }
};

struct InputFixture {
    SakuraTest::FakeWindow window;
    Sakura::Context ctx;

    InputFixture() {
        ctx.render = &window;
        ctx.pixelRatio = 1;
        ctx.framebufferSize = {800, 300};
        ctx.windowFocused = true;
    }

    void place(InputRoot& root) {
        ++ctx.frame;
        Sakura::Access::present(root, ctx.frame);
        Sakura::Access::layout(root, ctx, {0, 0, 800, 300});
        Sakura::Access::settleFocus(root);
    }
};

struct InputTexture : Render::Texture {
    InputTexture() : Texture(Config{.size = {400, 300}}) {}
    Result create() override { return Result::SUCCESS; }
    Result destroy() override { return Result::SUCCESS; }
    uint64_t raw() const override { return 1; }
};

U64 Count(const std::vector<InputEvent>& events, MouseEventType type, MouseButton button = MouseButton::Left) {
    return std::count_if(events.begin(), events.end(), [&](const auto& input) {
        const auto* event = std::get_if<MouseEvent>(&input);
        return event && event->type == type && (type == MouseEventType::Cancel || event->button == button);
    });
}

U64 FocusCount(const std::vector<InputEvent>& events, bool focused) {
    return std::count_if(events.begin(), events.end(), [&](const auto& input) {
        const auto* event = std::get_if<FocusEvent>(&input);
        return event && event->focused == focused;
    });
}

}  // namespace

namespace {
MouseEvent TouchWheel(ScrollPhase phase, F32 delta = -2) {
    return {.type = MouseEventType::Scroll, .position = {50, 50}, .scroll = {0, delta},
            .phase = phase, .precise = true, .touch = true};
}

struct TouchSink : Sakura::Component {
    std::vector<MouseEvent> events;
    std::function<void()> began;
    std::function<void()> cancelled;
    bool claim = true;
    void show(bool value) { setVisible(value); }
    bool mouseEvent(const MouseEvent& event) override {
        events.push_back(event);
        const bool handled = claim && event.type == MouseEventType::Scroll;
        auto callback = event.type == MouseEventType::Cancel ? cancelled :
            (event.type == MouseEventType::Scroll && event.phase == ScrollPhase::Began ? began : std::function<void()>{});
        if (callback) callback();
        return handled;
    }
};

struct TouchParent : Sakura::Component {
    Sakura::Component* selected = nullptr;
    Extent2D<F32> shift{};
    void adopt(Sakura::Component& child) { addChild(child); }
    void drop(Sakura::Component& child) { removeChild(child); }
    void show(bool value) { setVisible(value); }
    bool mouseEvent(const MouseEvent& event) override {
        auto local = event;
        local.position.x -= shift.x;
        local.position.y -= shift.y;
        return selected && deliver(*selected, local);
    }
};
}

TEST_CASE("Touch scrolling retains the actual recipient through custom parent routing", "[core][sakura][touch][ownership]") {
    TouchSink first, second;
    TouchParent root;
    root.adopt(first);
    root.adopt(second);
    root.selected = &first;
    root.shift = {10, 15};
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(root, TouchWheel(ScrollPhase::Began)));
    REQUIRE(gesture.active());
    root.selected = &second;
    root.shift = {30, 40};
    gesture.send(TouchWheel(ScrollPhase::Changed));
    gesture.send(TouchWheel(ScrollPhase::Ended, 0));
    gesture.send(TouchWheel(ScrollPhase::Momentum));
    gesture.send(TouchWheel(ScrollPhase::MomentumEnded, 0));
    REQUIRE(first.events.size() == 5);
    for (const auto& event : first.events) CHECK(event.position == Extent2D<F32>{40, 35});
    CHECK(second.events.empty());
    CHECK_FALSE(gesture.active());
}

TEST_CASE("Touch owner invalidation cannot transfer a gesture to a replacement", "[core][sakura][touch][ownership]") {
    auto first = std::make_unique<TouchSink>();
    TouchSink second;
    TouchParent root;
    root.adopt(*first);
    root.adopt(second);
    root.selected = first.get();
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(root, TouchWheel(ScrollPhase::Began)));
    SECTION("hidden owner") { first->show(false); }
    SECTION("hidden then restored owner") { first->show(false); first->show(true); }
    SECTION("hidden parent") { root.show(false); }
    SECTION("removed owner") { root.drop(*first); }
    SECTION("removed then reattached owner") { root.drop(*first); root.adopt(*first); }
    SECTION("destroyed owner") { first.reset(); }
    root.selected = &second;
    gesture.send(TouchWheel(ScrollPhase::Changed));
    gesture.send(TouchWheel(ScrollPhase::Momentum));
    gesture.send(TouchWheel(ScrollPhase::MomentumEnded, 0));
    CHECK_FALSE(gesture.active());
    CHECK(second.events.empty());
    if (first) {
        CHECK(std::count_if(first->events.begin(), first->events.end(), [](const auto& event) {
            return event.type == MouseEventType::Scroll;
        }) == 1);
        CHECK(first->events.back().type == MouseEventType::Cancel);
    }
}

TEST_CASE("Reset during a touch claim callback cannot revive the old root", "[core][sakura][touch][ownership]") {
    TouchSink first, second;
    TouchParent root, replacement;
    root.adopt(first);
    root.selected = &first;
    replacement.adopt(second);
    replacement.selected = &second;
    Sakura::TouchScroll gesture;
    first.began = [&] { gesture.reset(); };
    REQUIRE(gesture.begin(root, TouchWheel(ScrollPhase::Began)));
    CHECK_FALSE(gesture.active());
    REQUIRE(first.events.size() == 2);
    CHECK(first.events.back().type == MouseEventType::Cancel);
    gesture.send(TouchWheel(ScrollPhase::Changed));
    gesture.send(TouchWheel(ScrollPhase::Ended, 0));
    CHECK(second.events.empty());
    REQUIRE(gesture.begin(replacement, TouchWheel(ScrollPhase::Began)));
    gesture.send(TouchWheel(ScrollPhase::Changed));
    CHECK(second.events.size() == 2);
}

TEST_CASE("Removing a recipient inside its initial scroll callback invalidates the claim", "[core][sakura][touch][ownership]") {
    TouchSink first, second;
    TouchParent root;
    root.adopt(first);
    root.adopt(second);
    root.selected = &first;
    first.began = [&] {
        root.drop(first);
        root.selected = &second;
    };
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(root, TouchWheel(ScrollPhase::Began)));
    CHECK_FALSE(gesture.active());
    gesture.send(TouchWheel(ScrollPhase::Changed));
    gesture.send(TouchWheel(ScrollPhase::MomentumEnded, 0));
    CHECK(second.events.empty());
    CHECK(first.events.back().type == MouseEventType::Cancel);
}

TEST_CASE("A remount during previous-owner cancellation invalidates the incoming touch claim", "[core][sakura][touch][ownership]") {
    TouchSink first, second;
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(first, TouchWheel(ScrollPhase::Began)));
    first.cancelled = [&] { gesture.reset(); };
    REQUIRE(gesture.begin(second, TouchWheel(ScrollPhase::Began)));
    CHECK_FALSE(gesture.active());
    CHECK(second.events.empty());
}

TEST_CASE("Destroying the root in its touch claim callback leaves no retained recipient", "[core][sakura][touch][ownership]") {
    auto root = std::make_unique<TouchSink>();
    root->began = [&] { root.reset(); };
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(*root, TouchWheel(ScrollPhase::Began)));
    CHECK_FALSE(gesture.active());
    gesture.send(TouchWheel(ScrollPhase::Momentum));
    gesture.reset();
}

TEST_CASE("Unclaimed touch scroll remains available for fallback drag", "[core][sakura][touch][ownership]") {
    TouchSink sink;
    sink.claim = false;
    Sakura::TouchScroll gesture;
    CHECK_FALSE(gesture.begin(sink, TouchWheel(ScrollPhase::Began)));
    CHECK_FALSE(gesture.active());
    gesture.send(TouchWheel(ScrollPhase::Changed));
    CHECK(sink.events.size() == 1);
}

TEST_CASE("A custom touch owner cannot redirect later phases into a new child", "[core][sakura][touch][ownership]") {
    struct RoutingOwner : TouchParent {
        bool forward = false;
        bool mouseEvent(const MouseEvent& event) override {
            return forward ? TouchParent::mouseEvent(event) : event.type == MouseEventType::Scroll;
        }
    } root;
    TouchSink child;
    root.adopt(child);
    root.selected = &child;
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(root, TouchWheel(ScrollPhase::Began)));
    root.forward = true;
    gesture.send(TouchWheel(ScrollPhase::Ended, 0));
    CHECK(gesture.active());
    gesture.send(TouchWheel(ScrollPhase::Momentum, 0));
    CHECK(gesture.active());
    CHECK(child.events.empty());
    gesture.send(TouchWheel(ScrollPhase::Changed));
    CHECK_FALSE(gesture.active());
    for (const auto& event : child.events) CHECK(event.type == MouseEventType::Cancel);
}

TEST_CASE("Touch scroller ownership survives layout and cannot move to another layer", "[core][sakura][touch][ownership]") {
    InputFixture fixture;
    F32 firstValue = 0, secondValue = 0;
    Sakura::Scroller first, second;
    first.update({.contentHeight = 1000, .onScrollY = [&](F32 value) { firstValue = value; }});
    second.update({.contentHeight = 1000, .onScrollY = [&](F32 value) { secondValue = value; }});
    RegionLayer firstLayer(first, {0, 0, 400, 300});
    RegionLayer secondLayer(second, {0, 0, 400, 300});
    InputRoot root;
    root.setLayers({&firstLayer});
    fixture.place(root);
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(root, TouchWheel(ScrollPhase::Began)));
    CHECK(firstValue == 20);
    fixture.place(root);
    gesture.send(TouchWheel(ScrollPhase::Changed));
    CHECK(firstValue == 40);
    SECTION("replacement at the same position") { root.setLayers({&secondLayer}); }
    SECTION("overlay over the existing owner") { root.setLayers({&firstLayer, &secondLayer}); }
    fixture.place(root);
    gesture.send(TouchWheel(ScrollPhase::Momentum));
    CHECK(secondValue == 0);
    gesture.send(TouchWheel(ScrollPhase::MomentumEnded, 0));
    CHECK_FALSE(gesture.active());
}

TEST_CASE("Overlay ScrollArea fallback retains its Scroller rather than re-hit-testing content", "[core][sakura][touch][ownership]") {
    SakuraTest::Scene scene;
    struct Tall : TouchSink {
        Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32>) override { return {200, 1000}; }
    } content;
    content.claim = false;
    Sakura::ScrollArea area;
    area.update({.content = &content, .overlay = true});
    SakuraTest::Host host(area);
    host.place(scene.ctx, {0, 0, 200, 200});
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(area, TouchWheel(ScrollPhase::Began)));
    REQUIRE(gesture.active());
    content.events.clear();
    content.claim = true;
    host.place(scene.ctx, {0, 0, 200, 200});
    gesture.send(TouchWheel(ScrollPhase::Changed));
    gesture.send(TouchWheel(ScrollPhase::Ended, 0));
    gesture.send(TouchWheel(ScrollPhase::Momentum));
    CHECK(content.events.empty());
    gesture.reset();
    CHECK_FALSE(gesture.active());
}

TEST_CASE("New popups and modal layers cannot inherit existing touch scroll momentum", "[core][sakura][touch][ownership]") {
    InputFixture fixture;
    TouchSink owner, popup;
    RegionLayer back(owner, {0, 0, 400, 300});
    struct Modal : RegionLayer {
        using RegionLayer::RegionLayer;
        using Component::setVisible;
        bool modal() const override { return true; }
    } modal(popup, {0, 0, 400, 300});
    modal.setVisible(false);
    InputRoot root;
    root.setLayers({&back, &modal});
    fixture.place(root);
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(root, TouchWheel(ScrollPhase::Began)));
    bool blocked = false;
    SECTION("interactive popup") {
        root.popups.replace(back, Sakura::Access::alive(back), {{.content = &popup,
            .anchor = {0, 0, 400, 1}, .size = {400, 200}}});
    }
    SECTION("modal layer") {
        modal.setVisible(true);
        blocked = true;
    }
    fixture.place(root);
    popup.events.clear();
    gesture.send(TouchWheel(ScrollPhase::Momentum));
    CHECK(popup.events.empty());
    REQUIRE(owner.events.size() == 2);
    CHECK(owner.events.back().type == (blocked ? MouseEventType::Cancel : MouseEventType::Scroll));
    gesture.send(TouchWheel(ScrollPhase::MomentumEnded, 0));
    CHECK_FALSE(gesture.active());
}

TEST_CASE("Touch release reaches a scroller after layout moves it away from the pinned point", "[core][sakura][touch][ownership]") {
    InputFixture fixture;
    fixture.ctx.time = 1;
    Sakura::Scroller scroll;
    scroll.update({.contentHeight = 1000});
    RegionLayer layer(scroll, {0, 0, 300, 300});
    InputRoot root;
    root.setLayers({&layer});
    fixture.place(root);
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(root, TouchWheel(ScrollPhase::Began, 2)));
    REQUIRE(scroll.overscroll(fixture.ctx).y < 0);
    layer.region = {400, 0, 300, 300};
    Sakura::Access::invalidate(layer);
    fixture.place(root);
    gesture.send(TouchWheel(ScrollPhase::Ended, 0));
    for (int frame = 0; frame < 120; ++frame) {
        fixture.ctx.time += 1.0 / 60;
        fixture.place(root);
        scroll.overscroll(fixture.ctx);
    }
    CHECK(scroll.overscroll(fixture.ctx).y == 0);
    gesture.send(TouchWheel(ScrollPhase::MomentumEnded, 0));
    CHECK_FALSE(gesture.active());
}

TEST_CASE("Stopping a resumed fling cancels its touch owner", "[core][sakura][touch][ownership]") {
    TouchSink sink;
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(sink, TouchWheel(ScrollPhase::Began)));
    gesture.send(TouchWheel(ScrollPhase::Ended, 0));
    gesture.send(TouchWheel(ScrollPhase::Momentum));
    auto stop = TouchWheel(ScrollPhase::MomentumEnded, 0);
    stop.cancel = true;
    gesture.send(stop);
    CHECK_FALSE(gesture.active());
    REQUIRE_FALSE(sink.events.empty());
    CHECK(sink.events.back().type == MouseEventType::Cancel);
}

TEST_CASE("A reset from a resumed fling handler still cancels the live owner", "[core][sakura][touch][ownership]") {
    Sakura::TouchScroll gesture;
    struct ResettingSink : TouchSink {
        std::function<void()> momentum;
        bool mouseEvent(const MouseEvent& event) override {
            const bool handled = TouchSink::mouseEvent(event);
            if (event.type == MouseEventType::Scroll && event.phase == ScrollPhase::Momentum && momentum) {
                std::exchange(momentum, {})();
            }
            return handled;
        }
    } sink;
    REQUIRE(gesture.begin(sink, TouchWheel(ScrollPhase::Began)));
    gesture.send(TouchWheel(ScrollPhase::Ended, 0));
    sink.momentum = [&] { gesture.reset(); };
    gesture.send(TouchWheel(ScrollPhase::Momentum));
    CHECK_FALSE(gesture.active());
    REQUIRE_FALSE(sink.events.empty());
    CHECK(sink.events.back().type == MouseEventType::Cancel);
}

TEST_CASE("Dropping a released touch owner lets its stretch spring home", "[core][sakura][touch][ownership]") {
    InputFixture fixture;
    fixture.ctx.time = 1;
    Sakura::Scroller scroll;
    scroll.update({.contentHeight = 1000});
    RegionLayer layer(scroll, {0, 0, 300, 300});
    InputRoot root;
    root.setLayers({&layer});
    fixture.place(root);
    Sakura::TouchScroll gesture;
    REQUIRE(gesture.begin(root, TouchWheel(ScrollPhase::Began, 2)));
    gesture.send(TouchWheel(ScrollPhase::Ended, 0));
    const F32 released = scroll.overscroll(fixture.ctx).y;
    REQUIRE(released < 0);
    gesture.reset();
    CHECK(scroll.overscroll(fixture.ctx).y == released);
    for (int frame = 0; frame < 3; ++frame) {
        fixture.ctx.time += 1.0 / 60;
        fixture.place(root);
        scroll.overscroll(fixture.ctx);
    }
    const F32 springing = scroll.overscroll(fixture.ctx).y;
    CHECK(springing < 0);
    CHECK(springing > released);
    for (int frame = 0; frame < 120; ++frame) {
        fixture.ctx.time += 1.0 / 60;
        fixture.place(root);
        scroll.overscroll(fixture.ctx);
    }
    CHECK(scroll.overscroll(fixture.ctx).y == 0);
}

TEST_CASE("An overlay cannot swallow an elastic scroll release", "[core][sakura][input][ownership][wayland-scroll]") {
    InputFixture fixture;
    fixture.ctx.time = 1.0;
    Sakura::Scroller scroll;
    scroll.update({.contentHeight = 1000.0f});
    InputSink sink;
    RegionLayer back(scroll, {0, 0, 400, 300});
    struct Modal : RegionLayer {
        using RegionLayer::RegionLayer;
        bool modal() const override { return true; }
    } modal(sink, {0, 0, 400, 300});
    modal.setVisible(false);
    InputRoot root;
    root.setLayers({&back, &modal});
    fixture.place(root);
    MouseEvent wheel{.type = MouseEventType::Scroll, .position = {100, 100}, .scroll = {0, 4},
                     .phase = ScrollPhase::Began, .precise = true};
    REQUIRE(Sakura::Access::mouseEvent(root, wheel));
    const F32 stretched = scroll.overscroll(fixture.ctx).y;
    REQUIRE(stretched < 0.0f);
    SECTION("modal") {
        modal.setVisible(true);
        fixture.place(root);
        REQUIRE(root.target(wheel.position) == &modal);
    }
    SECTION("interactive popup") {
        root.popups.replace(back, Sakura::Access::alive(back), {{.content = &sink,
            .anchor = {0, 0, 400, 1}, .size = {400, 200}}});
        fixture.place(root);
        REQUIRE(root.target(wheel.position) == &root.popups);
    }
    // Opening the overlay does not pretend that fingers have lifted.
    fixture.ctx.time += 1.0;
    CHECK(scroll.overscroll(fixture.ctx).y == stretched);
    wheel.phase = ScrollPhase::Ended;
    wheel.scroll = {0, 0};
    Sakura::Access::mouseEvent(root, wheel);
    for (int frame = 0; frame < 120; ++frame) {
        fixture.ctx.time += 1.0 / 60.0;
        fixture.place(root);
        scroll.overscroll(fixture.ctx);
    }
    CHECK(scroll.overscroll(fixture.ctx).y == 0.0f);
}

TEST_CASE("A temporarily zero-sized canvas cancels held scrolling without mouse buttons", "[core][sakura][input][ownership][wayland-scroll]") {
    InputFixture fixture;
    fixture.ctx.time = 1.0;
    Sakura::Scroller scroll;
    scroll.update({.contentHeight = 1000.0f});
    RegionLayer layer(scroll, {0, 0, 400, 300});
    InputRoot root;
    root.setLayers({&layer});
    fixture.place(root);
    Sakura::InputRouter router;
    Sakura::InputFrame frame;
    frame.windowFocused = true;
    frame.pointer = Extent2D<F32>{100, 100};
    frame.events = {MouseEvent{.type = MouseEventType::Scroll, .position = {100, 100}, .scroll = {0, 4},
                              .phase = ScrollPhase::Began}};
    root.dispatch(router.route(frame, {0, 0, 800, 300}));
    REQUIRE(scroll.overscroll(fixture.ctx).y < 0.0f);
    ++frame.serial;
    frame.events = {MouseEvent{.type = MouseEventType::Scroll, .position = {100, 100},
                              .phase = ScrollPhase::Ended}};
    root.dispatch(router.route(frame, {0, 0, 0, 0}));
    CHECK(scroll.overscroll(fixture.ctx).y == 0.0f);
    ++frame.serial;
    frame.events.clear();
    root.dispatch(router.route(frame, {0, 0, 800, 300}));
    CHECK(scroll.overscroll(fixture.ctx).y == 0.0f);
}

TEST_CASE("Secondary releases preserve the primary layer capture and dock drag",
          "[core][sakura][input][ownership][review17]") {
    const auto secondary = GENERATE(MouseButton::Right, MouseButton::Middle);
    InputFixture fixture;
    std::vector<InputEvent> events;
    Sakura::SurfaceImage image;
    image.update({.onInput = [&](InputEvent event) { events.push_back(event); }});
    InputSink overlay;
    RegionLayer back(image, {0, 0, 400, 300});
    RegionLayer front(overlay, {400, 0, 400, 300});
    InputRoot root;
    root.setLayers({&back, &front});
    fixture.place(root);

    REQUIRE(root.send(MouseEventType::Click, 100, 100));
    root.drag.active = true;
    root.drag.pane = "primary";
    root.send(MouseEventType::Move, 500, 100);
    root.send(MouseEventType::Click, 500, 100, secondary);
    root.send(MouseEventType::Release, 500, 100, secondary);
    CHECK(root.captured == &back);
    CHECK(root.drag.active);
    CHECK(root.drag.pane == "primary");
    REQUIRE(root.send(MouseEventType::Release, 500, 100));
    CHECK(Count(events, MouseEventType::Release) == 1);
    CHECK(overlay.events.empty());
    CHECK(root.captured == nullptr);
    CHECK_FALSE(root.drag.active);
    root.drag.active = true;
    root.send(MouseEventType::Release, 500, 100, secondary);
    CHECK(root.drag.active);
    root.send(MouseEventType::Release, 500, 100);
    CHECK_FALSE(root.drag.active);
}

TEST_CASE("Layer capture balances each held button in either release order",
          "[core][sakura][input][ownership][review17]") {
    const auto first = GENERATE(MouseButton::Left, MouseButton::Right, MouseButton::Middle);
    const bool reverse = GENERATE(false, true);
    const auto second = static_cast<MouseButton>((static_cast<U64>(first) + 1) % 3);
    InputFixture fixture;
    std::vector<InputEvent> events;
    Sakura::SurfaceImage image;
    image.update({.onInput = [&](InputEvent event) { events.push_back(event); }});
    InputSink overlay;
    RegionLayer back(image, {0, 0, 400, 300});
    RegionLayer front(overlay, {400, 0, 400, 300});
    InputRoot root;
    root.setLayers({&back, &front});
    fixture.place(root);

    REQUIRE(root.send(MouseEventType::Click, 100, 100, first));
    REQUIRE(root.send(MouseEventType::Click, 100, 100, second));
    root.send(MouseEventType::Move, 500, 100);
    REQUIRE(root.send(MouseEventType::Release, 500, 100, reverse ? second : first));
    CHECK(root.captured == &back);
    REQUIRE(root.send(MouseEventType::Release, 500, 100, reverse ? first : second));
    for (const auto button : {first, second}) {
        CHECK(Count(events, MouseEventType::Click, button) == 1);
        CHECK(Count(events, MouseEventType::Release, button) == 1);
    }
    CHECK(overlay.events.empty());
    CHECK(root.captured == nullptr);
}

TEST_CASE("Layer capture and dock drags cancel all buttons on explicit cancellation or focus loss",
          "[core][sakura][input][ownership][review17]") {
    const bool focusLoss = GENERATE(false, true);
    InputFixture fixture;
    std::vector<InputEvent> events;
    Sakura::SurfaceImage image;
    image.update({.onInput = [&](InputEvent event) { events.push_back(event); }});
    InputSink overlay;
    RegionLayer back(image, {0, 0, 400, 300});
    RegionLayer front(overlay, {400, 0, 400, 300});
    InputRoot root;
    root.setLayers({&back, &front});
    fixture.place(root);
    Sakura::InputFrame frame;
    Sakura::InputRouter router;
    frame.pointer = Extent2D<F32>{100, 100};
    frame.buttons.fill(true);
    for (const auto button : {MouseButton::Left, MouseButton::Right, MouseButton::Middle}) {
        frame.events.push_back(MouseEvent{.type = MouseEventType::Click, .button = button, .position = {100, 100}});
    }
    root.dispatch(router.route(frame, {0, 0, 800, 300}));
    REQUIRE(root.captured == &back);
    root.drag.active = true;
    frame.advance(0.1f);
    if (focusLoss) frame.windowFocused = false;
    else frame.events.push_back(MouseEvent{.type = MouseEventType::Cancel});
    root.dispatch(router.route(frame, {0, 0, 800, 300}));
    CHECK(Count(events, MouseEventType::Cancel) == 1);
    CHECK(root.captured == nullptr);
    CHECK_FALSE(root.drag.active);
    overlay.events.clear();

    frame.advance(0.1f);
    frame.windowFocused = true;
    frame.pointer = Extent2D<F32>{500, 100};
    frame.buttons.fill(false);
    for (const auto button : {MouseButton::Left, MouseButton::Right, MouseButton::Middle}) {
        frame.events.push_back(MouseEvent{.type = MouseEventType::Release, .button = button, .position = {500, 100}});
    }
    root.dispatch(router.route(frame, {0, 0, 800, 300}));
    CHECK(Count(events, MouseEventType::Release) == 0);
    CHECK(std::none_of(overlay.events.begin(), overlay.events.end(), [](const auto& event) {
        return event.type == MouseEventType::Release;
    }));
}

TEST_CASE("Popup withdrawal blurs nested editors exactly once without changing visibility focus semantics",
          "[core][sakura][input][ownership][review17]") {
    const int boundary = GENERATE(0, 1, 2);
    InputFixture fixture;
    EditorContent content;
    Sakura::Popup popup;
    Sakura::Popup::Config config{.content = &content, .open = true};
    popup.update(config);
    RegionLayer layer(popup, {0, 0, 100, 20});
    InputRoot root;
    root.setLayers({&layer});
    fixture.place(root);
    content.editor.focus();
    REQUIRE(content.editor.focused());
    REQUIRE(root.editing());

    if (boundary == 0) {
        config.open = false;
        popup.update(config);
    } else if (boundary == 1) {
        popup.setVisible(false);
    } else {
        layer.content = nullptr;
        layer.update({});
    }
    for (U64 pass = 0; pass < 3; ++pass) fixture.place(root);
    CHECK_FALSE(content.visible());
    CHECK_FALSE(content.editor.focused());
    CHECK_FALSE(root.editing());
    CHECK(content.editor.blurs == 1);
}

TEST_CASE("Withdrawing popup content preserves focus outside it and ordinary hidden focus",
          "[core][sakura][input][ownership][review17]") {
    InputFixture fixture;
    FocusEditor underlying;
    EditorContent content;
    Sakura::Popup popup;
    popup.update({.content = &content, .open = true});
    RegionLayer back(underlying, {0, 0, 400, 300});
    RegionLayer front(popup, {400, 0, 100, 20});
    InputRoot root;
    root.setLayers({&back, &front});
    fixture.place(root);
    underlying.focus();
    popup.update({.content = &content, .open = false});
    fixture.place(root);
    CHECK(underlying.focused());
    CHECK(underlying.blurs == 0);
    CHECK(root.editing());
    underlying.setVisible(false);
    CHECK(underlying.focused());
    CHECK(underlying.blurs == 0);
}

TEST_CASE("Surface replacement cancels the old recipient and never transfers held input",
          "[core][sakura][input][ownership][review17]") {
    const int boundary = GENERATE(0, 1, 2);
    InputFixture fixture;
    std::vector<InputEvent> oldEvents;
    std::vector<InputEvent> newEvents;
    const auto texture = std::make_shared<InputTexture>();
    Sakura::SurfaceImage image;
    image.update({.texture = texture, .onInput = [&](InputEvent event) { oldEvents.push_back(event); }});
    RegionLayer layer(image, {0, 0, 400, 300});
    InputRoot root;
    root.setLayers({&layer});
    fixture.place(root);
    for (const auto button : {MouseButton::Left, MouseButton::Right, MouseButton::Middle}) {
        REQUIRE(root.send(MouseEventType::Click, 100, 100, button));
    }
    Sakura::Access::key(root, {KeyEventType::Press, KeyCode::Space});
    REQUIRE(FocusCount(oldEvents, true) == 1);
    Sakura::SurfaceImage::Config next{
        .texture = boundary == 0 ? std::make_shared<InputTexture>() : boundary == 1 ? nullptr : texture,
        .onInput = [&](InputEvent event) { newEvents.push_back(event); },
    };
    if (boundary == 2) next.onInput = {};
    image.update(next);
    CHECK(Count(oldEvents, MouseEventType::Cancel) == 1);
    CHECK(FocusCount(oldEvents, false) == 1);
    CHECK(Sakura::Access::focusOwner(root) == nullptr);
    CHECK(newEvents.empty());
    for (const auto button : {MouseButton::Left, MouseButton::Right, MouseButton::Middle}) {
        root.send(MouseEventType::Release, 100, 100, button);
    }
    CHECK(newEvents.empty());

    next.texture = std::make_shared<InputTexture>();
    next.onInput = [&](InputEvent event) { newEvents.push_back(event); };
    image.update(next);
    fixture.place(root);
    REQUIRE(root.send(MouseEventType::Click, 100, 100));
    REQUIRE(root.send(MouseEventType::Release, 100, 100));
    CHECK(Count(newEvents, MouseEventType::Click) == 1);
    CHECK(Count(newEvents, MouseEventType::Release) == 1);
    CHECK(Count(newEvents, MouseEventType::Cancel) == 0);
    CHECK(FocusCount(newEvents, true) == 1);
}

TEST_CASE("Same surface identity keeps a gesture through callback-only polling updates",
          "[core][sakura][input][ownership][review17]") {
    InputFixture fixture;
    std::vector<InputEvent> events;
    const auto texture = std::make_shared<InputTexture>();
    Sakura::SurfaceImage image;
    const auto update = [&] {
        image.update({.texture = texture, .onInput = [&](InputEvent event) { events.push_back(event); }});
    };
    update();
    RegionLayer layer(image, {0, 0, 400, 300});
    InputRoot root;
    root.setLayers({&layer});
    fixture.place(root);
    REQUIRE(root.send(MouseEventType::Click, 100, 100));
    update();
    fixture.place(root);
    update();
    REQUIRE(root.send(MouseEventType::Release, 500, 100));
    CHECK(Count(events, MouseEventType::Click) == 1);
    CHECK(Count(events, MouseEventType::Release) == 1);
    CHECK(Count(events, MouseEventType::Cancel) == 0);
    CHECK(FocusCount(events, true) == 1);
    CHECK(FocusCount(events, false) == 0);
}

TEST_CASE("Surface images deliver the release of every button they were pressed with",
          "[core][sakura][surface]") {
    const auto texture = std::make_shared<InputTexture>();
    std::vector<MouseEvent> events;
    Sakura::SurfaceImage image;
    image.update({
        .texture = texture,
        .onInput = [&](InputEvent event) {
            if (const auto* mouse = std::get_if<MouseEvent>(&event)) {
                events.push_back(*mouse);
            }
        },
    });
    SakuraTest::Host host(image);
    SakuraTest::Scene scene;
    host.place(scene.ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    for (const auto button : {MouseButton::Middle, MouseButton::Left, MouseButton::Right}) {
        events.clear();
        CHECK(host.send(SakuraTest::Mouse(MouseEventType::Click, 50.0f, 50.0f, button)));
        CHECK(host.send(SakuraTest::Mouse(MouseEventType::Release, 150.0f, 50.0f, button)));
        REQUIRE(events.size() == 2);
        CHECK(events[1].type == MouseEventType::Release);
        CHECK(events[1].button == button);
        CHECK_FALSE(host.send(SakuraTest::Mouse(MouseEventType::Release, 150.0f, 50.0f, button)));
    }
}
