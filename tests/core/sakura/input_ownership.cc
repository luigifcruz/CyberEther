#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/components/popup.hh>
#include <jetstream/render/sakura/components/surface_image.hh>

#include "harness.hh"
#include "scene.hh"
#include "render/sakura/detail/input_router.hh"
#include "render/sakura/detail/root_component.hh"

#include <algorithm>
#include <array>
#include <memory>
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
