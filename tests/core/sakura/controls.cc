#include <catch2/catch_test_macros.hpp>

#include "fonts.hh"
#include "component_harness.hh"

#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/checkbox.hh>
#include <jetstream/render/sakura/components/combo.hh>
#include <jetstream/render/sakura/components/progress_bar.hh>
#include <jetstream/render/sakura/components/qr_code.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/components/text_input.hh>

#include "harness.hh"
#include "scene.hh"

#include <limits>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {


Extent2D<F32> TextSize(const Sakura::Context& ctx, Sakura::Text::Config config,
                       Extent2D<F32> available = {kInfinite, kInfinite}) {
    Sakura::Text text;
    text.update(std::move(config));
    Host host(text);
    return host.measureOf(ctx, available);
}

F32 TextWidth(const Sakura::Context& ctx, const std::string& str) {
    return TextSize(ctx, {.str = str}).x;
}

struct Pair : Sakura::Component {
    Pair(Sakura::Component& first, Sakura::Component& second) : first(first), second(second) {
        addChild(first);
        addChild(second);
    }

    void place(const Sakura::Context& ctx, Rect bounds) {
        layoutRoot(ctx, bounds);
        Sakura::Access::settleFocus(*this);
    }

    bool send(const MouseEvent& event) {
        return mouseEventChildren(event);
    }

    bool click(F32 x, F32 y) {
        const bool pressed = send(Mouse(MouseEventType::Click, x, y));
        send(Mouse(MouseEventType::Release, x, y));
        return pressed;
    }

 protected:
    void layout(const Sakura::Context& ctx) override {
        const Rect bounds = frame();
        layoutChild(ctx, first, {bounds.x, bounds.y, bounds.width, kRowHeight});
        layoutChild(ctx, second, {bounds.x, bounds.y + kRowStride, bounds.width, kRowHeight});
    }

 private:
    static constexpr F32 kRowHeight = 25.0f;
    static constexpr F32 kRowStride = 40.0f;

    Sakura::Component& first;
    Sakura::Component& second;
};

}  // namespace

TEST_CASE("Buttons click on press and release inside", "[core][sakura][controls]") {
    const Scene scene;
    U64 clicked = 0;
    Sakura::Button button;
    button.update({.str = "Run", .onClick = [&] { ++clicked; }});
    Host host(button);
    host.place(scene.ctx, {10.0f, 10.0f, 100.0f, 30.0f});

    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 20.0f)));
    CHECK(clicked == 0);
    CHECK(host.send(Mouse(MouseEventType::Release, 60.0f, 25.0f)));
    CHECK(clicked == 1);

    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 60.0f, 25.0f)));
    CHECK(clicked == 1);
}

TEST_CASE("Buttons do not click when released outside", "[core][sakura][controls]") {
    const Scene scene;
    U64 clicked = 0;
    Sakura::Button button;
    button.update({.str = "Run", .onClick = [&] { ++clicked; }});
    Host host(button);
    host.place(scene.ctx, {10.0f, 10.0f, 100.0f, 30.0f});

    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 20.0f)));
    CHECK(host.send(Mouse(MouseEventType::Release, 300.0f, 300.0f)));
    CHECK(clicked == 0);

    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 50.0f, 20.0f, MouseButton::Right)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 50.0f, 20.0f, MouseButton::Right)));
    CHECK(clicked == 0);
}

TEST_CASE("Disabled buttons ignore clicks", "[core][sakura][controls]") {
    const Scene scene;
    U64 clicked = 0;
    Sakura::Button button;
    button.update({.str = "Run", .disabled = true, .onClick = [&] { ++clicked; }});
    Host host(button);
    host.place(scene.ctx, {10.0f, 10.0f, 100.0f, 30.0f});

    CHECK_FALSE(host.click(50.0f, 20.0f));
    CHECK(clicked == 0);

    button.update({.str = "Run", .onClick = [&] { ++clicked; }});
    CHECK(host.click(50.0f, 20.0f));
    CHECK(clicked == 1);
}

TEST_CASE("Buttons route double clicks to onDoubleClick instead of onClick",
          "[core][sakura][controls]") {
    const Scene scene;
    U64 clicked = 0;
    U64 doubled = 0;
    Sakura::Button button;
    button.update({.str = "Open", .onClick = [&] { ++clicked; }, .onDoubleClick = [&] { ++doubled; }});
    Host host(button);
    host.place(scene.ctx, {10.0f, 10.0f, 100.0f, 30.0f});

    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 20.0f, MouseButton::Left, 2)));
    CHECK(doubled == 1);
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 50.0f, 20.0f)));
    CHECK(clicked == 0);

    Sakura::Button single;
    single.update({.str = "Open", .onClick = [&] { ++clicked; }});
    Host other(single);
    other.place(scene.ctx, {10.0f, 10.0f, 100.0f, 30.0f});
    CHECK(other.send(Mouse(MouseEventType::Click, 50.0f, 20.0f, MouseButton::Left, 2)));
    CHECK(other.send(Mouse(MouseEventType::Release, 50.0f, 20.0f)));
    CHECK(clicked == 1);
}

TEST_CASE("Button measure wraps the label in padding unless a size is given",
          "[core][sakura][controls]") {
    Scene scene;
    const F32 label = TextWidth(scene.ctx, "Connect");
    REQUIRE(label > 0.0f);

    const auto measure = [&](Sakura::Button::Config config, Extent2D<F32> available = {500.0f, 500.0f}) {
        Sakura::Button button;
        config.str = "Connect";
        button.update(std::move(config));
        Host host(button);
        return host.measureOf(scene.ctx, available);
    };

    CHECK(measure({}) == Extent2D<F32>{label + 24.0f, 25.0f});
    CHECK(measure({.padding = Extent2D<F32>{4.0f, 2.0f}}) == Extent2D<F32>{label + 8.0f, 19.0f});
    CHECK(measure({.variant = Sakura::Button::Variant::Text}) == Extent2D<F32>{label, 15.0f});
    CHECK(measure({.size = {120.0f, 40.0f}}) == Extent2D<F32>{120.0f, 40.0f});
    CHECK(measure({.size = {Sakura::Fill, 0.0f}}, {300.0f, 500.0f}) == Extent2D<F32>{300.0f, 25.0f});
    CHECK(measure({.size = {Sakura::Fill, 0.0f}}, {kInfinite, 500.0f}) == Extent2D<F32>{label + 24.0f, 25.0f});

    scene.ctx.pixelRatio = 2.0f;
    const F32 doubled = TextWidth(scene.ctx, "Connect");
    CHECK(measure({}) == Extent2D<F32>{doubled + 48.0f, 50.0f});
    CHECK(measure({.size = {120.0f, 40.0f}}) == Extent2D<F32>{240.0f, 80.0f});
}

TEST_CASE("Navigation items fill the width and select on click", "[core][sakura][controls]") {
    const Scene scene;
    U64 selected = 0;
    Sakura::NavigationItem item;
    item.update({.label = "General", .onSelect = [&] { ++selected; }});
    Host host(item);

    CHECK(host.measureOf(scene.ctx, {240.0f, 500.0f}) == Extent2D<F32>{240.0f, 35.0f});

    host.place(scene.ctx, {0.0f, 0.0f, 240.0f, 35.0f});
    CHECK(host.click(200.0f, 20.0f));
    CHECK(selected == 1);
}

TEST_CASE("Buttons drop a held press on cancel so a later release inside does not click",
          "[core][sakura][controls]") {
    const Scene scene;
    U64 clicked = 0;
    Sakura::Button button;
    button.update({.str = "Run", .onClick = [&] { ++clicked; }});
    Host host(button);
    host.place(scene.ctx, {10.0f, 10.0f, 100.0f, 30.0f});

    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 20.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Cancel, 50.0f, 20.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 50.0f, 20.0f)));
    CHECK(clicked == 0);

    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 20.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Leave, 50.0f, 20.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 50.0f, 20.0f)));
    CHECK(clicked == 0);

    CHECK(host.click(50.0f, 20.0f));
    CHECK(clicked == 1);
}

TEST_CASE("Navigation items select on a single click and route double clicks to onDoubleClick",
          "[core][sakura][controls]") {
    const Scene scene;
    U64 selected = 0;
    U64 opened = 0;
    Sakura::NavigationItem item;
    item.update({.label = "General", .onSelect = [&] { ++selected; }, .onDoubleClick = [&] { ++opened; }});
    Host host(item);
    host.place(scene.ctx, {0.0f, 0.0f, 240.0f, 35.0f});

    CHECK(host.click(120.0f, 18.0f));
    CHECK(selected == 1);
    CHECK(opened == 0);

    CHECK(host.send(Mouse(MouseEventType::Click, 120.0f, 18.0f, MouseButton::Left, 2)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Release, 120.0f, 18.0f)));
    CHECK(selected == 1);
    CHECK(opened == 1);

    CHECK(host.send(Mouse(MouseEventType::Click, 120.0f, 18.0f)));
    CHECK(host.send(Mouse(MouseEventType::Release, 400.0f, 18.0f)));
    CHECK(selected == 1);

    CHECK_FALSE(host.click(120.0f, 18.0f, MouseButton::Right));
    CHECK(selected == 1);
    CHECK(opened == 1);
}

TEST_CASE("Checkboxes request the toggled value on click", "[core][sakura][controls]") {
    const Scene scene;
    std::vector<bool> changes;
    Sakura::Checkbox checkbox;
    checkbox.update({.label = "Enabled", .value = false, .onChange = [&](bool value) { changes.push_back(value); }});
    Host host(checkbox);
    host.place(scene.ctx, {0.0f, 0.0f, 200.0f, 25.0f});

    CHECK(host.click(10.0f, 10.0f));
    CHECK(changes == std::vector<bool>{true});

    checkbox.update({.label = "Enabled", .value = true, .onChange = [&](bool value) { changes.push_back(value); }});
    CHECK(host.click(150.0f, 10.0f));
    CHECK(changes == std::vector<bool>{true, false});

    host.send(Mouse(MouseEventType::Click, 10.0f, 10.0f));
    host.send(Mouse(MouseEventType::Release, 300.0f, 10.0f));
    CHECK(changes.size() == 2);
}

TEST_CASE("Checkbox measure is a square plus the spaced label", "[core][sakura][controls]") {
    const Scene scene;
    Sakura::Checkbox bare;
    bare.update({});
    Host bareHost(bare);
    CHECK(bareHost.measureOf(scene.ctx, {500.0f, 500.0f}) == Extent2D<F32>{25.0f, 25.0f});

    Sakura::Checkbox labeled;
    labeled.update({.label = "Enabled"});
    Host labeledHost(labeled);
    const F32 label = TextWidth(scene.ctx, "Enabled");
    REQUIRE(label > 0.0f);
    CHECK(labeledHost.measureOf(scene.ctx, {500.0f, 500.0f}) == Extent2D<F32>{25.0f + 8.0f + label, 25.0f});
}

TEST_CASE("Combo opens a dropdown below itself and reports the chosen option", "[core][sakura][controls]") {
    const Scene scene;
    std::vector<std::string> events;
    Sakura::Combo combo;
    combo.update({
        .id = "mode",
        .options = {"alpha", "beta", "gamma"},
        .value = "alpha",
        .onChange = [&](const std::string& value) { events.push_back("change:" + value); },
        .onSelect = [&](U64 index) { events.push_back("select:" + std::to_string(index)); },
    });
    LayerSlot mount;
    mount.hold(combo, {100.0f, 100.0f, 200.0f, 25.0f});
    LayerHost host({&mount});
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 150.0f}));

    using Queue = Render::InputQueue;
    InputDriver driver;
    CHECK(driver.run(host, scene.ctx, kScreen, {
        Queue::Pointer{Extent2D<F32>{150.0f, 112.0f}},
        Queue::Button{MouseButton::Left, true},
        Queue::Button{MouseButton::Left, false},
    }) > 1);
    CHECK(host.popups.interactiveAt({100.0f, 125.0f}));
    CHECK(host.popups.interactiveAt({299.0f, 209.0f}));
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 210.0f}));
    host.place(scene.ctx, kScreen);
    CHECK(events.empty());

    driver.run(host, scene.ctx, kScreen, {
        Queue::Pointer{Extent2D<F32>{150.0f, 167.0f}},
        Queue::Button{MouseButton::Left, true},
        Queue::Button{MouseButton::Left, false},
    });
    CHECK(events == std::vector<std::string>{"select:1", "change:beta"});
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 150.0f}));
    events.clear();

    Sakura::Combo second;
    second.update({
        .id = "batched",
        .options = {"alpha", "beta", "gamma"},
        .value = "alpha",
        .onChange = [&](const std::string& value) { events.push_back("change:" + value); },
        .onSelect = [&](U64 index) { events.push_back("select:" + std::to_string(index)); },
    });
    LayerSlot batchedMount;
    batchedMount.hold(second, {100.0f, 100.0f, 200.0f, 25.0f});
    LayerHost batchedHost({&batchedMount});
    batchedHost.place(scene.ctx, kScreen);
    InputDriver batched;
    batched.run(batchedHost, scene.ctx, kScreen, {
        Queue::Pointer{Extent2D<F32>{150.0f, 112.0f}},
        Queue::Button{MouseButton::Left, true},
        Queue::Button{MouseButton::Left, false},
        Queue::Pointer{Extent2D<F32>{150.0f, 167.0f}},
        Queue::Button{MouseButton::Left, true},
        Queue::Button{MouseButton::Left, false},
    });
    CHECK(events == std::vector<std::string>{"select:1", "change:beta"});
    CHECK_FALSE(batchedHost.popups.interactiveAt({150.0f, 150.0f}));
}

TEST_CASE("Combo closes its dropdown on a second click and stays shut while disabled",
          "[core][sakura][controls]") {
    const Scene scene;
    U64 changes = 0;
    Sakura::Combo combo;
    Sakura::Combo::Config config{
        .options = {"alpha", "beta"},
        .onChange = [&](const std::string&) { ++changes; },
    };
    combo.update(config);
    LayerSlot mount;
    mount.hold(combo, {100.0f, 100.0f, 200.0f, 25.0f});
    LayerHost host({&mount});
    host.place(scene.ctx, kScreen);

    host.click(150.0f, 112.0f);
    host.place(scene.ctx, kScreen);
    REQUIRE(host.popups.interactiveAt({150.0f, 150.0f}));
    host.click(150.0f, 112.0f);
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 150.0f}));

    config.disabled = true;
    combo.update(config);
    CHECK_FALSE(host.click(150.0f, 112.0f));
    host.place(scene.ctx, kScreen);
    CHECK_FALSE(host.popups.interactiveAt({150.0f, 150.0f}));
    CHECK(changes == 0);
}

TEST_CASE("Combo measure uses its configured width or the available width", "[core][sakura][controls]") {
    const Scene scene;
    Sakura::Combo combo;
    combo.update({.options = {"alpha"}});
    Host host(combo);

    CHECK(host.measureOf(scene.ctx, {320.0f, 500.0f}) == Extent2D<F32>{320.0f, 25.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, 500.0f}) == Extent2D<F32>{200.0f, 25.0f});
    combo.update({.options = {"alpha"}, .width = 150.0f});
    CHECK(host.measureOf(scene.ctx, {320.0f, 500.0f}) == Extent2D<F32>{150.0f, 25.0f});
}

TEST_CASE("QR codes click only inside the centered code", "[core][sakura][controls]") {
    const Scene scene;
    U64 clicked = 0;
    Sakura::QrCode code;
    code.update({.data = {1, 0, 0, 1}, .width = 2, .onClick = [&] { ++clicked; }});
    Host host(code);

    CHECK(host.measureOf(scene.ctx, {100.0f, 500.0f}) == Extent2D<F32>{100.0f, 24.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, 500.0f}) == Extent2D<F32>{24.0f, 24.0f});

    host.place(scene.ctx, {0.0f, 0.0f, 100.0f, 24.0f});
    CHECK(host.send(Mouse(MouseEventType::Click, 38.0f, 10.0f)));
    CHECK(clicked == 1);
    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 37.0f, 10.0f)));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 62.0f, 10.0f)));
    CHECK(clicked == 1);

    code.update({.data = {1, 0, 0, 1}, .width = 2, .onClick = [&] { clicked += 10; }});
    CHECK(host.send(Mouse(MouseEventType::Click, 50.0f, 10.0f)));
    CHECK(clicked == 11);
}

TEST_CASE("Progress bars fill the available width by default", "[core][sakura][controls]") {
    const Scene scene;
    Sakura::ProgressBar bar;
    bar.update({.value = 0.5f, .overlay = "50%"});
    Host host(bar);

    CHECK(host.measureOf(scene.ctx, {300.0f, 500.0f}) == Extent2D<F32>{300.0f, 20.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, 500.0f}) == Extent2D<F32>{200.0f, 20.0f});

    bar.update({.size = {150.0f, 0.0f}});
    CHECK(host.measureOf(scene.ctx, {300.0f, 500.0f}) == Extent2D<F32>{150.0f, 25.0f});
}

TEST_CASE("Unwrapped text measures to its longest line", "[core][sakura][controls]") {
    const Scene scene;
    const F32 shortLine = TextWidth(scene.ctx, "ab");
    const F32 longLine = TextWidth(scene.ctx, "abcdef");
    REQUIRE(shortLine > 0.0f);
    REQUIRE(longLine > shortLine);

    CHECK(TextSize(scene.ctx, {.str = "ab\nabcdef"}) == Extent2D<F32>{longLine, 30.0f});
    CHECK(TextSize(scene.ctx, {.str = "ab\nabcdef"}, {500.0f, kInfinite}) == Extent2D<F32>{longLine, 30.0f});
    CHECK(TextSize(scene.ctx, {.str = "ab", .scale = 2.0f}).y == 30.0f);
}

TEST_CASE("Text measures to the available width and never past it", "[core][sakura][controls]") {
    const Scene scene;
    const F32 natural = TextWidth(scene.ctx, "alpha beta gamma");
    REQUIRE(natural > 0.0f);

    CHECK(TextSize(scene.ctx, {.str = "ab", .wrapped = true}, {500.0f, kInfinite}) == Extent2D<F32>{500.0f, 15.0f});
    CHECK(TextSize(scene.ctx, {.str = "alpha beta gamma", .wrapped = true}) == Extent2D<F32>{natural, 15.0f});
    CHECK(TextSize(scene.ctx, {.str = "ab", .align = Sakura::Text::Align::Center}, {500.0f, kInfinite}).x == 500.0f);

    const F32 width = TextWidth(scene.ctx, "alpha beta") + 1.0f;
    CHECK(TextSize(scene.ctx, {.str = "alpha beta gamma", .wrapped = true}, {width, kInfinite}) ==
          Extent2D<F32>{width, 30.0f});
    CHECK(TextSize(scene.ctx, {.str = "alpha beta gamma"}, {width, kInfinite}) == Extent2D<F32>{width, 15.0f});
}

TEST_CASE("Text inputs take focus when clicked or configured to focus", "[core][sakura][controls]") {
    const Scene scene;
    Sakura::TextInput first;
    Sakura::TextInput second;
    first.update({.value = "alpha"});
    second.update({.value = "beta"});
    {
        Host firstHost(first);
        Host secondHost(second);
        CHECK(firstHost.measureOf(scene.ctx, {300.0f, 500.0f}) == Extent2D<F32>{300.0f, 25.0f});
        CHECK(firstHost.measureOf(scene.ctx, {kInfinite, 500.0f}) == Extent2D<F32>{200.0f, 25.0f});
        second.update({.value = "beta", .multiline = true, .height = 80.0f});
        CHECK(secondHost.measureOf(scene.ctx, {300.0f, 500.0f}) == Extent2D<F32>{300.0f, 80.0f});
        second.update({.value = "beta"});
    }

    const Rect bounds = {0.0f, 0.0f, 300.0f, 65.0f};
    Pair pair(first, second);
    pair.place(scene.ctx, bounds);
    CHECK_FALSE(first.focused());

    CHECK(pair.click(40.0f, 12.0f));
    CHECK(first.focused());
    CHECK(FocusOwner(pair) != nullptr);

    CHECK(pair.click(40.0f, 52.0f));
    CHECK(second.focused());
    CHECK_FALSE(first.focused());

    ClickFocus(pair, std::nullopt);
    CHECK_FALSE(second.focused());
    CHECK(FocusOwner(pair) == nullptr);

    first.update({.value = "alpha", .focus = true});
    pair.place(scene.ctx, bounds);
    CHECK(first.focused());

    second.update({.value = "beta", .focus = true});
    pair.place(scene.ctx, bounds);
    CHECK(first.focused());
    CHECK_FALSE(second.focused());
}
