#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fonts.hh"
#include "component_harness.hh"

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/combo.hh>
#include <jetstream/render/sakura/components/progress_bar.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/pill_button.hh>
#include <jetstream/render/sakura/components/qr_code.hh>
#include <jetstream/render/sakura/components/spinner.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/status_bar.hh>
#include <jetstream/render/sakura/components/table.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/components/text_input.hh>

#include "harness.hh"
#include "scene.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <variant>
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

TEST_CASE("Squeezed buttons clip their caption and disabled buttons fade it", "[core][sakura][controls]") {
    const Scene scene;
    Sakura::Button button;
    button.update({.str = "A caption far wider than the button"});
    Host host(button);
    host.place(scene.ctx, {0.0f, 0.0f, 40.0f, 25.0f});

    std::vector<DrawnLeaf> drawn;
    CollectDrawn(button, drawn);
    const auto caption = std::find_if(drawn.begin(), drawn.end(), [](const DrawnLeaf& leaf) {
        return std::holds_alternative<Sakura::Label::Config>(leaf.config);
    });
    REQUIRE(caption != drawn.end());
    REQUIRE(caption->clip.has_value());
    CHECK(caption->clip->right() <= 40.0f);

    CHECK(Find<Sakura::Text>(button).at(0)->config().opacity == 1.0f);
    button.update({.str = "Open", .disabled = true});
    CHECK(Find<Sakura::Text>(button).at(0)->config().opacity == 0.6f);
    button.update({.str = "Open", .variant = Sakura::Button::Variant::Text, .disabled = true});
    CHECK(Find<Sakura::Text>(button).at(0)->config().opacity == 1.0f);
}

TEST_CASE("Long status messages end before the toggle pill", "[core][sakura][controls]") {
    const Scene scene;
    Sakura::StatusBar bar;
    const std::string status = "Invalid YAML. Fix the document before applying changes.";
    bar.update({.id = "status", .status = status, .toggleText = "Console", .onToggle = [] {}});
    Host host(bar);
    host.place(scene.ctx, {0.0f, 0.0f, 200.0f, 24.0f});

    const auto& message = Find<Sakura::Label>(bar).at(0)->config().instances.at(0);
    CHECK(message.str != status);
    CHECK(message.str.ends_with("..."));

    host.place(scene.ctx, {0.0f, 0.0f, 90.0f, 24.0f});
    CHECK_FALSE(Find<Sakura::Label>(bar).at(0)->config().instances.at(0).visible);
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

TEST_CASE("Password text inputs draw a mask instead of their text", "[core][sakura][controls]") {
    const Scene scene;
    Sakura::TextInput input;
    input.update({.value = "hunter2", .password = true});
    Host host(input);
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 25.0f});

    std::vector<std::string> drawn;
    for (auto* label : Find<Sakura::Label>(input)) {
        for (const auto& instance : label->config().instances) {
            if (instance.visible && !instance.str.empty()) {
                drawn.push_back(instance.str);
            }
        }
    }
    REQUIRE_FALSE(drawn.empty());
    for (const auto& str : drawn) {
        CHECK(str.find("hunter") == std::string::npos);
    }
    CHECK(std::find(drawn.begin(), drawn.end(), "*******") != drawn.end());

    input.update({.value = "hunter2"});
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 25.0f});
    bool plain = false;
    for (auto* label : Find<Sakura::Label>(input)) {
        for (const auto& instance : label->config().instances) {
            plain = plain || (instance.visible && instance.str == "hunter2");
        }
    }
    CHECK(plain);
}

TEST_CASE("Disabled text inputs ignore clicks, show no cursor, and give up focus", "[core][sakura][controls]") {
    const Scene scene;
    std::vector<std::string> changes;
    Sakura::TextInput input;
    const auto configure = [&](bool disabled) {
        input.update({.value = "alpha", .disabled = disabled,
                      .onChange = [&](const std::string& value) { changes.push_back(value); }});
    };
    configure(false);
    Host host(input);
    const Rect bounds = {0.0f, 0.0f, 300.0f, 25.0f};
    host.place(scene.ctx, bounds);
    CHECK(Peek::CursorAt(input, {40.0f, 12.0f}) == SurfaceCursor::TextInput);
    CHECK(host.click(40.0f, 12.0f));
    REQUIRE(input.focused());

    configure(true);
    host.place(scene.ctx, bounds);
    CHECK_FALSE(input.focused());
    CHECK_FALSE(Peek::CursorAt(input, {40.0f, 12.0f}).has_value());
    CHECK(Sakura::Access::cursor(host, {40.0f, 12.0f}) == SurfaceCursor::Default);
    CHECK_FALSE(host.click(40.0f, 12.0f));
    host.place(scene.ctx, bounds);
    CHECK_FALSE(input.focused());
    Type(host, "b");
    CHECK(changes.empty());

    configure(true);
    input.update({.value = "alpha", .focus = true, .disabled = true});
    host.place(scene.ctx, bounds);
    CHECK_FALSE(input.focused());
}

TEST_CASE("Spinners measure their size and turn their arc as time passes", "[core][sakura][controls]") {
    auto ctx = MakeContext();
    Sakura::Spinner spinner;
    Host host(spinner);
    spinner.update({.diameter = 20.0f, .thickness = 2.0f, .period = 0.8f});
    CHECK(host.measureOf(ctx, {100.0f, 100.0f}) == Extent2D<F32>{20.0f, 20.0f});

    const auto head = [&](F64 time) {
        ctx.time = time;
        host.place(ctx, {10.0f, 10.0f, 20.0f, 20.0f});
        const auto lines = Direct<Sakura::Polyline>(spinner);
        REQUIRE(lines.size() == 1);
        const auto& instances = lines.front()->config().instances;
        REQUIRE(instances.size() == 2);
        CHECK(instances[0].closed);
        return instances[1].points.front();
    };
    const auto start = head(0.0);
    CHECK(start.x == Catch::Approx(29.0f).margin(0.01f));
    CHECK(start.y == Catch::Approx(20.0f).margin(0.01f));
    const auto quarter = head(0.2);
    CHECK(quarter.x == Catch::Approx(20.0f).margin(0.01f));
    CHECK(quarter.y == Catch::Approx(29.0f).margin(0.01f));
    const auto turned = head(0.8);
    CHECK(turned.x == Catch::Approx(start.x).margin(0.01f));
    CHECK(turned.y == Catch::Approx(start.y).margin(0.01f));
}

TEST_CASE("Spinners resolve their diameter and scale with the pixel ratio", "[core][sakura][controls]") {
    auto ctx = MakeContext();
    Sakura::Spinner spinner;
    Host host(spinner);
    spinner.update({});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{16.0f, 16.0f});
    CHECK(host.measureOf(ctx, {10.0f, 100.0f}) == Extent2D<F32>{10.0f, 10.0f});

    spinner.update({.diameter = 24.0f});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{24.0f, 24.0f});
    CHECK(host.measureOf(ctx, {100.0f, 12.0f}) == Extent2D<F32>{12.0f, 12.0f});

    spinner.update({.diameter = Sakura::Fill});
    CHECK(host.measureOf(ctx, {80.0f, 50.0f}) == Extent2D<F32>{50.0f, 50.0f});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{16.0f, 16.0f});

    auto dense = ctx;
    dense.pixelRatio = 2.0f;
    spinner.update({.diameter = 20.0f, .thickness = 2.0f, .period = 0.8f});
    CHECK(host.measureOf(dense, {kInfinite, kInfinite}) == Extent2D<F32>{40.0f, 40.0f});
    host.place(dense, {0.0f, 0.0f, 40.0f, 40.0f});
    const auto lines = Direct<Sakura::Polyline>(spinner);
    REQUIRE(lines.size() == 1);
    const auto& instances = lines.front()->config().instances;
    REQUIRE(instances.size() == 2);
    CHECK(instances[0].thickness == 4.0f);
    CHECK(instances[1].points.front().x == Catch::Approx(38.0f).margin(0.01f));
}

TEST_CASE("Spinner tracks are closed rings without a duplicated seam point at the track alpha",
          "[core][sakura][controls]") {
    auto ctx = MakeContext();
    Sakura::Spinner spinner;
    Host host(spinner);
    spinner.update({.diameter = 20.0f});
    host.place(ctx, {10.0f, 10.0f, 20.0f, 20.0f});
    const auto lines = Direct<Sakura::Polyline>(spinner);
    REQUIRE(lines.size() == 1);
    const auto& instances = lines.front()->config().instances;
    REQUIRE(instances.size() == 2);
    const auto& track = instances[0];
    REQUIRE(track.points.size() > 2);
    CHECK(track.closed);
    const auto first = track.points.front();
    const auto last = track.points.back();
    CHECK(std::hypot(last.x - first.x, last.y - first.y) > 0.5f);
    CHECK(track.color.a == Catch::Approx(instances[1].color.a * 0.2f));
}

TEST_CASE("Spinners repaint only while the clock ticks and settle when frozen or static",
          "[core][sakura][controls][retention]") {
    Scene scene;
    Sakura::Spinner spinner;
    Place place(spinner, {10.0f, 10.0f, 20.0f, 20.0f});
    Frames frames(place, kScreen, scene.context());
    F32 period = 0.9f;
    const auto update = [&] {
        spinner.update({.period = period});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    frames.timeStep = 0.05;
    for (U64 frame = 0; frame < 4; ++frame) {
        CAPTURE(frame);
        CHECK(RepaintOnly(frames.run(update)));
    }

    frames.timeStep = 0.0;
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    period = 0.0f;
    frames.timeStep = 0.05;
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Disabled pill buttons ignore clicks, drop hover and the hand cursor, and fade",
          "[core][sakura][controls]") {
    const Scene scene;
    auto ctx = scene.context();
    ctx.palette = DistinctPalette();
    U64 clicked = 0;
    Sakura::PillButton button;
    const auto configure = [&](bool disabled) {
        button.update({.icon = "+", .label = "Send", .disabled = disabled, .onClick = [&] { ++clicked; }});
    };
    const auto fill = [&] {
        return Find<Sakura::Box>(button).at(1)->config().instances.at(0).backgroundColor;
    };
    Host host(button);
    configure(false);
    host.place(ctx, {10.0f, 10.0f, 120.0f, 28.0f});
    const auto enabled = fill();
    CHECK(Peek::CursorAt(button, {50.0f, 24.0f}) == SurfaceCursor::Hand);
    CHECK(host.click(50.0f, 24.0f));
    CHECK(clicked == 1);
    host.send(Mouse(MouseEventType::Move, 50.0f, 24.0f));
    host.place(ctx, {10.0f, 10.0f, 120.0f, 28.0f});
    CHECK(fill().r == ctx.color("pill_hover_bg").r);

    configure(true);
    host.place(ctx, {10.0f, 10.0f, 120.0f, 28.0f});
    CHECK(fill().r == ctx.color("pill_bg").r);
    CHECK(fill().a == Catch::Approx(enabled.a * 0.6f));
    for (const auto* text : Find<Sakura::Text>(button)) {
        CHECK(text->config().opacity == 0.6f);
    }
    CHECK_FALSE(Peek::CursorAt(button, {50.0f, 24.0f}).has_value());
    CHECK_FALSE(host.click(50.0f, 24.0f));
    CHECK_FALSE(host.send(Mouse(MouseEventType::Move, 50.0f, 24.0f)));
    CHECK(clicked == 1);

    configure(false);
    host.place(ctx, {10.0f, 10.0f, 120.0f, 28.0f});
    CHECK(fill() == enabled);
    CHECK(host.click(50.0f, 24.0f));
    CHECK(clicked == 2);
}

TEST_CASE("Pill buttons repaint without rebuilding when they toggle disabled", "[core][sakura][controls]") {
    Scene scene;
    auto ctx = scene.context();
    ctx.palette = DistinctPalette();
    Sakura::PillButton button;
    Place place(button, {10.0f, 10.0f, 120.0f, 28.0f});
    Frames frames(place, kScreen, ctx);
    bool disabled = false;
    const auto update = [&] {
        button.update({.label = "Send", .disabled = disabled, .onClick = [] {}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    disabled = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 24.0f}, {60.0f, 24.0f}}, [&](const MouseEvent& event) {
        return place.send(event);
    }));

    disabled = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Tables shift their rows with the elastic overscroll and spring back", "[core][sakura][controls]") {
    std::vector<std::unique_ptr<Probe>> probes;
    Sakura::Table::Config config{.columns = {"Name", "Value"}};
    for (U64 row = 0; row < 10; ++row) {
        auto& name = *probes.emplace_back(std::make_unique<Probe>(Extent2D<F32>{38.0f, 10.0f}));
        auto& value = *probes.emplace_back(std::make_unique<Probe>(Extent2D<F32>{138.0f, 10.0f}));
        name.consume = false;
        value.consume = false;
        config.cellRows.push_back({&name, &value});
    }
    Sakura::Table table;
    table.update(config);
    Host host(table);
    auto ctx = MakeContext();
    ctx.time = 1.0;
    const Rect bounds = {0.0f, 0.0f, 400.0f, 100.0f};
    host.place(ctx, bounds);
    const F32 rest = probes[0]->laidFrame.y;
    const auto trackpad = [&](F32 dy, ScrollPhase phase) {
        auto event = Wheel(50.0f, 50.0f, 0.0f, dy);
        event.phase = phase;
        host.send(event);
        host.place(ctx, bounds);
    };

    trackpad(0.0f, ScrollPhase::Began);
    trackpad(3.0f, ScrollPhase::Changed);
    const F32 stretched = probes[0]->laidFrame.y;
    CHECK(stretched > rest);
    CHECK(probes[2]->laidFrame.y - probes[0]->laidFrame.y > 0.0f);

    trackpad(0.0f, ScrollPhase::Ended);
    for (U64 frame = 0; frame < 120; ++frame) {
        ctx.time += 1.0 / 60.0;
        host.place(ctx, bounds);
    }
    CHECK(probes[0]->laidFrame.y == rest);
}

TEST_CASE("Combo dropdown rows follow the elastic overscroll for drawing and picking", "[core][sakura][controls]") {
    const Scene scene;
    std::vector<std::string> options;
    for (U64 index = 0; index < 20; ++index) {
        options.push_back("option " + std::to_string(index));
    }
    std::optional<U64> selected;
    Sakura::Combo combo;
    combo.update({
        .id = "elastic",
        .options = options,
        .value = options[0],
        .onSelect = [&](U64 index) { selected = index; },
    });
    LayerSlot mount;
    mount.hold(combo, {100.0f, 100.0f, 200.0f, 25.0f});
    LayerHost host({&mount});
    auto ctx = scene.ctx;
    ctx.time = 1.0;
    host.place(ctx, kScreen);
    host.click(150.0f, 112.0f);
    host.place(ctx, kScreen);

    const auto row = [&](const std::string& str) -> std::optional<Rect> {
        for (auto* label : Find<Sakura::Label>(host.popups)) {
            for (const auto& instance : label->config().instances) {
                if (instance.str == str && instance.visible) {
                    return instance.rect;
                }
            }
        }
        return std::nullopt;
    };
    const auto rest = row("option 2");
    REQUIRE(rest.has_value());

    const F32 x = rest->x + 20.0f;
    const F32 y = rest->y + rest->height * 0.5f;
    auto began = Wheel(x, y, 0.0f, 0.0f);
    began.phase = ScrollPhase::Began;
    host.send(began);
    host.place(ctx, kScreen);
    auto changed = Wheel(x, y, 0.0f, 3.0f);
    changed.phase = ScrollPhase::Changed;
    host.send(changed);
    host.place(ctx, kScreen);

    const auto stretched = row("option 2");
    REQUIRE(stretched.has_value());
    REQUIRE(stretched->y > rest->y);
    host.click(x, stretched->y + stretched->height * 0.5f);
    REQUIRE(selected.has_value());
    CHECK(*selected == 2);
}

TEST_CASE("Reopening a combo dropdown drops the previous opening stretch", "[core][sakura][controls]") {
    const Scene scene;
    std::vector<std::string> options;
    for (U64 index = 0; index < 30; ++index) {
        options.push_back("option " + std::to_string(index));
    }
    Sakura::Combo combo;
    combo.update({.id = "reopen", .options = options, .value = options[0]});
    LayerSlot mount;
    mount.hold(combo, {100.0f, 100.0f, 200.0f, 25.0f});
    LayerHost host({&mount});
    auto ctx = scene.ctx;
    ctx.time = 1.0;
    host.place(ctx, kScreen);
    const auto row = [&]() -> std::optional<Rect> {
        for (auto* label : Find<Sakura::Label>(host.popups)) {
            for (const auto& instance : label->config().instances) {
                if (instance.str == "option 0" && instance.visible) {
                    return instance.rect;
                }
            }
        }
        return std::nullopt;
    };
    host.click(150.0f, 112.0f);
    host.place(ctx, kScreen);
    const auto rest = row();
    REQUIRE(rest.has_value());

    auto stretch = Wheel(rest->x + 20.0f, rest->y + rest->height * 0.5f, 0.0f, 4.0f);
    stretch.phase = ScrollPhase::Changed;
    host.send(stretch);
    host.place(ctx, kScreen);
    REQUIRE(row().has_value());
    REQUIRE(row()->y > rest->y + 1.0f);

    host.click(150.0f, 112.0f);
    host.place(ctx, kScreen);
    REQUIRE_FALSE(row().has_value());
    host.click(150.0f, 112.0f);
    host.place(ctx, kScreen);
    REQUIRE(row().has_value());
    CHECK(row()->y == Catch::Approx(rest->y));
}
