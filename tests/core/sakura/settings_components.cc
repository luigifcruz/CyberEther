#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/badge.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/combo.hh>
#include <jetstream/render/sakura/components/feature_card.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/panel.hh>
#include <jetstream/render/sakura/components/setting_card.hh>
#include <jetstream/render/sakura/components/sidebar_item.hh>
#include <jetstream/render/sakura/components/split_view.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/table.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_input.hh>
#include <jetstream/render/sakura/components/toggle.hh>

#include "component_harness.hh"
#include "harness.hh"
#include "scene.hh"

#include <algorithm>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

Sakura::Context Doubled(const Sakura::Context& ctx) {
    auto next = ctx;
    next.pixelRatio = 2.0f;
    return next;
}

bool Owns(const Sakura::Component& parent, const Sakura::Component& child) {
    const auto& children = Peek::Children(parent);
    return std::find(children.begin(), children.end(), &child) != children.end();
}

template<typename T>
const T* First(Sakura::Component& root) {
    const auto found = Find<T>(root);
    return found.empty() ? nullptr : found.front();
}

}  // namespace

TEST_CASE("Toggles flip on release inside and ignore cancelled or outside releases", "[core][sakura][settings]") {
    const Scene scene;
    std::vector<bool> changes;
    Sakura::Toggle toggle;
    toggle.update({.value = false, .onChange = [&](bool value) { changes.push_back(value); }});
    Host host(toggle);
    host.place(scene.ctx, {0.0f, 0.0f, 38.0f, 22.0f});

    host.send(Mouse(MouseEventType::Click, 10.0f, 10.0f));
    CHECK(changes.empty());
    host.send(Mouse(MouseEventType::Release, 10.0f, 10.0f));
    CHECK(changes == std::vector<bool>{true});

    host.send(Mouse(MouseEventType::Click, 10.0f, 10.0f));
    host.send(Mouse(MouseEventType::Release, 200.0f, 10.0f));
    host.send(Mouse(MouseEventType::Click, 10.0f, 10.0f));
    host.send(Mouse(MouseEventType::Cancel, 10.0f, 10.0f));
    host.send(Mouse(MouseEventType::Release, 10.0f, 10.0f));
    CHECK(changes.size() == 1);

    toggle.update({.value = false, .disabled = true, .onChange = [&](bool value) { changes.push_back(value); }});
    CHECK_FALSE(host.click(10.0f, 10.0f));
    CHECK(changes.size() == 1);
}

TEST_CASE("Toggles measure a fixed pill that scales with the pixel ratio", "[core][sakura][settings]") {
    const Scene scene;
    Sakura::Toggle toggle;
    toggle.update({});
    Host host(toggle);
    CHECK(host.measureOf(scene.ctx, {500.0f, 500.0f}) == Extent2D<F32>{38.0f, 22.0f});
    CHECK(host.measureOf(Doubled(scene.ctx), {500.0f, 500.0f}) == Extent2D<F32>{76.0f, 44.0f});
}

TEST_CASE("Toggles and sidebar items settle and repaint once per hover transition", "[core][sakura][settings]") {
    Scene scene;
    {
        Sakura::Toggle toggle;
        Place place(toggle, {10.0f, 10.0f, 38.0f, 22.0f});
        Frames frames(place, kScreen, scene.context());
        const auto update = [&] {
            toggle.update({.value = true, .onChange = [](bool) {}});
        };
        REQUIRE(frames.settle(update));
        CHECK(Retains(frames, update));
        CheckHoverRepaintsOnce(frames, place, update, {20.0f, 20.0f}, {400.0f, 300.0f});
    }
    {
        Sakura::SidebarItem item;
        Place place(item, {10.0f, 10.0f, 200.0f, 30.0f});
        Frames frames(place, kScreen, scene.context());
        const auto update = [&] {
            item.update({.icon = "*", .label = "General", .onSelect = [] {}});
        };
        REQUIRE(frames.settle(update));
        CHECK(Retains(frames, update));
        CheckHoverRepaintsOnce(frames, place, update, {50.0f, 20.0f}, {400.0f, 300.0f});
    }
}

TEST_CASE("Sidebar items select on release inside and fill the width at a fixed height",
          "[core][sakura][settings]") {
    const Scene scene;
    U64 selected = 0;
    Sakura::SidebarItem item;
    item.update({.icon = "*", .label = "General", .onSelect = [&] { ++selected; }});
    Host host(item);
    CHECK(host.measureOf(scene.ctx, {240.0f, 500.0f}) == Extent2D<F32>{240.0f, 30.0f});
    CHECK(host.measureOf(Doubled(scene.ctx), {240.0f, 500.0f}) == Extent2D<F32>{240.0f, 60.0f});
    host.place(scene.ctx, {0.0f, 0.0f, 240.0f, 30.0f});

    host.send(Mouse(MouseEventType::Click, 50.0f, 15.0f));
    CHECK(selected == 0);
    host.send(Mouse(MouseEventType::Release, 50.0f, 15.0f));
    CHECK(selected == 1);
    host.send(Mouse(MouseEventType::Click, 50.0f, 15.0f));
    host.send(Mouse(MouseEventType::Release, 500.0f, 15.0f));
    host.send(Mouse(MouseEventType::Click, 50.0f, 15.0f));
    host.send(Mouse(MouseEventType::Cancel, 50.0f, 15.0f));
    host.send(Mouse(MouseEventType::Release, 50.0f, 15.0f));
    CHECK(selected == 1);
}

TEST_CASE("Badges wrap their label in padding and repaint without rebuilding on change", "[core][sakura][settings]") {
    Scene scene;
    Sakura::Badge badge;
    std::string label = "Ready";
    Place place(badge, {10.0f, 10.0f, 120.0f, 22.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        badge.update({.label = label});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    label = "Checking...";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(frames.settle(update));

    Sakura::Badge measured;
    measured.update({.label = "Ready"});
    Host host(measured);
    const auto size = host.measureOf(scene.ctx, {500.0f, 500.0f});
    CHECK(size.y == 22.0f);
    CHECK(size.x > 20.0f);
    CHECK(host.measureOf(Doubled(scene.ctx), {500.0f, 500.0f}).y == 44.0f);
}

TEST_CASE("Setting card rows click on release inside the same row", "[core][sakura][settings]") {
    const Scene scene;
    std::vector<std::string> clicks;
    Sakura::SettingCard card;
    card.update({.rows = {
        {.label = "First", .description = "One", .onClick = [&] { clicks.push_back("first"); }},
        {.label = "Second", .description = "Two", .onClick = [&] { clicks.push_back("second"); }},
        {.label = "Static", .description = "Three"},
    }});
    Host host(card);
    const F32 row = host.measureOf(scene.ctx, {600.0f, kInfinite}).y / 3.0f;
    host.place(scene.ctx, {0.0f, 0.0f, 600.0f, row * 3.0f});

    CHECK(host.click(100.0f, row * 0.5f));
    CHECK(host.click(100.0f, row * 1.5f));
    CHECK_FALSE(host.click(100.0f, row * 2.5f));
    CHECK(clicks == std::vector<std::string>{"first", "second"});

    host.send(Mouse(MouseEventType::Click, 100.0f, row * 0.5f));
    host.send(Mouse(MouseEventType::Release, 100.0f, row * 1.5f));
    host.send(Mouse(MouseEventType::Click, 100.0f, row * 0.5f));
    host.send(Mouse(MouseEventType::Cancel, 100.0f, row * 0.5f));
    host.send(Mouse(MouseEventType::Release, 100.0f, row * 0.5f));
    CHECK(clicks.size() == 2);
}

TEST_CASE("Setting cards settle, repaint once per row hover, and park rows they no longer show",
          "[core][sakura][settings]") {
    Scene scene;
    Sakura::SettingCard card;
    Sakura::Toggle toggle;
    U64 rows = 3;
    Place place(card, {10.0f, 10.0f, 600.0f, 300.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        toggle.update({.value = true});
        std::vector<Sakura::SettingCard::Row> list = {
            {.label = "Clickable", .description = "Opens a page", .onClick = [] {}},
            {.label = "Without description", .control = &toggle},
        };
        if (rows > 2) {
            list.push_back({.label = "Third", .description = "Extra"});
        }
        card.update({.rows = std::move(list)});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CheckHoverRepaintsOnce(frames, place, update, {100.0f, 20.0f}, {700.0f, 500.0f});

    const U64 children = Peek::Children(card).size();
    rows = 2;
    frames.run(update);
    CHECK(Peek::Children(card).size() == children);
    CHECK(frames.settle(update));
}

TEST_CASE("Setting cards show the empty message, size wide controls, and collapse when blank",
          "[core][sakura][settings]") {
    const Scene scene;
    Sakura::SettingCard empty;
    empty.update({.empty = "Nothing here."});
    Host emptyHost(empty);
    CHECK(emptyHost.measureOf(scene.ctx, {600.0f, kInfinite}).y > 44.0f);

    Sakura::SettingCard blank;
    blank.update({});
    Host blankHost(blank);
    CHECK(blankHost.measureOf(scene.ctx, {600.0f, kInfinite}).y == 0.0f);

    Probe wide({400.0f, 20.0f});
    Probe narrow({400.0f, 20.0f});
    Sakura::SettingCard card;
    card.update({.rows = {
        {.label = "Wide", .control = &wide, .wide = true},
        {.label = "Narrow", .control = &narrow},
    }});
    Host host(card);
    host.place(scene.ctx, {0.0f, 0.0f, 600.0f, 200.0f});
    CHECK(Peek::Frame(narrow).width == 220.0f);
    CHECK(Peek::Frame(wide).width > 220.0f);
    CHECK(Peek::Frame(wide).width <= (600.0f - 36.0f) * 0.6f);
}

TEST_CASE("Feature cards place the action beside the header and the content below it",
          "[core][sakura][settings]") {
    Scene scene;
    Probe action({80.0f, 24.0f});
    Probe content({200.0f, 60.0f});
    Sakura::FeatureCard card;
    Place place(card, {0.0f, 0.0f, 600.0f, 400.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        card.update({.title = "Update available", .description = "Restart to apply it.", .action = &action,
                     .content = &content});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    const auto bounds = Peek::Frame(card);
    const auto button = Peek::Frame(action);
    const auto body = Peek::Frame(content);
    CHECK(button.right() == bounds.right() - 18.0f);
    CHECK(body.y > button.bottom());
    CHECK(body.width == bounds.width - 36.0f);
    CHECK(bounds.height >= body.bottom() - bounds.y);
}

TEST_CASE("Text line height spaces lines and weight reaches the label", "[core][sakura][settings]") {
    const Scene scene;
    Sakura::Text plain;
    plain.update({.str = "Line"});
    Host plainHost(plain);
    const F32 single = plainHost.measureOf(scene.ctx, {500.0f, 500.0f}).y;

    Sakura::Text spaced;
    spaced.update({.str = "Line", .lineHeight = 2.0f, .weight = 0.25f});
    Host spacedHost(spaced);
    CHECK(spacedHost.measureOf(scene.ctx, {500.0f, 500.0f}).y == std::round(single * 2.0f));
    spacedHost.place(scene.ctx, {0.0f, 0.0f, 200.0f, 40.0f});
    const auto* label = First<Sakura::Label>(spaced);
    REQUIRE(label != nullptr);
    CHECK(label->config().weight > 0.0f);
}

TEST_CASE("Panels center their content inside the maximum width", "[core][sakura][settings]") {
    const Scene scene;
    Probe content({100.0f, 40.0f});
    Sakura::Panel panel;
    panel.update({.content = &content, .maxWidth = 200.0f, .transparent = true});
    Host host(panel);
    host.place(scene.ctx, {0.0f, 0.0f, 600.0f, 100.0f});
    CHECK(Peek::Frame(content).x == 200.0f);
    CHECK(Peek::Frame(content).width == 200.0f);
}

TEST_CASE("Split view gutters set the gap on both sides of the divider", "[core][sakura][settings]") {
    const Scene scene;
    Probe left({50.0f, 20.0f});
    Probe right({50.0f, 20.0f});
    Sakura::SplitView split;
    split.update({.left = &left, .right = &right, .leftWidth = 100.0f, .height = Sakura::Fill, .gutter = 0.0f});
    Host host(split);
    host.place(scene.ctx, {0.0f, 0.0f, 400.0f, 100.0f});
    CHECK(Peek::Frame(left).width == 100.0f);
    CHECK(Peek::Frame(right).x == 101.0f);
}

TEST_CASE("Text inputs and combos draw an outline only when a border color is set", "[core][sakura][settings]") {
    const Scene scene;
    const auto bordered = [](Sakura::Component& root) {
        const auto boxes = Find<Sakura::Box>(root);
        return std::any_of(boxes.begin(), boxes.end(), [](const Sakura::Box* box) {
            return box->config().borderWidth > 0.0f && !box->config().instances.empty() &&
                   box->config().instances.front().visible;
        });
    };

    Sakura::TextInput input;
    input.update({.value = "a"});
    Host inputHost(input);
    inputHost.place(scene.ctx, {0.0f, 0.0f, 200.0f, 25.0f});
    CHECK_FALSE(bordered(input));
    input.update({.value = "a", .borderColorKey = "border", .rounding = 12.0f});
    inputHost.place(scene.ctx, {0.0f, 0.0f, 200.0f, 25.0f});
    CHECK(bordered(input));
    const auto boxes = Find<Sakura::Box>(input);
    CHECK(std::any_of(boxes.begin(), boxes.end(), [](const Sakura::Box* box) {
        return box->config().cornerRadius == 12.0f;
    }));

    Sakura::Combo combo;
    combo.update({.options = {"A"}, .value = "A"});
    Host comboHost(combo);
    comboHost.place(scene.ctx, {0.0f, 0.0f, 200.0f, 25.0f});
    CHECK_FALSE(bordered(combo));
    combo.update({.options = {"A"}, .value = "A", .borderColorKey = "border"});
    comboHost.place(scene.ctx, {0.0f, 0.0f, 200.0f, 25.0f});
    CHECK(bordered(combo));
}

TEST_CASE("Tables reclaim cells another parent mounted while their config was unchanged",
          "[core][sakura][settings]") {
    Sakura::Text cell;
    cell.update({.str = "Value"});
    Sakura::Table table;
    Sakura::Stack thief;
    const auto update = [&] {
        table.update({.columns = {"Name"}, .cellRows = {{&cell}}});
    };
    const auto owned = [&] {
        for (const auto* child : Peek::Children(table)) {
            if (child == &cell || Owns(*child, cell)) {
                return true;
            }
        }
        return false;
    };
    update();
    CHECK(owned());
    thief.update({.items = {{.child = &cell}}});
    CHECK_FALSE(owned());
    update();
    CHECK(owned());
}

TEST_CASE("Feature cards without an action measure the same every time", "[core][sakura][settings]") {
    const Scene scene;
    Sakura::FeatureCard card;
    card.update({.title = "Update available", .description = "Restart to apply it."});
    Host host(card);
    const auto first = host.measureOf(scene.ctx, {600.0f, kInfinite});
    CHECK(first.y > 0.0f);
    CHECK(host.measureOf(scene.ctx, {600.0f, kInfinite}) == first);
}

TEST_CASE("Height constrained setting cards ignore clicks on rows past their bounds", "[core][sakura][settings]") {
    const Scene scene;
    std::vector<std::string> clicks;
    Sakura::SettingCard card;
    card.update({.rows = {
        {.label = "First", .description = "One", .onClick = [&] { clicks.push_back("first"); }},
        {.label = "Second", .description = "Two", .onClick = [&] { clicks.push_back("second"); }},
    }});
    Host host(card);
    const F32 row = host.measureOf(scene.ctx, {600.0f, kInfinite}).y / 2.0f;
    host.place(scene.ctx, {0.0f, 0.0f, 600.0f, row});
    CHECK(host.click(100.0f, row * 0.5f));
    CHECK_FALSE(host.click(100.0f, row * 1.5f));
    CHECK(clicks == std::vector<std::string>{"first"});
}

TEST_CASE("Toggle knobs stay inside narrow bounds", "[core][sakura][settings]") {
    const Scene scene;
    Sakura::Toggle toggle;
    toggle.update({.value = true});
    Host host(toggle);
    host.place(scene.ctx, {0.0f, 0.0f, 10.0f, 22.0f});
    for (const auto* box : Find<Sakura::Box>(toggle)) {
        for (const auto& instance : box->config().instances) {
            CHECK(instance.rect.x >= 0.0f);
            CHECK(instance.rect.right() <= 10.0f);
        }
    }
}
