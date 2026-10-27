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
#include "render/sakura/metrics.hh"

#include <algorithm>
#include <cmath>
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

TEST_CASE("Setting and feature cards lay their children out over the card surface", "[core][sakura][settings]") {
    const Scene scene;
    BackdropProbe control({80.0f, 24.0f});
    Sakura::SettingCard setting;
    setting.update({.rows = {{.label = "Row", .control = &control}}});
    Host settingHost(setting);
    settingHost.place(scene.ctx, {0.0f, 0.0f, 600.0f, 100.0f});
    CHECK(control.backdrop == "welcome_card_bg");

    BackdropProbe action({80.0f, 24.0f});
    BackdropProbe content({200.0f, 40.0f});
    Sakura::FeatureCard feature;
    feature.update({.title = "Title", .action = &action, .content = &content});
    Host featureHost(feature);
    featureHost.place(scene.ctx, {0.0f, 0.0f, 600.0f, 300.0f});
    CHECK(action.backdrop == "welcome_card_bg");
    CHECK(content.backdrop == "welcome_card_bg");
}

TEST_CASE("Text shrinks a single line to fit the width only when asked", "[core][sakura][settings]") {
    const Scene scene;
    Sakura::Text text;
    Host host(text);
    const auto update = [&](bool shrink) {
        text.update({.str = "CyberEther", .shrinkToFit = shrink, .scale = 4.0f});
    };

    update(false);
    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(host.measureOf(scene.ctx, {natural.x * 0.5f, kInfinite}).y == natural.y);

    update(true);
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == natural);
    CHECK(host.measureOf(scene.ctx, {natural.x * 2.0f, kInfinite}) == natural);
    const auto half = host.measureOf(scene.ctx, {natural.x * 0.5f, kInfinite});
    CHECK(half.x <= natural.x * 0.5f);
    CHECK(half.x > natural.x * 0.4f);
    CHECK(half.y < natural.y * 0.6f);

    host.place(scene.ctx, {0.0f, 0.0f, natural.x * 0.5f, natural.y});
    const auto* label = First<Sakura::Label>(text);
    REQUIRE(label != nullptr);
    REQUIRE(label->config().instances.size() == 1);
    CHECK(label->config().instances.front().fontSize < Sakura::Typography::FontSize * 4.0f * 0.6f);
    CHECK(label->config().instances.front().rect.right() <= natural.x * 0.5f);

    host.place(scene.ctx, {0.0f, 0.0f, natural.x, natural.y});
    CHECK(First<Sakura::Label>(text)->config().instances.front().fontSize == Sakura::Typography::FontSize * 4.0f);
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

TEST_CASE("Text inputs size their line to the configured font size", "[core][sakura][settings]") {
    const Scene scene;
    Sakura::TextInput input;
    input.update({.hint = "Search"});
    Host host(input);
    const auto base = host.measureOf(scene.ctx, {300.0f, 500.0f});
    input.update({.hint = "Search", .fontSize = 21.0f});
    const auto large = host.measureOf(scene.ctx, {300.0f, 500.0f});
    CHECK(large.y - base.y == 21.0f - Sakura::Typography::FontSize);
}

TEST_CASE("Feature cards repaint on text changes, restructure once for new parts, and park removed ones",
          "[core][sakura][settings][retention]") {
    Scene scene;
    Probe action({80.0f, 24.0f});
    Probe content({200.0f, 60.0f});
    Sakura::FeatureCard card;
    Place place(card, {0.0f, 0.0f, 600.0f, 400.0f});
    Frames frames(place, kScreen, scene.context());
    std::string title = "Update available";
    std::string description = "Restart to apply it.";
    bool parts = false;
    const auto update = [&] {
        card.update({
            .title = title,
            .description = description,
            .action = parts ? &action : nullptr,
            .content = parts ? &content : nullptr,
        });
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    title = "Update ready";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    description = "Restart now to apply it.";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);

    parts = true;
    const auto added = frames.run(update);
    CHECK(added.restructured);
    CHECK_FALSE(added.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(frames.restructures == 1);
    CHECK(action.visible());
    CHECK(content.visible());

    parts = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK_FALSE(action.visible());
    CHECK_FALSE(content.visible());

    parts = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 1);
}

TEST_CASE("Feature cards scale their paddings with the pixel ratio", "[core][sakura][settings]") {
    const Scene scene;
    Probe action({80.0f, 24.0f});
    Probe content({200.0f, 60.0f});
    Sakura::FeatureCard card;
    card.update({.title = "Update available", .action = &action, .content = &content});
    Host host(card);
    host.place(scene.ctx, {0.0f, 0.0f, 600.0f, 400.0f});
    CHECK(Peek::Frame(action).right() == 600.0f - 18.0f);
    CHECK(Peek::Frame(content).x == 18.0f);
    CHECK(Peek::Frame(content).width == 600.0f - 36.0f);

    host.place(Doubled(scene.ctx), {0.0f, 0.0f, 600.0f, 400.0f});
    CHECK(Peek::Frame(action).right() == 600.0f - 36.0f);
    CHECK(Peek::Frame(content).x == 36.0f);
    CHECK(Peek::Frame(content).width == 600.0f - 72.0f);
    CHECK(Peek::Frame(content).y > Peek::Frame(action).bottom() + 36.0f);
}

TEST_CASE("Sidebar items repaint on selection, measure a natural width when unbounded, and scale with the pixel ratio",
          "[core][sakura][settings]") {
    Scene scene;
    Sakura::SidebarItem item;
    bool selected = false;
    Place place(item, {10.0f, 10.0f, 200.0f, 30.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        item.update({.icon = "*", .label = "General", .selected = selected});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    selected = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    selected = false;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);

    Sakura::SidebarItem measured;
    measured.update({.icon = "*", .label = "General"});
    Host host(measured);
    const auto label = [&] {
        for (auto* text : Find<Sakura::Text>(measured)) {
            if (text->config().str == "General") {
                return Peek::Frame(*text);
            }
        }
        return Rect{};
    };
    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(natural.y == 30.0f);
    CHECK(host.measureOf(scene.ctx, {240.0f, 500.0f}).x == 240.0f);
    host.place(scene.ctx, {0.0f, 0.0f, 400.0f, 30.0f});
    CHECK(label().x == 33.0f);
    CHECK(natural.x == std::ceil(label().right() + 9.0f));

    const auto doubled = Doubled(scene.ctx);
    const auto wide = host.measureOf(doubled, {kInfinite, kInfinite});
    CHECK(wide.y == 60.0f);
    host.place(doubled, {0.0f, 0.0f, 800.0f, 60.0f});
    CHECK(label().x == 66.0f);
    CHECK(wide.x == std::ceil(label().right() + 18.0f));
}

TEST_CASE("Setting cards repaint on label and description changes and scale with the pixel ratio",
          "[core][sakura][settings]") {
    Scene scene;
    Sakura::SettingCard card;
    Place place(card, {10.0f, 10.0f, 600.0f, 300.0f});
    Frames frames(place, kScreen, scene.context());
    std::string label = "Telemetry";
    std::string description = "Applies on restart.";
    const auto update = [&] {
        card.update({.rows = {{.label = label, .description = description}, {.label = "Second"}}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    label = "Diagnostics";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    description = "Applies immediately.";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);

    Probe wide({400.0f, 20.0f});
    Probe narrow({400.0f, 20.0f});
    Sakura::SettingCard scaled;
    scaled.update({.rows = {{.label = "Wide", .control = &wide, .wide = true}, {.label = "Narrow", .control = &narrow}}});
    Host host(scaled);
    host.place(Doubled(scene.ctx), {0.0f, 0.0f, 600.0f, 400.0f});
    CHECK(Peek::Frame(narrow).width == 264.0f);
    CHECK(Peek::Frame(narrow).right() == 600.0f - 36.0f);
    CHECK(Peek::Frame(wide).right() == 600.0f - 36.0f);
    const auto* separators = dynamic_cast<const Sakura::Box*>(Peek::Children(scaled).at(2));
    REQUIRE(separators != nullptr);
    REQUIRE(separators->config().instances.size() == 1);
    const Rect line = separators->config().instances.front().rect;
    CHECK(line.x == 36.0f);
    CHECK(line.width == 600.0f - 72.0f);
    CHECK(line.height == 2.0f);
}

TEST_CASE("Setting cards keep a viewport separator budget and cull rules outside the clip",
          "[core][sakura][settings]") {
    struct Window : Place {
        Window(Sakura::Component& child, Rect target) : Place(child, target) {
            setClipsChildren(true);
        }
    };

    const Scene scene;
    Sakura::SettingCard card;
    Window window(card, {0.0f, 0.0f, 600.0f, 300.0f});
    const auto rows = [&](U64 count, F32 top) {
        std::vector<Sakura::SettingCard::Row> list(count);
        for (auto& row : list) {
            row.label = "Row";
        }
        card.update({.rows = std::move(list)});
        window.move({0.0f, top, 600.0f, count * 60.0f});
        window.place(scene.ctx, {0.0f, 0.0f, 600.0f, 300.0f});
    };
    const auto separators = [&]() -> const Sakura::Box::Config& {
        const auto* box = dynamic_cast<const Sakura::Box*>(Peek::Children(card).at(2));
        REQUIRE(box != nullptr);
        return box->config();
    };

    rows(3, 0.0f);
    const U64 capacity = separators().capacity;
    CHECK(capacity > 2);
    CHECK(separators().instances.size() == 2);

    rows(400, -2000.0f);
    CHECK(separators().capacity == capacity);
    REQUIRE_FALSE(separators().instances.empty());
    CHECK(separators().instances.size() <= capacity);
    for (const auto& instance : separators().instances) {
        CHECK(instance.rect.bottom() >= 0.0f);
        CHECK(instance.rect.y < 300.0f);
    }

    rows(1, 0.0f);
    CHECK(separators().capacity == capacity);
    CHECK(separators().instances.empty());
}

TEST_CASE("Text inputs scale their font size with the pixel ratio and draw the hint in the configured font",
          "[core][sakura][settings]") {
    const Scene scene;
    Sakura::TextInput input;
    input.update({.hint = "Search", .fontSize = 21.0f, .fontName = Sakura::Typography::BodyFont});
    Host host(input);
    const auto single = host.measureOf(scene.ctx, {300.0f, 500.0f});
    CHECK(single.y == 31.0f);
    const auto doubled = host.measureOf(Doubled(scene.ctx), {600.0f, 500.0f});
    CHECK(doubled.y == 62.0f);

    host.place(Doubled(scene.ctx), {0.0f, 0.0f, 600.0f, doubled.y});
    bool found = false;
    for (const auto* label : Find<Sakura::Label>(input)) {
        if (label->config().instances.empty() || label->config().instances.front().str != "Search") {
            continue;
        }
        found = true;
        CHECK(label->config().fontName == Sakura::Typography::BodyFont);
        CHECK(label->config().instances.front().fontSize == 42.0f);
        CHECK(label->config().instances.front().rect.x == 24.0f);
    }
    CHECK(found);
}

TEST_CASE("Sidebar items indent their content and collapse the icon slot without an icon",
          "[core][sakura][settings]") {
    Scene scene;
    struct Placed {
        Extent2D<F32> natural = {0.0f, 0.0f};
        Rect label = {};
        U64 shown = 0;
    };
    const auto layoutOf = [&](const Sakura::SidebarItem::Config& config, const Sakura::Context& ctx, F32 width) {
        Sakura::SidebarItem item;
        item.update(config);
        Host host(item);
        Placed placed;
        placed.natural = host.measureOf(ctx, {kInfinite, kInfinite});
        host.place(ctx, {0.0f, 0.0f, width, placed.natural.y});
        for (auto* text : Find<Sakura::Text>(item)) {
            ++placed.shown;
            if (text->config().str == "General") {
                placed.label = Peek::Frame(*text);
            }
        }
        return placed;
    };

    const auto icon = layoutOf({.icon = "*", .label = "General", .indent = 12.0f}, scene.ctx, 400.0f);
    CHECK(icon.shown == 2);
    CHECK(icon.label.x == 9.0f + 12.0f + 16.0f + 8.0f);
    CHECK(icon.natural.x == std::ceil(icon.label.right() + 9.0f));

    const auto bare = layoutOf({.label = "General", .indent = 12.0f}, scene.ctx, 400.0f);
    CHECK(bare.shown == 1);
    CHECK(bare.label.x == 9.0f + 12.0f);
    CHECK(bare.natural.x == std::ceil(bare.label.right() + 9.0f));
    CHECK(bare.natural.x < icon.natural.x);

    const auto doubled = layoutOf({.label = "General", .indent = 12.0f}, Doubled(scene.ctx), 800.0f);
    CHECK(doubled.shown == 1);
    CHECK(doubled.label.x == 2.0f * (9.0f + 12.0f));
    CHECK(doubled.natural.x == std::ceil(doubled.label.right() + 18.0f));
}
