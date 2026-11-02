#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/pill_button.hh>
#include <jetstream/render/sakura/components/pill_input.hh>
#include <jetstream/render/sakura/components/text.hh>

#include "component_harness.hh"
#include "harness.hh"
#include "scene.hh"

#include "compositor/default/views/components/card_button.hh"
#include "compositor/default/views/components/compact_button.hh"
#include "compositor/default/views/components/file_field.hh"
#include "compositor/default/views/components/option_stepper.hh"
#include "compositor/default/views/components/flowgraph_cell.hh"
#include "compositor/default/views/components/pagination.hh"
#include "compositor/default/views/components/sidebar_heading.hh"
#include "compositor/default/views/components/sidebar_navigation.hh"
#include "compositor/default/views/components/sidebar_tiles.hh"
#include "compositor/default/views/components/status_badge.hh"

#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using Catch::Approx;

constexpr Rect kSlot = {20.0f, 20.0f, 240.0f, 60.0f};

Sakura::Context Doubled(const Sakura::Context& ctx) {
    auto next = ctx;
    next.pixelRatio = 2.0f;
    return next;
}

template<typename T>
void CheckFirstMeasure(T& component, const Sakura::Context& ctx, Extent2D<F32> available, Rect bounds) {
    Host host(component);
    const auto first = host.measureOf(ctx, available);
    CHECK(first.x > 0.0f);
    CHECK(first.y > 0.0f);
    host.place(ctx, bounds);
    CHECK(host.measureOf(ctx, available) == first);
    const auto doubled = host.measureOf(Doubled(ctx), {available.x * 2.0f, available.y * 2.0f});
    CHECK(doubled.y == Approx(first.y * 2.0f).margin(2.0f));
    CHECK(doubled.x >= first.x);
}

bool ClickAt(Place& place, Extent2D<F32> point) {
    const bool pressed = place.send(Mouse(MouseEventType::Click, point.x, point.y));
    const bool released = place.send(Mouse(MouseEventType::Release, point.x, point.y));
    return pressed || released;
}

Extent2D<F32> Middle(const Sakura::Component& component) {
    const Rect frame = Peek::Frame(component);
    return {frame.x + frame.width * 0.5f, frame.y + frame.height * 0.5f};
}

}  // namespace

TEST_CASE("Compact buttons measure before placement, scale, repaint on tone changes and fire the newest callback",
          "[core][sakura][view-components]") {
    Scene scene;
    {
        CompactButton button;
        button.update({.str = "Check", .tone = CompactButton::Tone::Primary});
        CheckFirstMeasure(button, scene.ctx, {kInfinite, kInfinite}, {0.0f, 0.0f, 120.0f, 40.0f});
    }

    CompactButton button;
    Place place(button, kSlot);
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    CompactButton::Tone tone = CompactButton::Tone::Default;
    std::vector<std::string> calls;
    std::string tag = "a";
    const auto update = [&] {
        button.update({.str = "Check", .tone = tone, .onClick = [&calls, tag] { calls.push_back(tag); }});
    };
    REQUIRE(frames.settle(update));
    for (const auto next : {CompactButton::Tone::Primary, CompactButton::Tone::Warning, CompactButton::Tone::Danger,
                            CompactButton::Tone::Default}) {
        CAPTURE(static_cast<int>(next));
        tone = next;
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(Retains(frames, update));
    }

    tag = "b";
    CHECK(frames.run(update).clean());
    CHECK(ClickAt(place, Middle(button)));
    CHECK(calls == std::vector<std::string>{"b"});
}

TEST_CASE("Status badges measure before placement, scale and repaint on tone changes",
          "[core][sakura][view-components]") {
    Scene scene;
    {
        StatusBadge badge;
        badge.update({.label = "Loaded", .tone = StatusBadge::Tone::Success});
        CheckFirstMeasure(badge, scene.ctx, {kInfinite, kInfinite}, {0.0f, 0.0f, 120.0f, 30.0f});
    }

    StatusBadge badge;
    Place place(badge, kSlot);
    Frames frames(place, kScreen, scene.context());
    StatusBadge::Tone tone = StatusBadge::Tone::Neutral;
    const auto update = [&] {
        badge.update({.label = "Loaded", .tone = tone});
    };
    REQUIRE(frames.settle(update));
    for (const auto next : {StatusBadge::Tone::Success, StatusBadge::Tone::Warning, StatusBadge::Tone::Error,
                            StatusBadge::Tone::Neutral}) {
        CAPTURE(static_cast<int>(next));
        tone = next;
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(Retains(frames, update));
    }
}

TEST_CASE("Card buttons round into a pill and keep their padding at natural and filling sizes",
          "[core][sakura][view-components]") {
    Scene scene;
    {
        CardButton button;
        button.update({.str = "New Flowgraph", .tone = CardButton::Tone::Primary});
        CheckFirstMeasure(button, scene.ctx, {300.0f, kInfinite}, {0.0f, 0.0f, 300.0f, 40.0f});
    }

    for (const F32 height : {0.0f, Sakura::Fill, 40.0f}) {
        for (const F32 width : {0.0f, Sakura::Fill}) {
            CAPTURE(height, width);
            CardButton button;
            button.update({.str = "Open", .width = width, .height = height});
            Host host(button);
            const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
            CHECK(natural.y > 0.0f);
            CHECK(std::isfinite(natural.x));
            const auto bounded = host.measureOf(scene.ctx, {300.0f, 120.0f});
            CHECK(bounded.x <= 300.0f);
            CHECK(bounded.y <= 120.0f);
            if (height == Sakura::Fill) {
                CHECK(bounded.y == 120.0f);
            }
            host.place(scene.ctx, {0.0f, 0.0f, bounded.x, bounded.y});
            const auto texts = Find<Sakura::Text>(button);
            REQUIRE(texts.size() == 1);
            CHECK(Peek::Frame(*texts.front()).x >= 12.0f);
            const auto boxes = Find<Sakura::Box>(button);
            REQUIRE_FALSE(boxes.empty());
            for (const auto* box : boxes) {
                const auto& instances = box->config().instances;
                if (std::any_of(instances.begin(), instances.end(), [](const auto& instance) {
                        return instance.visible;
                    })) {
                    CHECK(box->config().cornerRadius >= bounded.y * 0.5f);
                }
            }
        }
    }
}

TEST_CASE("Card buttons repaint on tone changes and fire the newest callback", "[core][sakura][view-components]") {
    Scene scene;
    CardButton button;
    Place place(button, kSlot);
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    CardButton::Tone tone = CardButton::Tone::Card;
    std::string tint;
    std::vector<std::string> calls;
    std::string tag = "a";
    const auto update = [&] {
        button.update({.str = "Open", .tone = tone, .hoverTint = tint, .onClick = [&calls, tag] { calls.push_back(tag); }});
    };
    REQUIRE(frames.settle(update));
    for (const auto next : {CardButton::Tone::Primary, CardButton::Tone::Action, CardButton::Tone::Card}) {
        CAPTURE(static_cast<int>(next));
        tone = next;
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(Retains(frames, update));
    }
    tint = "blue";
    CHECK(Kept(frames.run(update)));
    CHECK(Retains(frames, update));

    tag = "b";
    CHECK(frames.run(update).clean());
    CHECK(ClickAt(place, Middle(button)));
    CHECK(calls == std::vector<std::string>{"b"});
}

TEST_CASE("Pagination measures before placement, scales exactly, repaints on page changes and fires the newest callback",
          "[core][sakura][view-components]") {
    Scene scene;
    {
        Pagination pager;
        pager.update({.page = 1, .pages = 5});
        Host host(pager);
        const auto first = host.measureOf(scene.ctx, {kInfinite, kInfinite});
        CHECK(first == Extent2D<F32>{7.0f * Pagination::ButtonSize + 6.0f * Pagination::Spacing, Pagination::ButtonSize});
        host.place(scene.ctx, {0.0f, 0.0f, first.x, first.y});
        CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == first);
        CHECK(host.measureOf(Doubled(scene.ctx), {kInfinite, kInfinite}) == Extent2D<F32>{first.x * 2.0f, first.y * 2.0f});
    }

    Pagination pager;
    Place place(pager, {20.0f, 20.0f, 400.0f, 40.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    U64 page = 1;
    std::vector<std::string> calls;
    std::string tag = "a";
    const auto update = [&] {
        pager.update({.page = page, .pages = 5, .onSelect = [&calls, tag](U64 target) {
                           calls.push_back(tag + std::to_string(target));
                       }});
    };
    REQUIRE(frames.settle(update));
    page = 2;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    tag = "b";
    CHECK(frames.run(update).clean());
    const auto buttons = Find<Sakura::Button>(pager);
    REQUIRE(buttons.size() == 7);
    CHECK(ClickAt(place, Middle(*buttons[4])));
    CHECK(ClickAt(place, Middle(*buttons[3])));
    CHECK(calls == std::vector<std::string>{"b3"});
}

TEST_CASE("Sidebar helpers measure before placement and scale their insets with the pixel ratio",
          "[core][sakura][view-components]") {
    Scene scene;
    {
        SidebarHeading heading;
        heading.update({.label = "General", .title = true});
        CheckFirstMeasure(heading, scene.ctx, {kInfinite, kInfinite}, {0.0f, 0.0f, 300.0f, 60.0f});
    }

    Probe item({200.0f, 30.0f});
    item.prepare();
    SidebarNavigation navigation;
    navigation.update({.title = "Settings", .sections = {{.heading = "Basics", .items = {&item}}}});
    Host host(navigation);
    const auto first = host.measureOf(scene.ctx, {300.0f, 600.0f});
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 600.0f});
    CHECK(host.measureOf(scene.ctx, {300.0f, 600.0f}) == first);
    const F32 inset = item.laidFrame.x;
    CHECK(inset == SidebarNavigation::Padding);
    const auto doubled = Doubled(scene.ctx);
    host.place(doubled, {0.0f, 0.0f, 600.0f, 1200.0f});
    CHECK(item.laidFrame.x == inset * 2.0f);

    Probe sidebar;
    Probe page;
    SidebarTiles tiles;
    tiles.update({.sidebar = &sidebar, .page = &page});
    Host tileHost(tiles);
    tileHost.place(scene.ctx, {0.0f, 0.0f, 1200.0f, 800.0f});
    const F32 width = sidebar.laidFrame.width;
    CHECK(width > SidebarTiles::SidebarWidth * 0.8f);
    CHECK(width < SidebarTiles::SidebarWidth * 1.2f);
    tileHost.place(doubled, {0.0f, 0.0f, 2400.0f, 1600.0f});
    CHECK(sidebar.laidFrame.width == Approx(width * 2.0f).margin(2.0f));
}

TEST_CASE("Flowgraph cells measure before placement, scale, repaint on favorite changes and fire the newest callbacks",
          "[core][sakura][view-components]") {
    Scene scene;
    {
        FlowgraphCell cell;
        cell.update({.title = "First Steps", .accessed = "Today"});
        CheckFirstMeasure(cell, scene.ctx, {320.0f, kInfinite}, {0.0f, 0.0f, 320.0f, 80.0f});
    }

    FlowgraphCell cell;
    Place place(cell, {20.0f, 20.0f, 320.0f, 80.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    bool favorite = false;
    std::vector<std::string> calls;
    std::string tag = "a";
    const auto update = [&] {
        cell.update({
            .title = "First Steps",
            .accessed = "Today",
            .favorite = favorite,
            .onOpen = [&calls, tag] { calls.push_back(tag + "open"); },
            .onToggleFavorite = [&calls, tag] { calls.push_back(tag + "star"); },
        });
    };
    REQUIRE(frames.settle(update));
    favorite = true;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    tag = "b";
    CHECK(frames.run(update).clean());
    const auto buttons = Find<Sakura::Button>(cell);
    REQUIRE(buttons.size() == 1);
    CHECK(ClickAt(place, Middle(*buttons.front())));
    CHECK(ClickAt(place, {60.0f, 60.0f}));
    CHECK(calls == std::vector<std::string>{"bstar", "bopen"});
}

TEST_CASE("Option steppers step within their options, reset to the fallback and disable the ends",
          "[core][sakura][view-components]") {
    const Scene scene;
    std::vector<U64> picks;
    OptionStepper stepper;
    U64 selected = 0;
    const auto update = [&] {
        stepper.update({.options = {"15 fps", "30 fps", "60 fps"}, .selected = selected, .fallback = 1,
                        .onSelect = [&](U64 index) { picks.push_back(index); }});
    };
    update();
    Host host(stepper);
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{132.0f, 34.0f});
    const Rect bounds = {0.0f, 0.0f, 132.0f, 34.0f};
    host.place(scene.ctx, bounds);
    CHECK(ShownText(stepper, "15 fps") != nullptr);

    host.click(17.0f, 17.0f);
    CHECK(picks.empty());
    host.click(bounds.right() - 17.0f, 17.0f);
    host.click(66.0f, 17.0f);
    CHECK(picks == std::vector<U64>{1, 1});

    picks.clear();
    selected = 2;
    update();
    host.place(scene.ctx, bounds);
    host.click(bounds.right() - 17.0f, 17.0f);
    host.click(17.0f, 17.0f);
    CHECK(picks == std::vector<U64>{1});

    picks.clear();
    selected = 9;
    update();
    host.place(scene.ctx, bounds);
    CHECK(ShownText(stepper, "60 fps") != nullptr);
}

TEST_CASE("File fields edit the path and browse from an icon button beside it", "[core][sakura][view-components]") {
    const Scene scene;
    std::vector<std::string> calls;
    FileField field;
    field.update({.value = "/etc/cert.pem", .hint = "None",
                  .onChange = [&](const std::string& value) { calls.push_back("change:" + value); },
                  .onBrowse = [&] { calls.push_back("browse"); }});
    Host host(field);
    const auto size = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(size == Extent2D<F32>{260.0f, 34.0f});
    host.place(scene.ctx, {0.0f, 0.0f, size.x, size.y});

    auto inputs = Find<Sakura::PillInput>(field);
    auto pills = Find<Sakura::PillButton>(field);
    REQUIRE(inputs.size() == 1);
    REQUIRE(pills.size() == 1);
    CHECK(Peek::Frame(*inputs[0]).width == 218.0f);
    CHECK(Peek::Frame(*pills[0]).width == 34.0f);
    CHECK(Peek::Frame(*pills[0]).right() == size.x);
    CHECK(ShownText(*pills[0], ICON_LC_FOLDER_OPEN) != nullptr);
    const auto browse = Middle(*pills[0]);
    CHECK(host.click(browse.x, browse.y));
    CHECK(calls == std::vector<std::string>{"browse"});
}

TEST_CASE("Danger compact buttons are outlined red pills", "[core][sakura][view-components]") {
    const Scene scene;
    auto ctx = scene.context();
    ctx.palette = DistinctPalette();
    CompactButton button;
    button.update({.str = "Remove", .icon = ICON_LC_TRASH_2, .tone = CompactButton::Tone::Danger});
    Host host(button);
    const auto size = host.measureOf(ctx, {kInfinite, kInfinite});
    CHECK(size.y == 34.0f);
    host.place(ctx, {0.0f, 0.0f, size.x, size.y});
    const auto boxes = Direct<Sakura::Box>(button);
    REQUIRE(boxes.size() == 4);
    CHECK(boxes[0]->config().instances.front().visible);
    CHECK(boxes[2]->config().instances.front().backgroundColor == ctx.color("tint_red_bg"));
    CHECK(boxes[3]->config().borderColor == ctx.color("tint_red_border"));
    auto* label = ShownText(button, "Remove");
    REQUIRE(label != nullptr);
    CHECK(label->config().colorKey == "tint_red_text");
}
