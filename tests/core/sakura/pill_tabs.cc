#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/pill_icon_button.hh>
#include <jetstream/render/sakura/components/pill_tabs.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "component_harness.hh"
#include "harness.hh"
#include "scene.hh"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

TEST_CASE("Pill tabs size to their items, select on release, and settle", "[core][sakura][pill-tabs]") {
    Scene scene;
    std::vector<U64> picks;
    Sakura::PillTabs tabs;
    const auto update = [&] {
        tabs.update({
            .items = {{.icon = "*", .label = "Flowgraphs"}, {.icon = "*", .label = "Plugins"}},
            .selected = 0,
            .onSelect = [&](U64 index) { picks.push_back(index); },
        });
    };
    update();
    Host host(tabs);
    auto doubled = scene.ctx;
    doubled.pixelRatio = 2.0f;
    const auto natural = host.measureOf(scene.ctx, {1000.0f, 500.0f});
    CHECK(natural.y == 40.0f);
    CHECK(natural.x < 400.0f);
    CHECK(host.measureOf(scene.ctx, {80.0f, 500.0f}).x == 80.0f);
    CHECK(host.measureOf(doubled, {1000.0f, 500.0f}).y == 80.0f);
    host.place(scene.ctx, {0.0f, 0.0f, natural.x, natural.y});

    const auto labels = Find<Sakura::Text>(tabs);
    REQUIRE(labels.size() == 4);
    const auto second = Peek::Frame(*labels[3]);
    host.send(Mouse(MouseEventType::Click, second.x + 2.0f, 20.0f));
    CHECK(picks.empty());
    host.send(Mouse(MouseEventType::Release, second.x + 2.0f, 20.0f));
    REQUIRE(picks.size() == 1);
    CHECK(picks[0] == 1);
    host.send(Mouse(MouseEventType::Click, second.x + 2.0f, 20.0f));
    host.send(Mouse(MouseEventType::Release, 5.0f, 20.0f));
    CHECK(picks.size() == 1);
    host.send(Mouse(MouseEventType::Click, natural.x + 50.0f, 20.0f));
    host.send(Mouse(MouseEventType::Release, natural.x + 50.0f, 20.0f));
    CHECK(picks.size() == 1);

    Sakura::PillTabs settled;
    Place place(settled, {10.0f, 10.0f, natural.x, natural.y});
    Frames frames(place, kScreen, scene.context());
    const auto again = [&] {
        settled.update({.items = {{.icon = "*", .label = "Flowgraphs"}, {.icon = "*", .label = "Plugins"}},
                        .selected = 0,
                        .onSelect = [](U64) {}});
    };
    REQUIRE(frames.settle(again));
    CHECK(Retains(frames, again));
    CheckHoverRepaintsOnce(frames, place, again, {10.0f + second.x + 2.0f, 30.0f}, {400.0f, 300.0f});
}

TEST_CASE("Pill tabs shrink in compact mode", "[core][sakura][pill-tabs]") {
    const Scene scene;
    Sakura::PillTabs tabs;
    tabs.update({.items = {{.label = "All"}, {.label = "Plugins"}}});
    Host host(tabs);
    const auto plain = host.measureOf(scene.ctx, {1000.0f, 500.0f});
    CHECK(plain.y == 40.0f);

    tabs.update({.items = {{.label = "All"}, {.label = "Plugins"}}, .compact = true});
    const auto compact = host.measureOf(scene.ctx, {1000.0f, 500.0f});
    CHECK(compact.y == 32.0f);
    const auto labels = Find<Sakura::Text>(tabs);
    const auto label = [&](const std::string& str) {
        return *std::find_if(labels.begin(), labels.end(), [&](auto* text) { return text->config().str == str; });
    };
    CHECK(label("All")->config().font == Sakura::Text::Font::Body);
    CHECK(label("Plugins")->config().font == Sakura::Text::Font::Body);
    CHECK(label("All")->config().weight == label("Plugins")->config().weight);
    CHECK(label("Plugins")->config().colorKey == "text_primary");
    CHECK(label("All")->config().colorKey == "text_primary");
    CHECK(label("Plugins")->config().opacity < 1.0f);
    CHECK(label("All")->config().opacity == 1.0f);

    tabs.update({.items = {{.label = "All"}, {.label = "Plugins"}}});
    CHECK(label("All")->config().font == Sakura::Text::Font::Sans);
    CHECK(label("Plugins")->config().font == Sakura::Text::Font::Sans);
    CHECK(label("Plugins")->config().opacity == 1.0f);
}

TEST_CASE("Compact pill tabs switch the selection with a repaint and keep one label font",
          "[core][sakura][pill-tabs][retention]") {
    Scene scene;
    Sakura::PillTabs tabs;
    U64 selected = 0;
    Place place(tabs, {10.0f, 10.0f, 400.0f, 32.0f});
    Frames frames(place, kScreen, scene.context());
    const auto update = [&] {
        tabs.update({
            .items = {{.label = "All"}, {.label = "Basics"}, {.label = "Receivers"}},
            .selected = selected,
            .compact = true,
        });
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    const auto fonts = [&] {
        std::vector<Sakura::Text::Font> result;
        for (auto* text : Find<Sakura::Text>(tabs)) {
            result.push_back(text->config().font);
        }
        return result;
    };
    const auto before = fonts();
    for (const U64 next : {1, 2, 0}) {
        CAPTURE(next);
        selected = next;
        CHECK(RepaintOnly(frames.run(update)));
        CHECK(Retains(frames, update));
        CHECK(fonts() == before);
    }
}

TEST_CASE("Pill tabs ignore cancelled presses, keep moves quiet, and restructure once for new items",
          "[core][sakura][pill-tabs][retention]") {
    Scene scene;
    std::vector<U64> picks;
    Sakura::PillTabs tabs;
    Place place(tabs, {10.0f, 10.0f, 600.0f, 40.0f});
    Frames frames(place, kScreen, scene.context());
    const std::vector<Sakura::PillTabs::Item> all = {{.label = "All"}, {.label = "Basics"}, {.label = "Receivers"}};
    U64 count = 2;
    const auto update = [&] {
        tabs.update({
            .items = std::vector<Sakura::PillTabs::Item>(all.begin(), all.begin() + static_cast<I64>(count)),
            .selected = 0,
            .onSelect = [&](U64 index) { picks.push_back(index); },
        });
    };
    const auto send = [&](const MouseEvent& event) {
        return place.send(event);
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{550.0f, 30.0f}, {560.0f, 30.0f}, {400.0f, 300.0f}}, send));

    CHECK(place.send(Mouse(MouseEventType::Click, 20.0f, 30.0f)));
    place.send(Mouse(MouseEventType::Cancel, 20.0f, 30.0f));
    CHECK_FALSE(place.send(Mouse(MouseEventType::Release, 20.0f, 30.0f)));
    CHECK(picks.empty());
    REQUIRE(frames.settle(update));

    count = 3;
    const auto grown = frames.run(update);
    CHECK(grown.restructured);
    CHECK_FALSE(grown.rebuilt);
    REQUIRE(frames.settle(update));
    CHECK(frames.restructures == 1);

    count = 2;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    count = 3;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 1);
}

TEST_CASE("Pill tabs scale their padding and spacing with the pixel ratio", "[core][sakura][pill-tabs]") {
    const Scene scene;
    Sakura::PillTabs tabs;
    tabs.update({.items = {{.label = "All"}, {.label = "Plugins"}}});
    Host host(tabs);
    const auto text = [&](const std::string& str) {
        for (auto* found : Find<Sakura::Text>(tabs)) {
            if (found->config().str == str) {
                return Peek::Frame(*found);
            }
        }
        return Rect{};
    };

    for (const F32 ratio : {1.0f, 2.0f}) {
        CAPTURE(ratio);
        auto ctx = scene.ctx;
        ctx.pixelRatio = ratio;
        host.place(ctx, {0.0f, 0.0f, 2000.0f, 40.0f * ratio});
        const Rect label = text("All");
        CHECK(label.x == 14.0f * ratio);
        CHECK(text("Plugins").x == std::ceil(label.right() + 14.0f * ratio) + 6.0f * ratio + 14.0f * ratio);
    }
}

TEST_CASE("Pill tabs ellipsize labels that do not fit, keep their natural width, and restore the label when it fits",
          "[core][sakura][pill-tabs]") {
    const std::string full = "Receivers and Transmitters";
    Scene scene;
    Sakura::PillTabs tabs;
    const auto update = [&] {
        tabs.update({.items = {{.label = "All"}, {.label = full}}, .compact = true});
    };
    update();
    Host host(tabs);
    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    REQUIRE(natural.x > 0.0f);
    const auto label = [&] {
        const auto labels = Find<Sakura::Text>(tabs);
        REQUIRE(labels.size() == 2);
        return labels[1];
    };

    const Rect narrow = {0.0f, 0.0f, natural.x - 60.0f, natural.y};
    host.place(scene.ctx, narrow);
    const std::string fitted = label()->config().str;
    CHECK(fitted != full);
    CHECK(fitted.size() > 3);
    CHECK(fitted.substr(fitted.size() - 3) == "...");
    CHECK(Peek::Frame(*label()).right() <= narrow.right());
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}).x == natural.x);

    update();
    host.place(scene.ctx, narrow);
    CHECK(label()->config().str == fitted);

    host.place(scene.ctx, {0.0f, 0.0f, natural.x, natural.y});
    CHECK(label()->config().str == full);

    Sakura::PillTabs settled;
    Place place(settled, {10.0f, 10.0f, natural.x - 60.0f, natural.y});
    Frames frames(place, kScreen, scene.context());
    const auto again = [&] {
        settled.update({.items = {{.label = "All"}, {.label = full}}, .compact = true});
    };
    REQUIRE(frames.settle(again));
    CHECK(Retains(frames, again));
}

TEST_CASE("Settled pill tabs fire the newest select callback after a callback only change",
          "[core][sakura][pill-tabs]") {
    Scene scene;
    Sakura::PillTabs tabs;
    Place place(tabs, {10.0f, 10.0f, 300.0f, 40.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    std::vector<std::string> calls;
    std::string tag = "a";
    const auto update = [&] {
        tabs.update({
            .items = {{.icon = "*", .label = "Flowgraphs"}, {.icon = "*", .label = "Plugins"}},
            .selected = 0,
            .onSelect = [&calls, tag](U64 index) { calls.push_back(tag + std::to_string(index)); },
        });
    };
    REQUIRE(frames.settle(update));
    tag = "b";
    CHECK(frames.run(update).clean());

    const auto labels = Find<Sakura::Text>(tabs);
    REQUIRE(labels.size() == 4);
    const auto second = Peek::Frame(*labels[3]);
    const F32 y = second.y + second.height * 0.5f;
    place.send(Mouse(MouseEventType::Click, second.x + 2.0f, y));
    place.send(Mouse(MouseEventType::Release, second.x + 2.0f, y));
    CHECK(calls == std::vector<std::string>{"b1"});
}

TEST_CASE("Settled pill icon buttons fire the newest click callback after a callback only change",
          "[core][sakura][pill-icon-button]") {
    Scene scene;
    Sakura::PillIconButton button;
    Place place(button, {10.0f, 10.0f, 34.0f, 34.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    std::vector<std::string> calls;
    std::string tag = "a";
    const auto update = [&] {
        button.update({.icon = ICON_LC_INFO, .onClick = [&calls, tag] { calls.push_back(tag); }});
    };
    REQUIRE(frames.settle(update));
    tag = "b";
    CHECK(frames.run(update).clean());

    place.send(Mouse(MouseEventType::Click, 27.0f, 27.0f));
    place.send(Mouse(MouseEventType::Release, 27.0f, 27.0f));
    CHECK(calls == std::vector<std::string>{"b"});
}
