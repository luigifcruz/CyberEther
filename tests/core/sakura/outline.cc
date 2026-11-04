#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/aside.hh>
#include <jetstream/render/sakura/components/markdown.hh>
#include <jetstream/render/sakura/components/outline.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/scroll_area.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_markdown.hh>

#include "component_harness.hh"
#include "scene.hh"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;
using Catch::Approx;

namespace {

constexpr F32 kFrameStep = 0.05f;

const std::vector<Sakura::Outline::Entry> kEntries = {
    {.label = "First", .level = 0},
    {.label = "Second", .level = 0},
    {.label = "Nested", .level = 1},
    {.label = "Deeper", .level = 1},
    {.label = "Last", .level = 0},
};

Sakura::Context Doubled(Sakura::Context ctx) {
    ctx.pixelRatio = 2.0f;
    return ctx;
}

const Sakura::Polyline& Track(Sakura::Outline& outline) {
    return *Find<Sakura::Polyline>(outline).at(0);
}

const Sakura::Polyline& Marker(Sakura::Outline& outline) {
    return *Find<Sakura::Polyline>(outline).at(1);
}

Rect RowFrame(Sakura::Outline& outline, U64 index) {
    return Peek::Frame(*outline.row(index));
}

std::string Article() {
    std::string value = "Introduction paragraph.\n\n";
    for (U64 section = 0; section < 6; ++section) {
        value += "## Section " + std::to_string(section) + "\n\n";
        for (U64 line = 0; line < 12; ++line) {
            value += "Body line " + std::to_string(line) + ".\n\n";
        }
        value += "### Detail " + std::to_string(section) + "\n\nDetail body.\n\n";
    }
    return value;
}

std::optional<std::string> ActiveText(Sakura::Markdown& markdown) {
    const auto active = markdown.activeHeading(2, 3);
    if (!active.has_value()) {
        return std::nullopt;
    }
    return markdown.headings().at(*active).text;
}

}  // namespace

TEST_CASE("Outlines measure their rows before the first layout and rewrap when narrowed",
          "[core][sakura][outline]") {
    Scene scene;
    Sakura::Outline outline;
    outline.update({.entries = {{.label = "A heading long enough to wrap across several narrow lines"}}});
    Host host(outline);

    const auto wide = host.measureOf(scene.ctx, {800.0f, 500.0f});
    const auto narrow = host.measureOf(scene.ctx, {80.0f, 500.0f});
    CHECK(wide.x == 800.0f);
    CHECK(wide.y > 0.0f);
    CHECK(narrow.x == 80.0f);
    CHECK(narrow.y > wide.y);

    const auto natural = host.measureOf(scene.ctx, {kInfinite, kInfinite});
    CHECK(std::isfinite(natural.x));
    CHECK(natural.x > 0.0f);
    CHECK(host.measureOf(scene.ctx, {800.0f, 4.0f}).y == 4.0f);
}

TEST_CASE("Outlines scale their lengths with the pixel ratio", "[core][sakura][outline]") {
    Scene scene;
    Sakura::Outline outline;
    outline.update({.entries = kEntries, .active = 2});
    Host host(outline);

    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 300.0f});
    const F32 indented = RowFrame(outline, 2).x - RowFrame(outline, 0).x;
    const F32 thickness = Track(outline).config().instances.at(0).thickness;
    CHECK(indented == 12.0f);
    CHECK(thickness == 2.0f);

    Sakura::Outline doubled;
    doubled.update({.entries = kEntries, .active = 2});
    Host doubledHost(doubled);
    doubledHost.place(Doubled(scene.ctx), {0.0f, 0.0f, 600.0f, 600.0f});
    CHECK(RowFrame(doubled, 2).x - RowFrame(doubled, 0).x == 24.0f);
    CHECK(Track(doubled).config().instances.at(0).thickness == 3.0f);
}

TEST_CASE("Outline lines follow the indentation with diagonals and no repeated points", "[core][sakura][outline]") {
    Scene scene;
    Sakura::Outline outline;
    outline.update({.entries = kEntries});
    Host host(outline);
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 300.0f});

    const auto& points = Track(outline).config().instances.at(0).points;
    REQUIRE(points.size() == 6);
    for (U64 index = 1; index < points.size(); ++index) {
        CHECK(points[index].y > points[index - 1].y);
    }
    CHECK(points[0].x == points[1].x);
    CHECK(points[2].x == points[1].x + 12.0f);
    CHECK(points[3].x == points[2].x);
    CHECK(points[4].x == points[0].x);
    CHECK(points[5].x == points[0].x);
}

TEST_CASE("Outlines with a single level draw one straight segment", "[core][sakura][outline]") {
    Scene scene;
    Sakura::Outline outline;
    outline.update({.entries = {{.label = "One"}, {.label = "Two"}, {.label = "Three"}}});
    Host host(outline);
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 300.0f});
    CHECK(Track(outline).config().instances.at(0).points.size() == 2);
}

TEST_CASE("Outline markers share the line thickness and sit on the active row", "[core][sakura][outline]") {
    Scene scene;
    Sakura::Outline outline;
    outline.update({.entries = kEntries, .active = 1});
    Host host(outline);
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 300.0f});

    const auto& marker = Marker(outline).config().instances.at(0);
    const auto& track = Track(outline).config().instances.at(0);
    REQUIRE(marker.visible);
    REQUIRE(marker.points.size() >= 2);
    CHECK(marker.thickness == track.thickness);
    CHECK(marker.points.front().y == Approx(RowFrame(outline, 1).y - 5.0f).margin(1.0f));

    outline.update({.entries = kEntries});
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 300.0f});
    CHECK_FALSE(Marker(outline).config().instances.at(0).visible);
}

TEST_CASE("Outlines select on release inside the pressed row", "[core][sakura][outline]") {
    Scene scene;
    std::vector<U64> selected;
    Sakura::Outline outline;
    outline.update({.entries = kEntries, .onSelect = [&](U64 index) { selected.push_back(index); }});
    Host host(outline);
    host.place(scene.ctx, {0.0f, 0.0f, 300.0f, 300.0f});

    const Rect second = RowFrame(outline, 1);
    const Rect last = RowFrame(outline, 4);
    const F32 y = second.y + second.height * 0.5f;

    host.send(Mouse(MouseEventType::Click, 40.0f, y));
    CHECK(selected.empty());
    host.send(Mouse(MouseEventType::Release, 40.0f, y));
    CHECK(selected == std::vector<U64>{1});

    host.send(Mouse(MouseEventType::Click, 40.0f, y));
    host.send(Mouse(MouseEventType::Release, 40.0f, last.y + last.height * 0.5f));
    CHECK(selected.size() == 1);

    host.send(Mouse(MouseEventType::Click, 40.0f, y));
    host.send(Mouse(MouseEventType::Cancel, 40.0f, y));
    host.send(Mouse(MouseEventType::Release, 40.0f, y));
    CHECK(selected.size() == 1);

    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 40.0f, 295.0f)));
}

TEST_CASE("Outlines ignore the clipped part of a row", "[core][sakura][outline]") {
    Scene scene;
    std::vector<U64> selected;
    Sakura::Outline outline;
    outline.update({.entries = kEntries, .onSelect = [&](U64 index) { selected.push_back(index); }});
    Sakura::ScrollArea area;
    area.update({.id = "clipped", .content = &outline, .height = Sakura::Fill, .alwaysScrollbar = false,
                 .border = false});
    Place place(area, {0.0f, 0.0f, 300.0f, 200.0f});
    place.place(scene.ctx, kScreen);

    const Rect row = RowFrame(outline, 1);
    const F32 middle = row.y + row.height * 0.5f;
    place.move({0.0f, 0.0f, 300.0f, middle});
    place.place(scene.ctx, kScreen);

    place.send(Mouse(MouseEventType::Click, 40.0f, middle - 2.0f));
    place.send(Mouse(MouseEventType::Release, 40.0f, middle + 2.0f));
    CHECK(selected.empty());

    place.send(Mouse(MouseEventType::Click, 40.0f, middle - 2.0f));
    place.send(Mouse(MouseEventType::Release, 40.0f, middle - 2.0f));
    CHECK(selected == std::vector<U64>{1});
}

TEST_CASE("Outlines settle, repaint on hover, and repaint only while the marker glides",
          "[core][sakura][outline][retention]") {
    Scene scene;
    Sakura::Outline outline;
    Place place(outline, {10.0f, 10.0f, 260.0f, 240.0f});
    Frames frames(place, kScreen, scene.context());
    std::optional<U64> active = 0;
    const auto update = [&] {
        outline.update({.entries = kEntries, .active = active});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{200.0f, 248.0f}, {210.0f, 248.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));

    const Rect row = RowFrame(outline, 2);
    const Extent2D<F32> inside = {row.x + 20.0f, row.y + row.height * 0.5f};
    CheckHoverRepaintsOnce(frames, place, update, inside, {500.0f, 500.0f});
    CheckLeaveRepaintsOnce(frames, place, update, inside);

    frames.timeStep = 0.0;
    active = 3;
    const F32 before = Marker(outline).config().instances.at(0).points.front().y;
    CHECK(Kept(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(Marker(outline).config().instances.at(0).points.front().y == Approx(before).margin(0.5f));

    frames.timeStep = kFrameStep;
    bool moved = false;
    for (U64 frame = 0; frame < 3; ++frame) {
        const auto stats = frames.run(update);
        CHECK(Kept(stats));
        moved = moved || stats.repainted;
    }
    CHECK(moved);
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(Marker(outline).config().instances.at(0).points.front().y == Approx(RowFrame(outline, 3).y - 5.0f).margin(1.0f));
    CHECK(frames.restructures == 0);
}

TEST_CASE("Outlines restructure once when rows are added and repaint when they shrink",
          "[core][sakura][outline][retention]") {
    Scene scene;
    Sakura::Outline outline;
    Place place(outline, {10.0f, 10.0f, 260.0f, 240.0f});
    Frames frames(place, kScreen, scene.context());
    std::vector<Sakura::Outline::Entry> entries = {{.label = "One"}};
    const auto update = [&] {
        outline.update({.entries = entries});
    };
    REQUIRE(frames.settle(update));
    const U64 restructures = frames.restructures;

    entries.push_back({.label = "Two"});
    frames.run(update);
    CHECK(frames.restructures == restructures + 1);
    CHECK(Retains(frames, update));

    entries.pop_back();
    CHECK(Kept(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK_FALSE(outline.row(1));
}

TEST_CASE("Asides fill the available space and overlay the rail on the content", "[core][sakura][aside]") {
    Scene scene;
    Probe content({100.0f, 100.0f});
    Probe rail({100.0f, 100.0f});
    Sakura::Aside aside;
    aside.update({.content = &content, .rail = &rail, .railWidth = 200.0f, .railInset = 16.0f,
                  .minimumContentWidth = 400.0f});
    Host host(aside);

    CHECK(host.measureOf(scene.ctx, {800.0f, 600.0f}) == Extent2D<F32>{800.0f, 600.0f});
    CHECK(host.measureOf(scene.ctx, {kInfinite, kInfinite}) == Extent2D<F32>{0.0f, 0.0f});

    host.place(scene.ctx, {0.0f, 0.0f, 800.0f, 600.0f});
    CHECK(rail.visible());
    CHECK(content.laidFrame == Rect{0.0f, 0.0f, 800.0f, 600.0f});
    CHECK(rail.laidFrame == Rect{584.0f, 0.0f, 200.0f, 600.0f});

    Sakura::Aside doubled;
    Probe doubledContent({100.0f, 100.0f});
    Probe doubledRail({100.0f, 100.0f});
    doubled.update({.content = &doubledContent, .rail = &doubledRail, .railWidth = 200.0f, .railInset = 16.0f,
                    .minimumContentWidth = 400.0f});
    Host doubledHost(doubled);
    doubledHost.place(Doubled(scene.ctx), {0.0f, 0.0f, 1600.0f, 1200.0f});
    CHECK(doubledRail.laidFrame == Rect{1168.0f, 0.0f, 400.0f, 1200.0f});
}

TEST_CASE("Asides remove the rail when the content would get too narrow", "[core][sakura][aside]") {
    Scene scene;
    Probe content({100.0f, 100.0f});
    Probe rail({100.0f, 100.0f});
    Sakura::Aside aside;
    aside.update({.content = &content, .rail = &rail, .railWidth = 200.0f, .railInset = 16.0f,
                  .minimumContentWidth = 400.0f});
    Host host(aside);

    host.place(scene.ctx, {0.0f, 0.0f, 616.0f, 600.0f});
    CHECK(rail.visible());
    host.place(scene.ctx, {0.0f, 0.0f, 615.0f, 600.0f});
    CHECK_FALSE(rail.visible());
    CHECK(content.laidFrame == Rect{0.0f, 0.0f, 615.0f, 600.0f});
    host.place(scene.ctx, {0.0f, 0.0f, 616.0f, 600.0f});
    CHECK(rail.visible());
}

TEST_CASE("Asides cancel a held press when the rail hides", "[core][sakura][aside]") {
    Scene scene;
    Probe content({100.0f, 100.0f});
    Probe rail({100.0f, 100.0f});
    Sakura::Aside aside;
    aside.update({.content = &content, .rail = &rail, .railWidth = 200.0f, .minimumContentWidth = 400.0f});
    Host host(aside);
    host.place(scene.ctx, {0.0f, 0.0f, 800.0f, 600.0f});

    host.send(Mouse(MouseEventType::Click, 700.0f, 100.0f));
    REQUIRE(rail.count(MouseEventType::Click) == 1);
    const U64 cancels = rail.count(MouseEventType::Cancel);
    host.place(scene.ctx, {0.0f, 0.0f, 500.0f, 600.0f});
    CHECK(rail.count(MouseEventType::Cancel) == cancels + 1);
}

TEST_CASE("Asides settle, repaint when the rail hides, and remount a child that came back",
          "[core][sakura][aside][retention]") {
    Scene scene;
    Probe content({100.0f, 100.0f});
    Probe rail({100.0f, 100.0f});
    Sakura::Aside aside;
    Place place(aside, {0.0f, 0.0f, 800.0f, 600.0f});
    Frames frames(place, kScreen, scene.context());
    Sakura::Component* current = &rail;
    const auto update = [&] {
        aside.update({.content = &content, .rail = current, .railWidth = 200.0f, .minimumContentWidth = 400.0f});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));

    place.move({0.0f, 0.0f, 500.0f, 600.0f});
    CHECK(Kept(frames.run(update)));
    CHECK_FALSE(rail.visible());
    CHECK(Retains(frames, update));

    place.move({0.0f, 0.0f, 800.0f, 600.0f});
    CHECK(Kept(frames.run(update)));
    CHECK(rail.visible());
    CHECK(Retains(frames, update));

    current = nullptr;
    frames.run(update);
    CHECK_FALSE(rail.visible());
    CHECK(Retains(frames, update));

    current = &rail;
    frames.run(update);
    CHECK(rail.visible());
    CHECK(Retains(frames, update));
}

TEST_CASE("Markdown headings survive padding changes and clear with a new value", "[core][sakura][markdown]") {
    Scene scene;
    Sakura::Markdown markdown;
    Host host(markdown);
    markdown.update({.value = "## One\n\n### Two\n"});
    CHECK(markdown.headings().empty());
    host.place(scene.ctx, {0.0f, 0.0f, 400.0f, 300.0f});
    REQUIRE(markdown.headings().size() == 2);

    markdown.update({.value = "## One\n\n### Two\n", .padding = Sakura::Padding{1.0f, 2.0f, 3.0f, 4.0f}});
    CHECK(markdown.headings().size() == 2);

    markdown.update({
        .value = "## One\n\n### Two\n",
        .padding = Sakura::Padding{1.0f, 2.0f, 3.0f, 4.0f},
        .hero = Sakura::TextMarkdown::Hero{.title = "Title"},
    });
    CHECK(markdown.headings().empty());
    host.place(scene.ctx, {0.0f, 0.0f, 400.0f, 300.0f});
    REQUIRE(markdown.headings().size() == 2);

    markdown.update({.value = "## Other\n", .padding = Sakura::Padding{1.0f, 2.0f, 3.0f, 4.0f}});
    CHECK(markdown.headings().empty());
    host.place(scene.ctx, {0.0f, 0.0f, 400.0f, 300.0f});
    REQUIRE(markdown.headings().size() == 1);
    CHECK(markdown.headings().at(0).text == "Other");
}

TEST_CASE("Markdown tracks the active heading as it scrolls", "[core][sakura][markdown]") {
    Scene scene;
    Sakura::Markdown markdown;
    Place place(markdown, {0.0f, 0.0f, 500.0f, 240.0f});
    Frames frames(place, kScreen, scene.context());
    const std::string value = Article();
    const auto update = [&] {
        markdown.update({.value = value, .fontSize = 15.0f});
    };
    REQUIRE(frames.settle(update));

    CHECK(ActiveText(markdown) == "Section 0");

    F32 scrolled = 0.0f;
    std::string previous = *ActiveText(markdown);
    U64 changes = 0;
    for (U64 step = 0; step < 400 && scrolled < markdown.metrics().contentHeight; ++step) {
        place.send(Wheel(100.0f, 100.0f, -1.0f));
        frames.run(update);
        scrolled = markdown.metrics().scrollY;
        const auto active = ActiveText(markdown);
        REQUIRE(active.has_value());
        if (*active != previous) {
            previous = *active;
            ++changes;
        }
    }
    CHECK(changes >= 10);
    CHECK(ActiveText(markdown) == "Detail 5");
}

TEST_CASE("Markdown filters the active heading by level and reports none without headings",
          "[core][sakura][markdown]") {
    Scene scene;
    Sakura::Markdown markdown;
    Host host(markdown);
    markdown.update({.value = "# Title\n\nBody.\n"});
    host.place(scene.ctx, {0.0f, 0.0f, 400.0f, 300.0f});
    CHECK_FALSE(markdown.activeHeading(2, 3).has_value());
    CHECK(markdown.activeHeading(1, 3) == U64{0});
}

TEST_CASE("Markdown reveals a heading with a glide that a wheel scroll cancels", "[core][sakura][markdown]") {
    Scene scene;
    Sakura::Markdown markdown;
    Place place(markdown, {0.0f, 0.0f, 500.0f, 240.0f});
    Frames frames(place, kScreen, scene.context());
    frames.timeStep = kFrameStep;
    const std::string value = Article();
    const auto update = [&] {
        markdown.update({.value = value, .fontSize = 15.0f});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(markdown.metrics().scrollY == 0.0f);

    const auto& headings = markdown.headings();
    REQUIRE(headings.size() == 12);
    markdown.revealHeading(4);
    CHECK(markdown.activeHeading(2, 3) == U64{4});

    frames.run(update);
    frames.run(update);
    const F32 midway = markdown.metrics().scrollY;
    CHECK(midway > 0.0f);
    CHECK(markdown.activeHeading(2, 3) == U64{4});

    F32 last = midway;
    for (U64 frame = 0; frame < 16; ++frame) {
        frames.run(update);
        CHECK(markdown.metrics().scrollY >= last);
        last = markdown.metrics().scrollY;
    }
    CHECK(last > midway);
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(markdown.activeHeading(2, 3) == U64{4});
    const auto& settled = markdown.metrics();
    const F32 offset = settled.sourceLines.at(headings.at(4).line).top - settled.viewportTop;
    CHECK(offset >= 0.0f);
    CHECK(offset <= settled.sourceLines.at(headings.at(4).line).height * 2.0f);

    markdown.revealHeading(8);
    frames.run(update);
    frames.run(update);
    const F32 interrupted = markdown.metrics().scrollY;
    place.send(Wheel(100.0f, 100.0f, -1.0f));
    frames.run(update);
    const F32 wheeled = markdown.metrics().scrollY;
    frames.run(update);
    frames.run(update);
    CHECK(markdown.metrics().scrollY == wheeled);
    CHECK(wheeled != interrupted);
    CHECK(markdown.activeHeading(2, 3) != U64{8});
}

TEST_CASE("Markdown keeps a revealed heading active when the page cannot scroll it to the top",
          "[core][sakura][markdown]") {
    Scene scene;
    Sakura::Markdown markdown;
    Place place(markdown, {0.0f, 0.0f, 500.0f, 520.0f});
    Frames frames(place, kScreen, scene.context());
    frames.timeStep = kFrameStep;
    const std::string value = Article();
    const auto update = [&] {
        markdown.update({.value = value, .fontSize = 15.0f});
    };
    REQUIRE(frames.settle(update));

    markdown.revealHeading(10);
    for (U64 frame = 0; frame < 16; ++frame) {
        frames.run(update);
    }
    REQUIRE(frames.settle(update));
    const auto& metrics = markdown.metrics();
    REQUIRE(metrics.scrollY >= metrics.contentHeight - metrics.viewportHeight - 1.0f);
    CHECK(markdown.activeHeading(2, 3) == U64{10});
    CHECK(Retains(frames, update));
    CHECK(markdown.activeHeading(2, 3) == U64{10});

    place.send(Wheel(100.0f, 100.0f, 1.0f));
    for (U64 frame = 0; frame < 4; ++frame) {
        frames.run(update);
    }
    CHECK(markdown.activeHeading(2, 3) != U64{10});
}

TEST_CASE("Markdown glides stop as soon as smooth wheel input arrives", "[core][sakura][markdown]") {
    Scene scene;
    Sakura::Markdown markdown;
    Place place(markdown, {0.0f, 0.0f, 500.0f, 240.0f});
    Frames frames(place, kScreen, scene.context());
    frames.timeStep = kFrameStep;
    const std::string value = Article();
    const auto update = [&] {
        markdown.update({.value = value, .fontSize = 15.0f});
    };
    REQUIRE(frames.settle(update));
    markdown.revealHeading(8);
    frames.run(update);
    frames.run(update);
    const auto grids = Find<Sakura::TextMarkdown>(markdown);
    REQUIRE(grids.size() == 1);
    REQUIRE(grids.front()->gliding());

    auto wheel = Wheel(100.0f, 100.0f, -1.0f);
    wheel.smooth = true;
    place.send(wheel);
    CHECK_FALSE(grids.front()->gliding());
}

TEST_CASE("Markdown releases a revealed heading when the page reflows", "[core][sakura][markdown]") {
    Scene scene;
    Sakura::Markdown markdown;
    Place place(markdown, {0.0f, 0.0f, 500.0f, 240.0f});
    Frames frames(place, kScreen, scene.context());
    frames.timeStep = kFrameStep;
    const std::string value = Article();
    const auto update = [&] {
        markdown.update({.value = value, .fontSize = 15.0f});
    };
    REQUIRE(frames.settle(update));
    markdown.revealHeading(4);
    for (U64 frame = 0; frame < 16; ++frame) {
        frames.run(update);
    }
    REQUIRE(frames.settle(update));
    REQUIRE(markdown.activeHeading(2, 3) == U64{4});

    place.move({0.0f, 0.0f, 70.0f, 240.0f});
    frames.run(update);
    CHECK(markdown.activeHeading(2, 3) != U64{4});
}

TEST_CASE("Markdown drops a revealed heading on an explicit scroll", "[core][sakura][markdown]") {
    Scene scene;
    Sakura::Markdown markdown;
    Place place(markdown, {0.0f, 0.0f, 500.0f, 600.0f});
    Frames frames(place, kScreen, scene.context());
    const std::string value = "## One\n\nBody.\n\n## Two\n\nBody.\n";
    const auto update = [&] {
        markdown.update({.value = value, .fontSize = 15.0f});
    };
    REQUIRE(frames.settle(update));
    REQUIRE(markdown.metrics().scrollY == 0.0f);

    markdown.revealHeading(1);
    CHECK(markdown.activeHeading(2, 3) == U64{1});
    markdown.scrollToLine(0);
    CHECK(markdown.activeHeading(2, 3) == U64{0});
}

TEST_CASE("Markdown drops a revealed heading when its value changes", "[core][sakura][markdown]") {
    Scene scene;
    Sakura::Markdown markdown;
    Place place(markdown, {0.0f, 0.0f, 500.0f, 240.0f});
    Frames frames(place, kScreen, scene.context());
    std::string value = Article();
    const auto update = [&] {
        markdown.update({.value = value, .fontSize = 15.0f});
    };
    REQUIRE(frames.settle(update));
    markdown.revealHeading(4);
    value = "## Only\n\nBody.\n";
    frames.run(update);
    REQUIRE(markdown.headings().size() == 1);
    CHECK(markdown.activeHeading(2, 3) == U64{0});
}
