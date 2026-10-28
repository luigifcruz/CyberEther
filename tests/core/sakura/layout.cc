#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/div.hh>
#include <jetstream/render/sakura/components/grid.hh>
#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/components/overlay.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/reveal.hh>
#include <jetstream/render/sakura/components/panel.hh>
#include <jetstream/render/sakura/components/scroll_area.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/sakura/components/spacer.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/wave_line.hh>

#include "component_harness.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {

using Sakura::Anchor;
using Sakura::Div;
using Sakura::Grid;
using Sakura::Layer;
using Sakura::Overlay;
using Sakura::Panel;
using Sakura::Spacer;
using Sakura::Stack;

}  // namespace

TEST_CASE("Vertical stacks place items top to bottom with spacing", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({40.0f, 20.0f});
    Probe b({30.0f, 10.0f});
    Probe c({50.0f, 30.0f});
    Stack stack;
    stack.update({.items = {{.child = &a}, {.child = &b}, {.child = &c}}});
    Host host(stack);

    host.place(ctx, {10.0f, 20.0f, 200.0f, 300.0f});

    CHECK(a.laidFrame == Rect{10.0f, 20.0f, 200.0f, 20.0f});
    CHECK(b.laidFrame == Rect{10.0f, 48.0f, 200.0f, 10.0f});
    CHECK(c.laidFrame == Rect{10.0f, 66.0f, 200.0f, 30.0f});
}

TEST_CASE("Vertical stacks measure the summed height and available width", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({40.0f, 20.0f});
    Probe b({30.0f, 10.0f});
    Probe c({50.0f, 30.0f});
    Stack stack;
    stack.update({.items = {{.child = &a}, {.child = &b}, {.child = &c}}});
    Host host(stack);

    const auto bounded = host.measureOf(ctx, {200.0f, kInfinite});
    CHECK(bounded.x == 200.0f);
    CHECK(bounded.y == 76.0f);

    const auto unbounded = host.measureOf(ctx, {kInfinite, kInfinite});
    CHECK(unbounded.x == 50.0f);
    CHECK(unbounded.y == 76.0f);
}

TEST_CASE("Vertical stacks size the column to the chosen item and center every child at that width",
          "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({300.0f, 20.0f});
    Probe b({120.0f, 10.0f});
    Flow c(400.0f, 10.0f);
    Stack stack;
    stack.update({.spacing = 0.0f, .widthOf = &b, .items = {{.child = &a}, {.child = &b}, {.child = &c}}});
    Host host(stack);

    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}).x == 120.0f);
    const auto size = host.measureOf(ctx, {600.0f, kInfinite});
    CHECK(size.x == 600.0f);
    CHECK(size.y == 20.0f + 10.0f + std::ceil(4000.0f / 120.0f));
    host.place(ctx, {10.0f, 20.0f, 600.0f, size.y});
    CHECK(a.laidFrame == Rect{250.0f, 20.0f, 120.0f, 20.0f});
    CHECK(b.laidFrame == Rect{250.0f, 40.0f, 120.0f, 10.0f});
    CHECK(c.laidFrame == Rect{250.0f, 50.0f, 120.0f, std::ceil(4000.0f / 120.0f)});

    b.resize({900.0f, 10.0f});
    host.place(ctx, {10.0f, 20.0f, 600.0f, size.y});
    CHECK(a.laidFrame.x == 10.0f);
    CHECK(a.laidFrame.width == 600.0f);
    CHECK(c.laidFrame.width == 600.0f);

    Probe stray({50.0f, 10.0f});
    stack.update({.spacing = 0.0f, .widthOf = &stray, .items = {{.child = &a}, {.child = &b}, {.child = &c}}});
    host.place(ctx, {10.0f, 20.0f, 600.0f, size.y});
    CHECK(a.laidFrame.x == 10.0f);
    CHECK(a.laidFrame.width == 600.0f);
}

TEST_CASE("Stacks honour custom spacing and fixed item sizes", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({40.0f, 20.0f});
    Probe b({30.0f, 10.0f});
    Stack stack;
    stack.update({.spacing = 4.0f, .items = {{.child = &a, .size = 50.0f}, {.child = &b}}});
    Host host(stack);

    host.place(ctx, {0.0f, 0.0f, 100.0f, 300.0f});

    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 100.0f, 50.0f});
    CHECK(b.laidFrame == Rect{0.0f, 54.0f, 100.0f, 10.0f});
    CHECK(host.measureOf(ctx, {100.0f, kInfinite}).y == 64.0f);
}

TEST_CASE("Stack items grow proportionally into leftover space", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({40.0f, 20.0f});
    Probe b({30.0f, 10.0f});
    Probe c({30.0f, 12.0f});
    Stack stack;
    stack.update({.items = {{.child = &a, .grow = 1.0f}, {.child = &b, .grow = 3.0f}, {.child = &c}}});
    Host host(stack);

    host.place(ctx, {0.0f, 0.0f, 100.0f, 258.0f});

    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 100.0f, 70.0f});
    CHECK(b.laidFrame == Rect{0.0f, 78.0f, 100.0f, 160.0f});
    CHECK(c.laidFrame == Rect{0.0f, 246.0f, 100.0f, 12.0f});
}

TEST_CASE("Growing stack items never shrink below their size", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({40.0f, 20.0f});
    Probe b({30.0f, 30.0f});
    Stack stack;
    stack.update({.items = {{.child = &a, .grow = 1.0f}, {.child = &b}}});
    Host host(stack);

    host.place(ctx, {0.0f, 0.0f, 100.0f, 40.0f});

    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 100.0f, 20.0f});
    CHECK(b.laidFrame == Rect{0.0f, 28.0f, 100.0f, 30.0f});
}

TEST_CASE("Horizontal stacks center items on the cross axis", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({40.0f, 20.0f});
    Probe b({60.0f, 50.0f});
    Probe tall({30.0f, 500.0f});
    Stack stack;
    stack.update({.axis = Stack::Axis::Horizontal, .items = {{.child = &a}, {.child = &b}, {.child = &tall}}});
    Host host(stack);

    host.place(ctx, {10.0f, 20.0f, 300.0f, 100.0f});

    CHECK(a.laidFrame == Rect{10.0f, 60.0f, 40.0f, 20.0f});
    CHECK(b.laidFrame == Rect{58.0f, 45.0f, 60.0f, 50.0f});
    CHECK(tall.laidFrame == Rect{126.0f, 20.0f, 30.0f, 100.0f});
}

TEST_CASE("Horizontal stacks without centering fill the cross axis", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({40.0f, 20.0f});
    Probe b({60.0f, 50.0f});
    Stack stack;
    stack.update({.axis = Stack::Axis::Horizontal, .centerCross = false, .items = {{.child = &a}, {.child = &b}}});
    Host host(stack);

    host.place(ctx, {10.0f, 20.0f, 300.0f, 100.0f});

    CHECK(a.laidFrame == Rect{10.0f, 20.0f, 40.0f, 100.0f});
    CHECK(b.laidFrame == Rect{58.0f, 20.0f, 60.0f, 100.0f});
}

TEST_CASE("Horizontal stacks measure summed width and tallest item", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({40.0f, 20.0f});
    Probe b({60.0f, 50.0f});
    Stack stack;
    stack.update({.axis = Stack::Axis::Horizontal, .items = {{.child = &a}, {.child = &b, .size = 70.0f}}});
    Host host(stack);

    const auto measured = host.measureOf(ctx, {kInfinite, 100.0f});
    CHECK(measured.x == 118.0f);
    CHECK(measured.y == 50.0f);
}

TEST_CASE("Hidden stack items take no space", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({40.0f, 20.0f});
    Probe b({30.0f, 10.0f});
    Probe c({50.0f, 30.0f});
    Stack stack;
    stack.update({.items = {{.child = &a}, {.child = &b}, {.child = &c}}});
    b.setVisible(false);
    Host host(stack);

    host.place(ctx, {0.0f, 0.0f, 100.0f, 300.0f});

    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 100.0f, 20.0f});
    CHECK(c.laidFrame == Rect{0.0f, 28.0f, 100.0f, 30.0f});
    CHECK(b.layouts == 0);
    CHECK(host.measureOf(ctx, {100.0f, kInfinite}).y == 58.0f);
}

TEST_CASE("Stack spacing and fixed sizes scale with the pixel ratio", "[core][sakura][layout]") {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    Probe a({40.0f, 20.0f});
    Probe b({30.0f, 10.0f});
    Stack stack;
    stack.update({.items = {{.child = &a, .size = 15.0f}, {.child = &b}}});
    Host host(stack);

    host.place(ctx, {0.0f, 0.0f, 100.0f, 300.0f});

    CHECK(a.laidFrame == Rect{0.0f, 0.0f, 100.0f, 30.0f});
    CHECK(b.laidFrame == Rect{0.0f, 46.0f, 100.0f, 10.0f});
}

TEST_CASE("Spacers measure one stack spacing per extra line", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Spacer spacer;
    spacer.update({.lines = 3});
    Host host(spacer);

    const auto measured = host.measureOf(ctx, {120.0f, kInfinite});
    CHECK(measured.x == 120.0f);
    CHECK(measured.y == 2.0f * Stack::Spacing);

    spacer.update({.lines = 1});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}).y == 0.0f);
}

TEST_CASE("Overlay anchors measured items inside the bounds", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    struct Case {
        Anchor anchor;
        Rect expected;
    };
    const std::vector<Case> cases = {
        {Anchor::Fill, {10.0f, 20.0f, 400.0f, 300.0f}},
        {Anchor::TopLeft, {15.0f, 27.0f, 40.0f, 20.0f}},
        {Anchor::TopCenter, {195.0f, 27.0f, 40.0f, 20.0f}},
        {Anchor::TopRight, {375.0f, 27.0f, 40.0f, 20.0f}},
        {Anchor::Center, {195.0f, 167.0f, 40.0f, 20.0f}},
        {Anchor::BottomLeft, {15.0f, 307.0f, 40.0f, 20.0f}},
        {Anchor::BottomCenter, {195.0f, 307.0f, 40.0f, 20.0f}},
        {Anchor::BottomRight, {375.0f, 307.0f, 40.0f, 20.0f}},
    };
    std::vector<std::unique_ptr<Probe>> probes;
    Overlay::Config config;
    for (const auto& entry : cases) {
        probes.push_back(std::make_unique<Probe>(Extent2D<F32>{40.0f, 20.0f}));
        config.items.push_back({.child = probes.back().get(), .anchor = entry.anchor, .offset = {5.0f, 7.0f}});
    }
    Overlay overlay;
    overlay.update(config);
    Host host(overlay);

    host.place(ctx, {10.0f, 20.0f, 400.0f, 300.0f});

    for (U64 index = 0; index < cases.size(); ++index) {
        INFO("anchor " << index);
        CHECK(probes[index]->laidFrame == cases[index].expected);
    }
}

TEST_CASE("Overlay explicit sizes skip measuring the item", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe sized({40.0f, 20.0f});
    Probe partial({40.0f, 20.0f});
    Probe huge({1000.0f, 1000.0f});
    Overlay overlay;
    overlay.update({.items = {
        {.child = &sized, .anchor = Anchor::BottomRight, .size = {100.0f, 50.0f}},
        {.child = &partial, .anchor = Anchor::TopLeft, .size = {100.0f, 0.0f}},
        {.child = &huge, .anchor = Anchor::Center},
    }});
    Host host(overlay);

    host.place(ctx, {0.0f, 0.0f, 400.0f, 300.0f});

    CHECK(sized.measures == 0);
    CHECK(sized.laidFrame == Rect{300.0f, 250.0f, 100.0f, 50.0f});
    CHECK(partial.measures == 1);
    CHECK(partial.laidFrame == Rect{0.0f, 0.0f, 100.0f, 20.0f});
    CHECK(huge.laidFrame == Rect{0.0f, 0.0f, 400.0f, 300.0f});
}

TEST_CASE("Overlays measure the available space", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Overlay overlay;
    overlay.update({});
    Host host(overlay);

    const auto bounded = host.measureOf(ctx, {320.0f, 240.0f});
    CHECK(bounded.x == 320.0f);
    CHECK(bounded.y == 240.0f);

    const auto unbounded = host.measureOf(ctx, {kInfinite, kInfinite});
    CHECK(unbounded.x == 0.0f);
    CHECK(unbounded.y == 0.0f);
}

TEST_CASE("Panels measure the content height at the available width", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({120.0f, 45.0f});
    Panel panel;
    panel.update({.content = &content});
    Host host(panel);

    const auto measured = host.measureOf(ctx, {300.0f, kInfinite});
    CHECK(measured.x == 300.0f);
    CHECK(measured.y == 45.0f);

    content.resize({120.0f, 90.0f});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}).y == 90.0f);
}

TEST_CASE("Natural width panels measure content then clamp to the max width", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({120.4f, 45.0f});
    Panel panel;
    panel.update({.content = &content, .size = {0.0f, 0.0f}});
    Host host(panel);

    CHECK(host.measureOf(ctx, {300.0f, kInfinite}).x == 121.0f);
    CHECK(host.measureOf(ctx, {80.0f, kInfinite}).x == 80.0f);

    panel.update({.content = &content, .size = {0.0f, 0.0f}, .maxWidth = 100.0f});
    const auto clamped = host.measureOf(ctx, {300.0f, kInfinite});
    CHECK(clamped.x == 100.0f);
    CHECK(clamped.y == 45.0f);
}

TEST_CASE("Filling panels take the available or configured height", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({120.0f, 45.0f});
    Panel panel;
    Host host(panel);

    panel.update({.content = &content, .size = {Sakura::Fill, Sakura::Fill}});
    CHECK(host.measureOf(ctx, {300.0f, 200.0f}).y == 200.0f);
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}).y == 0.0f);

    panel.update({.content = &content, .size = {Sakura::Fill, 50.0f}});
    CHECK(host.measureOf(ctx, {300.0f, 200.0f}).y == 50.0f);
    CHECK(host.measureOf(ctx, {300.0f, 30.0f}).y == 30.0f);
}

TEST_CASE("Panels lay content over their frame and clip it", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({120.0f, 45.0f});
    Panel panel;
    panel.update({.content = &content});
    Place slot(panel, {50.0f, 50.0f, 100.0f, 80.0f});

    slot.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(content.laidFrame == Rect{50.0f, 50.0f, 100.0f, 80.0f});
    CHECK(content.laidClip == Rect{50.0f, 50.0f, 100.0f, 80.0f});

    slot.move({350.0f, 350.0f, 100.0f, 80.0f});
    slot.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(content.laidClip == Rect{350.0f, 350.0f, 50.0f, 50.0f});
}

TEST_CASE("Layers fill their frame with the content", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({10.0f, 10.0f});
    Layer layer;
    layer.update({.content = &content});
    Host host(layer);

    host.place(ctx, {10.0f, 10.0f, 300.0f, 200.0f});

    CHECK(content.laidFrame == Rect{10.0f, 10.0f, 300.0f, 200.0f});
    const auto measured = host.measureOf(ctx, {300.0f, 200.0f});
    CHECK(measured.x == 300.0f);
    CHECK(measured.y == 200.0f);
}

TEST_CASE("Layers cover points only while they have content", "[core][sakura][layout]") {
    Probe content({10.0f, 10.0f});
    Layer layer;
    CHECK_FALSE(layer.covers({5.0f, 5.0f}));

    layer.update({.content = &content});
    CHECK(layer.covers({5.0f, 5.0f}));
    CHECK(layer.covers({5000.0f, -5000.0f}));

    layer.update({.content = nullptr});
    CHECK_FALSE(layer.covers({5.0f, 5.0f}));
    CHECK_FALSE(content.visible());
}

TEST_CASE("Grids place items in padded columns and rows", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({10.0f, 20.0f});
    Probe b({10.0f, 30.0f});
    Probe c({10.0f, 10.0f});
    Grid grid;
    grid.update({.items = {&a, &b, &c}});
    Host host(grid);

    host.place(ctx, {0.0f, 0.0f, 200.0f, 400.0f});

    CHECK(a.laidFrame == Rect{4.0f, 2.0f, 92.0f, 30.0f});
    CHECK(b.laidFrame == Rect{104.0f, 2.0f, 92.0f, 30.0f});
    CHECK(c.laidFrame == Rect{4.0f, 36.0f, 92.0f, 10.0f});

    const auto measured = host.measureOf(ctx, {200.0f, kInfinite});
    CHECK(measured.x == 200.0f);
    CHECK(measured.y == 48.0f);
}

TEST_CASE("Grids without outer padding start at the frame edge", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({10.0f, 20.0f});
    Probe b({10.0f, 20.0f});
    Probe c({10.0f, 20.0f});
    Grid grid;
    grid.update({.columns = 3, .outerPadding = false, .items = {&a, &b, &c}});
    Host host(grid);

    host.place(ctx, {10.0f, 0.0f, 316.0f, 400.0f});

    CHECK(a.laidFrame == Rect{10.0f, 2.0f, 100.0f, 20.0f});
    CHECK(b.laidFrame == Rect{118.0f, 2.0f, 100.0f, 20.0f});
    CHECK(c.laidFrame == Rect{226.0f, 2.0f, 100.0f, 20.0f});
}

TEST_CASE("Grids with a minimum cell width drop columns as they narrow", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({10.0f, 20.0f});
    Probe b({10.0f, 20.0f});
    Probe c({10.0f, 20.0f});
    Grid grid;
    grid.update({.columns = 3, .minCellWidth = 100.0f, .outerPadding = false, .items = {&a, &b, &c}});
    Host host(grid);

    host.place(ctx, {0.0f, 0.0f, 1000.0f, 400.0f});
    CHECK(c.laidFrame.y == a.laidFrame.y);
    CHECK(c.laidFrame.width > 300.0f);

    host.place(ctx, {0.0f, 0.0f, 316.0f, 400.0f});
    CHECK(a.laidFrame == Rect{0.0f, 2.0f, 100.0f, 20.0f});
    CHECK(c.laidFrame == Rect{216.0f, 2.0f, 100.0f, 20.0f});

    host.place(ctx, {0.0f, 0.0f, 250.0f, 400.0f});
    CHECK(a.laidFrame == Rect{0.0f, 2.0f, 121.0f, 20.0f});
    CHECK(b.laidFrame == Rect{129.0f, 2.0f, 121.0f, 20.0f});
    CHECK(c.laidFrame == Rect{0.0f, 26.0f, 121.0f, 20.0f});
    CHECK(host.measureOf(ctx, {250.0f, 400.0f}).y == 48.0f);

    host.place(ctx, {0.0f, 0.0f, 60.0f, 400.0f});
    CHECK(a.laidFrame == Rect{0.0f, 2.0f, 60.0f, 20.0f});
    CHECK(b.laidFrame == Rect{0.0f, 26.0f, 60.0f, 20.0f});
    CHECK(c.laidFrame == Rect{0.0f, 50.0f, 60.0f, 20.0f});
}

TEST_CASE("Overflowing grids reserve the scrollbar and clip items", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({10.0f, 20.0f});
    Probe b({10.0f, 30.0f});
    Probe c({10.0f, 10.0f});
    Grid grid;
    grid.update({.items = {&a, &b, &c}});
    Place slot(grid, {0.0f, 0.0f, 200.0f, 40.0f});

    slot.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(a.laidFrame == Rect{4.0f, 2.0f, 86.0f, 30.0f});
    CHECK(b.laidFrame == Rect{98.0f, 2.0f, 86.0f, 30.0f});
    CHECK(c.laidFrame == Rect{4.0f, 36.0f, 86.0f, 10.0f});
    CHECK(c.laidClip == Rect{0.0f, 0.0f, 200.0f, 40.0f});
}

TEST_CASE("Grids honour an explicit size", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe a({10.0f, 20.0f});
    Grid grid;
    grid.update({.size = {150.0f, 60.0f}, .items = {&a}});
    Host host(grid);

    const auto measured = host.measureOf(ctx, {300.0f, kInfinite});
    CHECK(measured.x == 150.0f);
    CHECK(measured.y == 60.0f);
}

TEST_CASE("Divs measure content plus padding", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({50.0f, 20.0f});
    Div div;
    div.update({.content = &content, .padding = 10.0f});
    Host host(div);

    const auto bounded = host.measureOf(ctx, {300.0f, kInfinite});
    CHECK(bounded.x == 300.0f);
    CHECK(bounded.y == 40.0f);

    const auto unbounded = host.measureOf(ctx, {kInfinite, kInfinite});
    CHECK(unbounded.x == 70.0f);
    CHECK(unbounded.y == 40.0f);

    div.update({.content = &content, .size = {120.0f, 60.0f}, .padding = 10.0f});
    const auto sized = host.measureOf(ctx, {300.0f, kInfinite});
    CHECK(sized.x == 120.0f);
    CHECK(sized.y == 60.0f);
}

TEST_CASE("Divs inset content by the padding", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({50.0f, 20.0f});
    Div div;
    div.update({.content = &content, .padding = 10.0f});
    Host host(div);

    host.place(ctx, {10.0f, 10.0f, 200.0f, 100.0f});

    CHECK(content.laidFrame == Rect{20.0f, 20.0f, 180.0f, 80.0f});
}

TEST_CASE("Div waves tile sine rows inside the rounded corners", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({50.0f, 20.0f});
    Div div;
    div.update({.content = &content, .rounding = 16.0f});
    Host host(div);
    host.place(ctx, {0.0f, 0.0f, 560.0f, 280.0f});
    CHECK(Find<Sakura::Polyline>(div).empty());

    div.update({.content = &content, .rounding = 16.0f, .waves = true});
    host.place(ctx, {0.0f, 0.0f, 560.0f, 280.0f});
    const auto waves = Find<Sakura::Polyline>(div);
    REQUIRE(waves.size() == 1);
    const auto& config = waves[0]->config();
    REQUIRE(config.instances.size() == 10);
    const auto& first = config.instances.front().points;
    REQUIRE(first.size() > 4);
    CHECK(first[0].x == 0.0f);
    CHECK(first[0].y == 14.0f);
    CHECK(std::abs(first[3].y - 3.5f) < 0.01f);
    CHECK(first.back().x >= 560.0f);
    CHECK(config.instances.front().color.a < 1.0f);
    REQUIRE(config.clip.has_value());
    const F32 corner = 16.0f * (1.0f - 0.70710678f);
    CHECK(std::abs(config.clip->x - corner) < 0.01f);
    CHECK(std::abs(config.clip->bottom() - (280.0f - corner)) < 0.01f);
    CHECK(content.laidFrame == Rect{0.0f, 0.0f, 560.0f, 280.0f});
}

TEST_CASE("Div waves fit a viewport sized budget and only draw the rows inside the clip", "[core][sakura][layout]") {
    const auto ctx = MakeContext({7680.0f, 4320.0f});
    const auto vertices = [](const Sakura::Polyline::Config& config) {
        U64 total = 0;
        for (const auto& instance : config.instances) {
            total += 4 * instance.points.size();
        }
        return total;
    };
    Div div;
    div.update({.waves = true});
    Host host(div);
    host.place(ctx, {0.0f, 0.0f, 7680.0f, 4320.0f});
    const auto waves = Find<Sakura::Polyline>(div);
    REQUIRE(waves.size() == 1);
    const auto& full = waves[0]->config();
    REQUIRE_FALSE(full.instances.empty());
    CHECK(full.instances.back().points.front().y >= 4320.0f - 28.0f);
    CHECK(full.instances.back().points.back().x >= 7680.0f);
    CHECK(vertices(full) <= full.capacity);
    const U64 capacity = full.capacity;

    Div tall;
    tall.update({.waves = true});
    Place slot(tall, {0.0f, -2000.0f, 600.0f, 8000.0f});
    slot.place(ctx, {0.0f, 0.0f, 600.0f, 300.0f});
    const auto& clipped = Find<Sakura::Polyline>(tall).at(0)->config();
    CHECK(clipped.capacity == capacity);
    CHECK(clipped.instances.size() == 12);
    for (const auto& instance : clipped.instances) {
        CHECK(instance.points.front().x == 0.0f);
        CHECK(instance.points.front().y >= -28.0f);
        CHECK(instance.points.front().y <= 328.0f);
    }

    host.place(MakeContext(), {0.0f, 0.0f, 800.0f, 600.0f});
    CHECK(waves[0]->config().capacity == 65536);
}

TEST_CASE("Div clicks run the click callback", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({50.0f, 20.0f});
    content.consume = false;
    U64 clicks = 0;
    Div div;
    div.update({.content = &content, .onClick = [&clicks] { ++clicks; }});
    Place slot(div, {50.0f, 50.0f, 100.0f, 40.0f});
    slot.place(ctx, {0.0f, 0.0f, 400.0f, 400.0f});

    CHECK(slot.send(Mouse(MouseEventType::Click, 60.0f, 60.0f)));
    CHECK(clicks == 1);

    CHECK(slot.send(Mouse(MouseEventType::Click, 60.0f, 60.0f, MouseButton::Left, 2)));
    CHECK(clicks == 2);

    CHECK_FALSE(slot.send(Mouse(MouseEventType::Click, 10.0f, 10.0f)));
    CHECK_FALSE(slot.send(Mouse(MouseEventType::Click, 60.0f, 60.0f, MouseButton::Right)));
    CHECK_FALSE(slot.send(Mouse(MouseEventType::Release, 60.0f, 60.0f)));
    CHECK(clicks == 2);
}

TEST_CASE("Div double clicks prefer the double click callback", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    U64 clicks = 0;
    U64 doubles = 0;
    Div div;
    div.update({.onClick = [&clicks] { ++clicks; }, .onDoubleClick = [&doubles] { ++doubles; }});
    Host host(div);
    host.place(ctx, {0.0f, 0.0f, 100.0f, 40.0f});

    CHECK(host.send(Mouse(MouseEventType::Click, 10.0f, 10.0f, MouseButton::Left, 2)));
    CHECK(doubles == 1);
    CHECK(clicks == 0);

    CHECK(host.click(10.0f, 10.0f));
    CHECK(doubles == 1);
    CHECK(clicks == 1);
}

TEST_CASE("Divs without callbacks or with consuming content ignore clicks", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe content({50.0f, 20.0f});
    U64 clicks = 0;
    Div div;
    div.update({});
    Host host(div);
    host.place(ctx, {0.0f, 0.0f, 100.0f, 40.0f});

    CHECK_FALSE(host.click(10.0f, 10.0f));

    div.update({.content = &content, .onClick = [&clicks] { ++clicks; }});
    host.place(ctx, {0.0f, 0.0f, 100.0f, 40.0f});

    CHECK(host.click(10.0f, 10.0f));
    CHECK(content.count(MouseEventType::Click) == 1);
    CHECK(clicks == 0);
}

TEST_CASE("Scroller track insets move the track off its edge and pull both ends inward", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Sakura::Scroller scroll;
    Host host(scroll);
    scroll.update({.id = "scroll", .contentWidth = 600.0f, .contentHeight = 600.0f, .trackInset = {4.0f, 12.0f}});
    host.place(ctx, {10.0f, 10.0f, 200.0f, 100.0f});

    const auto boxes = Direct<Sakura::Box>(scroll);
    REQUIRE(boxes.size() >= 3);
    const Rect vertical = boxes[0]->config().instances.at(0).rect;
    const Rect horizontal = boxes[2]->config().instances.at(0).rect;
    CHECK(vertical == Rect{195.0f, 25.0f, 8.0f, 52.0f});
    CHECK(horizontal == Rect{25.0f, 95.0f, 152.0f, 8.0f});
}

TEST_CASE("Scrollers follow trackpad gestures point for point and keep the wheel step for mouse wheels",
          "[core][sakura][layout]") {
    auto ctx = MakeContext();
    Sakura::Scroller scroll;
    Host host(scroll);
    F32 scrollY = 0.0f;
    const auto update = [&] {
        scroll.update({.id = "scroll", .contentHeight = 2000.0f, .scrollY = scrollY, .wheelStep = 48.0f,
                       .onScrollY = [&](F32 next) { scrollY = next; }});
    };
    const auto place = [&] {
        update();
        host.place(ctx, {0.0f, 0.0f, 200.0f, 300.0f});
    };
    const auto wheel = [&](F32 dy, ScrollPhase phase) {
        auto event = Wheel(100.0f, 150.0f, dy);
        event.phase = phase;
        return host.send(event);
    };

    place();
    REQUIRE(wheel(-1.0f, ScrollPhase::None));
    CHECK(scrollY == 48.0f);
    place();
    REQUIRE(wheel(-0.5f, ScrollPhase::Changed));
    CHECK(scrollY == 53.0f);
    place();
    REQUIRE(wheel(-0.3f, ScrollPhase::Momentum));
    CHECK(scrollY == 56.0f);

    ctx.pixelRatio = 2.0f;
    place();
    REQUIRE(wheel(-1.0f, ScrollPhase::Changed));
    CHECK(scrollY == 76.0f);
}

namespace {

struct ElasticRig {
    Sakura::Context ctx = MakeContext();
    Sakura::Scroller scroll;
    Host host{scroll};
    F32 scrollY = 0.0f;
    U64 notified = 0;

    ElasticRig() {
        ctx.time = 1.0;
        place();
    }

    void place() {
        scroll.update({.id = "elastic",
                       .contentHeight = 1000.0f,
                       .scrollY = scrollY,
                       .wheelStep = 48.0f,
                       .onScrollY = [this](F32 next) { scrollY = next; },
                       .onOverscroll = [this] { ++notified; }});
        host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    }

    bool wheel(F32 dy, ScrollPhase phase) {
        auto event = Wheel(100.0f, 50.0f, dy);
        event.phase = phase;
        const bool handled = host.send(event);
        place();
        return handled;
    }

    bool precise(F32 dy) {
        auto event = Wheel(100.0f, 50.0f, dy);
        event.precise = true;
        const bool handled = host.send(event);
        place();
        return handled;
    }

    F32 offset() {
        return scroll.overscroll(ctx).y;
    }

    F32 advance(F64 seconds) {
        for (F64 elapsed = 0.0; elapsed < seconds; elapsed += 1.0 / 60.0) {
            ctx.time += 1.0 / 60.0;
            place();
        }
        return offset();
    }
};

}  // namespace

TEST_CASE("Trackpad drags stretch past the edge with resistance and spring back once released",
          "[core][sakura][layout]") {
    ElasticRig rig;
    REQUIRE(rig.wheel(0.0f, ScrollPhase::Began));
    REQUIRE(rig.wheel(3.0f, ScrollPhase::Changed));
    const F32 first = rig.offset();
    CHECK(first < 0.0f);
    CHECK(first > -30.0f);
    CHECK(rig.scrollY == 0.0f);
    REQUIRE(rig.wheel(3.0f, ScrollPhase::Changed));
    const F32 second = rig.offset();
    CHECK(second < first);
    CHECK(first - second < -first);
    CHECK(rig.notified > 0);

    CHECK(rig.advance(0.2) == second);
    CHECK_FALSE(rig.wheel(0.0f, ScrollPhase::Ended));
    F32 previous = rig.offset();
    for (U64 frame = 0; frame < 6; ++frame) {
        const F32 next = rig.advance(1.0 / 60.0);
        CHECK(next >= previous);
        previous = next;
    }
    CHECK(previous < 0.0f);
    CHECK(rig.advance(1.0) == 0.0f);
    CHECK(rig.scrollY == 0.0f);
}

TEST_CASE("Momentum after a stretched release never bounces again once the spring settles",
          "[core][sakura][layout]") {
    ElasticRig rig;
    rig.wheel(0.0f, ScrollPhase::Began);
    rig.wheel(6.0f, ScrollPhase::Changed);
    REQUIRE(rig.offset() < 0.0f);
    rig.wheel(0.0f, ScrollPhase::Ended);
    bool settled = false;
    for (U64 frame = 0; frame < 120; ++frame) {
        rig.wheel(1.0f, ScrollPhase::Momentum);
        const F32 offset = rig.advance(1.0 / 60.0);
        if (settled) {
            CHECK(offset == 0.0f);
        }
        settled = settled || offset == 0.0f;
    }
    CHECK(settled);
    CHECK(rig.scrollY == 0.0f);

    rig.wheel(0.0f, ScrollPhase::MomentumEnded);
    rig.wheel(0.0f, ScrollPhase::Began);
    rig.wheel(3.0f, ScrollPhase::Changed);
    CHECK(rig.offset() < 0.0f);
}

TEST_CASE("Stretched gestures that never report a lift release after a timeout", "[core][sakura][layout]") {
    ElasticRig rig;
    rig.wheel(0.0f, ScrollPhase::Began);
    rig.wheel(4.0f, ScrollPhase::Changed);
    const F32 held = rig.offset();
    REQUIRE(held < 0.0f);
    CHECK(rig.advance(0.25) == held);
    CHECK(rig.advance(1.5) == 0.0f);
}

TEST_CASE("Stretches that start after an idle period hold until the gesture times out", "[core][sakura][layout]") {
    ElasticRig rig;
    rig.ctx.time += 5.0;
    rig.wheel(0.0f, ScrollPhase::Began);
    rig.wheel(4.0f, ScrollPhase::Changed);
    const F32 held = rig.offset();
    REQUIRE(held < 0.0f);
    CHECK(rig.advance(0.2) == held);
    CHECK(rig.advance(1.5) == 0.0f);
}

TEST_CASE("Scrollers leave trackpad gestures on an axis they cannot scroll to the scroller below",
          "[core][sakura][layout]") {
    auto ctx = MakeContext();
    ctx.time = 1.0;
    Sakura::Scroller scroll;
    Host host(scroll);
    scroll.update({.id = "sideways", .contentWidth = 600.0f, .contentHeight = 100.0f});
    host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    const auto wheel = [&](F32 dx, F32 dy, ScrollPhase phase) {
        auto event = Wheel(100.0f, 50.0f, dx, dy);
        event.phase = phase;
        const bool handled = host.send(event);
        host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
        return handled;
    };

    CHECK(wheel(0.0f, 0.0f, ScrollPhase::Began));
    CHECK_FALSE(wheel(0.0f, -3.0f, ScrollPhase::Changed));
    CHECK_FALSE(wheel(0.0f, -3.0f, ScrollPhase::Changed));
    CHECK(wheel(-3.0f, 0.0f, ScrollPhase::Changed));
}

TEST_CASE("Momentum into an edge bounces out and settles while mouse wheels stay clamped",
          "[core][sakura][layout]") {
    ElasticRig rig;
    rig.scrollY = 895.0f;
    rig.place();
    rig.wheel(0.0f, ScrollPhase::Ended);
    rig.advance(0.1);
    rig.wheel(-2.0f, ScrollPhase::Momentum);
    CHECK(rig.scrollY == 900.0f);
    F32 peak = 0.0f;
    for (U64 frame = 0; frame < 20; ++frame) {
        rig.wheel(-1.0f, ScrollPhase::Momentum);
        peak = std::max(peak, rig.advance(1.0 / 60.0));
    }
    CHECK(peak > 0.0f);
    CHECK(rig.advance(1.5) == 0.0f);
    CHECK(rig.scrollY == 900.0f);

    rig.scrollY = 0.0f;
    rig.place();
    CHECK(rig.wheel(5.0f, ScrollPhase::None));
    CHECK(rig.scrollY == 0.0f);
    CHECK(rig.offset() == 0.0f);
}

TEST_CASE("Momentum heading back into the content scrolls right after a bounce", "[core][sakura][layout]") {
    ElasticRig rig;
    const auto peak = [&] {
        F32 highest = 0.0f;
        for (U64 frame = 0; frame < 10; ++frame) {
            highest = std::max(highest, rig.advance(1.0 / 60.0));
        }
        return highest;
    };
    rig.scrollY = 895.0f;
    rig.place();
    rig.wheel(-2.0f, ScrollPhase::Momentum);
    CHECK(rig.scrollY == 900.0f);
    CHECK(peak() > 0.0f);

    rig.wheel(-1.0f, ScrollPhase::Momentum);
    CHECK(rig.scrollY == 900.0f);

    rig.wheel(3.0f, ScrollPhase::Momentum);
    CHECK(rig.scrollY == 870.0f);
    CHECK(rig.advance(1.5) == 0.0f);

    rig.wheel(-4.0f, ScrollPhase::Momentum);
    CHECK(rig.scrollY == 900.0f);
    CHECK(peak() == 0.0f);

    rig.wheel(0.0f, ScrollPhase::MomentumEnded);
    rig.wheel(3.0f, ScrollPhase::Momentum);
    rig.wheel(-4.0f, ScrollPhase::Momentum);
    CHECK(rig.scrollY == 900.0f);
    CHECK(peak() > 0.0f);
}

TEST_CASE("Precise scrolling without gesture phases moves one to one and clamps without overscroll",
          "[core][sakura][layout]") {
    ElasticRig rig;
    rig.scrollY = 880.0f;
    rig.place();
    CHECK(rig.precise(-1.5f));
    CHECK(rig.scrollY == 895.0f);
    rig.precise(-4.0f);
    CHECK(rig.scrollY == 900.0f);
    CHECK(rig.offset() == 0.0f);
    CHECK(rig.advance(0.5) == 0.0f);
    rig.precise(2.0f);
    CHECK(rig.scrollY == 880.0f);
    CHECK(rig.notified == 0);
}

TEST_CASE("Scroll areas move their content with the stretch", "[core][sakura][layout]") {
    auto ctx = MakeContext();
    ctx.time = 1.0;
    Probe child({100.0f, 1000.0f});
    child.consume = false;
    Sakura::ScrollArea area;
    area.update({.id = "stretch", .content = &child, .height = 200.0f, .border = false});
    Host host(area);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    const F32 rest = child.laidFrame.y;

    auto event = Wheel(100.0f, 100.0f, 0.0f);
    event.phase = ScrollPhase::Began;
    host.send(event);
    event.scroll.y = 5.0f;
    event.phase = ScrollPhase::Changed;
    host.send(event);
    ctx.time += 1.0 / 60.0;
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    CHECK(child.laidFrame.y > rest);
}

TEST_CASE("Scroll areas drop the stretch and gesture of the previous id", "[core][sakura][layout]") {
    auto ctx = MakeContext();
    ctx.time = 1.0;
    Probe child({100.0f, 1000.0f});
    child.consume = false;
    Sakura::ScrollArea area;
    area.update({.id = "a", .content = &child, .height = 200.0f, .border = false});
    Host host(area);
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    const F32 rest = child.laidFrame.y;

    auto event = Wheel(100.0f, 100.0f, 0.0f);
    event.phase = ScrollPhase::Began;
    host.send(event);
    event.scroll.y = 5.0f;
    event.phase = ScrollPhase::Changed;
    host.send(event);
    ctx.time += 1.0 / 60.0;
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    REQUIRE(child.laidFrame.y > rest);

    area.update({.id = "b", .content = &child, .height = 200.0f, .border = false});
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    CHECK(child.laidFrame.y == rest);
    ctx.time += 1.0 / 60.0;
    host.place(ctx, {0.0f, 0.0f, 300.0f, 200.0f});
    CHECK(child.laidFrame.y == rest);

    event.scroll = {0.0f, 0.0f};
    event.phase = ScrollPhase::Ended;
    CHECK_FALSE(host.send(event));
}

TEST_CASE("Panels remount a child that came back from another owner", "[core][sakura][layout]") {
    const auto shown = [](Sakura::Component& root, Probe& probe) {
        const auto probes = Find<Probe>(root);
        return std::find(probes.begin(), probes.end(), &probe) != probes.end();
    };

    Probe child({50.0f, 20.0f});
    Sakura::Panel first;
    Sakura::Panel second;
    Place firstRoot(first, {0.0f, 0.0f, 200.0f, 100.0f});
    Place secondRoot(second, {0.0f, 0.0f, 200.0f, 100.0f});
    Frames firstFrames(firstRoot);
    Frames secondFrames(secondRoot);
    firstFrames.verify = false;
    secondFrames.verify = false;
    first.update({.content = &child});
    firstFrames.run();
    REQUIRE(shown(firstRoot, child));
    second.update({.content = &child});
    secondFrames.run();
    REQUIRE(shown(secondRoot, child));
    second.update({});
    secondFrames.run();
    first.update({.content = &child});
    firstFrames.run();
    CHECK(shown(firstRoot, child));
}

TEST_CASE("Dragging an inset scroller thumb in place keeps the scroll offset", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Sakura::Scroller scroll;
    Host host(scroll);
    F32 scrollY = 0.0f;
    const auto update = [&] {
        scroll.update({.id = "scroll",
                       .contentHeight = 900.0f,
                       .scrollY = scrollY,
                       .thickness = 12.0f,
                       .thumbInset = 3.0f,
                       .onScrollY = [&](F32 next) { scrollY = next; }});
    };
    update();
    host.place(ctx, {0.0f, 0.0f, 200.0f, 300.0f});

    const Rect thumb = Direct<Sakura::Box>(scroll)[1]->config().instances.at(0).rect;
    const F32 x = thumb.x + thumb.width * 0.5f;
    const F32 y = thumb.y + 4.0f;
    REQUIRE(host.send(Mouse(MouseEventType::Click, x, y)));
    host.send(Mouse(MouseEventType::Move, x, y));
    CHECK(scrollY == 0.0f);

    host.send(Mouse(MouseEventType::Move, x, y + 1000.0f));
    host.send(Mouse(MouseEventType::Release, x, y + 1000.0f));
    update();
    host.place(ctx, {0.0f, 0.0f, 200.0f, 300.0f});
    const F32 bottom = scrollY;
    const Rect end = Direct<Sakura::Box>(scroll)[1]->config().instances.at(0).rect;
    REQUIRE(host.send(Mouse(MouseEventType::Click, x, end.y + 4.0f)));
    host.send(Mouse(MouseEventType::Move, x, end.y + 4.0f));
    CHECK(scrollY == bottom);
}

TEST_CASE("Stacks reclaim a child another parent mounted while their config was unchanged",
          "[core][sakura][layout]") {
    Sakura::Spacer child;
    Sakura::Stack first;
    Sakura::Stack second;
    const auto owns = [](const Sakura::Component& parent, const Sakura::Component& component) {
        const auto& children = Peek::Children(parent);
        return std::find(children.begin(), children.end(), &component) != children.end();
    };

    first.update({.items = {{.child = &child}}});
    CHECK(owns(first, child));
    second.update({.items = {{.child = &child}}});
    CHECK(owns(second, child));
    CHECK_FALSE(owns(first, child));

    first.update({.items = {{.child = &child}}});
    CHECK(owns(first, child));
    CHECK(child.visible());
}

TEST_CASE("Horizontal stacks shrink flagged items when their children overflow", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe wide({300.0f, 20.0f});
    Probe fixed({80.0f, 20.0f});
    Sakura::Stack stack;
    stack.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = 0.0f,
        .items = {{.child = &wide, .shrink = 1.0f}, {.child = &fixed}},
    });
    Host host(stack);
    host.place(ctx, {0.0f, 0.0f, 200.0f, 20.0f});
    CHECK(Peek::Frame(wide).width == 120.0f);
    CHECK(Peek::Frame(fixed).x == 120.0f);
    CHECK(Peek::Frame(fixed).width == 80.0f);
}

TEST_CASE("Stacks spread the overflow an item cannot absorb across the other shrinking items",
          "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe narrow({20.0f, 20.0f});
    Probe wide({300.0f, 20.0f});
    Probe fixed({80.0f, 20.0f});
    Sakura::Stack stack;
    stack.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = 0.0f,
        .items = {{.child = &narrow, .shrink = 3.0f}, {.child = &wide, .shrink = 1.0f}, {.child = &fixed}},
    });
    Host host(stack);
    CHECK(host.measureOf(ctx, {200.0f, kInfinite}).x == 200.0f);
    host.place(ctx, {0.0f, 0.0f, 200.0f, 20.0f});
    CHECK(Peek::Frame(narrow).width == 0.0f);
    CHECK(Peek::Frame(wide).x == 0.0f);
    CHECK(Peek::Frame(wide).width == 120.0f);
    CHECK(Peek::Frame(fixed).x == 120.0f);
    CHECK(Peek::Frame(fixed).width == 80.0f);

    stack.update({
        .axis = Sakura::Stack::Axis::Vertical,
        .spacing = 0.0f,
        .items = {{.child = &narrow, .shrink = 3.0f}, {.child = &wide, .shrink = 1.0f}, {.child = &fixed}},
    });
    narrow.resize({20.0f, 10.0f});
    wide.resize({20.0f, 100.0f});
    fixed.resize({20.0f, 30.0f});
    CHECK(host.measureOf(ctx, {20.0f, 60.0f}).y == 60.0f);
    host.place(ctx, {0.0f, 0.0f, 20.0f, 60.0f});
    CHECK(Peek::Frame(narrow).height == 0.0f);
    CHECK(Peek::Frame(wide).height == 30.0f);
    CHECK(Peek::Frame(fixed).y == 30.0f);
    CHECK(Peek::Frame(fixed).height == 30.0f);
}

TEST_CASE("Horizontal stacks measure shrunk items at their shrunk width", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Flow wide(300.0f, 20.0f);
    Probe fixed({80.0f, 20.0f});
    Sakura::Stack stack;
    stack.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = 0.0f,
        .centerCross = false,
        .items = {{.child = &wide, .shrink = 1.0f}, {.child = &fixed}},
    });
    Host host(stack);
    CHECK(host.measureOf(ctx, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 50.0f});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{380.0f, 20.0f});

    auto doubled = ctx;
    doubled.pixelRatio = 2.0f;
    stack.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = 10.0f,
        .centerCross = false,
        .items = {{.child = &wide, .shrink = 1.0f}, {.child = &fixed}},
    });
    CHECK(host.measureOf(doubled, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 60.0f});

    host.place(ctx, {0.0f, 0.0f, 200.0f, 50.0f});
    CHECK(Peek::Frame(wide).width == 110.0f);
    CHECK(Peek::Frame(wide).height == 50.0f);
}

TEST_CASE("Wave lines draw one sine stroke across their width inside the amplitude", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Sakura::WaveLine wave;
    wave.update({.period = 20.0f});
    Host host(wave);
    const auto size = host.measureOf(ctx, {400.0f, 100.0f});
    CHECK(size.x == 400.0f);
    CHECK(size.y >= 12.0f);
    CHECK(size.y <= 16.0f);
    host.place(ctx, {10.0f, 20.0f, 400.0f, size.y});

    const auto lines = Find<Sakura::Polyline>(wave);
    REQUIRE(lines.size() == 1);
    const auto& config = lines[0]->config();
    REQUIRE(config.instances.size() == 1);
    const auto& points = config.instances[0].points;
    REQUIRE(points.size() > 200);
    CHECK(points.front().x == 10.0f);
    CHECK(points.back().x >= 400.0f);
    const F32 center = 20.0f + size.y * 0.5f;
    for (const auto& point : points) {
        CHECK(std::abs(point.y - center) <= 5.0f + 1e-3f);
    }
    wave.update({.period = 20.0f, .span = 0.5f});
    host.place(ctx, {10.0f, 20.0f, 400.0f, size.y});
    const auto& centered = Find<Sakura::Polyline>(wave).front()->config().instances.front().points;
    CHECK(centered.front().x == 110.0f);
    CHECK(centered.back().x <= 310.0f);
    CHECK(centered.back().x > 300.0f);
}

TEST_CASE("Wave lines size their vertex budget from the viewport so a full width wave is never dropped",
          "[core][sakura][layout]") {
    Sakura::WaveLine wave;
    wave.update({});
    Host host(wave);
    host.place(MakeContext(), {0.0f, 0.0f, 800.0f, 16.0f});
    const auto lines = Find<Sakura::Polyline>(wave);
    REQUIRE(lines.size() == 1);
    CHECK(lines[0]->config().capacity == 16384);

    host.place(MakeContext({7680.0f, 4320.0f}), {0.0f, 0.0f, 7680.0f, 16.0f});
    const auto& config = lines[0]->config();
    REQUIRE(config.instances.size() == 1);
    const auto& points = config.instances[0].points;
    CHECK(points.front().x == 0.0f);
    CHECK(points.back().x >= 7680.0f);
    CHECK(4 * points.size() > 16384);
    CHECK(4 * points.size() <= config.capacity);
}

TEST_CASE("Wave lines settle, repaint on period and span changes, and scale with the pixel ratio",
          "[core][sakura][layout][retention]") {
    Sakura::WaveLine wave;
    Place place(wave, {10.0f, 10.0f, 400.0f, 16.0f});
    Frames frames(place, kScreen, MakeContext());
    F32 period = 18.0f;
    F32 span = 1.0f;
    const auto update = [&] {
        wave.update({.period = period, .span = span});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    CHECK(QuietMoves(frames, update, {{50.0f, 15.0f}, {60.0f, 15.0f}}, [&](const MouseEvent& e) {
        return place.send(e);
    }));
    period = 24.0f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    span = 0.5f;
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == 0);

    Sakura::WaveLine scaled;
    scaled.update({.period = 20.0f});
    Host host(scaled);
    const auto ctx = MakeContext();
    auto doubled = ctx;
    doubled.pixelRatio = 2.0f;
    const auto single = host.measureOf(ctx, {400.0f, kInfinite});
    const auto dense = host.measureOf(doubled, {400.0f, kInfinite});
    CHECK(dense.y >= 2.0f * single.y - 2.0f);
    host.place(doubled, {0.0f, 0.0f, 400.0f, dense.y});
    const auto& instance = Find<Sakura::Polyline>(scaled).front()->config().instances.front();
    CHECK(instance.thickness == 3.2f);
    F32 reach = 0.0f;
    for (const auto& point : instance.points) {
        reach = std::max(reach, std::abs(point.y - dense.y * 0.5f));
    }
    CHECK(reach <= 10.0f + 1e-3f);
    CHECK(reach > 9.0f);
}

TEST_CASE("Overlay scroll areas give content the full width, draw the bar on top and leave the wheel to content first",
          "[core][sakura][layout]") {
    Probe content({0.0f, 600.0f});
    Sakura::ScrollArea area;
    Host host(area);
    const auto ctx = MakeContext();
    const auto place = [&](bool overlay) {
        area.update({.content = &content, .height = Sakura::Fill, .alwaysScrollbar = false, .border = false,
                     .overlay = overlay});
        host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    };

    place(false);
    const F32 gutter = Peek::Frame(content).width;
    CHECK(gutter < 200.0f);
    CHECK(dynamic_cast<Sakura::Scroller*>(Peek::Children(area).front()) != nullptr);

    place(true);
    CHECK(Peek::Frame(content).width == 200.0f);
    REQUIRE(dynamic_cast<Sakura::Scroller*>(Peek::Children(area).back()) != nullptr);
    const auto scrollers = Find<Sakura::Scroller>(area);
    REQUIRE(scrollers.size() == 1);
    const auto boxes = Direct<Sakura::Box>(*scrollers.front());
    REQUIRE_FALSE(boxes.empty());
    CHECK(boxes.front()->config().instances.at(0).rect.right() <= 200.0f);

    const F32 top = Peek::Frame(content).y;
    CHECK(host.send(Wheel(100.0f, 50.0f, -1.0f)));
    host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    CHECK(Peek::Frame(content).y == top);

    content.consume = false;
    CHECK(host.send(Wheel(100.0f, 50.0f, -1.0f)));
    host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    CHECK(Peek::Frame(content).y < top);
    CHECK_FALSE(host.send(Wheel(300.0f, 50.0f, -1.0f)));

    place(false);
    CHECK(Peek::Frame(content).width == gutter);
}

TEST_CASE("Scroll areas return to the top when their id changes and keep the offset otherwise",
          "[core][sakura][layout]") {
    Probe content({0.0f, 600.0f});
    content.consume = false;
    Sakura::ScrollArea area;
    Host host(area);
    const auto ctx = MakeContext();
    const auto place = [&](const std::string& id) {
        area.update({.id = id, .content = &content, .height = Sakura::Fill, .border = false});
        host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    };

    place("a");
    const F32 top = Peek::Frame(content).y;
    CHECK(host.send(Wheel(100.0f, 50.0f, -1.0f)));
    place("a");
    const F32 scrolled = Peek::Frame(content).y;
    CHECK(scrolled < top);
    place("a");
    CHECK(Peek::Frame(content).y == scrolled);
    place("b");
    CHECK(Peek::Frame(content).y == top);
}

TEST_CASE("Scroll area scrollbar insets move the track off its edge and scale with the pixel ratio",
          "[core][sakura][layout]") {
    Probe content({100.0f, 600.0f});
    Sakura::ScrollArea area;
    Host host(area);
    const auto track = [&](Extent2D<F32> inset, F32 ratio) {
        auto ctx = MakeContext();
        ctx.pixelRatio = ratio;
        area.update({.content = &content, .height = Sakura::Fill, .border = false, .scrollbarInset = inset});
        host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
        const auto scrollers = Find<Sakura::Scroller>(area);
        REQUIRE(scrollers.size() == 1);
        const auto boxes = Direct<Sakura::Box>(*scrollers.front());
        REQUIRE_FALSE(boxes.empty());
        return boxes.front()->config().instances.at(0).rect;
    };
    for (const F32 ratio : {1.0f, 2.0f}) {
        CAPTURE(ratio);
        const Rect plain = track({0.0f, 0.0f}, ratio);
        const Rect inset = track({4.0f, 12.0f}, ratio);
        CHECK(inset.x == plain.x - 4.0f * ratio);
        CHECK(inset.y == plain.y + 12.0f * ratio);
        CHECK(inset.height == plain.height - 24.0f * ratio);
    }
}

TEST_CASE("Reveal grows and shrinks its child with the gap and holds it for the minimum duration",
          "[core][sakura][layout]") {
    auto ctx = MakeContext();
    Probe child({100.0f, 40.0f});
    child.prepare();
    Sakura::Reveal reveal;
    reveal.update({.child = &child, .gap = 10.0f, .minimumDuration = 1.0});
    Probe after({100.0f, 10.0f});
    after.prepare();
    Stack stack;
    stack.update({.spacing = 0.0f, .centerCross = false, .items = {{.child = &reveal}, {.child = &after}}});
    Host host(stack);
    const auto height = [&](F64 time) {
        ctx.time = time;
        host.place(ctx, {0.0f, 0.0f, 200.0f, 500.0f});
        return after.currentFrame().y;
    };

    CHECK(height(0.0) == 0.0f);
    reveal.update({.child = &child, .shown = true, .gap = 10.0f, .minimumDuration = 1.0});
    CHECK(height(5.0) == 0.0f);
    CHECK(height(5.125) == 25.0f);
    CHECK(height(5.25) == 50.0f);

    reveal.update({.child = &child, .gap = 10.0f, .minimumDuration = 1.0});
    CHECK(height(5.5) == 50.0f);
    CHECK(height(6.0) == 50.0f);
    CHECK(height(6.125) == 25.0f);
    CHECK(height(6.25) == 0.0f);
    CHECK_FALSE(child.visible());

    reveal.update({.child = &child, .shown = true, .gap = 10.0f, .minimumDuration = 1.0});
    CHECK(height(20.0) == 0.0f);
    CHECK(height(20.25) == 50.0f);
    CHECK(child.currentFrame() == Rect{0.0f, 10.0f, 200.0f, 40.0f});

    reveal.update({.child = &child, .gap = 10.0f, .minimumDuration = 1.0, .linger = 2.0});
    CHECK(height(30.0) == 50.0f);
    CHECK(height(31.5) == 50.0f);
    CHECK(height(32.0) == 50.0f);
    CHECK(height(32.125) == 25.0f);
    CHECK(height(32.25) == 0.0f);
}

TEST_CASE("Reveal measures its child at its natural height, caps to the space and scales the gap",
          "[core][sakura][layout]") {
    auto ctx = MakeContext();
    Probe child({100.0f, 40.0f});
    child.prepare();
    Sakura::Reveal reveal;
    Host host(reveal);
    reveal.update({.child = &child, .shown = true, .gap = 10.0f});
    CHECK(child.visible());
    ctx.time = 1.0;
    host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    ctx.time = 2.0;
    host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});

    CHECK(host.measureOf(ctx, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 50.0f});
    CHECK(host.measureOf(ctx, {200.0f, 30.0f}) == Extent2D<F32>{200.0f, 30.0f});
    auto doubled = ctx;
    doubled.pixelRatio = 2.0f;
    CHECK(host.measureOf(doubled, {200.0f, kInfinite}) == Extent2D<F32>{200.0f, 60.0f});
    host.place(doubled, {0.0f, 0.0f, 200.0f, 60.0f});
    CHECK(child.currentFrame() == Rect{0.0f, 20.0f, 200.0f, 40.0f});
}

TEST_CASE("Reveal drops focus and held presses inside its child once it collapses", "[core][sakura][layout]") {
    auto ctx = MakeContext();
    Probe child({100.0f, 40.0f});
    child.prepare();
    Sakura::Reveal reveal;
    Host host(reveal);
    const auto place = [&](bool shown, F64 time) {
        reveal.update({.child = &child, .shown = shown});
        ctx.time = time;
        host.place(ctx, {0.0f, 0.0f, 200.0f, 100.0f});
    };
    place(true, 1.0);
    place(true, 2.0);
    REQUIRE(child.currentFrame().height == 40.0f);
    REQUIRE(Sakura::Access::focus(child));
    REQUIRE(FocusOwner(host) == &child);
    host.send(Mouse(MouseEventType::Click, 50.0f, 20.0f));
    const U64 cancels = child.count(MouseEventType::Cancel);

    place(false, 3.0);
    CHECK(child.visible());
    CHECK(FocusOwner(host) == &child);

    place(false, 4.0);
    CHECK_FALSE(child.visible());
    CHECK(FocusOwner(host) == nullptr);
    CHECK(child.count(MouseEventType::Cancel) == cancels + 1);
}

TEST_CASE("Grids scale the minimum cell width with the pixel ratio", "[core][sakura][layout]") {
    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    Probe a({10.0f, 20.0f});
    Probe b({10.0f, 20.0f});
    Probe c({10.0f, 20.0f});
    Grid grid;
    grid.update({.columns = 3, .minCellWidth = 100.0f, .outerPadding = false, .items = {&a, &b, &c}});
    Host host(grid);

    host.place(ctx, {0.0f, 0.0f, 632.0f, 400.0f});
    CHECK(a.laidFrame.y == c.laidFrame.y);
    CHECK(a.laidFrame.width == 200.0f);

    host.place(ctx, {0.0f, 0.0f, 500.0f, 400.0f});
    CHECK(c.laidFrame.y > a.laidFrame.y);
    CHECK(a.laidFrame.width == 242.0f);
}

TEST_CASE("Grabbing the scrollbar during spring-back takes over and the thumb follows the drag",
          "[core][sakura][layout]") {
    ElasticRig rig;
    REQUIRE(rig.wheel(0.0f, ScrollPhase::Began));
    REQUIRE(rig.wheel(5.0f, ScrollPhase::Changed));
    rig.wheel(0.0f, ScrollPhase::Ended);
    rig.advance(2.0 / 60.0);
    REQUIRE(rig.offset() < 0.0f);
    const auto thumb = [&] {
        return Direct<Sakura::Box>(rig.scroll)[1]->config().instances.at(0).rect;
    };
    const Rect grabbed = thumb();
    const F32 x = grabbed.x + grabbed.width * 0.5f;
    const F32 y = grabbed.y + grabbed.height * 0.5f;
    REQUIRE(rig.host.send(Mouse(MouseEventType::Click, x, y)));
    CHECK(rig.offset() == 0.0f);
    REQUIRE(rig.host.send(Mouse(MouseEventType::Move, x, y + 30.0f)));
    rig.place();
    CHECK(rig.scrollY > 100.0f);
    CHECK(thumb().y > grabbed.y + 10.0f);
    CHECK(rig.host.send(Mouse(MouseEventType::Release, x, y + 30.0f)));
}

TEST_CASE("Ending a trackpad gesture releases an outer scroller stretched past a nested one",
          "[core][sakura][layout]") {
    struct Nested : Sakura::Component {
        Sakura::Scroller outer;
        Sakura::Scroller inner;
        F32 y = 0.0f;
        F32 x = 0.0f;
        Nested() {
            addChild(outer);
            addChild(inner);
        }
        void tick() { invalidate(Dirty::Layout); }
        void layout(const Sakura::Context& ctx) override {
            outer.update({.id = "outer", .contentHeight = 1000.0f, .scrollY = y,
                          .onScrollY = [this](F32 next) { y = next; }});
            inner.update({.id = "inner", .contentWidth = 1000.0f, .scrollX = x,
                          .onScrollX = [this](F32 next) { x = next; }});
            layoutChild(ctx, outer, frame());
            layoutChild(ctx, inner, frame());
        }
    };
    auto ctx = MakeContext();
    ctx.time = 1.0;
    Nested nested;
    Host host(nested);
    const Rect bounds = {0.0f, 0.0f, 200.0f, 100.0f};
    host.place(ctx, bounds);
    const auto gesture = [&](F32 dy, ScrollPhase phase) {
        auto event = Wheel(100.0f, 50.0f, dy);
        event.phase = phase;
        host.send(event);
        nested.tick();
        host.place(ctx, bounds);
    };
    gesture(0.0f, ScrollPhase::Began);
    gesture(3.0f, ScrollPhase::Changed);
    const F32 stretched = nested.outer.overscroll(ctx).y;
    REQUIRE(stretched < 0.0f);
    CHECK(nested.inner.overscroll(ctx) == Extent2D<F32>{0.0f, 0.0f});

    gesture(0.0f, ScrollPhase::Ended);
    for (U64 frame = 0; frame < 6; ++frame) {
        ctx.time += 1.0 / 60.0;
        nested.tick();
        host.place(ctx, bounds);
    }
    CHECK(nested.outer.overscroll(ctx).y > stretched);
}
