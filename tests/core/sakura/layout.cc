#include <catch2/catch_test_macros.hpp>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/layout.hh>
#include <jetstream/render/sakura/components/scroller.hh>
#include <jetstream/render/sakura/components/shell.hh>

#include "component_harness.hh"

#include <algorithm>
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
using Sakura::SplitView;
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

TEST_CASE("Split views give the left pane its configured width", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe left({10.0f, 10.0f});
    Probe right({10.0f, 10.0f});
    SplitView split;
    split.update({.left = &left, .right = &right, .height = 120.0f});
    Host host(split);

    host.place(ctx, {0.0f, 0.0f, 500.0f, 300.0f});

    CHECK(left.laidFrame == Rect{0.0f, 4.0f, 220.0f, 296.0f});
    CHECK(right.laidFrame == Rect{233.0f, 4.0f, 267.0f, 296.0f});

    const auto measured = host.measureOf(ctx, {500.0f, kInfinite});
    CHECK(measured.x == 500.0f);
    CHECK(measured.y == 120.0f);
}

TEST_CASE("Split views clamp the left pane to narrow frames", "[core][sakura][layout]") {
    const auto ctx = MakeContext();
    Probe left({10.0f, 10.0f});
    Probe right({10.0f, 10.0f});
    SplitView split;
    split.update({.left = &left, .right = &right, .leftWidth = 100.0f});
    Host host(split);

    host.place(ctx, {0.0f, 0.0f, 80.0f, 50.0f});

    CHECK(left.laidFrame == Rect{0.0f, 4.0f, 80.0f, 46.0f});
    CHECK(right.laidFrame == Rect{93.0f, 4.0f, 0.0f, 46.0f});
    CHECK(left.laidClip == Rect{0.0f, 0.0f, 80.0f, 50.0f});
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

TEST_CASE("Panels and split views remount a child that came back from another owner", "[core][sakura][layout]") {
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

    Probe left({50.0f, 20.0f});
    Probe right({50.0f, 20.0f});
    Sakura::SplitView split;
    Sakura::Panel holder;
    Place splitRoot(split, {0.0f, 0.0f, 400.0f, 100.0f});
    Place holderRoot(holder, {0.0f, 0.0f, 200.0f, 100.0f});
    Frames splitFrames(splitRoot);
    Frames holderFrames(holderRoot);
    splitFrames.verify = false;
    holderFrames.verify = false;
    split.update({.left = &left, .right = &right});
    splitFrames.run();
    REQUIRE(shown(splitRoot, left));
    holder.update({.content = &left});
    holderFrames.run();
    holder.update({});
    holderFrames.run();
    split.update({.left = &left, .right = &right});
    splitFrames.run();
    CHECK(shown(splitRoot, left));
    CHECK(shown(splitRoot, right));
}
