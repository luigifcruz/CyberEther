#include <catch2/catch_test_macros.hpp>

#include "fonts.hh"
#include "component_harness.hh"
#include "scene.hh"

#include <jetstream/render/sakura/components/controls.hh>
#include <jetstream/render/sakura/components/table.hh>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace Jetstream;
using namespace SakuraTest;

namespace {


struct Cells {
    std::vector<std::unique_ptr<Probe>> probes;

    Probe& make(Extent2D<F32> desired = {38.0f, 10.0f}) {
        probes.push_back(std::make_unique<Probe>(desired));
        return *probes.back();
    }
};

Sakura::Table::Config ProbeRows(Cells& cells, U64 count, bool consume = true) {
    Sakura::Table::Config config{.columns = {"Name", "Value"}};
    for (U64 row = 0; row < count; ++row) {
        auto& name = cells.make();
        auto& value = cells.make({138.0f, 10.0f});
        name.consume = consume;
        value.consume = consume;
        config.cellRows.push_back({&name, &value});
    }
    return config;
}

Sakura::Table::Nodes Tree() {
    return {
        {.id = "a", .label = "Alpha", .cells = {"1"}, .children = {{.label = "Child", .cells = {"2"}}}},
        {.id = "b", .label = "Beta", .cells = {"3"}},
    };
}

}  // namespace

TEST_CASE("Table columns share the width in proportion to their natural cell widths",
          "[core][sakura][table]") {
    Cells cells;
    Sakura::Table table;
    auto config = ProbeRows(cells, 2);
    config.showHeaders = false;
    table.update(config);
    Host host(table);
    host.place(MakeContext(), {10.0f, 20.0f, 400.0f, 100.0f});

    CHECK(cells.probes[0]->laidFrame == Rect{16.0f, 24.0f, 88.0f, 15.0f});
    CHECK(cells.probes[1]->laidFrame == Rect{116.0f, 24.0f, 288.0f, 15.0f});
    CHECK(cells.probes[2]->laidFrame == Rect{16.0f, 47.0f, 88.0f, 15.0f});
    CHECK(cells.probes[3]->laidFrame == Rect{116.0f, 47.0f, 288.0f, 15.0f});
}

TEST_CASE("Table fixed column widths are honoured and the rest fills the remaining width",
          "[core][sakura][table]") {
    Cells cells;
    Sakura::Table table;
    auto config = ProbeRows(cells, 1);
    config.showHeaders = false;
    config.fixedColumnWidths = {0.0f, 150.0f};
    table.update(config);
    Host host(table);
    host.place(MakeContext(), {0.0f, 0.0f, 400.0f, 100.0f});

    CHECK(cells.probes[0]->laidFrame == Rect{6.0f, 4.0f, 238.0f, 15.0f});
    CHECK(cells.probes[1]->laidFrame == Rect{256.0f, 4.0f, 138.0f, 15.0f});

    auto ctx = MakeContext();
    ctx.pixelRatio = 2.0f;
    host.place(ctx, {0.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[1]->laidFrame.x == 112.0f);
    CHECK(cells.probes[1]->laidFrame.width == 276.0f);
}

TEST_CASE("Table rows grow to their tallest cell", "[core][sakura][table]") {
    Cells cells;
    Sakura::Table table;
    auto& tall = cells.make({38.0f, 40.0f});
    auto& shortCell = cells.make({138.0f, 10.0f});
    auto& next = cells.make();
    table.update({
        .columns = {"Name", "Value"},
        .cellRows = {{&tall, &shortCell}, {&next, nullptr}},
        .showHeaders = false,
    });
    Host host(table);

    CHECK(host.measureOf(MakeContext(), {400.0f, kInfinite}) == Extent2D<F32>{400.0f, 48.0f + 23.0f});

    host.place(MakeContext(), {0.0f, 0.0f, 400.0f, 100.0f});
    CHECK(tall.laidFrame.height == 40.0f);
    CHECK(shortCell.laidFrame.height == 40.0f);
    CHECK(next.laidFrame.y == 52.0f);
}

TEST_CASE("Table headers take the first row and push the body down", "[core][sakura][table]") {
    Cells cells;
    Sakura::Table table;
    auto config = ProbeRows(cells, 2);
    table.update(config);
    Host host(table);

    CHECK(host.measureOf(MakeContext(), {400.0f, kInfinite}).y == 69.0f);
    host.place(MakeContext(), {0.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[0]->laidFrame.y == 27.0f);
    CHECK(cells.probes[2]->laidFrame.y == 50.0f);

    config.showHeaders = false;
    table.update(config);
    CHECK(host.measureOf(MakeContext(), {400.0f, kInfinite}).y == 46.0f);
    host.place(MakeContext(), {0.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[0]->laidFrame.y == 4.0f);
}

TEST_CASE("Table measure follows the configured size and max height", "[core][sakura][table]") {
    Cells cells;
    Sakura::Table table;
    auto config = ProbeRows(cells, 4);
    Host host(table);
    const auto measure = [&](Sakura::Table::Config next, Extent2D<F32> available = {500.0f, kInfinite}) {
        table.update(std::move(next));
        return host.measureOf(MakeContext(), available);
    };

    CHECK(measure(config) == Extent2D<F32>{500.0f, 115.0f});

    auto sized = config;
    sized.size = {300.0f, 60.0f};
    CHECK(measure(sized) == Extent2D<F32>{300.0f, 60.0f});

    auto capped = config;
    capped.maxHeight = 50.0f;
    CHECK(measure(capped) == Extent2D<F32>{500.0f, 50.0f});

    auto filling = config;
    filling.size = {Sakura::Fill, Sakura::Fill};
    CHECK(measure(filling) == Extent2D<F32>{500.0f, 0.0f});
    CHECK(measure(filling, {500.0f, 80.0f}) == Extent2D<F32>{500.0f, 80.0f});

    auto natural = config;
    natural.size = {0.0f, 0.0f};
    CHECK(measure(natural) == Extent2D<F32>{50.0f + 150.0f, 115.0f});

    CHECK(measure(config, {kInfinite, kInfinite}) == Extent2D<F32>{50.0f + 150.0f, 115.0f});
}

TEST_CASE("Empty tables measure to just their header row", "[core][sakura][table]") {
    Sakura::Table table;
    Host host(table);
    const auto ctx = MakeContext();

    table.update({.columns = {"Key", "Value"}});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 23.0f});

    table.update({.columns = {"Key", "Value"}, .showHeaders = false});
    CHECK(host.measureOf(ctx, {300.0f, kInfinite}) == Extent2D<F32>{300.0f, 0.0f});

    table.update({.showHeaders = false});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}) == Extent2D<F32>{0.0f, 0.0f});

    host.place(ctx, {0.0f, 0.0f, 300.0f, 100.0f});
    CHECK_FALSE(host.click(50.0f, 10.0f));
    CHECK_FALSE(host.send(Wheel(50.0f, 10.0f, -1.0f)));
}

TEST_CASE("Table body cells are clipped to the body below the header", "[core][sakura][table]") {
    U64 clicked = 0;
    Cells cells;
    Sakura::Button button;
    button.update({.str = "Approve", .size = {Sakura::Fill, 0.0f}, .onClick = [&] { ++clicked; }});
    Sakura::Table::Config config{.columns = {"Name", "Value"}};
    config.cellRows.push_back({&cells.make({38.0f, 60.0f}), &button});
    for (U64 row = 0; row < 5; ++row) {
        config.cellRows.push_back({&cells.make({38.0f, 60.0f}), &cells.make({138.0f, 60.0f})});
    }
    for (const auto& probe : cells.probes) {
        probe->consume = false;
    }
    Sakura::Table table;
    table.update(config);
    Host host(table);
    const auto ctx = MakeContext();
    host.place(ctx, {0.0f, 0.0f, 400.0f, 100.0f});

    for (const auto& probe : cells.probes) {
        CHECK(probe->laidClip == Rect{0.0f, 23.0f, 400.0f, 77.0f});
    }
    CHECK(cells.probes[1]->laidFrame.y == 95.0f);

    REQUIRE(host.send(Wheel(50.0f, 50.0f, -1.0f)));
    host.place(ctx, {0.0f, 0.0f, 400.0f, 100.0f});
    REQUIRE(cells.probes[0]->laidFrame == Rect{6.0f, -21.0f, 88.0f, 60.0f});

    CHECK_FALSE(host.click(320.0f, 10.0f));
    CHECK(clicked == 0);
    CHECK(host.click(320.0f, 30.0f));
    CHECK(clicked == 1);
}

TEST_CASE("Wheel scrolling moves body rows under a fixed header and clamps at the end",
          "[core][sakura][table]") {
    Cells cells;
    Sakura::Table table;
    table.update(ProbeRows(cells, 10, false));
    Host host(table);
    const auto ctx = MakeContext();
    host.place(ctx, {0.0f, 0.0f, 400.0f, 100.0f});
    REQUIRE(cells.probes[0]->laidFrame.y == 27.0f);

    CHECK_FALSE(host.send(Wheel(50.0f, 10.0f, 0.0f)));
    CHECK(host.send(Wheel(50.0f, 50.0f, -1.0f)));
    host.place(ctx, {0.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[0]->laidFrame.y == 27.0f - 48.0f);
    CHECK(cells.probes[0]->laidClip == Rect{0.0f, 23.0f, 400.0f, 77.0f});

    for (int notch = 0; notch < 10; ++notch) {
        host.send(Wheel(50.0f, 50.0f, -1.0f));
        host.place(ctx, {0.0f, 0.0f, 400.0f, 100.0f});
    }
    CHECK(cells.probes[18]->laidFrame.y == 81.0f);
    CHECK(cells.probes[18]->laidFrame.bottom() + 4.0f == 100.0f);

    CHECK(host.send(Wheel(50.0f, 50.0f, 20.0f)));
    host.place(ctx, {0.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[0]->laidFrame.y == 27.0f);
}

TEST_CASE("Tables that fit their rows ignore the wheel", "[core][sakura][table]") {
    Cells cells;
    Sakura::Table table;
    table.update(ProbeRows(cells, 2, false));
    Host host(table);
    host.place(MakeContext(), {0.0f, 0.0f, 400.0f, 100.0f});

    CHECK_FALSE(host.send(Wheel(50.0f, 50.0f, -1.0f)));
    host.place(MakeContext(), {0.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[0]->laidFrame.y == 27.0f);
}

TEST_CASE("Table cell components receive clicks inside their row", "[core][sakura][table]") {
    U64 clicked = 0;
    Sakura::Button button;
    button.update({.str = "Approve", .size = {Sakura::Fill, 0.0f}, .onClick = [&] { ++clicked; }});
    Probe label({38.0f, 10.0f});
    label.consume = false;
    Sakura::Table table;
    table.update({
        .columns = {"Client", "Status"},
        .cellRows = {{&label, &button}},
        .fixedColumnWidths = {0.0f, 160.0f},
    });
    Host host(table);
    host.place(MakeContext(), {0.0f, 0.0f, 400.0f, 200.0f});

    CHECK(host.click(320.0f, 35.0f));
    CHECK(clicked == 1);
    CHECK_FALSE(host.click(320.0f, 150.0f));
    CHECK(clicked == 1);
}

TEST_CASE("Clicking an expandable tree row toggles its children", "[core][sakura][table]") {
    Sakura::Table table;
    table.update({.columns = {"Name", "Value"}, .nodes = Tree()});
    Host host(table);
    const auto ctx = MakeContext();
    const auto height = [&] {
        return host.measureOf(ctx, {400.0f, kInfinite}).y;
    };
    REQUIRE(height() == 69.0f);
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});

    CHECK(host.click(200.0f, 30.0f));
    CHECK(height() == 92.0f);
    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});

    CHECK_FALSE(host.click(200.0f, 76.0f));
    CHECK_FALSE(host.click(200.0f, 10.0f));
    CHECK(height() == 92.0f);

    CHECK(host.click(200.0f, 30.0f));
    CHECK(height() == 69.0f);

    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});
    CHECK_FALSE(host.click(200.0f, 30.0f, MouseButton::Right));
    CHECK(height() == 69.0f);
}

TEST_CASE("Tree rows start from the configured open state and keep user toggles across updates",
          "[core][sakura][table]") {
    auto nodes = Tree();
    nodes[0].open = true;
    Sakura::Table table;
    table.update({.columns = {"Name", "Value"}, .nodes = nodes});
    Host host(table);
    const auto ctx = MakeContext();
    const auto height = [&] {
        return host.measureOf(ctx, {400.0f, kInfinite}).y;
    };
    CHECK(height() == 92.0f);

    host.place(ctx, {0.0f, 0.0f, 400.0f, 200.0f});
    CHECK(host.click(200.0f, 30.0f));
    CHECK(height() == 69.0f);

    nodes[1].cells = {"4"};
    table.update({.columns = {"Name", "Value"}, .nodes = nodes});
    CHECK(height() == 69.0f);

    nodes[0].id = "renamed";
    table.update({.columns = {"Name", "Value"}, .nodes = nodes});
    CHECK(height() == 92.0f);
}

TEST_CASE("Tree rows indent their label column by depth", "[core][sakura][table]") {
    auto nodes = Tree();
    Sakura::Table table;
    table.update({.columns = {"Name", "Value"}, .nodes = nodes, .showHeaders = false});
    Host host(table);
    const auto ctx = MakeContext();

    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}).x == 39.0f + 12.0f + 12.0f);

    nodes[0].open = true;
    table.update({.columns = {"Name", "Value"}, .nodes = nodes, .showHeaders = false});
    CHECK(host.measureOf(ctx, {kInfinite, kInfinite}).x == 59.0f + 12.0f + 12.0f);
}

TEST_CASE("Table cursor marks expandable rows and resizable column borders", "[core][sakura][table]") {
    Sakura::Table tree;
    tree.update({.columns = {"Name", "Value"}, .nodes = Tree()});
    Host treeHost(tree);
    treeHost.place(MakeContext(), {0.0f, 0.0f, 400.0f, 200.0f});

    CHECK(Peek::CursorAt(tree, {200.0f, 30.0f}) == SurfaceCursor::Hand);
    CHECK(Peek::CursorAt(tree, {200.0f, 53.0f}) == SurfaceCursor::Default);
    CHECK(Peek::CursorAt(tree, {200.0f, 10.0f}) == SurfaceCursor::Default);
    CHECK_FALSE(Peek::CursorAt(tree, {200.0f, 150.0f}).has_value());

    Cells cells;
    Sakura::Table table;
    auto config = ProbeRows(cells, 1);
    config.showHeaders = false;
    config.resizable = true;
    table.update(config);
    Host host(table);
    host.place(MakeContext(), {10.0f, 0.0f, 400.0f, 100.0f});

    CHECK(Peek::CursorAt(table, {113.0f, 10.0f}) == SurfaceCursor::ResizeEW);
    CHECK(Peek::CursorAt(table, {106.0f, 10.0f}) == SurfaceCursor::ResizeEW);
    CHECK(Peek::CursorAt(table, {120.0f, 10.0f}) == SurfaceCursor::Default);
    CHECK(Peek::CursorAt(table, {10.0f, 10.0f}) == SurfaceCursor::Default);
}

TEST_CASE("Dragging a resizable column border overrides the column width", "[core][sakura][table]") {
    Cells cells;
    Sakura::Table table;
    auto config = ProbeRows(cells, 1);
    config.showHeaders = false;
    config.resizable = true;
    table.update(config);
    Host host(table);
    const auto ctx = MakeContext();
    host.place(ctx, {10.0f, 0.0f, 400.0f, 100.0f});
    REQUIRE(cells.probes[1]->laidFrame.x == 116.0f);

    CHECK(host.send(Mouse(MouseEventType::Click, 111.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, 161.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Release, 161.0f, 10.0f)));
    host.place(ctx, {10.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[0]->laidFrame.width == 138.0f);
    CHECK(cells.probes[1]->laidFrame == Rect{166.0f, 4.0f, 238.0f, 15.0f});
    CHECK(cells.probes[0]->count(MouseEventType::Click) == 0);

    host.send(Mouse(MouseEventType::Move, 300.0f, 10.0f));
    host.place(ctx, {10.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[1]->laidFrame.x == 166.0f);

    CHECK(host.send(Mouse(MouseEventType::Click, 160.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, -100.0f, 10.0f)));
    host.send(Mouse(MouseEventType::Cancel, -100.0f, 10.0f));
    host.place(ctx, {10.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[0]->laidFrame.width == 20.0f);
}

TEST_CASE("Table borders are not draggable unless the table is resizable", "[core][sakura][table]") {
    Cells cells;
    Sakura::Table table;
    auto config = ProbeRows(cells, 1, false);
    config.showHeaders = false;
    table.update(config);
    Host host(table);
    const auto ctx = MakeContext();
    host.place(ctx, {10.0f, 0.0f, 400.0f, 100.0f});

    CHECK_FALSE(host.send(Mouse(MouseEventType::Click, 111.0f, 10.0f)));
    host.send(Mouse(MouseEventType::Move, 161.0f, 10.0f));
    host.send(Mouse(MouseEventType::Release, 161.0f, 10.0f));
    host.place(ctx, {10.0f, 0.0f, 400.0f, 100.0f});
    CHECK(cells.probes[1]->laidFrame.x == 116.0f);
    CHECK(Peek::CursorAt(table, {111.0f, 10.0f}) == SurfaceCursor::Default);
}

TEST_CASE("Wrapped table text grows the row to fit the column", "[core][sakura][table]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    Sakura::Table table;
    Host host(table);
    Sakura::Table::Config config{
        .columns = {"Key", "Value"},
        .rows = {{"Driver", "alpha beta gamma delta epsilon zeta eta theta"}},
        .fixedColumnWidths = {80.0f, 120.0f},
        .showHeaders = false,
    };

    table.update(config);
    CHECK(host.measureOf(ctx, {200.0f, kInfinite}).y == 23.0f);

    config.wrapped = true;
    table.update(config);
    const F32 wrapped = host.measureOf(ctx, {200.0f, kInfinite}).y;
    CHECK(wrapped > 23.0f);
    CHECK(static_cast<U64>(wrapped - 8.0f) % 15 == 0);
}

TEST_CASE("Identical table updates settle without rebuilds", "[core][sakura][table]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    Sakura::Table table;
    Host host(table);
    Frames frames(host, {0.0f, 0.0f, 400.0f, 200.0f}, ctx);
    const auto update = [&] {
        table.update({.columns = {"Name", "Value"}, .nodes = Tree()});
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;
    for (int frame = 0; frame < 4; ++frame) {
        CHECK(frames.run(update).clean());
    }

    host.click(200.0f, 30.0f);
    CHECK(frames.settle(update));
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Column resize moves that keep the same width stay clean", "[core][sakura][table]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    Cells cells;
    auto config = ProbeRows(cells, 1);
    config.showHeaders = false;
    config.resizable = true;
    Sakura::Table table;
    Host host(table);
    Frames frames(host, {10.0f, 0.0f, 400.0f, 100.0f}, ctx);
    const auto update = [&] {
        table.update(config);
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;

    CHECK(host.send(Mouse(MouseEventType::Click, 111.0f, 10.0f)));
    CHECK(host.send(Mouse(MouseEventType::Move, 161.0f, 10.0f)));
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(frames.settle(update));
    for (int frame = 0; frame < 3; ++frame) {
        CHECK(host.send(Mouse(MouseEventType::Move, 161.0f, 10.0f)));
        CHECK(frames.run(update).clean());
    }

    CHECK(host.send(Mouse(MouseEventType::Move, 171.0f, 10.0f)));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(host.send(Mouse(MouseEventType::Move, 171.0f, 10.0f)));
    CHECK(frames.run(update).clean());
    CHECK(host.send(Mouse(MouseEventType::Release, 171.0f, 10.0f)));
    CHECK(cells.probes[1]->laidFrame.x == 176.0f);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Table cell text changes repaint without restructuring", "[core][sakura][table][retention]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    ctx.palette = DistinctPalette();
    Sakura::Table table;
    Host host(table);
    Frames frames(host, {0.0f, 0.0f, 400.0f, 200.0f}, ctx);
    std::string value = "1.0";
    const auto update = [&] {
        table.update({.columns = {"Name", "Value"}, .rows = {{"Gain", value}, {"Rate", "2.0"}}});
    };
    REQUIRE(frames.settle(update));
    CHECK(Retains(frames, update));
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    value = "3.0";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    value = "a much longer value";
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.rebuilds == rebuilds);
    CHECK(frames.restructures == restructures);
}

TEST_CASE("Adding table rows restructures once and removing them parks the cells", "[core][sakura][table][retention]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    ctx.palette = DistinctPalette();
    Sakura::Table table;
    Host host(table);
    Frames frames(host, {0.0f, 0.0f, 400.0f, 200.0f}, ctx);
    std::vector<std::vector<std::string>> rows = {{"Gain", "1.0"}};
    const auto update = [&] {
        table.update({.columns = {"Name", "Value"}, .rows = rows});
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    rows.push_back({"Rate", "2.0"});
    const auto added = frames.run(update);
    CHECK_FALSE(added.rebuilt);
    CHECK(added.restructured);
    CHECK(frames.restructures == restructures + 1);
    CHECK(Retains(frames, update));

    rows.pop_back();
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));

    rows.push_back({"Rate", "4.0"});
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == restructures + 1);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Adding table cell rows restructures once and removing them parks the cells",
          "[core][sakura][table][retention]") {
    MonoFontWindow window;
    auto ctx = MakeContext();
    ctx.render = &window;
    ctx.palette = DistinctPalette();
    Sakura::Button first;
    Sakura::Button second;
    Sakura::Table table;
    Host host(table);
    Frames frames(host, {0.0f, 0.0f, 400.0f, 200.0f}, ctx);
    std::vector<std::vector<Sakura::Component*>> cellRows = {{&first}};
    const auto update = [&] {
        first.update({.str = "First"});
        second.update({.str = "Second"});
        table.update({.columns = {"Action"}, .cellRows = cellRows});
    };
    REQUIRE(frames.settle(update));
    const U64 rebuilds = frames.rebuilds;
    const U64 restructures = frames.restructures;

    cellRows.push_back({&second});
    const auto added = frames.run(update);
    CHECK_FALSE(added.rebuilt);
    CHECK(added.restructured);
    CHECK(Retains(frames, update));

    cellRows.pop_back();
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(Retains(frames, update));
    CHECK(frames.restructures == restructures + 1);
    CHECK(frames.rebuilds == rebuilds);
}

TEST_CASE("Tree table row hover clears on a cancelled pointer and returns on the next move",
          "[core][sakura][table][retention]") {
    Scene scene;
    Sakura::Table table;
    Place place(table, {10.0f, 10.0f, 300.0f, 100.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    const auto update = [&] {
        table.update({.columns = {"Key", "Value"},
                      .nodes = {{.id = "a", .label = "a", .cells = {"1"}}, {.id = "b", .label = "b", .cells = {"2"}}},
                      .showHeaders = false});
    };
    const auto highlighted = [&] {
        const auto hovered = frames.ctx.color("header_hovered");
        for (const auto& leaf : Drawn(place)) {
            if (const auto* box = std::get_if<Sakura::Box::Config>(&leaf.config)) {
                for (const auto& instance : box->instances) {
                    if (instance.visible && instance.backgroundColor == hovered) {
                        return true;
                    }
                }
            }
        }
        return false;
    };
    REQUIRE(frames.settle(update));
    REQUIRE_FALSE(highlighted());

    place.send(Mouse(MouseEventType::Move, 50.0f, 15.0f));
    CHECK(RepaintOnly(frames.run(update)));
    REQUIRE(highlighted());
    CHECK(frames.run(update).clean());

    place.send(Mouse(MouseEventType::Cancel, 50.0f, 15.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK_FALSE(highlighted());
    CHECK(frames.run(update).clean());

    place.send(Mouse(MouseEventType::Move, 50.0f, 15.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK(highlighted());
    CHECK(frames.run(update).clean());
}

TEST_CASE("A cancelled column resize clears the tree row hover it started on", "[core][sakura][table][retention]") {
    Scene scene;
    Sakura::Table table;
    Place place(table, {10.0f, 10.0f, 300.0f, 100.0f});
    Frames frames(place, kScreen, scene.context());
    frames.verify = false;
    const auto update = [&] {
        table.update({.columns = {"Key", "Value"},
                      .nodes = {{.id = "a", .label = "a", .cells = {"1"}}, {.id = "b", .label = "b", .cells = {"2"}}},
                      .showHeaders = false,
                      .resizable = true});
    };
    const auto highlighted = [&] {
        const auto hovered = frames.ctx.color("header_hovered");
        for (const auto& leaf : Drawn(place)) {
            if (const auto* box = std::get_if<Sakura::Box::Config>(&leaf.config)) {
                for (const auto& instance : box->instances) {
                    if (instance.visible && instance.backgroundColor == hovered) {
                        return true;
                    }
                }
            }
        }
        return false;
    };
    REQUIRE(frames.settle(update));
    std::optional<F32> border;
    for (F32 x = 12.0f; x < 308.0f && !border.has_value(); x += 1.0f) {
        if (Peek::CursorAt(table, {x, 15.0f}) == SurfaceCursor::ResizeEW) {
            border = x;
        }
    }
    REQUIRE(border.has_value());

    place.send(Mouse(MouseEventType::Move, *border, 15.0f));
    frames.run(update);
    REQUIRE(highlighted());
    CHECK(place.send(Mouse(MouseEventType::Click, *border, 15.0f)));
    frames.run(update);

    place.send(Mouse(MouseEventType::Cancel, *border, 15.0f));
    CHECK(RepaintOnly(frames.run(update)));
    CHECK_FALSE(highlighted());
    CHECK(frames.run(update).clean());
}
